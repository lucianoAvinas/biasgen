import itertools

from biasgen.config import cupy_path
from collections.abc import Iterable
from typing import Any, Literal as L
from biasgen.typing_utils import Array

# CPU specification on default
import numpy as ap
from scipy.ndimage import zoom
_USE_GPU = False


def use_gpu(assign_bool: bool):
    """Set preferred array package in field_calc.py"""
    global _USE_GPU, ap, zoom
    
    if assign_bool:
        if cupy_path is None:
            raise ModuleNotFoundError('CuPy array package not found')

        import cupy as ap
        from cupyx.scipy.ndimage import zoom
        _USE_GPU = True

    else:
        import numpy as ap
        from scipy.ndimage import zoom
        _USE_GPU = False


def line_biotsav(x: Array[float, ('n', L[3])], p0: Array[float, ('T', L[3])], 
                 p1: Array[float, ('T', L[3])]):
    """
        x: points in space to evaluate the magnetic field
        p0: a list of line segment starting points
        p1: a list of line segment end points
    """

    # promote axis
    x = ap.expand_dims(x, 0)
    p0 = ap.expand_dims(p0, 1)
    p1 = ap.expand_dims(p1, 1)

    # compute intermediaries
    a = p1 - p0
    b = x - p0
    c = x - p1

    a_norm = ap.linalg.norm(a, axis=2, keepdims=True)
    b_norm = ap.linalg.norm(b, axis=2, keepdims=True)
    c_norm = ap.linalg.norm(c, axis=2, keepdims=True)

    vec = ap.cross(a, b)

    a_unit = a / a_norm
    b_unit = b / b_norm
    c_unit = c / c_norm

    numer = last_dot(b_unit - c_unit, a_unit, keepdims=True)
    denom = a_norm**2 * b_norm**2 - last_dot(b, a, keepdims=True)**2

    return vec *  numer / denom 


def last_dot(x: Array[Any, ('...', 'n')], y: Array[Any, ('...', 'n')], keepdims=False):
    """Takes dot product on last axis
        x: input 1 to dot product
        y: input 2 to dot product
        keepdims: whether to keep a dummy axis on the last dimension
    """

    dot_result = ap.einsum('...i,...i->...', x, y)

    if keepdims:
        dot_result = ap.expand_dims(dot_result, -1)
    
    return dot_result


def nan_cleanup(x: Array[Any, '...']):
    """Send all nan, inf values to 0"""
    ap.nan_to_num(x, False, 0, 0, 0)


def calc_mag_fields(img_dims: tuple[int, int, int], p01_t: tuple[Array[float, ('T', L[3])]], 
                    scale_fctr: float, smooth_order: int, sum_coils: bool):
    """
        img_dims: output image dimensions for scale_fctrs <= 1. Will be appropriately scaled
                  for scale_fcts > 1
        p01_t: an array of starting and ending points for 'T' line segments
        scale_fctr: a downscale/upscale factor to use when calculating magnetic fields
        smooth_order: an interpolation order to use when downscale/upscaling
        sum_coils: reduces the coil axis by summing every segment contribution. This will
                   speed up downscale/upscale calculations for large coil_lists
    """

    p0_t, p1_t = ap.array(p01_t)
    scaled_dims = tuple(int(ni*scale_fctr) for ni in img_dims)

    # produce coordinates. collect per coordinate values then transpose to (n,3).
    x = ap.indices(scaled_dims).reshape(3,-1).T

    # max normalize for numerical accuracy
    M = max(scaled_dims)
    coil_fields = line_biotsav(x/M, p0_t/M, p1_t/M)

    # evaluations which lie exactly on coil segments will be ignored
    nan_cleanup(coil_fields)

    # un-flatten the spatial axes
    T = p0_t.shape[0]
    coil_fields = coil_fields.reshape(T, *scaled_dims, 3)

    if sum_coils:  # contract coil contributions
        coil_fields = coil_fields.sum(axis=0)

        if scale_fctr < 1:
            field = ap.zeros((*img_dims, 3))

            # interpolate mag values on each spatial axis
            for i in range(3):  
                field[...,i] = scale_and_pad(coil_fields[...,i], img_dims, 
                                             scale_fctr, smooth_order)
        else:
            field = coil_fields
    else:
        if scale_fctr < 1:
            field = ap.zeros((T, *img_dims, 3))

            # interpolate mag values on each spatial axis (per coil segment)
            for t,i in itertools.product(range(T), range(3)):
                field[t,...,i] = scale_and_pad(coil_fields[t,...,i], img_dims, 
                                               scale_fctr, smooth_order)
        else:
            field = coil_fields
    
    if _USE_GPU:  # send to CPU if necessary
        field = ap.asnumpy(field)

    return field


def scale_and_pad(scale_img: Array[float, ('...')], orig_dims: Iterable[int], 
                  scale_fctr: float, smooth_order: int):
    """
        scale_img: downscaled image
        orig_dim: dimensions of original image
        scale_fctr: scaling factor used for downscaled image
        smooth_order: order of smoothness to interpolate with
    """
        
    # upscale image by 1 / downscale factor
    img = zoom(scale_img, 1/scale_fctr, order=smooth_order, mode='nearest')

    # calculate dimension offset between upsampled image and original image
    dim_diff = tuple(o-n for o,n in zip(orig_dims, img.shape))

    # pad smaller dims
    img = ap.pad(img, tuple((0, max(dim, 0)) for dim in dim_diff), 'edge')

    # crop bigger dims
    img = img[tuple([slice(0, dim if dim < 0 else None) for dim in dim_diff])]

    return img


def calc_coil_sens(img_dims: tuple[int, int, int], p01_t: Array[float, (L[2], 'T', L[3])], 
                   omega: Array[float, ('m', L[3])], uv: tuple[Array[float, L[3]]], 
                   scale_fctr: float, smooth_order: int):
    """
        img_dims: output image dimensions for scale_fctrs <= 1. Will be appropriately scaled
                  for scale_fcts > 1
        p01_t: an array of starting and ending points for 'T' line segments
        omega: grid of sampling coordinates in frequency domain
        uv: magnetic field readout and phase encoding directions
        scale_fctr: a downscale/upscale factor to use when calculating magnetic fields
        smooth_order: an interpolation order to use when downscale/upscaling
        sum_coils: reduces the coil axis by summing every segment contribution. This will
                   speed up downscale/upscale calculations for large coil_lists
    """
    omega = ap.array(omega)
    p0_t, p1_t = ap.array(p01_t)
    scaled_dims = tuple(int(ni*scale_fctr) for ni in img_dims)

    # produce coordinates. collect per coordinate values then transpose to (n,3).
    x = ap.indices(scaled_dims).reshape(3,-1).T

    # max normalize for numerical accuracy
    M = max(scaled_dims)
    x, p0_t, p1_t = x/M, p0_t/M, p1_t/M

    # norm intermediaries
    a = p1_t - p0_t 
    a_norm = ap.linalg.norm(a, axis=1)
    omega_norm = ap.linalg.norm(omega, axis=1)

    # calculate segment specific rotations
    u, v = ap.array(uv)
    uv_y, omega_z = uv_rots(u, v, a, a_norm, omega)

    # calculate sensitivity values
    p_mid = (p1_t + p0_t)/2
    sens = sinusoid_model(uv_y, a, p_mid, a_norm, omega, omega_z, omega_norm, x)

    # un-flatten the spatial axes
    T = p0_t.shape[0]
    sens = sens.reshape(T, *scaled_dims)

    if scale_fctr < 1:  # upscale sensitivity values
        sens = ap.array([scale_and_pad(sens_t, img_dims, scale_fctr, 
                                       smooth_order) for sens_t in sens])

    if _USE_GPU:  # send to CPU if necessary
        sens = ap.asnumpy(sens)

    return sens


def uv_rots(u: Array[float, L[3]], v: Array[float, L[3]], a: Array[float, ('T', L[3])], 
            a_norm: Array[float, 'T'], omega: Array[float, ('m', L[3])]):
    """
        u: magnetic field readout direction
        v: magnetic field phase encoding direction
        a: collection of line segment vectors. Holds length and direction information
        a_norm: norms of the vector line segments
        omega: grid of sampling coordinates in frequency domain
    """
   
    # promote axes
    a = ap.expand_dims(a, 1)
    u = ap.expand_dims(u, (0,1))
    v = ap.expand_dims(v, (0,1))
    omega = ap.expand_dims(omega, 0)
    a_norm = ap.expand_dims(a_norm, (1,2))

    # compute intermediaries
    a_hat = a/a_norm
    omega_z = last_dot(omega, a_hat, keepdims=True)  # scalar component of omega in z-basis
    omega_xdir = omega - omega_z * a_hat             # x-basis vector contribution of omega

    # calculate reciprocal omega_x norm
    omegax_rnorm = ap.linalg.norm(omega_xdir, axis=2, keepdims=True)

    # mask zero values when dividing
    nonzero_vals = omegax_rnorm > 0
    omegax_rnorm[nonzero_vals] = 1/omegax_rnorm[nonzero_vals] 

    # make omega_x unit norm
    # ignore divide by zero as a full align with 'a' will not contribute
    omega_xdir = omega_xdir*omegax_rnorm

    # construct y basis directions
    y_dirs = ap.cross(a_hat, omega_xdir)

    u_y = last_dot(y_dirs, u)
    v_y = last_dot(y_dirs, v)

    return u_y + 1j*v_y, omega_z[...,0]  # index into the dummy axes of omega_z
    

def sinusoid_model(uv_y: Array[complex, ('T', 'm')], a: Array[float, ('T',L[3])],
                   p_mid: Array[float, ('T', L[3])], a_norm: Array[float, 'T'], 
                   omega: Array[float, ('m', L[3])],  omega_z: Array[float, ('T', 'm')], 
                   omega_norm: Array[float, 'm'], x: Array[float, ('n', L[3])]):
    """
        uv_y: y basis component of readout and phase encoding directions. Basis y
              depends on each line segment vector
        a: collection of line segment vectors. Holds length and direction information
        p_mid: collection of line segment mid-points
        a_norm: norms of the vector line segments
        omega: grid of sampling coordinates in frequency domain
        omega_z: omega contribution along z basis component
        omega_norm: coordinate norms for sampling grid
        x: points in space to evaluate the sensitivity calculation
    """

    # promote axes
    a_norm = ap.expand_dims(a_norm, 1)
    omega_norm = ap.expand_dims(omega_norm, 0)

    omega_x = ap.sqrt(ap.clip(1 - (omega_z/omega_norm)**2, 0, 1)) 
    omega_rad = omega_x * ap.sinc(a_norm*omega_z/(2*ap.pi)) / omega_norm
    complex_comp = 1j * uv_y * ap.exp(1j*last_dot(p_mid[:,None], omega[None]))

    # promote axes
    x = ap.expand_dims(x, 1)
    omega = ap.expand_dims(omega, 0)

    inv_ft = ap.exp(-1j * last_dot(omega, x))

    return ap.sum(complex_comp[:,None] * inv_ft[None], axis=2)
