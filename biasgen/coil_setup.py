import numpy as np

from collections.abc import Iterable
from biasgen.typing_utils import Array
from typing import Any, Type, Union, Literal as L

from biasgen.field_calc import calc_mag_fields, calc_coil_sens


class CoilSegments:
    def __init__(self, coil_range: Array[float, ('d', L[2])],
                 interp_path: Array[float, ('T+1', 'd')] = None):
        """
            coil_range: defines the start to stop range of each axis
            point_path: defines the order which range should be interpolated.
                        Path of length T+1 defines T segments
        """

        self.coil_range = np.array(coil_range)

        # default path is a closed rectangle varying in first dimension
        if interp_path is None:
            d = self.coil_range.shape[0]

            all_ones = np.ones(d)
            all_zeros = np.zeros(d)
            one_one = np.array([i == 0 for i in range(d)])
            one_zero = np.array([i != 0 for i in range(d)])

            interp_path = [all_zeros, one_one, all_ones, 
                           one_zero, all_zeros]

        self.interp_path = np.array(interp_path)
        self.num_segs = self.interp_path.shape[0] - 1

    def get_point_paths(self, scale_fctr: float = 1.0):
        """scale_fctr: scales start and end points in the point path"""

        # define origin coordinate and a full range step
        start = self.coil_range[:,0]
        step = self.coil_range[:,1] - start

        # promote axes
        start = start[None]
        step = step[None]

        # interpolate and scale by scale factor
        point_paths = scale_fctr * (start + step * self.interp_path)

        return point_paths

    def get_point_pairs(self, scale_fctr: float = 1.0):
        """scale_fctr: scales start and end points for the point pairs"""

        point_paths = self.get_point_paths(scale_fctr)
        L = point_paths.shape[0]

        return np.array([point_paths[:L-1], point_paths[1:L]])

    def get_coord_paths(self, scale_fctr: float = 1.0):
        """scale_fctr: scales start and end points for coordinate path"""
        return self.get_point_paths(scale_fctr).T


def collect_point_pairs(coil_list: Iterable[Type[CoilSegments]], scale_fctr: float):
    """
        coil_segments: a list of CoilSegment objects to separate and collect
        scale_fctr: scaling factor to be used in the collection method
    """
    path_list = [coil.get_point_pairs(scale_fctr) for coil in coil_list]
    return np.concatenate(path_list, axis=1)


def cast_to_tuple(val: Union[Any, Iterable[Any]], length: int = 1):
    """
        val: value to check whether iterable
        length: if value is scalar repeat as a tuple of length 'length'
    """
    # duck type to tuple
    try:
        iter(val)
    except TypeError:
        val = (val,) * length
    return val


def cage_constructor(n_coils: Union[int, Iterable[int]], center: tuple[int, int, int], 
                     coil_height: float, length_to_space_ratio: float, 
                     ellip_axes: tuple[float, float], angle_start: float = -90., 
                     sweep_range: float = 360., axes_increment: tuple[float, float] = (0,0)):
    """
        n_coils: specifies the number of coils to use in constructor. A list of values
                 will specify a phased coil with number of sweeps equaling list length
        center: center coordinate of the coil cage. Assumes 3d coordinates
        coil_height: height of the coil cage. By default it is assumed the first axis
                     controls height.
        length_to_space_ratio: the ratio of rectangular coil length (not height) to 
                               empty space
        ellip_axes: the semimajor axes (a,b) that the constructor will use when sweeping
        angle_start: where to start sweeping the cage. Input is in degrees
        sweep_range: how many degrees to sweep. Input is in degrees
        radius_increment: how much to increment semimajor axes (a,b) per phase
    """
    
    if length_to_space_ratio == 0:
        interp_path = [(0,0,0), (1,0,0)]
    else:
        interp_path = None

    radian_fctr = np.pi/180
    angle_start = angle_start * radian_fctr
    a0, b0 = ellip_axes

    n_coils = cast_to_tuple(n_coils)

    c1, c2, c3 = center
    c1_st, c1_en = c1 - coil_height//2, c1 + coil_height//2

    coil_geoms = []

    for i,coils_in_sweep in enumerate(n_coils):
        sweep_angle = sweep_range/coils_in_sweep * radian_fctr
        half_length_angle = sweep_angle * length_to_space_ratio / 2
        
        theta = angle_start + sweep_angle * i / len(n_coils) # phase offset

        a = a0 + i * axes_increment[0]
        b = b0 + i * axes_increment[1]

        for _ in range(coils_in_sweep):
            c2_st = c2 + a*np.cos(theta + half_length_angle)
            c3_st = c3 + b*np.sin(theta + half_length_angle)

            c2_en = c2 + a*np.cos(theta - half_length_angle)
            c3_en = c3 + b*np.sin(theta - half_length_angle)

            theta += sweep_angle

            coil = CoilSegments([(c1_st, c1_en), (c2_st, c2_en),
                                 (c3_st, c3_en)], interp_path)
            coil_geoms.append(coil)

    return coil_geoms


class SensitivitySettings:
    def __init__(self, grid_lengths: Union[int, tuple[int, int, int]],  
                 grid_spacings: Union[int, tuple[int, int, int]], 
                 grid_shifts: Union[int, tuple[int, int, int]] = (0.,0.,0.), 
                 u: Array[float, L[3]] = None, v: Array[float, L[3]] = None):
        """
            grid_lengths: specifies the lengths of the sampling grid used in the sinusoidal model
            grid_spacings: angular frequency spacing between sampled points
            grid_shifts: starting shift from the origin. Shift is reflected across each axis.
            u: magnetic field readout direction
            v: magnetic field phase encoding direction
        """

        # cast grid specifics to lists
        grid_lengths = cast_to_tuple(grid_lengths, 3)
        grid_spacings = cast_to_tuple(grid_spacings, 3)
        grid_shifts = cast_to_tuple(grid_shifts, 3)

        # check user specified lengths
        if not all(len(gi) == 3 for gi in [grid_lengths, grid_spacings, grid_shifts]):
            raise ValueError('Grid inputs should be scalar or a list of length 3.')
        
        # readout direction will be set to the third axes at default
        if u is None:
            u = (0,0,1.)
        u = np.array(u)
        u = u / np.linalg.norm(u)

        # phase encoding direction will be set to the second axes at default
        if v is None:
            v = (0,1.,0)
        v = np.array(v)
        v = v / np.linalg.norm(v)

        # confirm perpendicular
        if not np.allclose((dot_val := np.dot(u,v)), 0):
            raise ValueError('Readout u and phase encoding v should be perpendicular.\n'
                             f'Currently there is a {dot_val:.3e} overlap between u and v.')
        self.uv = (u,v)

        omegas = []
        for o_len, o_spc, o_shft in zip(grid_lengths, grid_spacings, grid_shifts):
            center_val = [] if o_len % 2 == 0 else [0]
            o_hlen = o_len//2

            omegas.append([i*o_spc - o_shft for i in range(-o_hlen, 0)] + center_val +  \
                          [i*o_spc + o_shft for i in range(1, o_hlen+1)])

        # cartesian product of grid points
        omegas = np.array(np.meshgrid(*omegas), dtype=float)

        # collect same coordinate values and then transpose
        omegas = omegas.reshape(3,-1).T

        # remove origin if included
        smallest_ind = np.argmin(np.abs(omegas).sum(axis=1))
        if np.abs(omegas[smallest_ind]).sum() == 0:
            omegas = np.delete(omegas, smallest_ind, 0)

        self.omegas = omegas


def compute_field(coil_list: Iterable[Type[CoilSegments]], img_dims: tuple[int, int, int],
                  batch_sz: int = None, sum_coils: bool = False, collect_within_coil: bool = True,
                  scale_fctr: float = 1, smooth_order: int = 3):
    """
    coil_segments: a list of CoilSegment objects used in computing magnetic field
    img_dims: output image dimensions for scale_fctrs <= 1. Will be appropriately scaled
              for scale_fcts > 1
    batch_sz: a batch size for number coil segments to compute at once. Can help with
              memory issues for large resolution or large segment jobs.
    sum_coils: reduces the coil axis by summing every segment contribution. This will
               speed up downscale/upscale calculations for large coil_lists
    collect_within_coil: collects contributions per CoilSegment object. Does nothing when
                         sum_coils is True.
    scale_fctr: scaling factor to use on the original image dimensions
    smooth_order: order of smoothness to interpolate with
    """

    # collect all coil segments and produce point pairs
    p0_t, p1_t = collect_point_pairs(coil_list, scale_fctr)

    T = p0_t.shape[0]
    batch_sz = T if batch_sz is None else int(batch_sz)
    num_batches = np.round(T/batch_sz).astype(int)

    field_shape = (*img_dims, 3) if sum_coils else (T, *img_dims, 3)
    field = np.zeros(field_shape)
    for i in range(num_batches):
        # no need for modulo operator when batching
        sub_p0t = p0_t[i*batch_sz:(i+1)*batch_sz]
        sub_p1t = p1_t[i*batch_sz:(i+1)*batch_sz]

        field_batch = calc_mag_fields(img_dims, (sub_p0t, sub_p1t), scale_fctr, 
                                      smooth_order, sum_coils)
        if sum_coils:
            field += field_batch
        else:
            field[i*batch_sz:(i+1)*batch_sz] = field_batch

    # collecting segments per coil
    if collect_within_coil and not sum_coils:
        tot_coils = 0
        field_coll = []

        for coil in coil_list:
            ns = coil.num_segs
            field_coll.append(field[tot_coils:tot_coils+ns].sum(axis=0))
            tot_coils += ns

        field = np.array(field_coll)

    return field


def field_readout(field: Array[float, ('...', 3)], uv: tuple[Array[float, L[3]]]):
    """
       field: magnetic field tensor to readout with directions u and v
       uv: magnetic field readout and phase encoding directions
    """

    spatial_dot = lambda x,y: np.einsum('...i,...i->...', x, y)

    # promote and dot the last axis
    non_spat_ax = tuple(i for i in range(len(field.shape) - 1))

    u,v = np.array(uv)
    field_real = spatial_dot(field, np.expand_dims(u, non_spat_ax))
    field_im = spatial_dot(field, np.expand_dims(v, non_spat_ax))

    return field_real + 1.0j*field_im


def compute_sensitivity(coil_list: list[Type[CoilSegments]], sens_options: SensitivitySettings,
                        img_dims: tuple[int, int, int], batch_sz: int = None, 
                        collect_within_coil: bool = True, scale_fctr: float = 1, smooth_order: int = 3):
    """
        coil_segments: a list of CoilSegment objects used in computing coil sensitivities
        sens_options: a SensitivitySettings object which determines sampling grid and magnetic readout
        img_dims: output image dimensions for scale_fctrs <= 1. Will be appropriately scaled
                  for scale_fcts > 1
        batch_sz: a batch size for number coil segments to compute at once. Can help with
                  memory issues for large resolution or large segment jobs.
        collect_within_coil: collects segments by coil group. Assumes segments within the same
                             coil group recieve signal at similar intensity scalings
        scale_fctr: scaling factor to use on the original image dimensions
        smooth_order: order of smoothness to interpolate with
    """

    # collect all coil segments and produce point pairs
    p0_t, p1_t = collect_point_pairs(coil_list, scale_fctr)

    T = p0_t.shape[0]
    batch_sz = T if batch_sz is None else int(batch_sz)
    num_batches = np.round(T/batch_sz).astype(int)

    omegas = sens_options.omegas
    uv = sens_options.uv

    sens = np.zeros((T, *img_dims), dtype=complex)
    for i in range(num_batches):
        # no need for modulo operator when batching
        sub_p0t = p0_t[i*batch_sz:(i+1)*batch_sz]
        sub_p1t = p1_t[i*batch_sz:(i+1)*batch_sz]

        sens[i*batch_sz:(i+1)*batch_sz] = calc_coil_sens(img_dims, (sub_p0t, sub_p1t), 
                                                         omegas, uv, scale_fctr, 
                                                         smooth_order)
    # collecting segments per coil
    if collect_within_coil:
        tot_coils = 0
        sens_coll = []

        for coil in coil_list:
            ns = coil.num_segs
            sens_coll.append(sens[tot_coils:tot_coils+ns].sum(axis=0))
            tot_coils += ns

        sens = np.array(sens_coll)

    return sens


def bias_sum_of_squares(sens: Array[complex, ('T', '...')]):
    """sens: coil sensitivities to use when computing sum-of-squares bias"""
    return np.linalg.norm(np.absolute(sens), axis=0)
