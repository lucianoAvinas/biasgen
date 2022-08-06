import numpy as np
import matplotlib.pyplot as plt

from collections.abc import Iterable
from biasgen.typing_utils import Array
from typing import Union, Type, Literal as L
from biasgen.coil_setup import CoilSegments, cast_to_tuple

SAVE_DPI = 300


def view_coil_positioning(image_volume: Array[float, ('...')], coil_list: Iterable[Type[CoilSegments]], 
                          topdown_axis: int = 0, pnt_sz: float = 2, ln_wd: float = 2, 
                          full_save_path: str = None):
    """
        image_volume: 3 axes array of image slices
        coil_list: list of CoilSegments to plot
        topdown_axis: axis not used in 2D slice. Provides a topdown view from this axis
        pnt_size: size of start and end point radius. Size is relative to image dimensions
        ln_wd: size of line thickness between points. Size is relative to image dimensions
        full_save_path: full path and file name to save to
    """

    img_shape = image_volume.shape
    d = len(img_shape)

    fig, ax = plt.subplots(1, 1)
    fig.patch.set_facecolor('lightgray')

    ax.imshow(np.take(image_volume, img_shape[topdown_axis]//2, topdown_axis), 'gray')
    ax.axis('off')

    coil_list = cast_to_tuple(coil_list)

    for coil in coil_list:
        p0_t, p1_t = coil.get_point_pairs()
        for p0, p1 in zip(p0_t, p1_t):
            p0_out = np.hstack((p0[:topdown_axis], p0[topdown_axis+1:]))
            p1_out = np.hstack((p1[:topdown_axis], p1[topdown_axis+1:]))

            if all(p0_out == p1_out):
                circ = plt.Circle((p0_out[1], p0_out[0]), 
                                       radius=pnt_sz, fc='r')
                ax.add_patch(circ)
            else:
                ax.plot([p0_out[1], p1_out[1]], [p0_out[0], p1_out[0]],
                         color='red', linewidth=ln_wd)
        ax.autoscale()

    if full_save_path is None:
        plt.show()
    else:
        plt.savefig(full_save_path, bbox_inches='tight', dpi=SAVE_DPI)
        plt.close(fig)


# complex plotting: https://stackoverflow.com/a/56595416
def imshow_cmplx(ax: plt.Axes, cmplx_dat: Array[complex, ('...')], 
                 cmin: float = None, cmax: float = None):
    """
        ax: a matplotlib.Axes object to plot on
        cmplx_dat: a matrix (2 axes) of complex values
        cmin: minimum absolute value to plot with brightness
        cmax: maximum absolute value to plot with brightness
    """

    abs_dat = np.abs(cmplx_dat)
    cmin = abs_dat.min() if cmin is None else cmin
    cmax = abs_dat.max() if cmax is None else cmax

    bkg = np.zeros((*cmplx_dat.shape, 4))
    if cmax != cmin:
        bkg[:,:,-1] = np.clip(1 - (abs_dat - cmin)/(cmax - cmin), 0, 1)

    ax.imshow(np.angle(cmplx_dat), cmap='hsv', interpolation='none')  # phase data
    ax.imshow(bkg)  # modulus data


def view_center_axes(image_volumes: Array[Union[float, complex], ('T','...')], 
                     mask_volume: Array[bool, ('...')], title_names: Iterable[str] = None, 
                     full_save_path: str = None):
    """
        image_volumes: collection of 3 axes arrays
        mask: boolean array with same spatial size as image volume. Emphasizes important regions
        title_names: list of subplot title names
        full_save_path: full path and file name to save to
    """

    if len(image_volumes.shape) == 3:
        image_volumes = np.expand_dims(image_volumes, 0)

    elif len(image_volumes.shape) < 3:
        raise ValueError(f"Input 'image_volume' should have at least 3 axes")

    elif abs(len(image_volumes.shape) - len(mask_volume.shape)) > 1:
        raise ValueError("Inputs 'image_volume' and 'mask_volume' differ by more than 2 axes")

    T, *dims = image_volumes.shape
    fig, axs = plt.subplots(T, 3)

    # promote axes
    axs = np.expand_dims(axs,0) if T == 1 else axs

    axes_slices = []
    cmin, cmax = np.inf, -np.inf
    for i in range(3):
        imgs_slice = np.take(image_volumes, dims[i]//2, i+1)
        mask_slice = np.take(mask_volume, dims[i]//2, i)

        imgs_mask_sl, cmin_i, cmax_i = apply_mask(imgs_slice, mask_slice)

        cmin = min(cmin, cmin_i)
        cmax = max(cmax, cmax_i)
        axes_slices.append(imgs_mask_sl)

    is_complex = any(np.iscomplexobj(ax_sl) for ax_sl in axes_slices)
    title_names = cast_to_tuple(title_names, 3*T)

    # if only 3 unique titles, fill the rest with None
    if len(title_names) == 3:
        title_names = list(title_names) + [None] * 3*(T-1)

    title_names = np.array(title_names).reshape(T, 3)

    for i in range(T):
        for j in range(3):
            ax = axs[i,j]
            tl_name = title_names[i,j]
            img_slice = axes_slices[j][i]

            if is_complex:
                imshow_cmplx(ax, img_slice, cmin, cmax)
            else:        
                ax.imshow(img_slice, vmin=cmin, vmax=cmax, interpolation='none')

            ax.axis('off')
            ax.set_title(tl_name)

    if full_save_path is None:
        plt.show()
    else:
        plt.savefig(full_save_path, bbox_inches='tight', dpi=SAVE_DPI)
        plt.close(fig)


def apply_mask(image_collection: Array[Union[float, complex], ('...', 'X')],
               mask: Array[bool, 'X'] = None):
    """
        image_collection: arbitrary size array of float or complex data type
        mask: boolean array with same tail axes size as image_collection.
              Is used to compute maximum and minimum values of image_collection
              within the mask. For complex data the modulus is used for min/max values.
    """

    if mask is None:
        masked_collection = image_collection
        abs_values = np.abs(masked_collection)
    else:
        masked_values = image_collection[...,mask]

        masked_collection = np.zeros_like(image_collection)
        masked_collection[...,mask] = masked_values

        abs_values = np.abs(masked_values)

    return masked_collection, abs_values.min(), abs_values.max()
