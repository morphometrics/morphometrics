import numpy as np
from skimage.morphology import binary_dilation, cube, square
from skimage.segmentation import find_boundaries

from morphometrics.types import BinaryImage, LabelImage


def make_boundary_mask(
    label_image: LabelImage, boundary_dilation_size: int = 0
) -> BinaryImage:
    """Get the mask of the internal boundary pixels of a label image.

    Parameters
    ----------
    label_image : LabelImage
        The label image containing the segmented objects.
    boundary_dilation_size : int
        The size of the morphological dilation to apply to expand the edge
        size. If set to 0, no dilation is performed (i.e., boundaries are
        one pixel wide). The default value is 0.
    """
    boundary_mask = find_boundaries(label_image, mode="inner")

    if boundary_dilation_size > 0:
        if label_image.ndim == 2:
            footprint = square(boundary_dilation_size)
        elif label_image.ndim == 3:
            footprint = cube(boundary_dilation_size)
        else:
            raise ValueError("image must be 2D or 3D")
        boundary_mask = binary_dilation(boundary_mask, footprint=footprint)

        # set the boundary mask pixels expanded outside the original objects to 0
        boundary_mask[np.logical_not(label_image.astype(bool))] = False
    return boundary_mask
