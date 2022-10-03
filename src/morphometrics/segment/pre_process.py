from typing import List, Optional

import numpy as np
from skimage.exposure import rescale_intensity
from skimage.transform import rescale


def pre_process_image(
    raw_image: np.ndarray,
    raw_pixel_size: List[float],
    target_pixel_size: List[float],
    segmentation_mask: Optional[np.ndarray] = np.zeros(1),
):
    """Prepare an image for segmentation.

    Parameters
    ----------
    raw_image : np.ndarray
        The image to be pre-processed.
        All other pixels are set to 0.
    raw_pixel_size : List[float]
        The pixel size in each dimension of the raw image.
    target_pixel_size : List[float]
        The pixel size in each dimension of the segmentaton model was trained on.
    segmentation_mask : Optional[np.ndarray] = np.zeros(1)
        A mask set to non-zero in the regions where segmentation should be performed.
        All other pixels are set to 0.

    Returns
    -------
    rescaled_image : np.ndarray
        The pre-processed image.
    """
    # mask the tissue
    if segmentation_mask.any():
        raw_image[np.invert(segmentation_mask.astype(bool))] = 0

    # calculate the scale factor for the pixel size
    scale_factor = np.asarray(target_pixel_size) / np.asarray(raw_pixel_size)

    # rescale the image pixel size
    rescaled_image = rescale(raw_image, scale_factor)

    # rescale the intensities
    rescaled_image = rescale_intensity(rescaled_image, out_range=(0, 1))

    return rescaled_image
