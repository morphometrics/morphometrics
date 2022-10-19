from typing import List, Optional

import numpy as np
from skimage import measure
from skimage.transform import resize


def post_process_image(
    segmented_image: np.ndarray,
    threshold: int,
    segmentation_mask: Optional[np.ndarray] = None,
    shape: Optional[List] = None,
):
    """
    Post-processing the segmented image.

    Parameters
    ---------
    segmented_image: np.ndarray
        the segmented image (same object with the same pixel value)
    threshold: int
        the minimal number of pixels, representing the minimal size of objects that we keep.
        Any objects smaller than that size will be deleted (pixel value set to 0).
    segmentation_mask: Optional[np.ndarray] = None
        A mask set to non-zero in the regions where segmentation should be kept.
        All other pixels are set to 0.
    shape: Optional[List] = None
        the initial shape of the image, to resize the rescaled image back to its initial size,
        since we need to filter the size upon threshold, which is the number of pixels.

    Returns
    -------
    postprocessed_image: np.ndarray
        the post-processed image
    """
    # resize the segmented image to match the mask size
    if segmentation_mask is not None:
        resized_image = resize(segmented_image, segmentation_mask.shape, order=0)
        # mask the background
        resized_image[np.invert(segmentation_mask.astype(bool))] = 0
    elif shape is not None:
        resized_image = resize(segmented_image, shape, order=0)

    # filter out the segmented objects blow the threshold size
    labeled = measure.label(resized_image)
    props = measure.regionprops(labeled)
    postprocessed_image = labeled.copy()
    for idx in range(0, len(props)):
        if props[idx].area < threshold:
            postprocessed_image[labeled == props[idx].label] = 0

    return postprocessed_image
