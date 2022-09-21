import numpy as np
from skimage import measure
from skimage.transform import resize


def post_process_image(
    segmented_image: np.ndarray, segmentation_mask: np.ndarray, threshold: int
):
    """
    Post-processing the segmented image.

    Parameters
    ---------
    segmented_image: np.ndarray
        the segmented image (same object with the same pixel value)
    segmentation_mask: np.ndarray
        A mask set to non-zero in the regions where segmentation should be kept.
        All other pixels are set to 0.
    threshold: int
        the minimal number of pixels, representing the minimal size of objects that we keep.
        Any objects smaller than that size will be deleted (pixel value set to 0).

    Returns
    -------
    postprocessed_image: np.ndarray
        the post-processed image
    """
    # resize the segmented image to match the mask size
    resized_image = resize(segmented_image, segmentation_mask.shape, order=0)

    # mask the background
    resized_image[np.invert(segmentation_mask.astype(bool))] = 0

    # filter out the segmented objects blow the threshold size
    labeled = measure.label(resized_image)
    props = measure.regionprops(labeled)
    postprocessed_image = labeled.copy()
    for idx in range(0, len(props)):
        if props[idx].area < threshold:
            postprocessed_image[labeled == props[idx].label] = 0

    return postprocessed_image