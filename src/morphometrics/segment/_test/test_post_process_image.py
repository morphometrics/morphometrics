import numpy as np

from ..post_process import post_process_image


def test_post_process_image_mask():
    """
    Test post_process_image() with a mask.
    """
    segmented_image = np.array([1])
    threshold = 4
    segmentation_mask = np.array([1, 1, 1, 0, 0])

    postprocessed_image = post_process_image(
        segmented_image, threshold, segmentation_mask
    )
    true_postprocessed_image = np.array([0, 0, 0, 0, 0])

    assert (
        true_postprocessed_image == postprocessed_image
    ), "The postprocessing with a mask is wrong."


def test_post_process_image_no_mask():
    """
    Test post_process_image() without a mask.
    """
    segmented_image = np.array([1])
    threshold = 4
    shape = 5

    postprocessed_image = post_process_image(segmented_image, threshold, None, shape)
    true_postprocessed_image = np.array([1, 1, 1, 1, 1])
    assert (
        true_postprocessed_image == postprocessed_image
    ), "The postprocessing without a mask is wrong."
