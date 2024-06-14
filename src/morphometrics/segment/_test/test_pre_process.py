import numpy as np

from ..pre_process import pre_process_image


def test_pre_process_image_mask():
    """
    Test if pre_process_image() is masking correctly.
    """
    raw_image = np.array([1, 2, 3])
    raw_pixel_size = 1
    target_pixel_size = 1
    segmentation_mask = np.array([0, 1, 1])

    preprocessed_image = pre_process_image(
        raw_image, raw_pixel_size, target_pixel_size, segmentation_mask
    )

    true_preprocessed_image = np.array([0, 2, 3])
    assert (
        preprocessed_image == true_preprocessed_image
    ), "The image is not masked correctly."


def test_pre_process_image():
    """
    Test pre_process_image() without mask.
    """
    raw_image = np.random.rand(10, 10, 10) * 255
    raw_pixel_size = (10, 10, 10)
    target_pixel_size = (2, 2, 2)

    preprocessed_image = pre_process_image(raw_image, raw_pixel_size, target_pixel_size)
    assert (
        preprocessed_image.shape == target_pixel_size
    ), "The image is not rescaled correctly."
    assert (
        preprocessed_image.all() <= 1 and preprocessed_image.all() >= 0
    ), "The intensity of the image is not scaled correctly."
