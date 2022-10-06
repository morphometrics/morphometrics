import numpy as np
import pyclesperanto_prototype as cle
import pytest

from morphometrics.label.image_utils import (
    _lower_triangle_to_full_array,
    expand_bounding_box,
    expand_selected_labels,
    expand_selected_labels_using_crop,
    get_mask_bounding_box_3d,
    touch_matrix_from_label_image,
)
from morphometrics.utils.environment_utils import on_ci, on_macos, on_windows

full_array = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]])
lower_triangle_array = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]])


@pytest.mark.parametrize("array", [lower_triangle_array, full_array])
def test__lower_triangle_to_full_array(array):
    result = _lower_triangle_to_full_array(array)
    expected_full_array = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]])
    np.testing.assert_equal(result, expected_full_array)


@pytest.mark.skipif(
    on_ci and (on_windows or on_macos), reason="openCL doesn't work on windows/mac CI"
)
def test_expand_selected_labels_2d():
    label_image = np.zeros((100, 100), dtype=int)
    label_image[30:70, 30:70] = 1
    label_image[45:55, 45:55] = 0

    label_image[49:51, 49:51] = 2

    expanded_image = expand_selected_labels(
        label_image=label_image, label_values_to_expand=2, expansion_amount=1
    )

    # check that the unexpanded labels were not expanded
    np.testing.assert_equal(label_image.shape, expanded_image.shape)
    np.testing.assert_equal(label_image == 1, expanded_image == 1)
    cpu_devices = cle.available_device_names(dev_type="cpu")
    print("Available CPU OpenCL devices:" + str(cpu_devices))

    assert expanded_image[50, 52] == 2

    # expand to fill the void
    expanded_image_2 = expand_selected_labels(
        label_image=label_image, label_values_to_expand=2, expansion_amount=5
    )
    np.testing.assert_equal(label_image.shape, expanded_image_2.shape)
    np.testing.assert_equal(label_image == 1, expanded_image_2 == 1)

    # check that the entire void was filled with 2
    assert np.sum(expanded_image_2 == 2) == 10**2


@pytest.mark.skipif(
    on_ci and (on_windows or on_macos), reason="openCL doesn't work on windows/mac CI"
)
def test_expand_selected_labels_3d():
    label_image = np.zeros((100, 100, 100), dtype=int)
    label_image[30:70, 30:70, 30:70] = 1
    label_image[45:55, 45:55, 45:55] = 0

    label_image[49:51, 49:51, 49:51] = 2

    expanded_image = expand_selected_labels(
        label_image=label_image, label_values_to_expand=2, expansion_amount=1
    )

    np.testing.assert_equal(label_image.shape, expanded_image.shape)
    np.testing.assert_equal(label_image == 1, expanded_image == 1)

    assert expanded_image[50, 52, 50] == 2

    # expand to fill the void
    expanded_image_2 = expand_selected_labels(
        label_image=label_image, label_values_to_expand=2, expansion_amount=5
    )
    np.testing.assert_equal(label_image.shape, expanded_image_2.shape)
    np.testing.assert_equal(label_image == 1, expanded_image_2 == 1)

    # check that the entire void was filled with 2
    assert np.sum(expanded_image_2 == 2) == 10**3


@pytest.mark.skipif(
    on_ci and (on_windows or on_macos), reason="openCL doesn't work on windows/mac CI"
)
def test_expand_selected_labels_4d():
    """Should raise a value error with label ndim > 3"""
    label_image = np.zeros((10, 10, 10, 10), dtype=int)

    with pytest.raises(ValueError):
        _ = expand_selected_labels(
            label_image=label_image, label_values_to_expand=2, expansion_amount=1
        )


@pytest.mark.skipif(
    on_ci and (on_windows or on_macos), reason="openCL doesn't work on windows/mac CI"
)
def test_expand_selected_labels_background_mask_2d():
    label_image = np.zeros((100, 100), dtype=int)
    label_image[49:51, 49:51] = 2

    background_mask = np.ones_like(label_image, dtype=bool)
    background_mask[45:55, 45:55] = 0

    expanded_image = expand_selected_labels(
        label_image=label_image,
        label_values_to_expand=2,
        expansion_amount=10,
        background_mask=background_mask,
    )

    # check that the shape was preserved
    np.testing.assert_equal(label_image.shape, expanded_image.shape)

    # check that label 2 was only expanded to the edge of the boundary mask
    assert np.sum(expanded_image == 2) == 10**2


@pytest.mark.skipif(
    on_ci and (on_windows or on_macos), reason="openCL doesn't work on windows/mac CI"
)
def test_expand_selected_labels_background_mask_3d():
    label_image = np.zeros((100, 100, 100), dtype=int)
    label_image[49:51, 49:51, 49:51] = 2

    background_mask = np.ones_like(label_image, dtype=bool)
    background_mask[45:55, 45:55, 45:55] = 0

    expanded_image = expand_selected_labels(
        label_image=label_image,
        label_values_to_expand=2,
        expansion_amount=10,
        background_mask=background_mask,
    )

    # check that the shape was preserved
    np.testing.assert_equal(label_image.shape, expanded_image.shape)

    # check that label 2 was only expanded to the edge of the boundary mask
    assert np.sum(expanded_image == 2) == 10**3


def test_get_mask_bounding_box_3d():
    mask_indices = np.array([[20, 30], [22, 32], [34, 44]])
    expected_bounding_box = mask_indices.copy()
    expected_bounding_box[:, 1] = expected_bounding_box[:, 1] - 1

    mask_image = np.zeros((50, 50, 50), dtype=int)
    mask_image[
        mask_indices[0, 0] : mask_indices[0, 1],
        mask_indices[1, 0] : mask_indices[1, 1],
        mask_indices[2, 0] : mask_indices[2, 1],
    ] = 1

    bounding_box = get_mask_bounding_box_3d(mask_image)
    np.testing.assert_allclose(bounding_box, expected_bounding_box)
    assert bounding_box.dtype == np.int


def test_expand_bounding_box():
    bounding_box = np.array([[10, 20], [30, 40], [50, 60]])
    expected_bounding_box = np.array([[7, 23], [27, 43], [47, 63]])
    expanded_bounding_box = expand_bounding_box(
        bounding_box=bounding_box, expansion_amount=3
    )
    np.testing.assert_equal(expanded_bounding_box, expected_bounding_box)


def test_expand_bounding_box_with_clipping():
    """Test that the bounding box is properly clipped to the image shape"""
    bounding_box = np.array([[2, 20], [30, 40], [50, 60]])
    expected_bounding_box = np.array([[0, 23], [27, 43], [47, 60]])
    expanded_bounding_box = expand_bounding_box(
        bounding_box=bounding_box, expansion_amount=3, image_shape=(61, 61, 61)
    )
    np.testing.assert_equal(expanded_bounding_box, expected_bounding_box)


@pytest.mark.skipif(
    on_ci and (on_windows or on_macos), reason="openCL doesn't work on windows/mac CI"
)
def test_expand_selected_labels_using_crop_3d():
    label_image = np.zeros((100, 100, 100), dtype=int)
    label_image[30:70, 30:70, 30:70] = 1
    label_image[45:55, 45:55, 45:55] = 0
    label_image[49:51, 49:51, 49:51] = 2
    label_image[0, 0, 90:100] = 3

    expanded_image = expand_selected_labels(
        label_image=label_image, label_values_to_expand=2, expansion_amount=1
    )

    np.testing.assert_equal(label_image.shape, expanded_image.shape)
    np.testing.assert_equal(label_image == 1, expanded_image == 1)

    assert expanded_image[50, 52, 50] == 2

    # expand to fill the void
    expanded_image_2 = expand_selected_labels_using_crop(
        label_image=label_image, label_values_to_expand=2, expansion_amount=5
    )
    np.testing.assert_equal(label_image.shape, expanded_image_2.shape)
    np.testing.assert_equal(label_image == 1, expanded_image_2 == 1)

    # check that the entire void was filled with 2
    assert np.sum(expanded_image_2 == 2) == 10**3

    # test that the bounding box doesn't extend beyond the image
    _ = expand_selected_labels_using_crop(
        label_image=label_image, label_values_to_expand=3, expansion_amount=5
    )


@pytest.mark.skipif(
    on_ci and (on_windows or on_macos), reason="openCL doesn't work on windows/mac CI"
)
def test_touch_matrix_from_label_image_2d():
    im_width = 100
    im_height = 100
    im = np.zeros((im_height, im_width), dtype=int)

    im[0:50, 0:50] = 0
    im[0:50, 50:im_width] = 1
    im[50:im_height, 50:im_width] = 2
    im[50:im_height, 0:50] = 3

    touch_matrix, background_mask = touch_matrix_from_label_image(im)

    expected_touch_matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    np.testing.assert_equal(touch_matrix.todense(), expected_touch_matrix)
    np.testing.assert_equal(background_mask, [True, False, True])
