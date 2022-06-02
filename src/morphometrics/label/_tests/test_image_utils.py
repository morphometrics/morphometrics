import numpy as np
import pytest

from morphometrics.label.image_utils import (
    expand_bounding_box,
    expand_selected_labels,
    get_mask_bounding_box_3d,
)
from morphometrics.utils.environment_utils import on_ci


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

    assert expanded_image[50, 52] == 2

    # expand to fill the void
    expanded_image_2 = expand_selected_labels(
        label_image=label_image, label_values_to_expand=2, expansion_amount=5
    )
    np.testing.assert_equal(label_image.shape, expanded_image_2.shape)
    np.testing.assert_equal(label_image == 1, expanded_image_2 == 1)

    # check that the entire void was filled with 2
    assert np.sum(expanded_image_2 == 2) == 10 ** 2


@pytest.mark.skipif(on_ci, reason="openCL tests not working on CI")
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
    assert np.sum(expanded_image_2 == 2) == 10 ** 3


@pytest.mark.skipif(on_ci, reason="openCL tests not working on CI")
def test_expand_selected_labels_4d():
    """Should raise a value error with label ndim > 3"""
    label_image = np.zeros((10, 10, 10, 10), dtype=int)

    with pytest.raises(ValueError):
        _ = expand_selected_labels(
            label_image=label_image, label_values_to_expand=2, expansion_amount=1
        )


@pytest.mark.skipif(on_ci, reason="openCL tests not working on CI")
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
    assert np.sum(expanded_image == 2) == 10 ** 2


@pytest.mark.skipif(on_ci, reason="openCL tests not working on CI")
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
    assert np.sum(expanded_image == 2) == 10 ** 3


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
