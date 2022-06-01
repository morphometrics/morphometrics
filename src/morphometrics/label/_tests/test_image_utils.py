import numpy as np
import pytest

from morphometrics.label.image_utils import expand_selected_labels
from morphometrics.utils.environment_utils import on_ci


@pytest.mark.skipif(on_ci, reason="openCL tests not working on CI")
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
