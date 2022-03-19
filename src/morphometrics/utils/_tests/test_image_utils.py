import numpy as np
import pytest

from morphometrics.utils.image_utils import make_boundary_mask

label_image = np.array(
    [
        [0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0],
    ]
)


def test_make_boundary_mask_no_dilation():
    expected_label_image = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    )
    boundary_mask = make_boundary_mask(label_image, boundary_dilation_size=0)
    np.testing.assert_array_equal(boundary_mask, expected_label_image)


def test_make_boundary_mask_dilation():
    expected_label_image = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    )
    boundary_mask = make_boundary_mask(label_image, boundary_dilation_size=3)
    np.testing.assert_array_equal(boundary_mask, expected_label_image)


def test_make_boundary_mask_bad_dim():
    """Test that making a boundary mask with ndim > 3 image raises an error"""
    rng = np.random.default_rng(42)
    label_image = rng.integers(0, 127, (30, 30, 30, 30))
    with pytest.raises(ValueError):
        _ = make_boundary_mask(label_image, boundary_dilation_size=3)
