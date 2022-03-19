from typing import Tuple

import numpy as np
import pytest
from skimage.segmentation import find_boundaries

from morphometrics.measure.intensity import (
    measure_boundary_intensity,
    measure_intensity_features,
    measure_internal_intensity,
)


def _make_test_intensity_image(label_image):
    intensity_image = np.zeros_like(label_image, dtype=float)
    intensity_image[label_image == 1] = 0.1

    label_2_mask = label_image == 2
    label_2_boundaries = find_boundaries(label_2_mask, mode="inner")
    label_2_internal = label_2_mask.copy()
    label_2_internal[label_2_boundaries] = False

    intensity_image[label_2_boundaries] = 0.5
    intensity_image[label_2_internal] = 1

    expected_internal_intensity = np.array([0.1, 1])
    expected_boundary_intensity = np.array([0.1, 0.5])

    return intensity_image, expected_boundary_intensity, expected_internal_intensity


def make_test_label_and_intensity_images_2d() -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    label_image = np.zeros((40, 40), dtype=int)
    label_image[10:20, 10:20] = 1
    label_image[25:30, 25:30] = 2

    (
        intensity_image,
        expected_boundary_intensity,
        expected_internal_intensity,
    ) = _make_test_intensity_image(label_image)

    return (
        label_image,
        intensity_image,
        expected_boundary_intensity,
        expected_internal_intensity,
    )


def make_test_label_and_intensity_images_3d() -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    label_image = np.zeros((40, 40, 40), dtype=int)
    label_image[10:20, 10:20, 10:20] = 1
    label_image[25:30, 25:30, 25:30] = 2

    (
        intensity_image,
        expected_boundary_intensity,
        expected_internal_intensity,
    ) = _make_test_intensity_image(label_image)

    return (
        label_image,
        intensity_image,
        expected_boundary_intensity,
        expected_internal_intensity,
    )


expected_internal_intensity_columns = {
    "internal_intensity_mean",
    "internal_intensity_min",
    "internal_intensity_max",
}
expected_boundary_intensity_columns = {
    "boundary_intensity_mean",
    "boundary_intensity_min",
    "boundary_intensity_max",
}
expected_intensity_features_columns = expected_internal_intensity_columns.union(
    expected_boundary_intensity_columns
)


@pytest.mark.parametrize(
    "test_data_func",
    [make_test_label_and_intensity_images_2d, make_test_label_and_intensity_images_3d],
)
def test_measure_internal_intensity(test_data_func):
    label_image, intensity_image, _, expected_internal_intensity = test_data_func()

    measurements = measure_internal_intensity(
        label_image=label_image,
        intensity_image=intensity_image,
    )
    np.testing.assert_almost_equal(
        measurements["internal_intensity_mean"], expected_internal_intensity
    )

    assert set(measurements.columns) == expected_internal_intensity_columns


@pytest.mark.parametrize(
    "test_data_func",
    [make_test_label_and_intensity_images_2d, make_test_label_and_intensity_images_3d],
)
def test_measure_boundary_intensity(test_data_func):
    label_image, intensity_image, expected_boundary_intensity, _ = test_data_func()

    measurements = measure_boundary_intensity(
        label_image=label_image,
        intensity_image=intensity_image,
    )
    np.testing.assert_almost_equal(
        measurements["boundary_intensity_mean"], expected_boundary_intensity
    )

    assert set(measurements.columns) == expected_boundary_intensity_columns


@pytest.mark.parametrize(
    "test_data_func",
    [make_test_label_and_intensity_images_2d, make_test_label_and_intensity_images_3d],
)
def test_measure_intensity_features(test_data_func):
    (
        label_image,
        intensity_image,
        expected_boundary_intensity,
        expected_internal_intensity,
    ) = test_data_func()

    measurements = measure_intensity_features(
        label_image=label_image,
        intensity_image=intensity_image,
    )
    np.testing.assert_almost_equal(
        measurements["boundary_intensity_mean"], expected_boundary_intensity
    )
    np.testing.assert_almost_equal(
        measurements["internal_intensity_mean"], expected_internal_intensity
    )

    assert set(measurements.columns) == expected_intensity_features_columns
