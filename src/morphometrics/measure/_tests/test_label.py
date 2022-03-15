from typing import Set, Tuple

import numpy as np
import pandas as pd
import pytest

from morphometrics.measure.label import (
    measure_surface_properties_from_labels,
    regionprops,
)
from morphometrics.types import LabelImage


def make_test_label_image_2d() -> Tuple[LabelImage, Set[int]]:
    """Make a 2D label image with two squares

    Returns
    -------
    label_image : LabelImage
        A 2D label image with two squares.
    label_indices : Set[int]
        The set of labels in the label image.
    """
    label_image = np.zeros((50, 50), dtype=int)

    # 10 x 10 square
    label_image[10:20, 10:20] = 1

    # 15 x 15 square
    label_image[30:45, 30:45] = 2

    return label_image, {1, 2}


def make_test_label_image_3d() -> Tuple[LabelImage, Set[int]]:
    """Make a label image with two cubes

    Returns
    -------
    label_image : LabelImage
        A 3D label image with two cubes.
    label_indices : Set[int]
        The set of labels in the label image.
    """
    label_image = np.zeros((50, 50, 50), dtype=int)

    # 10 x 10 x 10 cube -> surface area of 600
    label_image[10:20, 10:20, 10:20] = 1

    # 15 x 15 x 15 cube -> surface area of 1350
    label_image[30:45, 30:45, 30:45] = 2

    return label_image, {1, 2}


def is_valid_label_feature_table(
    measurement_table: pd.DataFrame, label_indices: Set[int]
) -> bool:
    assert len(measurement_table) == len(label_indices)
    assert set(measurement_table.index) == label_indices
    assert measurement_table.index.name == "label"

    return True


def test_measure_surface_properties_2d():
    """This tests that measure_surface_properties
    raises a ValueError when a 2D label image is passed
    """
    label_image, label_indices = make_test_label_image_2d()

    with pytest.raises(ValueError):
        _ = measure_surface_properties_from_labels(label_image)


def test_measure_surface_properties_3d():
    """This tests that measure_surface_properties
    executes and returns data of the expected shape/type.
    This does not test the accuracy of the measurements.
    """
    label_image, label_indices = make_test_label_image_3d()
    measurement_table = measure_surface_properties_from_labels(label_image)
    assert is_valid_label_feature_table(measurement_table, label_indices=label_indices)


def test_regionprops_2d():
    """This tests that regionprops executes in 2D
    and returns data of the expected shape/type.
    This does not test the accuracy of the measurements.
    """
    label_image, label_indices = make_test_label_image_2d()
    rng = np.random.default_rng(42)
    intensity_image = rng.random(label_image.shape)

    measurement_table = regionprops(
        intensity_image=intensity_image,
        label_image=label_image,
        size=True,
        intensity=True,
        perimeter=True,
        shape=True,
        position=True,
        moments=True,
    )
    assert is_valid_label_feature_table(measurement_table, label_indices)


def test_regionprops_3d():
    """This tests that regionprops executes in 3D
    and returns data of the expected shape/type.
    This does not test the accuracy of the measurements.
    """
    label_image, label_indices = make_test_label_image_3d()
    rng = np.random.default_rng(42)
    intensity_image = rng.random(label_image.shape)

    measurement_table = regionprops(
        intensity_image=intensity_image,
        label_image=label_image,
        size=True,
        intensity=True,
        perimeter=False,
        shape=True,
        position=True,
        moments=True,
    )
    assert is_valid_label_feature_table(measurement_table, label_indices)
