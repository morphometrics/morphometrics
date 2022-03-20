from typing import List

from .data import cylinders_and_spheres, random_3d_image, simple_labeled_cube


def make_simple_labeled_cube() -> "List[napari.types.LayerDataTuple]":  # noqa: F821
    """Create a simple labeled cube"""
    return [(simple_labeled_cube(), {}, "labels")]


def make_random_3d_image() -> "List[napari.types.LayerDataTuple]":  # noqa: F821
    return [(random_3d_image(), {}, "image")]


def make_cylinders_and_spheres() -> "List[napari.types.LayerDataTuple]":  # noqa: F821
    label_image, label_table, intensity_image = cylinders_and_spheres()

    return [
        (label_image, {"features": label_table}, "labels"),
        (intensity_image, {}, "image"),
    ]
