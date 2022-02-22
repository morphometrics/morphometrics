from typing import List

from .data import random_3d_image, simple_labeled_cube


def make_simple_labeled_cube() -> "List[napari.types.LayerDataTuple]":  # noqa: F821
    """Create a simple labeled cube"""
    return [(simple_labeled_cube(), {}, "labels")]


def make_random_3d_image() -> "List[napari.types.LayerDataTuple]":  # noqa: F821
    return [(random_3d_image(), {}, "image")]
