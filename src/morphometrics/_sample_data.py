from typing import List

from .data import simple_labeled_cube


def make_simple_labeled_cube() -> "List[napari.types.LayerDataTuple]":  # noqa: F821
    """Create a simple labeled cube"""
    return [(simple_labeled_cube(), {}, "labels")]
