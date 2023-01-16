from dataclasses import dataclass
from typing import List, Optional, Tuple

import napari
import numpy as np

from morphometrics._gui.label_curator.label_curator import CurationMode, LabelCurator


@dataclass
class MockMouseEvent:
    position: Tuple[float, float]
    view_direction: Optional[Tuple[float, float]]
    dims_displayed: Tuple[int, int]
    modifiers: List[str]


def test_labeling_model_mouse_selection(make_napari_viewer):
    # make the viewer
    viewer = make_napari_viewer()

    # make the labels_layer
    label_image = np.zeros((10, 10), dtype=int)
    label_image[0:5, 0:5] = 1
    label_image[0:5, 5:10] = 2
    labels_layer = napari.layers.Labels(label_image)

    label_curator = LabelCurator(viewer=viewer, labels_layer=labels_layer, mode="clean")
    assert label_curator.mode == CurationMode.CLEAN
    label_cleaning_model = label_curator._cleaning_model
    assert label_cleaning_model.enabled

    # select one label
    selection_event = MockMouseEvent(
        position=(2.5, 2.5), view_direction=None, dims_displayed=(0, 1), modifiers=[]
    )
    label_cleaning_model._on_click_selection(layer=labels_layer, event=selection_event)
    assert label_cleaning_model._selected_labels == {1}

    # select a second label
    selection_event = MockMouseEvent(
        position=(2.5, 7.5),
        view_direction=None,
        dims_displayed=(0, 1),
        modifiers=["shift"],
    )
    label_cleaning_model._on_click_selection(layer=labels_layer, event=selection_event)
    assert label_cleaning_model._selected_labels == {1, 2}

    # don't deselect label when clicking background and holding shift
    selection_event = MockMouseEvent(
        position=(7.5, 7.5),
        view_direction=None,
        dims_displayed=(0, 1),
        modifiers=["shift"],
    )
    label_cleaning_model._on_click_selection(layer=labels_layer, event=selection_event)
    assert label_cleaning_model._selected_labels == {1, 2}

    # deselect label when clicking background
    selection_event = MockMouseEvent(
        position=(7.5, 7.5), view_direction=None, dims_displayed=(0, 1), modifiers=[]
    )
    label_cleaning_model._on_click_selection(layer=labels_layer, event=selection_event)
    assert label_cleaning_model._selected_labels == set()
