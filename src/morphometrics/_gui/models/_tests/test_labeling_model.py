from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from napari.layers import Image, Labels

from morphometrics._gui.models.labeling_model import LabelingModel


def test_labeling_model():
    labeling_model = LabelingModel()
    assert labeling_model.labels_layer is None
    assert labeling_model.background_mask_layer is None
    assert labeling_model.curating is False

    labels_layer = Labels(np.random.randint(0, 5, (10, 10, 10)))
    labeling_model.labels_layer = labels_layer
    assert labeling_model.labels_layer is labels_layer

    background_mask_layer = Image(np.random.randint(0, 1, (10, 10, 10)).astype(bool))
    labeling_model.background_mask_layer = background_mask_layer
    assert labeling_model.background_mask_layer is background_mask_layer

    labeling_model.curating = True
    assert labeling_model.curating is True


@dataclass
class MockMouseEvent:
    position: Tuple[float, float]
    view_direction: Optional[Tuple[float, float]]
    dims_displayed: Tuple[int, int]
    modifiers: List[str]


def test_labeling_model_mouse_selection():
    labeling_model = LabelingModel()

    # make the labels_layer
    label_image = np.zeros((10, 10), dtype=int)
    label_image[0:5, 0:5] = 1
    label_image[0:5, 5:10] = 2
    labels_layer = Labels(label_image)

    # add the labels layer and start curating
    labeling_model.labels_layer = labels_layer
    labeling_model.curating = True

    # select one label
    selection_event = MockMouseEvent(
        position=(2.5, 2.5), view_direction=None, dims_displayed=(0, 1), modifiers=[]
    )
    labeling_model._on_click_selection(layer=labels_layer, event=selection_event)
    assert labeling_model._selected_labels == {1}

    # select a second label
    selection_event = MockMouseEvent(
        position=(2.5, 7.5),
        view_direction=None,
        dims_displayed=(0, 1),
        modifiers=["shift"],
    )
    labeling_model._on_click_selection(layer=labels_layer, event=selection_event)
    assert labeling_model._selected_labels == {1, 2}

    # don't deselect label when clicking background and holding shift
    selection_event = MockMouseEvent(
        position=(7.5, 7.5),
        view_direction=None,
        dims_displayed=(0, 1),
        modifiers=["shift"],
    )
    labeling_model._on_click_selection(layer=labels_layer, event=selection_event)
    assert labeling_model._selected_labels == {1, 2}

    # deselect label when clicking background
    selection_event = MockMouseEvent(
        position=(7.5, 7.5), view_direction=None, dims_displayed=(0, 1), modifiers=[]
    )
    labeling_model._on_click_selection(layer=labels_layer, event=selection_event)
    assert labeling_model._selected_labels == set()
