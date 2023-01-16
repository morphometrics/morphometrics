import napari
import numpy as np

from morphometrics._gui.label_curator.label_curator import CurationMode, LabelCurator


def test_empty_label_curator(make_napari_viewer):
    """Test initializing the label curator model without a labels layer."""
    viewer = make_napari_viewer()
    label_curator = LabelCurator(viewer=viewer)
    assert label_curator.initialized is False
    assert label_curator.labels_layer is None
    assert label_curator.mode == CurationMode.PAINT

    # changing mode shouldn't enable the mode
    label_curator.mode = "clean"
    assert label_curator.mode == CurationMode.CLEAN
    assert label_curator._cleaning_model.enabled is False

    # adding a labels layer should enable the mode
    labels_layer = napari.layers.Labels(np.zeros((10, 10), dtype=int))
    label_curator.labels_layer = labels_layer
    assert label_curator.initialized
    assert label_curator.mode == CurationMode.CLEAN
    assert label_curator._cleaning_model.enabled


def test_label_curator(make_napari_viewer):
    """Test initializing the curator with a labels layer"""
    viewer = make_napari_viewer()
    labels_layer = napari.layers.Labels(np.zeros((10, 10), dtype=int))
    label_curator = LabelCurator(viewer=viewer, labels_layer=labels_layer)
    assert label_curator.initialized is True
    assert label_curator.labels_layer is labels_layer
    assert label_curator.mode == CurationMode.PAINT
    assert label_curator._painting_model.enabled

    # changing the mode should deactivate the old one
    # and activate the new one
    label_curator.mode = "clean"
    assert label_curator.mode == CurationMode.CLEAN
    assert label_curator._cleaning_model.enabled
    assert label_curator._painting_model.enabled is False
