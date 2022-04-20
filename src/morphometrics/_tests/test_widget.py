import numpy as np
import pytest

from morphometrics._gui._qt.annotation_widgets import QtClusterAnnotatorWidget
from morphometrics._gui._qt.measurement_widgets import QtMeasurementWidget


# make_napari_viewer is a pytest fixture that returns a napari viewer object
@pytest.mark.parametrize("widget", [QtMeasurementWidget, QtClusterAnnotatorWidget])
def test_creating_widget(make_napari_viewer, widget):
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()
    viewer.add_image(np.random.random((100, 100)))

    # create our widget, passing in the viewer
    _ = widget(viewer)
