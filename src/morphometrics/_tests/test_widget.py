import numpy as np
import pytest

import morphometrics
from morphometrics._gui._qt.annotation_widgets import QtClusterAnnotatorWidget
from morphometrics._gui._qt.labeling_widgets import QtLabelingWidget
from morphometrics._gui._qt.measurement_widgets import QtMeasurementWidget


# make_napari_viewer is a pytest fixture that returns a napari viewer object
@pytest.mark.parametrize(
    "widget", [QtMeasurementWidget, QtClusterAnnotatorWidget, QtLabelingWidget]
)
def test_creating_widget(make_napari_viewer, widget):
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()
    viewer.add_image(np.random.random((100, 100)))

    # create our widget, passing in the viewer
    _ = widget(viewer)


MY_PLUGIN_NAME = "morphometrics"
# the name of your widget(s)
MY_WIDGET_NAMES = [
    "Annotate clustered features",
    "Measure region properties",
    "Curate labels",
]


@pytest.mark.parametrize("widget_name", MY_WIDGET_NAMES)
def test_something_with_viewer(widget_name, make_napari_viewer, napari_plugin_manager):
    napari_plugin_manager.register(morphometrics, name=MY_PLUGIN_NAME)
    viewer = make_napari_viewer()
    num_dw = len(viewer.window._dock_widgets)
    viewer.window.add_plugin_dock_widget(
        plugin_name=MY_PLUGIN_NAME, widget_name=widget_name
    )
    assert len(viewer.window._dock_widgets) == num_dw + 1
