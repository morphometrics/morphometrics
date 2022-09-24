import warnings
from typing import Any, Dict, List, Union

import napari
import numpy as np
from magicgui import magicgui
from napari_skimage_regionprops._table import add_table
from qtpy.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from superqt.collapsible import QCollapsible

from morphometrics.measure import _measurements, measure_selected


class QtSingleMeasurement(QWidget):
    def __init__(self, name: str):
        super().__init__()
        self._measurement_name = name
        self._label_widget = QLabel(name)
        self._check_box = QCheckBox()

        self.setLayout(QHBoxLayout())
        self.layout().addWidget(self._check_box)
        self.layout().addWidget(self._label_widget)

        self.layout().setSpacing(0)
        self.layout().setContentsMargins(0,0,0,0)

    @property
    def measurement_name(self) -> str:
        return self._measurement_name

    @property
    def include_measurement(self) -> bool:
        return self._check_box.isChecked()


class QtMeasurementSet(QWidget):
    def __init__(self, name: str, choices: List[str]):
        super().__init__()
        self._measurement_name = name

        self._measurement_section = QCollapsible(
            title=self._measurement_name, parent=self
        )
        self._choice_widgets = [QtSingleMeasurement(name=choice) for choice in choices]
        for choice_widget in self._choice_widgets:
            self._measurement_section.addWidget(choice_widget)

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self._measurement_section)

        self.layout().setSpacing(0)
        self.layout().setContentsMargins(0,0,0,0)

        self._measurement_section.layout().layout().setSpacing(0)
        self._measurement_section.layout().setContentsMargins(0,0,0,0)

    @property
    def measurement_name(self) -> str:
        return self._measurement_name

    @property
    def include_measurement(self) -> bool:
        return np.any([widget.include_measurement for widget in self._choice_widgets])

    @property
    def measurement_choices(self) -> Dict[str, bool]:
        return {
            widget.measurement_name: widget.include_measurement
            for widget in self._choice_widgets
        }


def create_measurement_widgets(
    measurements: List[Dict[str, Any]]
) -> List[Union[QtSingleMeasurement, QtMeasurementSet]]:
    widgets = []
    for measurement_name, measurement_config in measurements.items():
        if measurement_config["type"] == "single":
            widgets.append(QtSingleMeasurement(name=measurement_name))
        elif measurement_config["type"] == "set":
            widgets.append(
                QtMeasurementSet(
                    name=measurement_name, choices=measurement_config["choices"]
                )
            )
    return widgets


class QtMeasurementWidget(QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self._viewer = viewer

        # create the layer selection widgets
        self._layer_selection_widget = magicgui(
            self._select_layers,
            intensity_image={"choices": self._get_image_layers},
            label_image={"choices": self._get_labels_layers},
            auto_call=True,
            call_button=False,
        )
        self._layer_selection_widget()

        # create the measurement widgets
        self.measurement_widgets = create_measurement_widgets(_measurements)

        self._run_button = QPushButton("Run", self)
        self._run_button.clicked.connect(self._run)

        # add widgets to the layout
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self._layer_selection_widget.native)
        for widget in self.measurement_widgets:
            self.layout().addWidget(widget)
        self.layout().addWidget(self._run_button)

    @property
    def measurement_selection(self) -> List[Union[str, Dict[str, Any]]]:
        measurement_selection = []
        for widget in self.measurement_widgets:
            if not widget.include_measurement:
                continue
            if isinstance(widget, QtSingleMeasurement):
                measurement_selection.append(widget.measurement_name)
            elif isinstance(widget, QtMeasurementSet):
                measurement_selection.append(
                    {
                        "name": widget.measurement_name,
                        "choices": widget.measurement_choices,
                    }
                )
            else:
                raise TypeError(
                    "Widgets should be QtSingleMeasurement or QtMeasurementSet"
                )

        return measurement_selection

    def _select_layers(
        self, intensity_image: napari.layers.Image, label_image: napari.layers.Labels
    ):
        self._intensity_image_layer = intensity_image
        self._label_image_layer = label_image

    def _get_image_layers(self, combo_widget):
        return [
            layer
            for layer in self._viewer.layers
            if isinstance(layer, napari.layers.Image)
        ]

    def _get_labels_layers(self, combo_widget):
        return [
            layer
            for layer in self._viewer.layers
            if isinstance(layer, napari.layers.Labels)
        ]

    def _run(self):
        for widget in self.measurement_widgets:
            print(f"{widget.measurement_name}: {widget.include_measurement}")

            widget_is_set = isinstance(widget, QtMeasurementSet)
            print(f"{widget.measurement_name} is set: {widget_is_set}")

        labels = self._label_image_layer.data
        image = self._intensity_image_layer.data

        # deal with dimensionality of data
        if len(image.shape) > len(labels.shape):
            dim = 0
            subset = ""
            while len(image.shape) > len(labels.shape):
                current_dim_value = self._viewer.dims.current_step[dim]
                dim = dim + 1
                image = image[current_dim_value]
                subset = subset + ", " + str(current_dim_value)
            warnings.warn(
                "Not the full image was analysed, just the subset ["
                + subset[2:]
                + "] according to selected timepoint / slice."
            )

        measurement_table = measure_selected(
            label_image=labels,
            intensity_image=image,
            measurement_selection=self.measurement_selection,
        )

        self._label_image_layer.properties = measurement_table.reset_index()
        add_table(self._label_image_layer, self._viewer)
