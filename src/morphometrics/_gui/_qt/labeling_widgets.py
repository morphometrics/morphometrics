from typing import List, Optional

import napari
import numpy as np
import pyqtgraph as pg
from magicgui import magicgui, widgets
from napari.components.viewer_model import ViewerModel
from napari.qt.threading import FunctionWorker, thread_worker
from napari.types import LayerDataTuple
from napari.utils.notifications import show_info
from qtpy.QtWidgets import (
    QGroupBox,
    QLabel,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)
from skimage.exposure import histogram
from skimage.morphology import flood

from ...label.image_utils import expand_selected_labels_using_crop
from ..models.labeling_model import LabelingModel
from .multiple_viewer_widget import MultipleViewerWidget


class QtLabelingWidget(QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self._viewer = viewer
        self._model = LabelingModel()
        self._model.events.curating.connect(self._on_curating_change)

        # create the label selection widget
        self._label_selection_widget = magicgui(
            self._toggle_curating,
            labels_layer={"choices": self._get_valid_labels_layers},
            background_mask_layer={"choices": self._get_valid_image_layers},
            call_button="start curating",
        )

        # create the label expansion widget
        self._label_expansion_widget = magicgui(
            self._expand_selected_labels_widget_function,
            pbar={"visible": False, "max": 0, "label": "working..."},
            call_button="expand labels",
        )
        self._label_expansion_widget.native.setVisible(False)

        # add widgets to layout
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self._label_selection_widget.native)
        self.layout().addWidget(self._label_expansion_widget.native)

    def _expand_selected_labels_widget_function(
        self,
        pbar: widgets.ProgressBar,
        labels_to_expand: str,
        expansion_amount: int = 3,
    ) -> FunctionWorker[LayerDataTuple]:
        """

        Parameters
        ----------
        pbar : widgets.ProgressBar
            The progress bar to be displayed while the computation is running.
            This is supplied by magicgui.
        labels_to_expand : str
            The label values to expand as a comma separated string.
        expansion_amount : int
            The radius of the expansion in pixels.

        Returns
        -------
        function_worker : FunctionWorker[LayerDataTuple]
            The FunctionWorker that will return the new labels layer data when the computation has completed.
        """
        label_values_to_expand = labels_to_expand.replace(" ", "").split(",")
        label_values_to_expand = [int(value) for value in label_values_to_expand]

        # get the values from the selected labels layer
        label_image = self._model.labels_layer.data
        label_layer_name = self._model.labels_layer.name
        if self._model.background_mask_layer is not None:
            background_mask = self._model.background_mask_layer.data
        else:
            background_mask = None

        @thread_worker(connect={"returned": pbar.hide})
        def _expand_selected_labels() -> LayerDataTuple:
            new_labels = expand_selected_labels_using_crop(
                label_image=label_image,
                label_values_to_expand=label_values_to_expand,
                expansion_amount=expansion_amount,
                background_mask=background_mask,
            )
            layer_kwargs = {"name": label_layer_name}
            return (new_labels, layer_kwargs, "labels")

        # show progress bar and return worker
        pbar.show()

        return _expand_selected_labels()

    def _on_layer_update(self):
        """callback function that is called when self.layer is updated"""

    def _toggle_curating(
        self,
        labels_layer: napari.layers.Labels,
        background_mask_layer: napari.layers.Image,
    ):
        self._model.labels_layer = labels_layer
        self._model.background_mask_layer = background_mask_layer
        self._model.curating = True

    def _get_valid_labels_layers(self, combo_box) -> List[napari.layers.Labels]:
        return [
            layer
            for layer in self._viewer.layers
            if isinstance(layer, napari.layers.Labels)
        ]

    def _get_valid_image_layers(self, combo_box) -> List[Optional[napari.layers.Image]]:

        valid_layers = [
            layer
            for layer in self._viewer.layers
            if isinstance(layer, napari.layers.Image)
        ]
        valid_layers.append(None)

        return valid_layers

    def _on_curating_change(self):
        if self._model.curating is True:
            self._label_expansion_widget.native.setVisible(True)
        else:
            self._label_expansion_widget.native.setVisible(False)


class HistogramPlot:
    """Class for displaying a histogram plot in a Qt Widget.

    This widget will calculate and display a histogram of image intensities.
    You can add an overlay to show thresholding values.

    The widget is stored in HistogramPlot.widget.
    """

    # maximum height of the plot widget
    _MAXIMUM_HEIGHT: float = 200

    def __init__(self):
        self.widget = pg.PlotWidget()
        self.histogram_plot = self.widget.plot()

        # fill threshold region
        self.threshold_fill = pg.LinearRegionItem(values=(0, 0.5))
        self.widget.addItem(self.threshold_fill)

        # add the click intensity line
        self.click_value_line = pg.InfiniteLine(pos=0.3)
        self.widget.addItem(self.click_value_line)

        self._setup_histogram_plot()
        self.threshold_visible = False

    def _setup_histogram_plot(self):
        self.widget.setXRange(0, 1, padding=0.1)
        self.widget.setXRange(0, 1, padding=0.1)
        self.widget.setTitle("intensity histogram")
        self.widget.setMaximumHeight(self._MAXIMUM_HEIGHT)
        self.widget.setVisible(False)

        # set the plot line color to white
        self.histogram_plot.setPen((255, 255, 255))

        # make the threshold line and region no movable
        self.threshold_fill.setMovable(False)
        self.click_value_line.setMovable(False)

    @property
    def threshold_visible(self) -> bool:
        return self._threshold_visible

    @threshold_visible.setter
    def threshold_visible(self, threshold_visible: bool):
        # make the line and region invisible
        self.threshold_fill.setVisible(threshold_visible)
        self.click_value_line.setVisible(threshold_visible)

        self._threshold_visible = threshold_visible

    def plot_threshold(
        self, click_value: float, min_threshold: float, max_threshold: float
    ):
        self.threshold_fill.setRegion((min_threshold, max_threshold))
        self.click_value_line.setValue(click_value)

    def plot_image_histogram(self, image: np.ndarray):
        self.histogram_plot.clear()
        histogram_values, bins = histogram(image, nbins=100, normalize=True)
        self.histogram_plot.setData(x=bins, y=histogram_values)


class FloodFillWidget(QWidget):
    # string on the button to enable the mode
    ENABLE_STRING: str = "enable"

    # string on the button to disable the mode
    DISABLE_STRING: str = "disable"

    def __init__(self, main_viewer: napari.Viewer, ortho_viewers: List[ViewerModel]):
        super().__init__()
        self._viewer = main_viewer
        self._ortho_viewers = ortho_viewers
        self._enabled = False
        self._pan_zoom = False
        self._new_value = 1

        self._labels_layer_selection_widget = magicgui(
            self._select_labels_layer,
            labels_layer={"choices": self._get_labels_layers},
            auto_call=True,
            call_button=False,
        )
        self._labels_layer_selection_widget()

        self._image_layer_selection_widget = magicgui(
            self._select_image_layer,
            image_layer={"choices": self._get_image_layers},
            auto_call=True,
            call_button=False,
        )
        self._image_layer_selection_widget()

        # button to enable flood fill mode
        self._enabled_button = QPushButton("Enable")
        self._enabled_button.setCheckable(True)
        self._enabled_button.clicked.connect(self._enabled_button_clicked)

        # histogram plot widget
        self.histogram = HistogramPlot()

        self._viewer.layers.events.inserted.connect(self._reset_layer_choices)
        self._viewer.layers.events.removed.connect(self._reset_layer_choices)

        # create the layout
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(QLabel("Flood Fill"))
        self.layout().addWidget(self._labels_layer_selection_widget.native)
        self.layout().addWidget(self._image_layer_selection_widget.native)
        self.layout().addWidget(self._enabled_button)
        self.layout().addWidget(self.histogram.widget)

        # add spacer widget and reduce spacing so widget is compact.
        self.layout().addStretch(1)
        self.layout().setSpacing(0)
        self.layout().setContentsMargins(0, 0, 0, 0)

    def _on_activate(self):
        pass

    def _on_deactivate(self):
        self._enabled_button.setChecked(False)
        self._on_disable()

    def _enable_histogram_plot(self, image: np.ndarray):
        self.histogram.plot_image_histogram(image=image)
        self.histogram.widget.setVisible(True)

    def _disable_histogram_plot(self):
        self.histogram.widget.setVisible(False)

    def _select_labels_layer(self, labels_layer: Optional[napari.layers.Labels]):
        self._labels_layer = labels_layer

    def _get_labels_layers(self, combo_widget):
        """Get the labels layers in the viewer"""
        return [
            layer
            for layer in self._viewer.layers
            if isinstance(layer, napari.layers.Labels)
        ]

    def _select_image_layer(self, image_layer: Optional[napari.layers.Image]):
        self._image_layer = image_layer

    def _get_image_layers(self, combo_widget):
        """Get the labels layers in the viewer"""
        return [
            layer
            for layer in self._viewer.layers
            if isinstance(layer, napari.layers.Image)
        ]

    def _reset_layer_choices(self):
        self._labels_layer_selection_widget.reset_choices()
        self._image_layer_selection_widget.reset_choices()

    def _enabled_button_clicked(self, event=None):
        enabled = self._enabled_button.isChecked()
        if enabled is True:
            self._on_enable()
        else:
            self._on_disable()

    def _on_enable(self):
        if (self._image_layer is None) or (self._labels_layer is None):
            print("invalid state")
            return
        else:
            self._labels_layer.mouse_drag_callbacks.append(self.flood_fill)
            self._ortho_viewers[0].layers[
                self._labels_layer.name
            ].mouse_drag_callbacks.append(self.flood_fill)
            self._ortho_viewers[1].layers[
                self._labels_layer.name
            ].mouse_drag_callbacks.append(self.flood_fill)
            self._labels_layer.mode = "pan_zoom"
            self._viewer.layers.selection = [self._labels_layer]
            self._labels_layer.bind_key("Space", self._toggle_pan_zoom)
            self._ortho_viewers[0].layers[self._labels_layer.name].bind_key(
                "Space", self._toggle_pan_zoom
            )
            self._ortho_viewers[1].layers[self._labels_layer.name].bind_key(
                "Space", self._toggle_pan_zoom
            )
            self._enabled_button.setText(self.DISABLE_STRING)

            self._enable_histogram_plot(image=self._image_layer.data)

    def _on_disable(self):
        if self.flood_fill in self._labels_layer.mouse_drag_callbacks:
            self._labels_layer.mouse_drag_callbacks.remove(self.flood_fill)
            self._ortho_viewers[0].layers[
                self._labels_layer.name
            ].mouse_drag_callbacks.remove(self.flood_fill)
            self._ortho_viewers[1].layers[
                self._labels_layer.name
            ].mouse_drag_callbacks.remove(self.flood_fill)
        self._labels_layer.bind_key("Space", None)
        self._ortho_viewers[0].layers[self._labels_layer.name].bind_key("Space", None)
        self._ortho_viewers[1].layers[self._labels_layer.name].bind_key("Space", None)
        self._enabled_button.setText(self.ENABLE_STRING)
        self._disable_histogram_plot()

    def flood_fill(self, layer, event):
        if self._pan_zoom is True:
            # do not flood fill in pan/zooming
            return

        if len(event.dims_displayed) != 2:
            show_info("flood fill can only be applied to 2D views")
            return

        # get the value (need to nd-ify)
        click_world = event.position
        click_data = np.asarray(layer.world_to_data(click_world), dtype=int)

        initial_tolerance = 0.1

        initial_click_position = np.asarray(event.pos, dtype=float)
        layer.interactive = False

        do_not_clobber_mask = layer.data > 0

        mask = flood(
            self._image_layer.data, tuple(click_data), tolerance=initial_tolerance
        )
        mask[do_not_clobber_mask] = 0

        coordinates = np.argwhere(mask)
        previous_index = tuple(zip(coordinates.T))
        previous_values = layer.data[mask]

        layer.data[mask] = self._new_value
        layer.refresh()

        # add the threshold to the plot
        click_value = self._image_layer.data[tuple(click_data)]
        min_threshold = click_value - initial_tolerance
        max_threshold = click_value + initial_tolerance
        self.histogram.plot_threshold(
            click_value=click_value,
            min_threshold=min_threshold,
            max_threshold=max_threshold,
        )
        self.histogram.threshold_visible = True

        yield

        while event.type == "mouse_move":

            if len(previous_index) > 0:
                # reset the data
                self._labels_layer.data[previous_index] = previous_values

            current_click_position = np.asarray(event.pos, dtype=float)
            click_vector = current_click_position - initial_click_position

            offset = -click_vector[1]

            tolerance = np.clip(initial_tolerance + (offset / 300), 0, 1)

            # update the threshold plot
            min_threshold = np.clip(click_value - tolerance, 0, 1)
            max_threshold = np.clip(click_value + tolerance, 0, 1)
            self.histogram.plot_threshold(
                click_value=click_value,
                min_threshold=min_threshold,
                max_threshold=max_threshold,
            )

            mask = flood(self._image_layer.data, tuple(click_data), tolerance=tolerance)
            mask[do_not_clobber_mask] = 0

            coordinates = np.argwhere(mask)
            previous_index = tuple(zip(coordinates.T))
            previous_values = layer.data[mask]

            # set the new data
            layer.data[mask] = self._new_value
            layer.refresh()

            yield
        layer._save_history((previous_index, previous_values, self._new_value))
        self._new_value += 1

        layer.interactive = True
        self.histogram.threshold_visible = False

    def _toggle_pan_zoom(self, layer):
        """Toggle pan zoom mode.

        This is used to allow panning and zoom while in the flood fill
        mouse callback. This is bound to the Space bar by default.
        """
        self._pan_zoom = True
        yield
        self._pan_zoom = False


class PanZoomWidget(QWidget):
    def __init__(self, main_viewer: napari.Viewer, ortho_viewers: List[ViewerModel]):
        super().__init__()
        self.label = QLabel("pan/zoom")

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.label)
        self.layout().addStretch(1)

    def _on_activate(self):
        pass

    def _on_deactivate(self):
        pass


labeling_modes = {"pan/zoom": PanZoomWidget, "flood fill": FloodFillWidget}


class LabelingModeWidget(QWidget):
    def __init__(self, main_viewer: napari.Viewer, ortho_viewers: List[ViewerModel]):
        super().__init__()
        mode_box_layout = QVBoxLayout()
        mode_widget_layout = QVBoxLayout()

        mode_buttons = []
        self.labeling_widgets = dict()
        for name, mode_widget in labeling_modes.items():
            mode_button = QRadioButton(name)
            mode_button.toggled.connect(self._on_mode_button_clicked)
            mode_box_layout.addWidget(mode_button)
            mode_buttons.append(mode_button)

            widget = mode_widget(main_viewer=main_viewer, ortho_viewers=ortho_viewers)
            mode_widget_layout.addWidget(widget)
            widget.setVisible(False)
            self.labeling_widgets[name] = widget
        self.mode_buttons = mode_buttons

        self.mode_box = QGroupBox("labeling mode")
        mode_box_layout.addStretch(1)
        self.mode_box.setLayout(mode_box_layout)

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.mode_box)

        for widget in self.labeling_widgets.values():
            self.layout().addWidget(widget)
        self.layout().addStretch(1)

    def _on_mode_button_clicked(self):
        button = self.sender()
        button_name = button.text()
        widget = self.labeling_widgets[button_name]
        if button.isChecked():
            widget._on_activate()
            widget.setVisible(True)
        else:
            widget._on_deactivate()
            widget.setVisible(False)


class MultiCanvasLabelerWidget(MultipleViewerWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__(viewer)
        # self.flood_fill_widget = FloodFillWidget(
        #     main_viewer=viewer, ortho_viewers=self.ortho_viewer_models
        # )
        # self.addWidget(self.flood_fill_widget)
        self.labeling_mode_widget = LabelingModeWidget(
            main_viewer=viewer, ortho_viewers=self.ortho_viewer_models
        )
        self.addWidget(self.labeling_mode_widget)

        self.viewer.axes.visible = True
        for model in self.ortho_viewer_models:
            model.axes.visible = True
