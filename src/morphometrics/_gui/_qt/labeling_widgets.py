from typing import List, Optional

import napari
from magicgui import magicgui, widgets
from napari.qt.threading import FunctionWorker, thread_worker
from napari.types import LayerDataTuple
from qtpy.QtWidgets import QVBoxLayout, QWidget

from ...label.image_utils import expand_selected_labels_using_crop
from ..models.labeling_model import LabelingModel


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
        pass

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
