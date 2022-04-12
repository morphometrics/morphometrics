from typing import Optional

import napari
from magicgui import magicgui
from qtpy.QtWidgets import QVBoxLayout, QWidget


class QtClusterAnnotatorWidget(QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self._viewer = viewer

        # create the layer selection widgets
        self._layer_selection_widget = magicgui(
            self._select_layer,
            labels_layer={"choices": self._get_labels_layer},
            auto_call=True,
            call_button=False,
        )
        self._layer_selection_widget()

        # add widgets to the layout
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self._layer_selection_widget.native)

    def _select_layer(self, labels_layer: Optional[napari.layers.Labels]):
        self._layer = labels_layer
        print(self._layer)

    def _get_labels_layer(self, combo_widget):
        """Get the labels layers that have an associated anndata object stored
        in the metadata.
        """
        return [
            layer
            for layer in self._viewer.layers
            if (isinstance(layer, napari.layers.Labels)) and ("adata" in layer.metadata)
        ]
