from typing import Optional, Union

import napari
from napari.utils.events import EmitterGroup, Event


class LabelingModel:
    def __init__(self):
        self.events = EmitterGroup(
            source=self, labels_layer=Event, background_mask_layer=Event
        )
        self._labels_layer = None
        self._background_mask_layer = None

    @property
    def labels_layer(self) -> Optional[napari.layers.Labels]:
        return self._labels_layer

    @labels_layer.setter
    def labels_layer(self, layer: Optional[napari.layers.Labels]) -> None:
        self._labels_layer = layer
        self.events.labels_layer()

    @property
    def background_mask_layer(
        self,
    ) -> Optional[Union[napari.layers.Image, napari.layers.Labels]]:
        return self._background_mask_layer

    @background_mask_layer.setter
    def background_mask_layer(
        self, layer: Optional[Union[napari.layers.Image, napari.layers.Labels]]
    ):
        self._background_mask_layer = layer
        self.events.background_mask_layer()
