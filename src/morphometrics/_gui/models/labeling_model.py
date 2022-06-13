from typing import Optional, Union

import napari
from napari.utils.events import EmitterGroup, Event


class LabelingModel:
    def __init__(self):
        self.events = EmitterGroup(
            source=self, labels_layer=Event, background_mask_layer=Event, curating=Event
        )
        self._labels_layer = None
        self._background_mask_layer = None
        self._curating = False

    @property
    def labels_layer(self) -> Optional[napari.layers.Labels]:
        """The selected labels layer for curation.

        Returns
        -------
        layer : Optional[napari.layers.Labels]
            The selected labels layer that will be curated. If no layer is selected,
            returns None.
        """
        return self._labels_layer

    @labels_layer.setter
    def labels_layer(self, layer: Optional[napari.layers.Labels]) -> None:
        """The selected labels layer for curation.

        Parameters
        ----------
        layer : Optional[napari.layers.Labels]
            The selected labels layer that will be curated. If no layer is selected,
            set to None.
        """
        if layer is self._labels_layer:
            # if the layer hasn't changed, don't perform the update
            return None
        self._labels_layer = layer
        self.events.labels_layer()

    @property
    def background_mask_layer(
        self,
    ) -> Optional[Union[napari.layers.Image, napari.layers.Labels]]:
        """The selected labels layer for curation.

        Returns
        -------
        layer : Optional[napari.layers.Labels]
            The selected labels layer that will be curated. If no layer is selected,
            returns None.
        """
        return self._background_mask_layer

    @background_mask_layer.setter
    def background_mask_layer(
        self, layer: Optional[Union[napari.layers.Image, napari.layers.Labels]]
    ):
        """The selected image layer for backgroundmaskiung.

        Parameters
        ----------
        layer : Optional[napari.layers.Labels]
            The selected image layer that be used to mask off background when expanding
            labels
        """
        if layer is self._background_mask_layer:
            # if the layer hasn't changed, don't perform the update
            return None
        self._background_mask_layer = layer
        self.events.background_mask_layer()

    @property
    def curating(self) -> bool:
        return self._curating

    @curating.setter
    def curating(self, curating: bool) -> None:
        if self._curating == curating:
            return
        self._curating = curating
        self.events.curating()

    def set_paint_mode(self, n_edit_dimensions: int = 3) -> None:
        """Set the labels layer to paint mode with a specified n_edit_dimensions

        Sets the

        Parameeters
        -----------
        n_edit_dimensions : int
            The number of dimensions to be painted. Default value is 3.
        """
        if self.labels_layer is None:
            return
        self.labels_layer.n_edit_dimensions = n_edit_dimensions
        self.labels_layer.preserve_labels = True
        self.labels_layer.mode = "paint"
