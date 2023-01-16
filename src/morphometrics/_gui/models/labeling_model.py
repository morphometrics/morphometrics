from itertools import cycle
from typing import Dict, Optional, Tuple, Union

import glasbey
import napari
import numpy as np
from napari.utils.events import EmitterGroup, Event, EventedSet


class ColormapManager:
    def __init__(self):
        default_colormap, highlight_colormap = self._initialize_colormaps()
        self._default_colormap = default_colormap
        self._highlight_colormap = highlight_colormap

    def _initialize_colormaps(
        self,
    ) -> Tuple[Dict[Optional[int], np.ndarray], Dict[Optional[int], np.ndarray]]:

        colors = np.asarray(
            glasbey.create_palette(
                palette_size=50,
                as_hex=False,
                chroma_bounds=(40, 120),
                lightness_bounds=(50, 90),
            )
        )
        colors_highlight = np.column_stack((colors, np.ones((len(colors),))))
        colors_highlight_cycler = cycle(colors_highlight)

        colormap_highlight = {i: next(colors_highlight_cycler) for i in range(1000)}
        colormap_highlight[0] = np.array([0, 0, 0, 0])
        colormap_highlight[None] = np.array([0, 0, 0, 1])

        colors_default = colors_highlight.copy()
        colors_default[:, 3] = 0.5
        colors_default[:, 0:3] = colors_default[:, 0:3] * 0.5

        colors_default_cylcer = cycle(colors_default)
        colormap_default = {i: next(colors_default_cylcer) for i in range(1000)}
        colormap_default[0] = np.array([0, 0, 0, 0])
        colormap_default[None] = np.array([0, 0, 0, 1])

        return colormap_default, colormap_highlight

    def create_highlighted_colormap(
        self, label_indices: np.ndarray
    ) -> Dict[Optional[int], np.ndarray]:
        label_indices = np.asarray(label_indices)
        new_colormap = self._default_colormap.copy()
        for index in label_indices:
            new_colormap[index] = self._highlight_colormap[index]
        return new_colormap


class LabelingModel:
    def __init__(self):
        self.events = EmitterGroup(
            source=self, labels_layer=Event, background_mask_layer=Event, curating=Event
        )
        self._labels_layer = None
        self._background_mask_layer = None
        self._curating = False

        # managers for selecting labels
        self._selected_labels = EventedSet()
        self._colormap_manager = ColormapManager()

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
    def curating(self, curating: bool):
        if self._curating == curating:
            return
        self._curating = curating
        if curating:
            self._on_curation_start()
        else:
            self._on_curation_end()
        self.events.curating()

    def _on_curation_start(self):
        # add the events
        self._selected_labels.events.changed.connect(self._on_selection_changed)
        self.labels_layer.mouse_drag_callbacks.append(self._on_click_selection)

        self.labels_layer.color_mode = "direct"
        self.labels_layer.color = self._colormap_manager._default_colormap

    def _on_curation_end(self):
        self._selected_labels.events.changed.disconnect(self._on_selection_changed)
        self.labels_layer.mouse_drag_callbacks.remove(self._on_click_selection)

    def _on_click_selection(self, layer: napari.layers.Labels, event: Event):
        """Mouse callback for selecting labels"""
        label_index = layer.get_value(
            position=event.position,
            view_direction=event.view_direction,
            dims_displayed=event.dims_displayed,
            world=True,
        )
        if (label_index is None) or (label_index == layer._background_label):
            # the background or outside the layer was clicked, clear the selection
            if "shift" not in event.modifiers:
                # don't clear the selection if the shift key was held
                self._selected_labels.clear()
            return
        if "shift" in event.modifiers:
            self._selected_labels.symmetric_difference_update([label_index])
        else:
            self._selected_labels._set.clear()
            self._selected_labels.update([label_index])

    def _on_selection_changed(self, event: Event):
        if not self.curating:
            # don't do anything if not curating
            return
        self.labels_layer.color = self._colormap_manager.create_highlighted_colormap(
            list(self._selected_labels)
        )
