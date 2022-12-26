from enum import Enum
from itertools import cycle
from typing import Dict, Optional, Tuple, Union

import glasbey
import napari
import numpy as np
from napari.layers import Labels
from napari.utils.events import EmitterGroup, Event

from morphometrics._gui.label_curator.label_cleaning import LabelCleaningModel
from morphometrics._gui.label_curator.label_painting import LabelPaintingModel


class CurationMode(Enum):
    PAINT = "paint"
    CLEAN = "clean"
    EXPLORE = "explore"


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


class LabelCurator:
    def __init__(
        self,
        viewer: napari.Viewer,
        labels_layer: Optional[Labels] = None,
        mode: Union[CurationMode, str] = "paint",
    ):
        self._viewer = viewer
        self._mode = CurationMode(mode)
        self._labels_layer = labels_layer
        self._initialized = False

        self.events = EmitterGroup(source=self, labels_layer=Event, mode=Event)

        # add the manager for coloring the labels
        self._colormap_manager = ColormapManager()

        # add the models for each mode
        self._cleaning_model = LabelCleaningModel(curator_model=self)
        self._painting_model = LabelPaintingModel(curator_model=self)

        # set everything up.
        self._enable_current_mode()

    @property
    def initialized(self) -> bool:
        """Flag set to True when the curator is set up and can be used.

        To be initialized, the curator needs a valid layer and the current
        mode needs to be set up.
        """
        return self._initialized

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
        self._disable_current_mode()
        self._labels_layer = layer
        self._enable_current_mode()
        self.events.labels_layer()

    @property
    def mode(self) -> CurationMode:
        """The current curation mode."""
        return self._mode

    @mode.setter
    def mode(self, mode: Union[str, CurationMode]) -> None:
        """The current curation mode.

        Parameters
        mode : Union[str, CurationMode]
            The new mode. Should be one of:
            "paint", "clean", or "explore"
        """
        if not isinstance(mode, CurationMode):
            mode = CurationMode(mode)

        if mode == self._mode:
            # don't do anything if the mode is unchanged
            return
        self._disable_current_mode()
        self._mode = mode
        self._enable_current_mode()
        self.events.mode()

    def _enable_current_mode(self) -> None:
        """Initialize the current curation mode.

        This is separated from the mode setter because sometimes it may need
        to be called separately.
        """
        if self.labels_layer is None:
            # if the labels layer isn't set, the curator isn't initialized
            self._initialized = False
            return

        if self.mode == CurationMode.PAINT:
            self._painting_model.enabled = True
        elif self.mode == CurationMode.CLEAN:
            self._cleaning_model.enabled = True
        elif self.mode == CurationMode.EXPLORE:
            raise NotImplementedError
        else:
            raise ValueError("Unknown mode")

        self._initialized = True

    def _disable_current_mode(self) -> None:
        """Disable the current curation mode"""
        if self.mode == CurationMode.PAINT:
            self._painting_model.enabled = False
        elif self.mode == CurationMode.CLEAN:
            self._cleaning_model.enabled = False
        elif self.mode == CurationMode.EXPLORE:
            raise NotImplementedError
        else:
            raise ValueError("Unknown mode")
        self._initialized = False
