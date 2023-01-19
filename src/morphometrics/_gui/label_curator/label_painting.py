from typing import TYPE_CHECKING

from napari.utils.events import EmitterGroup, Event, EventedSet

if TYPE_CHECKING:
    from morphometrics._gui.label_curator.label_curator import LabelCurator


class LabelPaintingModel:
    def __init__(self, curator_model: "LabelCurator", enabled: bool = False):
        self._enabled = False
        self._curator_model = curator_model

        self.events = EmitterGroup(source=self, enabled=Event)

        self._selected_labels = EventedSet()
        self.enabled = enabled

    @property
    def enabled(self) -> bool:
        """Flag set to true when the label cleaning mode is active"""
        return self._enabled

    @enabled.setter
    def enabled(self, enabled: bool) -> None:
        if enabled == self.enabled:
            return

        if enabled:
            self._on_enable()
        else:
            self._on_disable()
        self._enabled = enabled
        self.events.enabled()

    def _on_enable(self):
        # add the events
        self._curator_model.labels_layer.mode = "paint"
        self._curator_model.labels_layer.color_mode = "auto"

    def _on_disable(self):
        self._curator_model.labels_layer.mode = "pan_zoom"
