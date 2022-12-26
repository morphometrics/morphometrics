from typing import TYPE_CHECKING

import napari
from napari.utils.events import EmitterGroup, Event, EventedSet

if TYPE_CHECKING:
    from morphometrics._gui.label_curator.label_curator import LabelCurator


class LabelCleaningModel:
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
        self._selected_labels.events.changed.connect(self._on_selection_changed)
        self._curator_model.labels_layer.mouse_drag_callbacks.append(
            self._on_click_selection
        )

        # set the labels layer coloring mode
        self._curator_model.labels_layer.color_mode = "direct"
        self._curator_model.labels_layer.color = (
            self._curator_model._colormap_manager._default_colormap
        )

    def _on_disable(self):
        self._selected_labels.events.changed.disconnect(self._on_selection_changed)
        self._curator_model.labels_layer.mouse_drag_callbacks.remove(
            self._on_click_selection
        )

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
        """Update the colormap based on a new selection"""
        if not self.enabled:
            # don't do anything if not curating
            return
        self._curator_model.labels_layer.color = (
            self._curator_model._colormap_manager.create_highlighted_colormap(
                list(self._selected_labels)
            )
        )
