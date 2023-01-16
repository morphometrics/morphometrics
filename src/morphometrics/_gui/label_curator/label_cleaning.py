from typing import TYPE_CHECKING, List

import napari
import numpy as np
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
        # get the indices of the validated labels
        features = self._curator_model.labels_layer.features
        hide_indices = features.loc[features["mm_validated"]]["index"].values
        self._curator_model.labels_layer.color = (
            self._curator_model._colormap_manager.create_highlighted_colormap(
                list(self._selected_labels), hide_indices=hide_indices
            )
        )

    def merge_selected_labels(self):
        """Merge the selected label values.

        The merged segments will be given the maximum value of the
        selected labels.
        """
        if len(self._selected_labels) < 2:
            # need at least two labels to merge
            return
        max_selected_label = np.max(list(self._selected_labels))
        non_max_labels = list(self._selected_labels.difference([max_selected_label]))

        self._update_labels(
            labels_to_update=non_max_labels, new_value=max_selected_label
        )

        # update the selection
        self._selected_labels._set.clear()
        self._selected_labels.update([max_selected_label])

    def delete_selected_labels(self):
        """Delete the selected labels.

        The deleted segments will be given the layer's background label value.
        """
        if len(self._selected_labels) == 0:
            # nothing to delete
            return
        self._update_labels(
            labels_to_update=list(self._selected_labels),
            new_value=self._curator_model.labels_layer._background_label,
        )

        # clear the selection
        self._selected_labels.clear()

    def toggle_selected_label_validated(self):
        """Toggle the validated value in the layer features table."""
        labels_layer = self._curator_model.labels_layer
        for label_value in list(self._selected_labels):
            features_index = labels_layer._label_index[label_value]
            labels_layer.features.at[features_index, "mm_validated"] = np.logical_not(
                labels_layer.features.at[features_index, "mm_validated"]
            )

        # get the indices of the validated labels
        features = labels_layer.features
        hide_indices = features.loc[features["mm_validated"]]["index"].values

        # set the colors
        self._curator_model.labels_layer.color = (
            self._curator_model._colormap_manager.create_highlighted_colormap(
                list(self._selected_labels), hide_indices=hide_indices
            )
        )

    def _update_labels(self, labels_to_update: List[int], new_value: int):
        labels_layer = self._curator_model.labels_layer
        update_mask = np.zeros_like(labels_layer.data, dtype=bool)
        for label in list(self._selected_labels):
            update_mask = np.logical_or(update_mask, labels_layer.data == label)

        # record the data to be changed for the undo
        coordinates = np.argwhere(update_mask)
        previous_index = tuple(zip(coordinates.T))
        previous_values = labels_layer.data[update_mask]

        # set the new data
        labels_layer.data[update_mask] = new_value
        labels_layer.refresh()

        # record the changes to the undo
        labels_layer._save_history((previous_index, previous_values, new_value))
