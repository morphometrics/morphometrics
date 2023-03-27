from typing import TYPE_CHECKING, Dict, List

import napari
import numpy as np
from napari.utils.colormaps import color_dict_to_colormap
from napari.utils.events import EmitterGroup, Event, EventedSet

if TYPE_CHECKING:
    from morphometrics._gui.label_curator.label_curator import LabelCurator


class LabelCleaningModel:
    def __init__(self, curator_model: "LabelCurator", enabled: bool = False):
        self._enabled = False
        self._curator_model = curator_model

        self.events = EmitterGroup(source=self, enabled=Event)

        self._selected_labels = EventedSet()
        self._selected_validated_labels = EventedSet()
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

        # check if the validated labels layer is set
        validated_labels_layer_set = self._curator_model._ortho_viewers is not None

        # add the lables layer events
        self._selected_labels.events.changed.connect(self._on_selection_changed)
        self._curator_model.labels_layer.mouse_drag_callbacks.append(
            self._on_click_selection
        )

        # add the validated labels layer events
        if self._curator_model.validated_labels_layer is not None:
            self._selected_validated_labels.events.changed.connect(
                self._on_selection_changed
            )
            self._curator_model.validated_labels_layer.mouse_drag_callbacks.append(
                self._on_click_selection
            )

        labels_layer_name = self._curator_model.labels_layer.name
        if validated_labels_layer_set:
            validated_labels_layer_name = (
                self._curator_model.validated_labels_layer.name
            )
        if validated_labels_layer_set:
            for viewer in self._curator_model._ortho_viewers:
                viewer.layers[labels_layer_name].mouse_drag_callbacks.append(
                    self._on_click_selection
                )
                if validated_labels_layer_set:
                    viewer.layers[
                        validated_labels_layer_name
                    ].mouse_drag_callbacks.append(self._on_click_selection)
        # set the labels layer coloring mode
        self._curator_model.labels_layer.color_mode = "direct"

        if validated_labels_layer_set:
            self._curator_model.validated_labels_layer.color_mode = "direct"

        # set the colormaps
        self._set_all_labels_colormamaps(
            labels_colormap=self._curator_model._colormap_manager._default_colormap,
            validated_labels_colormap=self._curator_model._colormap_manager._default_colormap,
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

        if layer.name == self._curator_model.labels_layer.name:
            selection_set = self._selected_labels
        elif layer.name == self._curator_model.validated_labels_layer.name:
            selection_set = self._selected_validated_labels
        else:
            return

        if (label_index is None) or (label_index == layer._background_label):
            # the background or outside the layer was clicked, clear the selection
            if "shift" not in event.modifiers:
                # don't clear the selection if the shift key was held
                selection_set.clear()
            return
        if "shift" in event.modifiers:
            selection_set.symmetric_difference_update([label_index])
        else:
            selection_set._set.clear()
            selection_set.update([label_index])

    def _on_selection_changed(self, event: Event):
        """Update the colormap based on a new selection"""
        if not self.enabled:
            # don't do anything if not curating
            return

        selected_labels = event.source
        # get the indices of the validated labels
        new_colormap = (
            self._curator_model._colormap_manager.create_highlighted_colormap(
                list(selected_labels)
            )
        )

        if selected_labels is self._selected_labels:
            self._set_labels_layer_colormap(
                labels_colormap=new_colormap,
            )
        elif selected_labels is self._selected_validated_labels:
            self._set_validated_labels_layer_colormap(labels_colormap=new_colormap)
        else:
            raise ValueError("unknown selction model")

    def _set_all_labels_colormamaps(
        self,
        labels_colormap: Dict[int, np.ndarray],
        validated_labels_colormap: Dict[int, np.ndarray],
    ) -> None:
        """Set the label colormaps for all viewers"""
        self._set_labels_layer_colormap(labels_colormap)
        self._set_validated_labels_layer_colormap(validated_labels_colormap)

    def _set_labels_layer_colormap(
        self, labels_colormap: Dict[int, np.ndarray]
    ) -> None:
        self._curator_model.labels_layer.color = labels_colormap

        if self._curator_model._ortho_viewers is None:
            # we can return early if there aren't orthoviewers
            return

        # update the orthoviewers
        labels_layer_name = self._curator_model.labels_layer.name

        for viewer in self._curator_model._ortho_viewers:
            viewer.layers[labels_layer_name].color = labels_colormap

        # # set the main viewers
        # self._fast_set_labels_colormap(layer=self._curator_model.labels_layer, colormap=labels_colormap)
        #
        # if self._curator_model._ortho_viewers is None:
        #     # we can return early if there aren't orthoviewers
        #     return
        #
        # # update the orthoviewers
        # labels_layer_name = self._curator_model.labels_layer.name
        #
        # for viewer in self._curator_model._ortho_viewers:
        #     self._fast_set_labels_colormap(
        #         layer=viewer.layers[labels_layer_name],
        #         colormap=labels_colormap
        #     )

    def _set_validated_labels_layer_colormap(
        self, labels_colormap: Dict[int, np.ndarray]
    ) -> None:
        if self._curator_model.validated_labels_layer is not None:
            self._curator_model.validated_labels_layer.color = labels_colormap

        if self._curator_model._ortho_viewers is None:
            # we can return early if there aren't orthoviewers
            return

        # update the orthoviewers
        validated_labels_layer_name = self._curator_model.validated_labels_layer.name
        for viewer in self._curator_model._ortho_viewers:
            viewer.layers[validated_labels_layer_name].color = labels_colormap

    def _fast_set_labels_colormap(self, layer, colormap):
        if layer._background_label not in colormap:
            colormap[layer._background_label] = np.array([0, 0, 0, 0])
        if None not in colormap:
            colormap[None] = np.array([0, 0, 0, 1])

        # colors = {
        #     label: transform_color(color_str)[0]
        #     for label, color_str in colormap.items()
        # }

        layer._color = colormap
        # set the colormap
        custom_colormap, label_color_index = color_dict_to_colormap(layer.color)
        #
        # # layer._colormap = custom_colormap
        # # layer._label_color_index = label_color_index
        # #
        # # layer._selected_color = layer.get_color(layer.selected_label)

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
        validated_labels_layer = self._curator_model.validated_labels_layer

        selected_layer = self._curator_model._viewer.layers.selection.active
        if selected_layer is labels_layer:
            selected_labels = list(self._selected_labels)
            to_layer = validated_labels_layer
        elif selected_layer is validated_labels_layer:
            selected_labels = list(self._selected_validated_labels)
            to_layer = labels_layer
        else:
            # invalid layer selected
            return

        for label_value in selected_labels:
            # add the mask to the new layer
            label_mask = selected_layer.data == label_value
            to_layer.data[label_mask] = label_value

            # remove the mask from the originating layer
            if selected_layer is validated_labels_layer:
                selected_layer.data[label_mask] = 0

        labels_layer.refresh()
        validated_labels_layer.refresh()
        self._selected_labels.clear()
        self._selected_validated_labels.clear()

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
