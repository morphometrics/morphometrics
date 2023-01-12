# from functools import partial
from typing import List, Optional

import napari
from magicgui import magicgui, widgets
from napari.layers.labels._labels_utils import first_nonzero_coordinate
from napari.qt.threading import FunctionWorker, thread_worker
from napari.types import LayerDataTuple
from qtpy.QtGui import QCursor
from qtpy.QtWidgets import QVBoxLayout, QWidget

from ...label.image_utils import expand_selected_labels_using_crop
from ..models.labeling_model import LabelingModel
from .radial_menu import ButtonSpecification, RadialMenu


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

        self._radial_menu: Optional[RadialMenu] = None

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

    def _on_curating_change(self) -> None:
        if self._model.curating is True:
            self._label_expansion_widget.native.setVisible(True)
            self._attach_viewer_callbacks()
        else:
            self._label_expansion_widget.native.setVisible(False)

    def _attach_viewer_callbacks(self) -> None:
        # callback for the radial manu
        self._viewer.bind_key("a", self._open_radial_menu)

    def _dettach_viewer_callbacks(self) -> None:
        self._viewer.bind_key("a", None)

    def _open_radial_menu(self, viewer: napari.viewer.Viewer):
        global_position = QCursor.pos()
        canvas_widget = viewer.window._qt_window._qt_viewer.canvas.native
        canvas_widget_position = canvas_widget.mapFromGlobal(global_position)

        if not (canvas_widget.rect().contains(canvas_widget_position)):
            # only show the menu if the mouse is in the canvas
            return

        if self._radial_menu is not None:
            self._radial_menu.kill()
            self._radial_menu = None

        (
            self._menu_coordinate,
            self._menu_label_index,
        ) = self._get_label_coordinates_under_cursor()
        if self._menu_coordinate is not None:
            self._menu_label_value = self._model.labels_layer.get_value(
                self._menu_coordinate
            )
        button_list = self._get_radial_menu_button_list()
        self._radial_menu = RadialMenu(
            canvas_widget, canvas_widget_position, buttonList=button_list
        )
        self._radial_menu.show()

    def _get_radial_menu_button_list(self) -> List[ButtonSpecification]:
        def hover():
            print("on")

        def unHover():
            print("off")

        button_list = [
            # ButtonSpecification(
            #   name="paint mode 2D", onClick=partial(self._model.set_paint_mode, n_edit_dimensions=2)
            # ),
            # ButtonSpecification(
            #     name="paint mode 3D", onClick=partial(self._model.set_paint_mode, n_edit_dimensions=3)
            # ),
            ButtonSpecification(
                name="expand object", onClick=self._expand_menu_function
            ),
            ButtonSpecification(
                name="delete object",
                onClick=self._delete_label_under_cursor,
                onHoverTrue=hover,
                onHoverFalse=unHover,
            ),
        ]
        return button_list

    def _expand_menu_function(self):
        self._label_expansion_widget(labels_to_expand=str(self._menu_label_value))

    def _delete_label_under_cursor(self) -> None:
        """Delete the label currently under the cursor."""
        if self._menu_coordinate is None:
            # Nothing to delete if there isn't a label under the coordinate
            return
        old_n_edit_dim = self._model.labels_layer.n_edit_dimensions
        old_preserve_labels = self._model.labels_layer.preserve_labels
        self._model.labels_layer.preserve_labels = False
        self._model.labels_layer.n_edit_dimensions = self._viewer.dims.ndisplay
        self._model.labels_layer.fill(self._menu_coordinate, 0)

        # restore settings
        self._model.labels_layer.n_edit_dimensions = old_n_edit_dim
        self._model.labels_layer.preserve_labels = old_preserve_labels

    def _get_label_coordinates_under_cursor(self):
        """Return the data coordinate of the first label under the cursor 2D or 3D.

        In 2D, this is just the cursor position transformed by the layer's
        world_to_data transform.

        In 3D, a ray is cast in data coordinates, and the coordinate of the first
        nonzero value along that ray is returned. If the ray only contains zeros,
        None is returned.

        Returns
        -------
        coordinates : array of int or None
            The data coordinates for the first .
        """
        cursor_position = self._viewer.cursor.position
        view_direction = self._viewer.camera.view_direction
        ndim = len(self._model.labels_layer._dims_displayed)
        if ndim == 2:
            coordinates = self._model.labels_layer.world_to_data(cursor_position)
        else:  # 3d
            start, end = self._model.labels_layer.get_ray_intersections(
                position=cursor_position,
                view_direction=view_direction,
                dims_displayed=self._model.labels_layer._dims_displayed,
                world=True,
            )
            if start is None and end is None:
                return None, None
            coordinates = first_nonzero_coordinate(
                self._model.labels_layer.data, start, end
            )
        label_index = self._model.labels_layer.get_value(
            position=cursor_position,
            view_direction=view_direction,
            dims_displayed=self._model.labels_layer._dims_displayed,
            world=True,
        )
        return coordinates, label_index
