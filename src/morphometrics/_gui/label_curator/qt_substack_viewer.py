from typing import Optional, Tuple

import napari
import numpy as np
from napari import Viewer
from napari.components.viewer_model import ViewerModel
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QSplitter, QWidget

from morphometrics._gui._qt.multiple_viewer_widget import QtViewerWrapper


class SubstackViewerWidget(QSplitter):
    """The main widget of the example.

    todo: add model for ortho view config
    """

    def __init__(self, viewer: napari.Viewer, side_widget: Optional[QWidget] = None):
        super().__init__()
        self.viewer = viewer

        self.substack_viewer_model = ViewerModel(title="model1")
        self._block = False

        # connect the viewer sync events
        # self._connect_main_viewer_events()
        # self._connect_ortho_viewer_events()

        # add the viewer widgets
        qt_viewers, viewer_splitter = self._setup_ortho_view_qt(
            self.substack_viewer_model, viewer
        )
        self.substack_qt_viewer = qt_viewers
        self.addWidget(viewer_splitter)

        # add a side widget if one was provided
        if side_widget is not None:
            self.addWidget(side_widget)

    def add_image(self, image: np.ndarray):
        self.substack_viewer_model.add_image(image)

    def add_labels(self, label_image: np.ndarray):
        self.substack_viewer_model.add_labels(label_image)

    def _connect_main_viewer_events(self):
        """Connect the update functions to the main viewer events.

        These events sync the ortho viewers with changes in the main viewer.
        """
        self.viewer.layers.events.inserted.connect(self._layer_added)
        self.viewer.layers.events.removed.connect(self._layer_removed)
        self.viewer.layers.events.moved.connect(self._layer_moved)
        self.viewer.layers.selection.events.active.connect(
            self._layer_selection_changed
        )
        self.viewer.dims.events.current_step.connect(self._point_update)
        self.viewer.dims.events.order.connect(self._order_update)
        self.viewer.events.reset_view.connect(self._reset_view)

    def _connect_ortho_viewer_events(self):
        """Connect the update functions to the orthoviewer events.

        These events sync the main viewer with changes in the ortho viewer.
        """
        for model in self.ortho_viewer_models:
            model.dims.events.current_step.connect(self._point_update)
            model.events.status.connect(self._status_update)

    def _setup_ortho_view_qt(
        self, substack_viewer_model: ViewerModel, main_viewer: Viewer
    ) -> Tuple[QtViewerWrapper, QSplitter]:
        # create the QtViewer objects
        qt_viewer = QtViewerWrapper(main_viewer, substack_viewer_model)

        # create and populate the QSplitter for the orthoview QtViewer
        viewer_splitter = QSplitter()
        viewer_splitter.setOrientation(Qt.Vertical)
        viewer_splitter.addWidget(qt_viewer)
        viewer_splitter.setContentsMargins(0, 0, 0, 0)

        return qt_viewer, viewer_splitter

    def _status_update(self, event):
        self.viewer.status = event.value

    def _reset_view(self):
        for model in self.ortho_viewer_models:
            model.reset_view()

    def _layer_selection_changed(self, event):
        """
        update of current active layer
        """
        if self._block:
            return

        if event.value is None:
            for model in self.ortho_viewer_models:
                model.layers.selection.active = None
            return

        for model in self.ortho_viewer_models:
            model.layers.selection.active = model.layers[event.value.name]

    def _point_update(self, event):
        all_viewer_models = [self.viewer] + self.ortho_viewer_models
        for model in all_viewer_models:
            if model.dims is event.source:
                continue
            model.dims.current_step = event.value

    def _order_update(self):
        """Set the dims order for each of the ortho viewers.

        This is used to set the displayed dimensions in each orthview.

        todo: make configurable via ortho view config
        """
        order = list(self.viewer.dims.order)
        if len(order) <= 2:
            for model in self.ortho_viewer_models:
                model.dims.order = order
            return

        order[-3:] = order[-2], order[-3], order[-1]
        self.ortho_viewer_models[1].dims.order = order
        order = list(self.viewer.dims.order)
        order[-3:] = order[-1], order[-2], order[-3]
        self.ortho_viewer_models[0].dims.order = order

        # order[-3:] = order[-2], order[-3], order[-1]
        # self.ortho_viewer_models[0].dims.order = order
        # order = list(self.viewer.dims.order)
        # order[-3:] = order[-1], order[-2], order[-3]
        # self.ortho_viewer_models[1].dims.order = order

    # def _layer_added(self, event):
    #     """add layer to additional viewers and connect all required events.
    #
    #     todo: make configurable with model
    #     """
    #     self.ortho_viewer_models[0].layers.insert(
    #         event.index, copy_layer(event.value, "model1")
    #     )
    #     self.ortho_viewer_models[1].layers.insert(
    #         event.index, copy_layer(event.value, "model2")
    #     )
    #     for name in get_property_names(event.value):
    #         getattr(event.value.events, name).connect(
    #             own_partial(self._property_sync, name)
    #         )
    #
    #     if isinstance(event.value, Labels):
    #         event.value.events.set_data.connect(self._set_data_refresh)
    #         self.ortho_viewer_models[0].layers[
    #             event.value.name
    #         ].events.set_data.connect(self._set_data_refresh)
    #         self.ortho_viewer_models[1].layers[
    #             event.value.name
    #         ].events.set_data.connect(self._set_data_refresh)
    #
    #     event.value.events.name.connect(self._sync_name)
    #
    #     self._order_update()

    def _sync_name(self, event):
        """sync name of layers"""
        index = self.viewer.layers.index(event.source)
        for model in self.ortho_viewer_models:
            self.model.layers[index].name = event.source.name

    def _sync_data(self, event):
        """sync data modification from additional viewers"""
        if self._block:
            return
        all_viewer_models = [self.viewer] + self.ortho_viewer_models
        for model in all_viewer_models:
            layer = model.layers[event.source.name]
            if layer is event.source:
                continue
            try:
                self._block = True
                layer.data = event.source.data
            finally:
                self._block = False

    def _set_data_refresh(self, event):
        """
        synchronize data refresh between layers
        """
        if self._block:
            return
        all_viewer_models = [self.viewer] + self.ortho_viewer_models
        for model in all_viewer_models:
            layer = model.layers[event.source.name]
            if layer is event.source:
                continue
            try:
                self._block = True
                layer.refresh()
            finally:
                self._block = False

    def _layer_removed(self, event):
        """remove layer in all viewers"""
        for model in self.ortho_viewer_models:
            model.layers.pop(event.index)

    def _layer_moved(self, event):
        """update order of layers"""
        dest_index = (
            event.new_index if event.new_index < event.index else event.new_index + 1
        )
        for model in self.ortho_viewer_models:
            model.layers.move(event.index, dest_index)

    def _property_sync(self, name, event):
        """Sync layers properties (except the name)"""
        if event.source not in self.viewer.layers:
            return
        try:
            self._block = True
            for model in self.ortho_viewer_models:
                setattr(
                    model.layers[event.source.name],
                    name,
                    getattr(event.source, name),
                )
        finally:
            self._block = False
