from copy import deepcopy
from typing import List, Optional, Tuple

import napari
from napari import Viewer
from napari.components.viewer_model import ViewerModel
from napari.layers import Image, Labels, Layer
from napari.qt import QtViewer
from napari.utils.events.event import WarningEmitter
from packaging.version import parse as parse_version
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QSplitter, QWidget

NAPARI_GE_4_16 = parse_version(napari.__version__) > parse_version("0.4.16")


def copy_layer_le_4_16(layer: Layer, name: str = ""):
    res_layer = deepcopy(layer)
    # this deepcopy is not optimal for labels and images layers
    if isinstance(layer, (Image, Labels)):
        res_layer.data = layer.data

    res_layer.metadata["viewer_name"] = name

    res_layer.events.disconnect()
    res_layer.events.source = res_layer
    for emitter in res_layer.events.emitters.values():
        emitter.disconnect()
        emitter.source = res_layer
    return res_layer


def copy_layer(layer: Layer, name: str = ""):
    if NAPARI_GE_4_16:
        return copy_layer_le_4_16(layer, name)

    res_layer = Layer.create(*layer.as_layer_data_tuple())
    res_layer.metadata["viewer_name"] = name
    return res_layer


def get_property_names(layer: Layer):
    klass = layer.__class__
    res = []
    for event_name, event_emitter in layer.events.emitters.items():
        if isinstance(event_emitter, WarningEmitter):
            continue
        if event_name in ("thumbnail", "name"):
            continue
        if (
            isinstance(getattr(klass, event_name, None), property)
            and getattr(klass, event_name).fset is not None
        ):
            res.append(event_name)
    return res


class own_partial:
    """
    Workaround for deepcopy not copying partial functions
    (Qt widgets are not serializable)
    """

    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        return self.func(*(self.args + args), **{**self.kwargs, **kwargs})

    def __deepcopy__(self, memodict={}):
        return own_partial(
            self.func,
            *deepcopy(self.args, memodict),
            **deepcopy(self.kwargs, memodict),
        )


class QtViewerWrapper(QtViewer):
    def __init__(self, main_viewer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.main_viewer = main_viewer

    def _qt_open(
        self,
        filenames: list,
        stack: bool,
        plugin: str = None,
        layer_type: str = None,
        **kwargs,
    ):
        """for drag and drop open files"""
        self.main_viewer.window._qt_viewer._qt_open(
            filenames, stack, plugin, layer_type, **kwargs
        )


class MultipleViewerWidget(QSplitter):
    """The main widget of the example.

    todo: add model for ortho view config
    """

    def __init__(self, viewer: napari.Viewer, side_widget: Optional[QWidget] = None):
        super().__init__()
        self.viewer = viewer

        viewer_model1 = ViewerModel(title="model1")
        viewer_model2 = ViewerModel(title="model2")
        self.ortho_viewer_models = [viewer_model1, viewer_model2]
        self._block = False

        # connect the viewer sync events
        self._connect_main_viewer_events()
        self._connect_ortho_viewer_events()

        # add the viewer widgets
        qt_viewers, viewer_splitter = self._setup_ortho_view_qt(
            self.ortho_viewer_models, viewer
        )
        self.ortho_qt_viewers = qt_viewers
        self.addWidget(viewer_splitter)

        # add a side widget if one was provided
        if side_widget is not None:
            self.addWidget(side_widget)

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
        self, ortho_viewer_models: List[ViewerModel], main_viewer: Viewer
    ) -> Tuple[List[QtViewerWrapper], QSplitter]:
        # create the QtViewer objects
        qt_viewers = [
            QtViewerWrapper(main_viewer, model) for model in ortho_viewer_models
        ]

        # create and populate the QSplitter for the orthoview QtViewer
        viewer_splitter = QSplitter()
        viewer_splitter.setOrientation(Qt.Vertical)
        for qt_viewer in qt_viewers:
            viewer_splitter.addWidget(qt_viewer)
        viewer_splitter.setContentsMargins(0, 0, 0, 0)

        return qt_viewers, viewer_splitter

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
        self.ortho_viewer_models[0].dims.order = order
        order = list(self.viewer.dims.order)
        order[-3:] = order[-1], order[-2], order[-3]
        self.ortho_viewer_models[1].dims.order = order

    def _layer_added(self, event):
        """add layer to additional viewers and connect all required events.

        todo: make configurable with model
        """
        self.ortho_viewer_models[0].layers.insert(
            event.index, copy_layer(event.value, "model1")
        )
        self.ortho_viewer_models[1].layers.insert(
            event.index, copy_layer(event.value, "model2")
        )
        for name in get_property_names(event.value):
            getattr(event.value.events, name).connect(
                own_partial(self._property_sync, name)
            )

        if isinstance(event.value, Labels):
            event.value.events.set_data.connect(self._set_data_refresh)
            self.ortho_viewer_models[0].layers[
                event.value.name
            ].events.set_data.connect(self._set_data_refresh)
            self.ortho_viewer_models[1].layers[
                event.value.name
            ].events.set_data.connect(self._set_data_refresh)

        event.value.events.name.connect(self._sync_name)

        self._order_update()

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
