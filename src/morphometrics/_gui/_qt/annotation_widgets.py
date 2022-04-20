from typing import Dict, List, Optional

import napari
from magicgui import magicgui
from qtpy.QtCore import Qt
from qtpy.QtGui import QFont
from qtpy.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..models.annotation_model import ClusterAnnotationModel, TableSource

LABEL_HOTKEYS = {
    0: "q",
    1: "w",
    2: "e",
    3: "r",
    4: "t",
    5: "y",
    6: "u",
    7: "i",
    8: "o",
    9: "p",
}


class QtClusterAnnotatorWidget(QWidget):

    _start_annotating_string = "start annotating"
    _stop_annotating_string = "stop annotating"

    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self._viewer = viewer

        # mapping of the label values to hotkey
        self._labels_to_hotkeys = dict()

        # mapping of hotkey to label value
        self._hotkeys_to_labels = dict()

        self.model = ClusterAnnotationModel()

        # connect the model events
        self.model.events.layer.connect(self._on_layer_update)
        self.model.events.selected_sample.connect(self._on_selected_sample)
        self.model.events.annotating.connect(self._on_annotating)
        self.model.events.annotation_classes.connect(self._on_annotation_classes)
        self.model.events.auto_advance.connect(self._on_auto_advance)
        self.model.events.sample_annotated.connect(self._on_selected_sample)

        # create the start annotating widget
        self._start_annotating_widget = magicgui(
            self.model.start_annotation,
            group_by={"choices": self.model._get_group_by_keys},
            call_button=self._start_annotating_string,
        )

        # create the layer selection widgets
        self._layer_selection_widget = magicgui(
            self._select_layer,
            labels_layer={"choices": self._get_labels_layer},
            auto_call=True,
            call_button=False,
        )
        self._feature_table_selection_widget = magicgui(
            self._select_table_source,
            table_source={"choices": self._table_table_source_choices},
            auto_call=True,
            call_button=False,
        )

        self._layer_selection_widget()

        self._sample_selection_widget = QtSampleSelectWidget(
            auto_advance=self.model.auto_advance
        )
        self._sample_selection_widget.setVisible(False)
        self._sample_selection_widget._auto_advance_cb.clicked.connect(
            self._on_auto_advance_clicked
        )

        self._label_widget = QtLabelSelectWidget()
        self._label_widget.setVisible(False)

        # add widgets to the layout
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self._layer_selection_widget.native)
        self.layout().addWidget(self._feature_table_selection_widget.native)
        self.layout().addWidget(self._start_annotating_widget.native)
        self.layout().addWidget(self._sample_selection_widget)
        self.layout().addWidget(self._label_widget)

    def _on_annotating(self, event=None):
        """Update the GUI when changing annotating state"""
        if self.model.annotating is True:
            self._setup_annotation()
            self._start_annotating_widget.call_button.text = (
                self._stop_annotating_string
            )
        else:
            self._teardown_annotation()
            self._start_annotating_widget.call_button.text = (
                self._start_annotating_string
            )

    def _select_layer(self, labels_layer: Optional[napari.layers.Labels]):
        self.model.layer = labels_layer

    def _on_layer_update(self, event=None):
        """When the model updates the selected layer, update the relevant widgets."""
        self._feature_table_selection_widget.reset_choices()
        self._start_annotating_widget.reset_choices()

    def _get_labels_layer(self, combo_widget):
        """Get the labels layers that have an associated anndata object stored
        in the metadata.
        """
        return [
            layer
            for layer in self._viewer.layers
            if (isinstance(layer, napari.layers.Labels)) and ("adata" in layer.metadata)
        ]

    def _select_table_source(self, table_source: TableSource):
        if table_source is None:
            # if  table source is selectable, don't update the model
            return
        self.model.table_source = table_source

    def _table_table_source_choices(self, combo_widget) -> List[TableSource]:
        table_source_choices = []

        if self.model.layer is not None:
            if "adata" in self.model.layer.metadata:
                table_source_choices.append(TableSource.ANNDATA.value)
            if len(self.model.layer.features) > 0:
                table_source_choices.append(TableSource.LAYER_FEATURES.value)

        return table_source_choices

    def _on_annotation_classes(self, annotation_classes: List[str]):
        annotation_classes = self.model.annotation_classes
        self._labels_to_hotkeys = {
            label_value: LABEL_HOTKEYS[label_index]
            for label_index, label_value in enumerate(annotation_classes)
        }
        self._hotkeys_to_labels = {
            LABEL_HOTKEYS[label_index]: label_value
            for label_index, label_value in enumerate(annotation_classes)
        }
        self._label_widget.set_labels(self._hotkeys_to_labels)
        self._attach_label_hotkeys_and_callbacks()

    def _attach_label_hotkeys_and_callbacks(self):
        for label_button in self._label_widget._label_buttons:
            label_value = label_button.text()
            hotkey = self._labels_to_hotkeys[label_value]
            label_callback = getattr(self, f"_on_label_{hotkey}")

            # add the labeling callback to the button press
            label_button.clicked.connect(label_callback)

            # add the hotkey to the viewer
            self._viewer.bind_key(hotkey, label_callback)

    def _detach_label_hotkeys_and_callbacks(self):
        for label_button in self._label_widget._label_buttons:
            label_value = label_button.text()
            hotkey = self._labels_to_hotkeys[label_value]
            label_callback = getattr(self, f"_on_label_{hotkey}")

            # remove the labeling callback from the button press
            label_button.clicked.disconnect(label_callback)

            # disconnect the hotkey from the viewer
            self._viewer.bind_key(hotkey, None)

    def _setup_annotation(self):
        """Set up the GUI for annotation"""
        # setup sample selection widget
        self._connect_sample_selection_widget_events()
        self._sample_selection_widget.setVisible(True)
        self._on_selected_sample()

        # make the labeling widget visible
        self._label_widget.setVisible(True)

    def _teardown_annotation(self):
        """Clean up after annotation is completed.

        This includes:
            - hiding widgets
            - disconnecting callbacks and events
            - adding the annotations back into the original anndata object
        """
        # remove the viewer callbacks
        self._detach_label_hotkeys_and_callbacks()

        # tear down sample selection widget
        self._disconnect_sample_selection_widget_events()
        self._sample_selection_widget.setVisible(False)

        # hide the labeling widget
        self._label_widget.setVisible(False)
        self._label_widget.clear_labels()

    def _on_selected_sample(self):
        if self.model._sample_data is None:
            return
        sample_annotation_table = self.model.sample_annotation_table
        n_labeled = sample_annotation_table[self.model._label_column].count()
        progress = (n_labeled / self.model.n_samples) * 100

        # update the progress bar
        self._sample_selection_widget._labeling_progress_bar.setValue(progress)

        # update the label
        format_string = self._sample_selection_widget._selection_label_format_string
        new_current_selection = format_string.format(
            n_labeled=self.model.selected_sample, n_total=self.model.n_samples - 1
        )
        self._sample_selection_widget._current_selection_label.setText(
            new_current_selection
        )

    def _connect_sample_selection_widget_events(self):
        self._sample_selection_widget._previous_sample_button.clicked.connect(
            self.model.previous_sample
        )
        self._sample_selection_widget._next_sample_button.clicked.connect(
            self.model.next_sample
        )

    def _disconnect_sample_selection_widget_events(self):
        self._sample_selection_widget._previous_sample_button.clicked.disconnect(
            self.model.previous_sample
        )
        self._sample_selection_widget._next_sample_button.clicked.disconnect(
            self.model.next_sample
        )

    def _on_auto_advance(self, event=None):
        self._sample_selection_widget._auto_advance_cb.setChecked(
            self.model.auto_advance
        )

    def _on_auto_advance_clicked(self):
        self.model._auto_advance = (
            self._sample_selection_widget._auto_advance_cb.isChecked()
        )

    def _on_label_q(self, event=None):
        label_value = self._hotkeys_to_labels["q"]
        self.model._annotate_selected_sample(label_value)

    def _on_label_w(self, event=None):
        label_value = self._hotkeys_to_labels["w"]
        self.model._annotate_selected_sample(label_value)

    def _on_label_e(self, event=None):
        label_value = self._hotkeys_to_labels["e"]
        self.model._annotate_selected_sample(label_value)

    def _on_label_r(self, event=None):
        label_value = self._hotkeys_to_labels["r"]
        self.model._annotate_selected_sample(label_value)

    def _on_label_t(self, event=None):
        label_value = self._hotkeys_to_labels["t"]
        self.model._annotate_selected_sample(label_value)

    def _on_label_y(self, event=None):
        label_value = self._hotkeys_to_labels["y"]
        self.model._annotate_selected_sample(label_value)

    def _on_label_u(self, event=None):
        label_value = self._hotkeys_to_labels["u"]
        self.model._annotate_selected_sample(label_value)

    def _on_label_i(self, event=None):
        label_value = self._hotkeys_to_labels["i"]
        self.model._annotate_selected_sample(label_value)

    def _on_label_o(self, event=None):
        label_value = self._hotkeys_to_labels["o"]
        self.model._annotate_selected_sample(label_value)

    def _on_label_p(self, event=None):
        label_value = self._hotkeys_to_labels["p"]
        self.model._annotate_selected_sample(label_value)


class QtSampleSelectWidget(QWidget):

    _default_selection_label = "current selection:"
    _selection_label_format_string = (
        _default_selection_label + " {n_labeled} / {n_total}"
    )

    def __init__(self, auto_advance: bool = True):
        super().__init__()

        self._title_widget = QLabel("Sample:")
        self._title_widget.setFont(QFont("Arial", 16))

        self._current_selection_label = QLabel(self._default_selection_label)

        self._labeling_progress_bar = QProgressBar(
            minimum=0, maximum=100, textVisible=True
        )

        self._previous_sample_button = QPushButton("prev.")
        self._previous_sample_button.sizePolicy().setHorizontalStretch(1)
        self._next_sample_button = QPushButton("next")
        self._next_sample_button.sizePolicy().setHorizontalStretch(1)

        button_row = QHBoxLayout()
        button_row.addWidget(self._previous_sample_button)
        button_row.addWidget(self._next_sample_button)

        self._auto_advance_cb = QCheckBox()
        self._auto_advance_cb.setTristate(False)
        self._auto_advance_cb.setChecked(auto_advance)
        self._auto_advance_label = QLabel("auto advance:")
        self._auto_advance_label.setToolTip(
            "automatically advance to next label after annotation"
        )
        self._auto_advance_label.setAlignment(Qt.AlignLeft)

        # setup the layout
        self.grid_layout = QGridLayout(self)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_layout.setSpacing(2)
        self.grid_layout.setColumnMinimumWidth(0, 94)
        self.grid_layout.setSpacing(4)
        self.setLayout(self.grid_layout)

        # add widgets to layout
        # addWidget(widget, row, column, [row_span, column_span])
        self.grid_layout.addWidget(self._title_widget, 0, 0, 1, 4)
        self.grid_layout.addWidget(self._current_selection_label, 1, 0)
        self.grid_layout.addWidget(QLabel("annotation progress:"), 2, 0)
        self.grid_layout.addWidget(self._labeling_progress_bar, 2, 1)
        self.grid_layout.addLayout(button_row, 3, 0, 1, 4)
        self.grid_layout.addWidget(self._auto_advance_label, 4, 0, 1, 3)
        self.grid_layout.addWidget(self._auto_advance_cb, 4, 1, 1, 1)

        self.grid_layout.setRowStretch(5, 1)

    def update_auto_advance(self, auto_advance: bool):
        self._auto_advance_cb.blockSignals(True)
        self._auto_advance_cb.setChecked(auto_advance)
        self._auto_advance_cb.blockSignals(False)


class QtLabelSelectWidget(QWidget):
    def __init__(self):
        super().__init__()

        self._label_buttons = []

        self._title_widget = QLabel("Label:")
        self._title_widget.setFont(QFont("Arial", 16))

        hotkey_label = QLabel("hot key")
        hotkey_label.setStyleSheet("font-weight: bold")
        hotkey_label.setAlignment(Qt.AlignCenter)

        value_label = QLabel("label value")
        value_label.setStyleSheet("font-weight: bold")
        value_label.setAlignment(Qt.AlignCenter)

        # setup the layout
        self.grid_layout = QGridLayout(self)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_layout.setSpacing(2)
        self.grid_layout.setColumnMinimumWidth(0, 94)
        self.grid_layout.setSpacing(4)
        self.setLayout(self.grid_layout)

        # add widgets to layout
        # addWidget(widget, row, column, [row_span, column_span])
        self.grid_layout.addWidget(self._title_widget, 0, 0, 1, 4)
        self.grid_layout.addWidget(hotkey_label, 1, 0)
        self.grid_layout.addWidget(value_label, 1, 1)

        self.grid_layout.setRowStretch(2, 1)

    def set_labels(self, labels: Dict[str, str]):
        for label_index, (label_hotkey, label_value) in enumerate(labels.items()):
            grid_row = label_index + 2
            label_hotkey_qlabel = QLabel(label_hotkey)
            label_hotkey_qlabel.setAlignment(Qt.AlignCenter)
            label_button = QPushButton(label_value)
            self._label_buttons.append(label_button)

            # add the elements to the layout
            self.grid_layout.addWidget(label_hotkey_qlabel, grid_row, 0)
            self.grid_layout.addWidget(label_button, grid_row, 1)

        n_labels = len(labels)
        self.grid_layout.setRowStretch(n_labels + 1, 1)

    def clear_labels(self):
        n_rows = len(self._label_buttons)

        for label_index in range(n_rows):
            for column in range(2):
                grid_row = label_index + 2
                layout = self.grid_layout.itemAtPosition(grid_row, column)
                if layout is not None:
                    layout.widget().deleteLater()
                    self.grid_layout.removeItem(layout)
        self._label_buttons = []
