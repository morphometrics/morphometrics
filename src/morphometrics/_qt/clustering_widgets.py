from typing import Dict, List, Optional

import anndata
import napari
import pandas as pd
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

from ..explore.sample import sample_anndata

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
        self._layer = None
        self._label_column = "label_value"
        self._sample_data = None
        self._annotating = False
        self._group_by = None
        self._n_samples_per_group = 10
        self._random_seed = 42

        # mapping of the label values to hotkey
        self._labels_to_hotkeys = dict()

        # mapping of hotkey to label value
        self._hotkeys_to_labels = dict()

        # index of the label in the label image for the selected observation
        self._selected_sample = 0

        # create the start annotating widget
        self._start_annotating_widget = magicgui(
            self.start_annotation,
            group_by={"choices": self._get_group_by_keys},
            call_button=self._start_annotating_string,
        )

        # create the layer selection widgets
        self._layer_selection_widget = magicgui(
            self._select_layer,
            labels_layer={"choices": self._get_labels_layer},
            auto_call=True,
            call_button=False,
        )
        self._layer_selection_widget()

        self._sample_selection_widget = QtSampleSelectWidget()
        self._sample_selection_widget.setVisible(False)
        self._sample_selection_widget._auto_advance_cb.clicked.connect(
            self._on_auto_advance_clicked
        )

        self._label_widget = QtLabelSelectWidget()
        self._label_widget.setVisible(False)

        self.auto_advance = True

        # add widgets to the layout
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self._layer_selection_widget.native)
        self.layout().addWidget(self._start_annotating_widget.native)
        self.layout().addWidget(self._sample_selection_widget)
        self.layout().addWidget(self._label_widget)

    @property
    def layer(self) -> Optional[napari.layers.Labels]:
        return self._layer

    @layer.setter
    def layer(self, layer: Optional[napari.layers.Layer]):
        self._layer = layer
        self._start_annotating_widget.reset_choices()

    @property
    def data(self) -> Optional[anndata.AnnData]:
        if self._layer is None:
            return None
        else:
            return self._layer.metadata["adata"]

    @property
    def annotating(self) -> bool:
        return self._annotating

    @annotating.setter
    def annotating(self, annotating: bool):
        if annotating == self.annotating:
            # if the value is not changing, return
            return
        if self.layer is None:
            self._annotating = False

        if annotating is True:
            self._initialize_annotation()
        else:
            self._teardown_annotation()
        self._annotating = annotating

    @property
    def n_samples(self) -> Optional[int]:
        if self._sample_data is None:
            return None
        return len(self._sample_data)

    @property
    def selected_sample(self) -> int:
        return self._selected_sample

    @selected_sample.setter
    def selected_sample(self, selected_sample):
        if (self._layer is None) or (self._sample_data is None):
            return
        self._selected_sample = selected_sample
        selected_sample_row = self.selected_sample_row
        label_value = selected_sample_row.obs["label"].values[0]
        self._layer.selected_label = label_value
        self._update_sample_selection_widget()

    def next_sample(self):
        """Incrememnt the selected sample"""
        if self._sample_data is None:
            # do nothing if the sample data hasn't been set
            return
        self.selected_sample = (self.selected_sample + 1) % self.n_samples

    def previous_sample(self):
        """Decrement the selected sample"""
        if self._sample_data is None:
            # do nothing if the sample data hasn't been set
            return
        self.selected_sample = (
            (self.selected_sample - 1) + self.n_samples
        ) % self.n_samples

    @property
    def selected_sample_row(self) -> Optional[anndata.AnnData]:
        """The AnnData row corresponded to the currently selected observation"""
        if self._sample_data is None:
            return None
        else:
            return self._sample_data[self.selected_sample, :]

    @property
    def auto_advance(self) -> bool:
        """Flag set to true when the next sample is automatically selected after making an annotation."""
        return self._auto_advance

    @auto_advance.setter
    def auto_advance(self, auto_advance: bool):
        self._sample_selection_widget.update_auto_advance(auto_advance)
        self._auto_advance = auto_advance

    def _select_layer(self, labels_layer: Optional[napari.layers.Labels]):
        self.layer = labels_layer

    def _get_labels_layer(self, combo_widget):
        """Get the labels layers that have an associated anndata object stored
        in the metadata.
        """
        return [
            layer
            for layer in self._viewer.layers
            if (isinstance(layer, napari.layers.Labels)) and ("adata" in layer.metadata)
        ]

    def start_annotation(
        self,
        labels: str = "true_positive, false_negative, false_positive",
        group_by: Optional[str] = None,
        n_samples_per_group: int = 10,
        random_seed: int = 42,
    ):
        if self._layer is None:
            return

        if self.annotating is False:
            if not self._validate_n_samples(
                group_by=group_by, n_samples_per_group=n_samples_per_group
            ):
                raise ValueError(
                    "n_samples_per_group is greater than the number of observations"
                )
            # set the labels in the QtLabelSelectWidget
            self._set_labels(labels)

            self._group_by = group_by
            self._n_samples_per_group = n_samples_per_group
            self._random_seed = random_seed

            self.annotating = True
            self.selected_sample = 0
            # self._start_annotating_widget.call_button.text = self._stop_annotating_string
        else:
            self.annotating = False
            # self._start_annotating_widget.call_button.text = self._start_annotating_string

    def _set_labels(self, label_string: str):
        labels = label_string.replace(" ", "").split(",")
        self._labels_to_hotkeys = {
            label_value: LABEL_HOTKEYS[label_index]
            for label_index, label_value in enumerate(labels)
        }
        self._hotkeys_to_labels = {
            LABEL_HOTKEYS[label_index]: label_value
            for label_index, label_value in enumerate(labels)
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

    def _initialize_annotation(self):
        if self.data is None:
            return

        if self._label_column not in self.data.obs.columns:
            self.data.obs[self._label_column] = ""

        self._sample_data = sample_anndata(
            self.data,
            group_by=self._group_by,
            n_samples_per_group=self._n_samples_per_group,
            random_seed=self._random_seed,
        )

        # setup sample selection widget
        self._connect_sample_selection_widget_events()
        self._sample_selection_widget.setVisible(True)
        self._initialize_sample_selection_widget()

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

        self._update_adata_obs(self.data.obs, self._sample_data.obs[self._label_column])
        self._sample_data = None

    def _initialize_sample_selection_widget(self):
        if self._sample_data is None:
            self._n_samples = 0
            return
        self._n_samples = len(self._sample_data)
        self._update_sample_selection_widget()

    def _update_sample_selection_widget(self):
        obs = self._sample_data.obs
        non_empty_rows = obs.loc[obs[self._label_column] != ""]
        n_labeled = len(non_empty_rows)
        progress = (n_labeled / self._n_samples) * 100

        # update the progress bar
        self._sample_selection_widget._labeling_progress_bar.setValue(progress)

        # update the label
        format_string = self._sample_selection_widget._selection_label_format_string
        new_current_selection = format_string.format(
            n_labeled=self.selected_sample, n_total=self._n_samples - 1
        )
        self._sample_selection_widget._current_selection_label.setText(
            new_current_selection
        )

    def _update_adata_obs(self, obs: pd.DataFrame, new_column: pd.Series):
        column_name = new_column.name
        for index, value in new_column.iteritems():
            obs.at[index, column_name] = value

    def _connect_sample_selection_widget_events(self):
        self._sample_selection_widget._previous_sample_button.clicked.connect(
            self.previous_sample
        )
        self._sample_selection_widget._next_sample_button.clicked.connect(
            self.next_sample
        )

    def _disconnect_sample_selection_widget_events(self):
        self._sample_selection_widget._previous_sample_button.clicked.disconnect(
            self.previous_sample
        )
        self._sample_selection_widget._next_sample_button.clicked.disconnect(
            self.next_sample
        )

    def _validate_n_samples(
        self, group_by: Optional[str], n_samples_per_group: int
    ) -> bool:
        """Check if the number of samples works with the selected table.

        Note: returns False if self._layer is None.
        """
        if self._layer is None:
            return False
        obs = self.data.obs
        if group_by is not None:
            max_counts = obs[group_by].value_counts().min()
        else:
            max_counts = len(obs)
        return n_samples_per_group <= max_counts

    def _get_group_by_keys(self, combo_widget=None) -> List[str]:
        if self._layer is None:
            return []
        return self.data.obs.columns.tolist()

    def _on_auto_advance_clicked(self):
        self._auto_advance = self._sample_selection_widget._auto_advance_cb.isChecked()

    def _label_selected_observation(self, label_value):
        self._sample_data.obs.iat[
            self.selected_sample,
            self._sample_data.obs.columns.get_loc(self._label_column),
        ] = label_value
        print(self._sample_data.obs)

        self._update_sample_selection_widget()
        if self.auto_advance is True:
            self.next_sample()

    def _on_label_q(self, event=None):
        label_value = self._hotkeys_to_labels["q"]
        self._label_selected_observation(label_value)

    def _on_label_w(self, event=None):
        label_value = self._hotkeys_to_labels["w"]
        self._label_selected_observation(label_value)

    def _on_label_e(self, event=None):
        label_value = self._hotkeys_to_labels["e"]
        self._label_selected_observation(label_value)

    def _on_label_r(self, event=None):
        label_value = self._hotkeys_to_labels["r"]
        self._label_selected_observation(label_value)

    def _on_label_t(self, event=None):
        label_value = self._hotkeys_to_labels["t"]
        self._label_selected_observation(label_value)

    def _on_label_y(self, event=None):
        label_value = self._hotkeys_to_labels["y"]
        self._label_selected_observation(label_value)

    def _on_label_u(self, event=None):
        label_value = self._hotkeys_to_labels["u"]
        self._label_selected_observation(label_value)

    def _on_label_i(self, event=None):
        label_value = self._hotkeys_to_labels["i"]
        self._label_selected_observation(label_value)

    def _on_label_o(self, event=None):
        label_value = self._hotkeys_to_labels["o"]
        self._label_selected_observation(label_value)

    def _on_label_p(self, event=None):
        label_value = self._hotkeys_to_labels["p"]
        self._label_selected_observation(label_value)


class QtSampleSelectWidget(QWidget):

    _default_selection_label = "current selection:"
    _selection_label_format_string = (
        _default_selection_label + " {n_labeled} / {n_total}"
    )

    def __init__(self):
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
        self.grid_layout.addWidget(QLabel("labeling progress:"), 2, 0)
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
