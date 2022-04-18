from typing import List, Optional

import anndata
import napari
from magicgui import magicgui
from qtpy.QtCore import Qt
from qtpy.QtGui import QFont
from qtpy.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..explore.sample import sample_anndata


class QtClusterAnnotatorWidget(QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self._viewer = viewer
        self._layer = None
        self._sample_data = None
        self._annotating = False
        self._group_by = None
        self._n_samples_per_group = 10
        self._random_seed = 42

        # create the start annotating widget
        self._start_annotating_widget = magicgui(
            self.start_annotation, group_by={"choices": self._get_group_by_keys}
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
        self._label_widget = QtLabelSelectWidget()

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
            self._stop_annotation()
        self._annotating = annotating

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
        group_by: Optional[str] = None,
        n_samples_per_group: int = 10,
        random_seed: int = 42,
    ):
        if self._layer is None:
            return
        if not self._validate_n_samples(
            group_by=group_by, n_samples_per_group=n_samples_per_group
        ):
            raise ValueError(
                "n_samples_per_group is greater than the number of observations"
            )

        self._group_by = group_by
        self._n_samples_per_group = n_samples_per_group
        self._random_seed = random_seed

        self.annotating = True

    def _initialize_annotation(self):
        if self.data is None:
            return
        self._sample_data = sample_anndata(
            self.data,
            group_by=self._group_by,
            n_samples_per_group=self._n_samples_per_group,
            random_seed=self._random_seed,
        )

    def _stop_annotation(self):
        pass

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

    @property
    def data(self) -> Optional[anndata.AnnData]:
        if self._layer is None:
            return None
        else:
            return self._layer.metadata["adata"]


class QtSampleSelectWidget(QWidget):
    def __init__(self):
        super().__init__()

        self._title_widget = QLabel("Sample:")
        self._title_widget.setFont(QFont("Arial", 16))

        self._previous_sample_button = QPushButton("prev.")
        self._previous_sample_button.sizePolicy().setHorizontalStretch(1)
        self._next_sample_button = QPushButton("next")
        self._next_sample_button.sizePolicy().setHorizontalStretch(1)

        button_row = QHBoxLayout()
        button_row.addWidget(self._previous_sample_button)
        button_row.addWidget(self._next_sample_button)

        self._auto_advance_cb = QCheckBox()
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
        self.grid_layout.addLayout(button_row, 1, 0, 1, 4)
        self.grid_layout.addWidget(self._auto_advance_label, 2, 0, 1, 3)
        self.grid_layout.addWidget(self._auto_advance_cb, 2, 1, 1, 1)

        self.grid_layout.setRowStretch(3, 1)


class QtLabelSelectWidget(QWidget):
    def __init__(self):
        super().__init__()

        self._title_widget = QLabel("Label:")
        self._title_widget.setFont(QFont("Arial", 16))

        hotkey_label = QLabel("hot key")
        hotkey_label.setStyleSheet("font-weight: bold")

        value_label = QLabel("label value")
        value_label.setStyleSheet("font-weight: bold")

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
        self.grid_layout.addWidget(QLabel("q"), 2, 0)
        self.grid_layout.addWidget(QPushButton("A"), 2, 1)

        self.grid_layout.setRowStretch(3, 1)
