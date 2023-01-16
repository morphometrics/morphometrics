from typing import List, Optional

import napari
from magicgui import magicgui, widgets
from napari.qt.threading import FunctionWorker, thread_worker
from napari.types import LayerDataTuple
from napari.utils.events import Event
from qtpy.QtWidgets import (
    QGroupBox,
    QLabel,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from morphometrics._gui.label_curator.label_cleaning import LabelCleaningModel
from morphometrics._gui.label_curator.label_curator import CurationMode, LabelCurator
from morphometrics.label.image_utils import expand_selected_labels_using_crop


class QtPaintWidget(QWidget):
    def __init__(self, curator_model: LabelCurator, visible: bool = False, parent=None):
        super().__init__(parent=parent)
        self._curator_model = curator_model
        self.label = QLabel("paint")

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.label)
        self.layout().addStretch(1)

        self.setVisible(self._curator_model._painting_model.enabled)
        self._curator_model._painting_model.events.enabled.connect(
            self._on_enabled_changed
        )

    def _on_activate(self):
        pass

    def _on_deactivate(self):
        pass

    def _on_enabled_changed(self, event: Event) -> None:
        self.setVisible(self._curator_model._painting_model.enabled)


class QtCleanWidget(QWidget):
    def __init__(self, model: LabelCleaningModel, parent=None):
        super().__init__(parent=parent)
        self._model = model
        self.label = QLabel("clean")

        # button for merging labels
        self._merge_button = QPushButton("merge selected labels")
        self._merge_button.clicked.connect(self._model.merge_selected_labels)

        # button for deleting labels
        self._delete_button = QPushButton("delete selected labels")
        self._delete_button.clicked.connect(self._model.delete_selected_labels)

        # button for toggling validated
        self._validate_button = QPushButton("toggle validation of selected labels")
        self._validate_button.clicked.connect(
            self._model.toggle_selected_label_validated
        )

        # widget for expanding labels

        self._label_expansion_widget = magicgui(
            self._expand_selected_labels_widget_function,
            pbar={"visible": False, "max": 0, "label": "working..."},
            call_button="expand labels",
        )
        expand_group_layout = QVBoxLayout()
        expand_group_layout.addStretch(1)
        expand_group_layout.addWidget(self._label_expansion_widget.native)
        self.expand_labels_box = QGroupBox("expand labels")
        self.expand_labels_box.setLayout(expand_group_layout)

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self._merge_button)
        self.layout().addWidget(self._delete_button)
        self.layout().addWidget(self._validate_button)
        self.layout().addWidget(self.expand_labels_box)
        self.layout().addStretch(1)

        self.setVisible(self._model.enabled)
        self._model.events.enabled.connect(self._on_enabled_changed)

    def _on_activate(self):
        pass

    def _on_deactivate(self):
        pass

    def _on_enabled_changed(self, event: Event) -> None:
        self.setVisible(self._model.enabled)

    def _expand_selected_labels_widget_function(
        self,
        pbar: widgets.ProgressBar,
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
        label_values_to_expand = list(self._model._selected_labels)

        # get the values from the selected labels layer
        label_image = self._model._curator_model.labels_layer.data
        label_layer_name = self._model._curator_model.labels_layer.name
        # if self._model.background_mask_layer is not None:
        #     background_mask = self._curator_model.background_mask_layer.data
        # else:
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


class QtExploreWidget(QWidget):
    def __init__(self, curator_model: LabelCurator, parent=None):
        super().__init__(parent=parent)
        self._curator_model = curator_model
        self.label = QLabel("explore")

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.label)
        self.layout().addStretch(1)

    def _on_activate(self):
        pass

    def _on_deactivate(self):
        pass


class QtLabelingModeWidget(QWidget):
    def __init__(self, curator_model: LabelCurator):
        super().__init__()
        self._curator_model = curator_model
        self.mode_widgets = {
            CurationMode.PAINT: QtPaintWidget(curator_model=curator_model, parent=self),
            CurationMode.CLEAN: QtCleanWidget(
                model=curator_model._cleaning_model, parent=self
            ),
            CurationMode.EXPLORE: QtExploreWidget(
                curator_model=curator_model, parent=self
            ),
        }
        mode_buttons = []
        mode_button_layout = QVBoxLayout()
        for mode, mode_widget in self.mode_widgets.items():
            mode_button = QRadioButton(mode.value)
            mode_button.toggled.connect(self._on_mode_button_clicked)
            mode_button_layout.addWidget(mode_button)
            mode_buttons.append(mode_button)
            mode_widget.setVisible(False)
        self.mode_buttons = mode_buttons

        self.mode_box = QGroupBox("labeling mode")
        mode_button_layout.addStretch(1)
        self.mode_box.setLayout(mode_button_layout)

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.mode_box)

        for widget in self.mode_widgets.values():
            self.layout().addWidget(widget)
        self.layout().addStretch(1)

    def _on_mode_button_clicked(self):
        button = self.sender()
        button_name = button.text()
        self._curator_model.mode = button_name
        # widget = self.mode_widgets[CurationMode(button_name)]
        # if button.isChecked():
        #     widget._on_activate()
        #     widget.setVisible(True)
        # else:
        #     widget._on_deactivate()
        #     widget.setVisible(False)

    def _on_paint_enabled_changed(self, event: Event) -> None:
        self.mode_widgets[CurationMode.PAINT].setVisible(
            self._curator_model._painting_model.enabled
        )


class QtLabelingWidget(QWidget):
    def __init__(self, viewer: napari.Viewer, parent: Optional[QWidget] = None):
        super().__init__(parent=parent)
        self._viewer = viewer
        self._model = LabelCurator(viewer=viewer)

        # make the label selection widget
        self._label_selection_widget = magicgui(
            self._set_labels_layer,
            labels_layer={"choices": self._get_valid_labels_layers},
            auto_call=True,
            call_button=None,
        )

        # get the curation mode widget
        self.curation_mode_widget = QtLabelingModeWidget(curator_model=self._model)

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self._label_selection_widget.native)
        self.layout().addWidget(self.curation_mode_widget)

    def _set_labels_layer(
        self,
        labels_layer: Optional[napari.layers.Labels],
    ) -> None:
        self._model.labels_layer = labels_layer

    def _get_valid_labels_layers(self, combo_box) -> List[napari.layers.Labels]:
        """Helper function that returns a list of the valid labels layers
        for the combobox."""
        return [
            layer
            for layer in self._viewer.layers
            if isinstance(layer, napari.layers.Labels)
        ]
