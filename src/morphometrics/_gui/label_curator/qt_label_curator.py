from typing import List, Optional

import napari
from magicgui import magicgui
from napari.utils.events import Event
from qtpy.QtWidgets import QGroupBox, QLabel, QRadioButton, QVBoxLayout, QWidget

from morphometrics._gui.label_curator import CurationMode, LabelCurator


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
    def __init__(self, curator_model: LabelCurator, parent=None):
        super().__init__(parent=parent)
        self._curator_model = curator_model
        self.label = QLabel("clean")

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.label)
        self.layout().addStretch(1)

        self.setVisible(self._curator_model._cleaning_model.enabled)
        self._curator_model._cleaning_model.events.enabled.connect(
            self._on_enabled_changed
        )

    def _on_activate(self):
        pass

    def _on_deactivate(self):
        pass

    def _on_enabled_changed(self, event: Event) -> None:
        self.setVisible(self._curator_model._cleaning_model.enabled)


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
            CurationMode.CLEAN: QtCleanWidget(curator_model=curator_model, parent=self),
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
