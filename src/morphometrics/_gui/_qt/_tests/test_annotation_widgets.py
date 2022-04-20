import anndata
import numpy as np
import pandas as pd

from morphometrics._gui._qt.annotation_widgets import (
    LABEL_HOTKEYS,
    QtClusterAnnotatorWidget,
)


def test_creating_widget(make_napari_viewer):

    rng = np.random.default_rng(42)
    n_rows = 9
    categories = ["cat", "dog", "cow"]

    # create the anndata object
    X = rng.random((n_rows, 5))
    label_values = np.arange(1, n_rows + 1)
    obs = pd.DataFrame({"class": 3 * categories, "label": label_values})
    adata = anndata.AnnData(X=X, obs=obs)

    # make a fake labels image
    label_image = np.zeros((10, 10), dtype=np.uint8)
    for i in label_values:
        label_image[i, i] = i

    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()
    labels_layer = viewer.add_labels(label_image, metadata={"adata": adata})

    # create our widget, passing in the viewer
    widget = QtClusterAnnotatorWidget(viewer)

    assert widget._viewer is viewer

    # start annotating
    model = widget.model
    model.layer = labels_layer
    model.table_source = "anndata"
    n_samples = 2
    annotation_classes = [str(i + 1) for i in range(10)]
    model.start_annotation(
        annotation_classes=annotation_classes,
        group_by="class",
        n_samples_per_group=n_samples,
    )
    assert model.annotating is True

    # annotate the first sample with the first class
    model.auto_advance = True
    assert model.selected_sample == 0
    widget._on_label_q()
    assert model.annotations.iloc[0] == annotation_classes[0]

    # verify the selected sample advanced
    assert model.selected_sample == 1

    # verify that all of the hotkey callbacks work
    for hotkey in LABEL_HOTKEYS.values():
        callback_function = getattr(widget, f"_on_label_{hotkey}")
        callback_function()

    # turn off auto advance from the GUI
    current_index = model.selected_sample
    model.auto_advance = False
    widget._on_label_q()
    assert model.selected_sample == current_index

    # stop annotating
    model.annotating = False
    assert widget._sample_selection_widget.isVisible() is False
    assert widget._label_widget.isVisible() is False
