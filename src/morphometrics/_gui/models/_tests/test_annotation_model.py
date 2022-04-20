import anndata
import napari
import numpy as np
import pandas as pd
import pytest

from morphometrics._gui.models.annotation_model import ClusterAnnotationModel


def make_labels_layer_with_adata():
    rng = np.random.default_rng(42)
    n_rows = 100
    categories = ["cat", "dog", "cow"]

    # create the anndata object
    X = rng.random((n_rows, 5))
    obs = pd.DataFrame(
        {"class": rng.choice(categories, n_rows), "label": np.arange(1, n_rows + 1)}
    )
    adata = anndata.AnnData(X=X, obs=obs)

    label_image = rng.integers(0, 5, (30, 30))
    return napari.layers.Labels(label_image, metadata={"adata": adata})


def make_labels_layer_with_features():
    rng = np.random.default_rng(42)
    n_rows = 100
    categories = ["cat", "dog", "cow"]

    # create the anndata object
    df = pd.DataFrame(
        {
            "measurement_0": rng.random((n_rows,)),
            "measurement_1": rng.random((n_rows,)),
            "measurement_2": rng.random((n_rows,)),
            "label": np.arange(1, n_rows + 1),
            "class": rng.choice(categories, n_rows),
        }
    )
    label_image = rng.integers(0, 5, (30, 30))
    return napari.layers.Labels(label_image, features=df)


labels_adata = make_labels_layer_with_adata()
labels_features = make_labels_layer_with_features()


@pytest.mark.parametrize(
    "labels_layer,table_source",
    [(labels_adata, "anndata"), (labels_features, "layer_features")],
)
def test_cluster_annotation_model(labels_layer, table_source):
    model = ClusterAnnotationModel()
    model.layer = labels_layer

    assert model.annotating is False

    model.table_source = table_source
    n_samples = 10
    annotation_classes = ["true positive", "false negative", "false_positive"]
    model.start_annotation(
        annotation_classes=annotation_classes,
        group_by="class",
        n_samples_per_group=n_samples,
    )
    assert model.annotating is True
    np.testing.assert_equal(annotation_classes, model.annotation_classes)

    model.auto_advance = True
    current_sample = model.selected_sample

    # annotate a sample
    annotation_value = annotation_classes[0]
    model._annotate_selected_sample(annotation_value)

    # verify the sample advanced to the next one
    assert model.selected_sample == (current_sample + 1)
    assert model.annotations.iloc[0] == annotation_value

    # verify the going to the previous sample wraps around
    model.selected_sample = 0
    model.previous_sample()
    assert model.selected_sample == ((n_samples * len(annotation_classes)) - 1)

    # get the index of the annotated sample
    annotated_sample_index = model.annotations.index.tolist()[0]

    # stop annotating
    model.annotating = False

    # verify the annotation was inserted into the original table
    if table_source == "anndata":
        annotation_table = labels_layer.metadata["adata"].obs
    else:
        annotation_table = labels_layer.features
    assert (
        annotation_table.at[annotated_sample_index, "label_value"] == annotation_value
    )
