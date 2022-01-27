import numpy as np
import pandas as pd

from morphometrics.types import (
    BinaryImage,
    FeatureImage,
    IntensityImage,
    LabelImage,
    LabelMeasurementTable,
)


def test_intensity_image():
    np_im = np.random.random((10, 10, 10))
    assert isinstance(np_im, IntensityImage)

    list_im = np_im.tolist()
    assert not isinstance(list_im, IntensityImage)


def test_label_image():
    np_im = np.random.random((10, 10, 10))
    assert isinstance(np_im, LabelImage)

    list_im = np_im.tolist()
    assert not isinstance(list_im, LabelImage)


def test_binary_image():
    np_im = np.random.random((10, 10, 10))
    assert isinstance(np_im, BinaryImage)

    list_im = np_im.tolist()
    assert not isinstance(list_im, BinaryImage)


def test_feature_image():
    np_im = np.random.random((10, 10, 10))
    assert isinstance(np_im, FeatureImage)

    list_im = np_im.tolist()
    assert not isinstance(list_im, FeatureImage)


def test_label_measurements_table():
    table_dict = {"label": np.array([0, 1, 2]), "volume": np.array([10, 20, 30])}
    table_df = pd.DataFrame(table_dict).set_index("label")

    assert isinstance(table_df, LabelMeasurementTable)
    assert not isinstance(table_dict, LabelMeasurementTable)
