import anndata
import numpy as np
import pandas as pd
import pytest

from morphometrics.utils.anndata_utils import iterate_over_anndata, table_to_anndata


def make_test_label_measurements_table_no_nan():
    records = {
        "label_index": [1, 2, 4, 5],
        "area": [10, 10, 20, 20],
        "intensity": [0.1, 0.5, 0.3, 1],
        "class": ["cat", "dog", "cat", "dog"],
    }

    measurement_table = pd.DataFrame(records).set_index("label_index")
    expected_x = measurement_table[["area", "intensity"]].to_numpy()
    return measurement_table, expected_x


def make_test_label_measurements_table_nan(fill_na_value: float = 3):
    records = {
        "label_index": [1, 2, 4, 5],
        "area": [10, 10, 20, 20],
        "intensity": [np.nan, 0.5, 0.3, np.nan],
        "class": ["cat", "dog", "cat", "dog"],
    }

    measurement_table = pd.DataFrame(records).set_index("label_index")
    expected_x = (
        measurement_table[["area", "intensity"]].fillna(fill_na_value).to_numpy()
    )
    return measurement_table, expected_x, fill_na_value


@pytest.mark.parametrize("columns_to_obs", ["class", ["class"]])
def test_label_measurements_to_anndata_no_obs(columns_to_obs):
    measurement_table, expected_x = make_test_label_measurements_table_no_nan()

    adata = table_to_anndata(measurement_table, columns_to_obs=columns_to_obs)

    assert set(adata.obs.columns) == {"class"}
    assert set(adata.var.index) == {"area", "intensity"}
    np.testing.assert_allclose(adata.X, expected_x)


def test_label_measurements_to_anndata_obs():
    measurement_table, expected_x = make_test_label_measurements_table_no_nan()

    obs = pd.DataFrame(
        {"label_index": [1, 2, 4, 5], "category": ["big", "small", "big", "small"]}
    ).set_index("label_index")

    adata = table_to_anndata(measurement_table, columns_to_obs="class", obs=obs)
    assert set(adata.obs.columns) == {"class", "category"}
    assert set(adata.var.index) == {"area", "intensity"}
    np.testing.assert_allclose(adata.X, expected_x)


def test_label_measurements_to_anndata_fill_nan():
    (
        measurement_table,
        expected_x,
        fill_na_value,
    ) = make_test_label_measurements_table_nan()

    adata = table_to_anndata(
        measurement_table, columns_to_obs="class", fill_na_value=fill_na_value
    )

    assert set(adata.obs.columns) == {"class"}
    assert set(adata.var.index) == {"area", "intensity"}
    np.testing.assert_allclose(adata.X, expected_x)


def test_iterate_over_anndata():
    rng = np.random.default_rng(42)
    n_rows = 100
    categories = ["cat", "dog", "cow"]

    # create the anndata object
    X = rng.random((n_rows, 5))
    obs = pd.DataFrame({"class": rng.choice(categories, n_rows)})
    adata = anndata.AnnData(X=X, obs=obs)

    n_iterations = 0
    for row in iterate_over_anndata(adata):
        assert len(row) == 1
        assert isinstance(row, anndata.AnnData)
        n_iterations += 1
    assert n_iterations == n_rows
