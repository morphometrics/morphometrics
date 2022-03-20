import anndata
import numpy as np
import pandas as pd
import pytest

from morphometrics.utils.anndata_utils import sample_by_obs_column, table_to_anndata


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


def test_sample_by_obs_column():
    rng = np.random.default_rng(42)
    n_rows = 100
    categories = ["cat", "dog", "cow"]

    # create the anndata object
    X = rng.random((n_rows, 5))
    obs = pd.DataFrame({"class": rng.choice(categories, n_rows)})
    adata = anndata.AnnData(X=X, obs=obs)

    n_samples = 10
    sampled_adata, sample_map = sample_by_obs_column(
        adata, column_name="class", n_samples=n_samples
    )

    assert isinstance(sampled_adata, anndata.AnnData)
    assert sampled_adata.X.shape == (30, 5)

    sampled_obs = sampled_adata.obs
    for category in categories:
        assert len(sampled_obs.loc[sampled_obs["class"] == category]) == n_samples

    assert set(sample_map.keys()) == set(categories)
    for key, value in sample_map.items():
        assert len(value) == n_samples


def test_sample_by_obs_column_too_many_samples():
    """Test that all items are returned when a greater number of samples than
    rows in adata are requested.
    """
    rng = np.random.default_rng(42)
    categories = ["cow", "cat"]
    n_per_category = 10
    n_rows = len(categories) * n_per_category

    # create the anndata object
    X = rng.random((n_rows, 5))
    obs_list = []
    for category in categories:
        obs_list.append(np.repeat(category, n_per_category))
    obs = pd.DataFrame({"class": np.concatenate(obs_list)})
    adata = anndata.AnnData(X=X, obs=obs)

    n_samples = n_per_category + 1
    sampled_adata, sample_map = sample_by_obs_column(
        adata, column_name="class", n_samples=n_samples
    )

    assert isinstance(sampled_adata, anndata.AnnData)
    assert sampled_adata.X.shape == (20, 5)

    sampled_obs = sampled_adata.obs
    for category in categories:
        assert len(sampled_obs.loc[sampled_obs["class"] == category]) == n_per_category

    assert set(sample_map.keys()) == set(categories)
    for key, value in sample_map.items():
        assert len(value) == n_per_category
