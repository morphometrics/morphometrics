import anndata
import numpy as np
import pandas as pd
import pytest

from morphometrics.explore.sample import sample_anndata, sample_pandas


def test_sample_by_obs_column():
    rng = np.random.default_rng(42)
    n_rows = 100
    categories = ["cat", "dog", "cow"]

    # create the anndata object
    X = rng.random((n_rows, 5))
    obs = pd.DataFrame({"class": rng.choice(categories, n_rows)})
    adata = anndata.AnnData(X=X, obs=obs)

    n_samples = 10
    sampled_adata = sample_anndata(
        adata, group_by="class", n_samples_per_group=n_samples
    )

    assert isinstance(sampled_adata, anndata.AnnData)
    assert sampled_adata.X.shape == (30, 5)

    sampled_obs = sampled_adata.obs
    for category in categories:
        assert len(sampled_obs.loc[sampled_obs["class"] == category]) == n_samples


def test_sample_by_obs_column_too_many_samples():
    """Test that a ValueError is raised if too many samples are requested."""
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
    with pytest.raises(ValueError):
        _ = sample_anndata(adata, group_by="class", n_samples_per_group=n_samples)


def test_sample_from_pandas_dataframe_groupby():
    rng = np.random.default_rng(42)
    n_rows = 100
    categories = ["cat", "dog", "cow"]

    # create the anndata object
    df = pd.DataFrame(
        {
            "measurement_0": rng.random((n_rows,)),
            "measurement_1": rng.random((n_rows,)),
            "measurement_2": rng.random((n_rows,)),
            "class": rng.choice(categories, n_rows),
        }
    )

    n_samples = 10
    sampled_rows = sample_pandas(df, group_by="class", n_samples_per_group=n_samples)

    assert isinstance(sampled_rows, pd.DataFrame)
    assert len(sampled_rows) == (len(categories) * n_samples)

    value_counts = sampled_rows["class"].value_counts()
    assert value_counts["cat"] == n_samples
    assert value_counts["dog"] == n_samples
    assert value_counts["cow"] == n_samples


def test_sample_from_pandas_dataframe_no_groupby():
    rng = np.random.default_rng(42)
    n_rows = 100
    categories = ["cat", "dog", "cow"]

    # create the anndata object
    df = pd.DataFrame(
        {
            "measurement_0": rng.random((n_rows,)),
            "measurement_1": rng.random((n_rows,)),
            "measurement_2": rng.random((n_rows,)),
            "class": rng.choice(categories, n_rows),
        }
    )

    n_samples = 10
    sampled_rows = sample_pandas(df, group_by=None, n_samples_per_group=n_samples)

    assert isinstance(sampled_rows, pd.DataFrame)
    assert len(sampled_rows) == n_samples
