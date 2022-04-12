import anndata
import numpy as np
import pandas as pd
import pytest

from morphometrics.explore.sample import sample_anndata


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
