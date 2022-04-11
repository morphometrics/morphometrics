from typing import Any, Dict, List, Optional, Tuple, Union

import anndata
import numpy as np
import pandas as pd

from ..types import LabelMeasurementTable


def table_to_anndata(
    measurement_table: Union[pd.DataFrame, LabelMeasurementTable],
    columns_to_obs: Optional[List[str]] = None,
    obs: Optional[pd.DataFrame] = None,
    fill_na_value: float = 0,
) -> anndata.AnnData:
    """Convert a LabelMeasurementTable to anndata.AnnData object.

    This is useful for clustering and data exploration workflows.
    For details on the AnnData table, see the docs:
    https://anndata.readthedocs.io/en/latest/


    Parameters
    ----------
    measurement_table : LabelMeasurementTable
        The measurements to be convered to an AnnData object.
    columns_to_obs : Optional[List[str]]
        A list of column names that should be stored in obs instead of X.
        Typically, these are measurements and annotations you do not want
        to use in clustering or downstream analysis. If None, no columns
        will be moved to obs. The default value is None.
    obs : Optional[pd.DataFrame]
        Data to be stored in AnnData.obs.
    fill_na_value : float
        The value to replace nan with in X. Default value is 0.

    Returns
    -------
    adata : anndata.AnnData
        The converted AnnData object.
    """

    #
    if columns_to_obs is not None:
        if isinstance(columns_to_obs, str):
            columns_to_obs = [columns_to_obs]
        obs_from_measurement_table = pd.concat(
            [measurement_table.pop(column) for column in columns_to_obs], axis=1
        )
        if obs is not None:
            obs = pd.concat((obs, obs_from_measurement_table), axis=1)
        else:
            obs = obs_from_measurement_table

    var = pd.DataFrame(index=measurement_table.columns)
    X = measurement_table.fillna(fill_na_value).to_numpy()

    return anndata.AnnData(X=X, var=var, obs=obs)


def sample_by_obs_column(
    adata: anndata.AnnData,
    column_name: str,
    n_samples: int = 10,
    random_seed: float = 42,
) -> Tuple[anndata.AnnData, Dict[Any, np.ndarray]]:
    """Get random samples from each category in a categorical column in AnnData.obs.

    This is useful for sampling observations from clusters.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object to sample from.
    column_name : str
        The name of the obs column to sample from. n_samples from each category
        in the obs column will be drawn.
    n_samples : int
        The number of samples to draw per category. If n_samples is greater
        than the number of observations in a given category, all observations
        from that category will be drawn.
    random_seed : float
        The seed to be used for the random sampling. Sampling is performed
        by the numpy default_rng:
        https://numpy.org/doc/stable/reference/random/generator.html
    """
    obs_column_values = adata.obs[column_name].to_numpy()
    obs_column_categories = adata.obs[column_name].unique()

    sample_map = dict()
    rng = np.random.default_rng(random_seed)
    for category in obs_column_categories:
        category_indices = np.argwhere(obs_column_values == category).ravel()

        if n_samples > len(category_indices):
            n_category_samples = len(category_indices)
        else:
            n_category_samples = n_samples
        sample_map[category] = rng.choice(
            category_indices, n_category_samples, replace=False
        )
    all_sample_indices = np.concatenate([value for value in sample_map.values()])
    return adata[all_sample_indices, :], sample_map
