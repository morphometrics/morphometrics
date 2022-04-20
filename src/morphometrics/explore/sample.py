from typing import Optional

import anndata
import pandas as pd


def sample_anndata(
    adata: anndata.AnnData,
    group_by: Optional[str] = None,
    n_samples_per_group: int = 10,
    random_seed: int = 42,
) -> anndata.AnnData:
    """Sample observations from an AnnData table.

    Parameters
    ----------
    adata : anndata.AnnData
        The table to sample from.
    group_by: Optional[str]
        The obs key to group by. If None, the whole table
        will be treated as a single group.
        Default value is None.
    n_samples_per_group : int
        The number of samples to take per group.
    random_seed: int
        The seed for the random number generator used for sampling.

    Returns
    -------
    sampled_rows : anndata.AnnData
        The subset of the AnnData table that was sampled. This is a view
        of the original AnnData table.
    """

    sampled_rows = sample_pandas(
        df=adata.obs,
        group_by=group_by,
        n_samples_per_group=n_samples_per_group,
        random_seed=random_seed,
    )
    sampled_indices = sampled_rows.index.tolist()
    row_numbers = adata.obs.index.get_indexer(sampled_indices)

    return adata[row_numbers, :]


def sample_pandas(
    df: pd.DataFrame,
    group_by: Optional[str] = None,
    n_samples_per_group: int = 10,
    random_seed: int = 42,
):
    """Sample observations from a pandas DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The table to sample from.
    group_by: Optional[str]
        The obs key to group by. If None, the whole table
        will be treated as a single group.
        Default value is None.
    n_samples_per_group : int
        The number of samples to take per group.
    random_seed: int
        The seed for the random number generator used for sampling.

    Returns
    -------
    sampled_rows : pd.DataFrame
        The subset of the DataFrame that was sampled. This is a view
        of the original table.
    """
    if group_by is not None:
        sampled_rows = df.groupby(group_by).sample(
            n=n_samples_per_group, random_state=random_seed, replace=False
        )
    else:
        sampled_rows = df.sample(
            n=n_samples_per_group, random_state=random_seed, replace=False
        )
    return sampled_rows
