from typing import Optional

import anndata


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
    if group_by is not None:
        sampled_rows = adata.obs.groupby(group_by).sample(
            n=n_samples_per_group, random_state=random_seed, replace=False
        )
    else:
        sampled_rows = adata.obs.sample(
            n=n_samples_per_group, random_state=random_seed, replace=False
        )

    sampled_indices = sampled_rows.index.tolist()
    row_numbers = adata.obs.index.get_indexer(sampled_indices)

    return adata[row_numbers, :]
