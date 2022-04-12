from typing import Iterator, List, Optional, Union

import anndata
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


def iterate_over_anndata(adata: anndata.AnnData) -> Iterator[anndata.AnnData]:
    """Iterate over an AnnData table.

    Parameters
    ----------
    adata : anndata.AnnData
        The table to iterate over.

    Yields
    ------
    adata_row : anndata.AnnData
        A row from the AnnData table
    """
    n_rows = len(adata)

    for row_index in range(n_rows):
        yield adata[row_index, :]
