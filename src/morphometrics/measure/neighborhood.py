from typing import List, Optional, Union

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, csr_matrix, issparse

NEIGHBORHOOD_MEASUREMENT_NAME_TEMPLATES: List[str] = [
    "{measurement_name}_mean",
    "{measurement_name}_stdev",
    "{measurement_name}_min",
    "{measurement_name}_max",
    "{measurement_name}_median",
    "{measurement_name}_25th_percentile",
    "{measurement_name}_75th_percentile",
]


def _calculate_neighborhood_statistics(
    neighborhood_graph: np.ndarray, measurements: np.ndarray
) -> np.ndarray:
    output_shape = (7,) + measurements.shape
    neighborhood_measurements = np.zeros(output_shape, dtype=measurements.dtype)
    for row_index, row in enumerate(neighborhood_graph):
        neighbor_measurements = measurements[row.astype(np.bool_), :]
        neighborhood_measurements[0, row_index, :] = np.mean(
            neighbor_measurements, axis=0
        )
        neighborhood_measurements[1, row_index, :] = np.std(
            neighbor_measurements, axis=0
        )
        neighborhood_measurements[2, row_index, :] = np.min(
            neighbor_measurements, axis=0
        )
        neighborhood_measurements[3, row_index, :] = np.max(
            neighbor_measurements, axis=0
        )
        neighborhood_measurements[4, row_index, :] = np.median(
            neighbor_measurements, axis=0
        )
        neighborhood_measurements[5, row_index, :] = np.percentile(
            neighbor_measurements, 0.25
        )
        neighborhood_measurements[6, row_index, :] = np.percentile(
            neighbor_measurements, 0.75
        )
    return neighborhood_measurements


def _convert_neighborhood_measurement_array_to_table(
    neighborhood_measurement_array: np.ndarray,
    measurement_column_names: np.ndarray,
    table_index: Optional[List[Union[int, str]]] = None,
) -> pd.DataFrame:
    """Convert an array of neighborhood measurements to a table.

    Parameters
    ----------
    neighborhood_measurement_array : np.ndarray
        The (m, n, p) array with m neighborhood statistics,
        n objects, and p measurements.
    measurement_column_names : np.ndarray
        (p,) array containing the names for the p measurements.
    table_index : Optional[List[Union[int, str]]]
        The dataframe index for the resulting table.
        The default value is None.
    """
    # convert the measurement array to a flat 2D array
    flat_array = np.column_stack(neighborhood_measurement_array)

    # create the column names
    column_names = []
    for column_template in NEIGHBORHOOD_MEASUREMENT_NAME_TEMPLATES:
        for measurement_name in measurement_column_names:
            column_names.append(
                column_template.format(measurement_name=measurement_name)
            )

    return pd.DataFrame(flat_array, columns=column_names, index=table_index)


def calculate_neighborhood_statistics(
    neighbor_graph: Union[csr_matrix, csc_matrix, np.ndarray],
    measurements: pd.DataFrame,
    include_self_in_neighborhood: bool = True,
) -> pd.DataFrame:
    """Calculate and concatenate the neighborhood statistics of
    a set of measurements.

    Included measurements:
        mean
        standard deviation
        min
        max
        median
        25th percentile
        75th percentile

    Parameters
    ----------
    neighbor_graph : Union[csr_matrix, csc_matrix, np.ndarray]
        The graph encoding the neighborhoods. Should be symmetric
        and undirected.
    measurements : pd.DataFrame
        The measurements to calculate the neighborhood statistics of.
    include_self_in_neighborhood : bool
        If set to True, an object will be included in calculates of
        its neighborhood statistics. Default value is True.

    Returns
    -------
    neighborhood_measurements : pd.DataFrame
        The original measurements appended with the neighborhood statistics.
        Neighborhood statistics are stored as {measurment_name}_{statistic_name}.
    """
    if issparse(neighbor_graph):
        neighbor_graph = neighbor_graph.toarray()

    # set the "self" included in neighborhood as requested
    # make a copy to prevent mutating the input array
    neighbor_graph = neighbor_graph.copy()
    if include_self_in_neighborhood is True:
        np.fill_diagonal(neighbor_graph, 1)
    else:
        np.fill_diagonal(neighbor_graph, 0)

    # calculate the neighborhood statistics
    measurement_array = measurements.to_numpy()
    neighborhood_measurements = _calculate_neighborhood_statistics(
        neighborhood_graph=neighbor_graph, measurements=measurement_array
    )

    # convert the measurements into a table
    column_names = measurements.columns.tolist()
    table_index = measurements.index
    neighborhood_measurement_table = _convert_neighborhood_measurement_array_to_table(
        neighborhood_measurement_array=neighborhood_measurements,
        measurement_column_names=column_names,
        table_index=table_index,
    )

    return pd.concat([measurements, neighborhood_measurement_table], axis=1)
