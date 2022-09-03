import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csc_matrix, csr_matrix

from morphometrics.measure.neighborhood import (
    _calculate_neighborhood_statistics,
    calculate_neighborhood_statistics,
)


def test__calculate_neighborhood_measurements():
    neighbor_graph = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)

    measurements = np.array([[1, 1], [0.5, 2], [0, 3]])
    neighborhood_measurements = _calculate_neighborhood_statistics(
        neighborhood_graph=neighbor_graph,
        measurements=measurements,
    )

    expected_shape = (7,) + measurements.shape
    np.testing.assert_array_equal(neighborhood_measurements.shape, expected_shape)

    # check the means are correct
    expected_means = np.array([[0.5, 2], [0.5, 2], [0.5, 2]])
    np.testing.assert_allclose(neighborhood_measurements[0, ...], expected_means)


neighbor_graph_without_self = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
neighbor_graph_with_self = neighbor_graph_without_self.copy()
np.fill_diagonal(neighbor_graph_with_self, 1)
csc_neighbor_graph_without_self = csc_matrix(neighbor_graph_without_self)
csr_neighbor_graph_without_self = csr_matrix(neighbor_graph_without_self)


@pytest.mark.parametrize(
    "neighbor_graph",
    [
        neighbor_graph_with_self,
        neighbor_graph_without_self,
        csc_neighbor_graph_without_self,
        csr_neighbor_graph_without_self,
    ],
)
def test_calculate_neighborhood_statistics_no_self(neighbor_graph):
    """test calculating neighborhood statistics when self isn't included
    in the neighbhood.
    """
    initial_neighbor_graph = neighbor_graph.copy()

    measurements_array = np.array([[1, 1], [0.5, 2], [0, 3]])
    measurement_table = pd.DataFrame(
        measurements_array, columns=["measurement_0", "measurement_1"]
    )
    neighborhood_measurement = calculate_neighborhood_statistics(
        neighbor_graph=neighbor_graph,
        measurements=measurement_table,
        include_self_in_neighborhood=False,
    )
    expected_number_of_columns = measurement_table.shape[1] * 8
    assert neighborhood_measurement.shape[0] == measurement_table.shape[0]
    assert neighborhood_measurement.shape[1] == expected_number_of_columns

    # check the mean values
    expected_means = np.array([0.5, 0.5, 0.5])
    np.testing.assert_allclose(
        expected_means, neighborhood_measurement["measurement_0_mean"].values
    )

    # verify the original neighbor graph wasn't mutated
    if isinstance(initial_neighbor_graph, np.ndarray):
        np.testing.assert_allclose(neighbor_graph, initial_neighbor_graph)
    else:
        np.testing.assert_allclose(neighbor_graph.A, initial_neighbor_graph.A)


@pytest.mark.parametrize(
    "neighbor_graph",
    [
        neighbor_graph_with_self,
        neighbor_graph_without_self,
        csc_neighbor_graph_without_self,
        csr_neighbor_graph_without_self,
    ],
)
def test_calculate_neighborhood_statistics_with_self(neighbor_graph):
    """test calculating neighborhood statistics when self is included
    in the neighbhood.
    """
    initial_neighbor_graph = neighbor_graph.copy()

    measurements_array = np.array([[1, 1], [0.5, 2], [0, 3]])
    measurement_table = pd.DataFrame(
        measurements_array, columns=["measurement_0", "measurement_1"]
    )
    neighborhood_measurement = calculate_neighborhood_statistics(
        neighbor_graph=neighbor_graph,
        measurements=measurement_table,
        include_self_in_neighborhood=True,
    )
    expected_number_of_columns = measurement_table.shape[1] * 8
    assert neighborhood_measurement.shape[0] == measurement_table.shape[0]
    assert neighborhood_measurement.shape[1] == expected_number_of_columns

    # check the mean values
    expected_means = np.array([0.75, 0.5, 0.25])
    np.testing.assert_allclose(
        expected_means, neighborhood_measurement["measurement_0_mean"].values
    )

    # verify the original neighbor graph wasn't mutated
    if isinstance(initial_neighbor_graph, np.ndarray):
        np.testing.assert_allclose(neighbor_graph, initial_neighbor_graph)
    else:
        np.testing.assert_allclose(neighbor_graph.A, initial_neighbor_graph.A)
