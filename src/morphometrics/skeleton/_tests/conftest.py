from typing import Callable, Tuple

import networkx as nx
import numpy as np
import pytest
from morphosamplers.spline import Spline3D

from morphometrics.skeleton.constants import (
    EDGE_COORDINATES_KEY,
    EDGE_SPLINE_KEY,
    NODE_COORDINATE_KEY,
)


@pytest.fixture
def make_bare_skeleton_graph() -> Callable[[], Tuple[nx.Graph, np.ndarray]]:
    """Make a skeleton graph with no properties."""

    def factory_function():
        node_coordinates = np.array(
            [
                [10, 25, 25],
                [20, 25, 25],
                [40, 35, 25],
                [40, 15, 25],
                [20, 30, 30],
                [20, 45, 45],
            ]
        )
        edges = [(0, 1), (1, 2), (1, 3), (4, 5)]
        skeleton_graph = nx.Graph(edges)
        return skeleton_graph, node_coordinates

    return factory_function


@pytest.fixture
def make_valid_skeleton_graph(make_bare_skeleton_graph) -> Callable[[], nx.Graph]:
    """Make a skeleton graph with all required properties."""

    def factory_function() -> nx.Graph:
        skeleton_graph, node_coordinates = make_bare_skeleton_graph()

        # add the node coordinates
        node_attributes = {}
        for node_index in skeleton_graph.nodes(data=False):
            node_attributes[node_index] = {
                NODE_COORDINATE_KEY: node_coordinates[node_index]
            }
        nx.set_node_attributes(skeleton_graph, node_attributes)

        # add the edge properties
        edge_attributes = {}
        for start_node, end_node in skeleton_graph.edges(data=False):
            start_point = node_coordinates[start_node]
            end_point = node_coordinates[end_node]
            line_length = np.linalg.norm(end_point - start_point)
            n_skeleton_points = int(line_length) // 2
            edge_coordinates = np.linspace(
                start_point, end_point, n_skeleton_points
            ).astype(int)
            edge_spline = Spline3D(
                points=edge_coordinates,
            )
            edge_attributes[(start_node, end_node)] = {
                EDGE_COORDINATES_KEY: node_coordinates[[start_node, end_node]],
                EDGE_SPLINE_KEY: edge_spline,
            }
        nx.set_edge_attributes(skeleton_graph, edge_attributes)
        return skeleton_graph

    return factory_function
