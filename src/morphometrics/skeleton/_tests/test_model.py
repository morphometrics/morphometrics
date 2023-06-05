import networkx as nx
import numpy as np

from morphometrics.skeleton.constants import NODE_COORDINATE_KEY
from morphometrics.skeleton.model import Skeleton3D


def test_skeleton_model_instantiation(make_valid_skeleton_graph):
    """Test that the Skeleton3D class can be instantiated."""
    graph = make_valid_skeleton_graph()
    skeleton = Skeleton3D(graph=graph)

    # the graph should be a copy of the original graph
    assert skeleton.graph is not graph

    # the graph should be identical to the original graph
    assert skeleton.graph.edges(data=False) == graph.edges(data=False)


def test_skeleton_model_parse(make_valid_skeleton_graph):
    """Test that the Skeleton3D class can be instantiated with the parser."""

    # get the skeleton graph and node coordinates
    skeleton = make_valid_skeleton_graph()

    # update an edge property
    nx.set_edge_attributes(skeleton, {(0, 1): {"validated": False}})

    # parse the skeleton
    scale = (1, 2, 1)
    parsed_skeleton = Skeleton3D.parse(
        graph=skeleton,
        edge_attributes={"validated": True},
        edge_coordinates_key="edge_coordinates",
        node_coordinate_key="node_coordinate",
        scale=scale,
    )

    # verify that the graph is a copy
    assert parsed_skeleton.graph is not skeleton

    # check that the original edge attributes were not overwritten
    assert parsed_skeleton.graph.edges[(0, 1)]["validated"] is False

    # check the default value was given to the edge attribute with missing values
    assert parsed_skeleton.graph.edges[(1, 2)]["validated"] is True

    # check that the scale was applied to the node coordinates
    original_node_coordinates = np.stack(
        [node_data[NODE_COORDINATE_KEY] for _, node_data in skeleton.nodes(data=True)]
    )
    parsed_node_coordinates = np.stack(
        [
            node_data[NODE_COORDINATE_KEY]
            for _, node_data in parsed_skeleton.nodes(data=True)
        ]
    )
    scaled_original_coordinates = original_node_coordinates * scale
    np.testing.assert_allclose(parsed_node_coordinates, scaled_original_coordinates)
