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

    # update an edge attribute
    nx.set_edge_attributes(skeleton, {(0, 1): {"validated": False}})

    # add an edge attribute that should be deleted by the parser
    # because it isn't passed as an edge_attribute
    bad_attribute_key = "bad_attribute"
    bad_attribute_edge = (0, 1)
    nx.set_edge_attributes(skeleton, {bad_attribute_edge: {bad_attribute_key: False}})

    # add a node attribute
    nx.set_node_attributes(skeleton, {0: {"good_node": True}})

    # add a node attribute that should be deleted by the parser
    # because it isn't passed as a node_attribute
    bad_attribute_node = 0
    nx.set_node_attributes(skeleton, {bad_attribute_node: {bad_attribute_key: True}})

    # parse the skeleton
    scale = (1, 2, 1)
    parsed_skeleton = Skeleton3D.parse(
        graph=skeleton,
        edge_attributes={"validated": True},
        node_attributes={"good_node": False},
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

    # check the edge attribute not passed in edge_attributes is removed
    assert bad_attribute_key not in parsed_skeleton.graph.edges[bad_attribute_edge]

    # check that the original node attributes were not overwritten
    assert parsed_skeleton.nodes()[0]["good_node"] is True

    # check that the rest of the nodes got the default value
    assert parsed_skeleton.nodes()[1]["good_node"] is False

    # check that the node attribute not in node_attributes is removed
    assert bad_attribute_key not in parsed_skeleton.nodes()[bad_attribute_node]

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
