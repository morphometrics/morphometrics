from copy import deepcopy
from typing import Any, Dict, List, Tuple, Union

import networkx as nx
import numpy as np
from cfgv import Optional
from morphosamplers.sampler import (
    generate_2d_grid,
    place_sampling_grids,
    sample_volume_at_coordinates,
)
from morphosamplers.spline import Spline3D

from morphometrics.skeleton.constants import (
    EDGE_COORDINATES_KEY,
    EDGE_SPLINE_KEY,
    NODE_COORDINATE_KEY,
)


class Skeleton3D:
    def __init__(self, graph: nx.Graph):
        self.graph = deepcopy(graph)

    def nodes(self, data: bool = True):
        """Passthrough for nx.Graph.nodes"""
        return self.graph.nodes(data=data)

    def edges(self, data: bool = True):
        """Passthrough for nx.Graph.edges"""
        return self.graph.edges(data=data)

    @property
    def node_coordinates(self) -> np.ndarray:
        """Coordinates of the nodes.

        Index matched to nx.Graph.nodes()
        """
        node_data = self.nodes(data=True)
        coordinates = [data[NODE_COORDINATE_KEY] for _, data in node_data]
        return np.stack(coordinates)

    def sample_points_on_edge(
        self, start_node: int, end_node: int, u: List[float], derivative_order: int = 0
    ):
        spline = self.graph[start_node][end_node][EDGE_SPLINE_KEY]
        return spline.sample(u=u, derivative_order=derivative_order)

    def sample_slices_on_edge(
        self,
        image: np.ndarray,
        image_voxel_size: Tuple[float, float, float],
        start_node: int,
        end_node: int,
        slice_pixel_size: float,
        slice_width: int,
        slice_spacing: float,
        interpolation_order: int = 1,
    ) -> np.ndarray:
        # get the spline object
        spline = self.graph[start_node][end_node][EDGE_SPLINE_KEY]

        # get the positions along the spline
        positions = spline.sample(separation=slice_spacing)
        orientations = spline.sample_orientations(separation=slice_spacing)

        # get the sampling coordinates
        sampling_shape = (slice_width, slice_width)
        grid = generate_2d_grid(
            grid_shape=sampling_shape, grid_spacing=(slice_pixel_size, slice_pixel_size)
        )
        sampling_coords = place_sampling_grids(grid, positions, orientations)

        # convert the sampling coordinates into the image indices
        sampling_coords = sampling_coords / np.array(image_voxel_size)

        return sample_volume_at_coordinates(
            image, sampling_coords, interpolation_order=interpolation_order
        )

    def sample_image_around_node(
        self,
        node_index: int,
        image: np.ndarray,
        image_voxel_size: Tuple[float, float, float],
        bounding_box_shape: Union[float, Tuple[float, float, float]] = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract an axis-aligned bounding box from an image around a node.

        Parameters
        ----------
        node_index : int
            The index of the node to sample around.
        image : np.ndarray
            The image to sample from.
        image_voxel_size : Tuple[float, float, float]
            Size of the image voxel in each axis. Should convert to the same
            scale as the skeleton graph.
        bounding_box_shape : Union[float, Tuple[float, float, float]]
            The shape of the bounding box to extract. Size should be specified
            in the coordinate system of the skeleton. If a single float is provided,
            a cube with edge-length bounding_box_shape will be extracted. Otherwise,
            provide a tuple with one element for each axis.

        Returns
        -------
        sub_volume : np.ndarray
            The extracted bounding box.
        bounding_box : np.ndarray
            (2, 3) array with the coordinates of the
            upper left and lower right hand corners of the bounding box.
        """

        # get the node coordinates
        node_coordinate = self.graph.nodes(data=NODE_COORDINATE_KEY)[node_index]

        # convert node coordinate to
        graph_to_image_factor = 1 / np.array(image_voxel_size)
        node_coordinate_image = node_coordinate * graph_to_image_factor

        # convert the bounding box to image coordinates
        if isinstance(bounding_box_shape, int) or isinstance(bounding_box_shape, float):
            bounding_box_shape = (
                bounding_box_shape,
                bounding_box_shape,
                bounding_box_shape,
            )
        grid_shape = np.asarray(bounding_box_shape) * graph_to_image_factor
        bounding_box_min = np.clip(
            node_coordinate_image - (grid_shape / 2), a_min=[0, 0, 0], a_max=image.shape
        )
        bounding_box_max = np.clip(
            node_coordinate_image + (grid_shape / 2), a_min=[0, 0, 0], a_max=image.shape
        )
        bounding_box = np.stack([bounding_box_min, bounding_box_max]).astype(int)

        # sample the image
        sub_volume = image[
            bounding_box[0, 0] : bounding_box[1, 0],
            bounding_box[0, 1] : bounding_box[1, 1],
            bounding_box[0, 2] : bounding_box[1, 2],
        ]

        return np.asarray(sub_volume), bounding_box

    def shortest_path(self, start_node: int, end_node: int) -> Optional[List[int]]:
        return nx.shortest_path(self.graph, source=start_node, target=end_node)

    @classmethod
    def parse(
        cls,
        graph: nx.Graph,
        edge_attributes: Optional[Dict[str, Any]] = None,
        node_attributes: Optional[Dict[str, Any]] = None,
        edge_coordinates_key: str = EDGE_COORDINATES_KEY,
        node_coordinate_key: str = NODE_COORDINATE_KEY,
        scale: Tuple[float, float, float] = (1, 1, 1),
    ):
        # make a copy of the graph so we don't clobber the original attributes
        graph = deepcopy(graph)

        scale = np.asarray(scale)
        if edge_attributes is None:
            edge_attributes = {}
        if node_attributes is None:
            node_attributes = {}

        # parse the edge attributes
        parsed_edge_attributes = {}
        for start_index, end_index, attributes in graph.edges(data=True):
            # remove attribute not specified
            keys_to_delete = [
                key
                for key in attributes
                if ((key not in edge_attributes) and (key != edge_coordinates_key))
            ]
            for key in keys_to_delete:
                del attributes[key]

            for expected_key, default_value in edge_attributes.items():
                # add expected keys that are missing
                if expected_key not in attributes:
                    attributes.update({expected_key: default_value})

            # make the edge spline
            coordinates = np.asarray(attributes[edge_coordinates_key]) * scale
            spline = Spline3D(points=coordinates)
            parsed_edge_attributes.update(
                {
                    (start_index, end_index): {
                        EDGE_COORDINATES_KEY: coordinates,
                        EDGE_SPLINE_KEY: spline,
                    }
                }
            )
        nx.set_edge_attributes(graph, parsed_edge_attributes)

        # parse the node attributes
        parsed_node_attributes = {}
        for node_index, attributes in graph.nodes(data=True):
            # remove attribute not specified
            keys_to_delete = [
                key
                for key in attributes
                if ((key not in node_attributes) and (key != node_coordinate_key))
            ]
            for key in keys_to_delete:
                del attributes[key]

            for expected_key, default_value in node_attributes.items():
                # add expected keys that are missing
                if expected_key not in attributes:
                    attributes.update({expected_key: default_value})
            # add the node coordinates
            coordinate = np.asarray(attributes[node_coordinate_key])
            coordinate = coordinate * scale

            parsed_node_attributes.update(
                {node_index: {NODE_COORDINATE_KEY: coordinate}}
            )
        nx.set_node_attributes(graph, parsed_node_attributes)

        return cls(graph=graph)
