from typing import Optional, Tuple

import napari
import numpy as np
from morphosamplers.sampler import (
    generate_3d_grid,
    place_sampling_grids,
    sample_volume_at_coordinates,
)
from scipy.spatial.transform import Rotation


class SubStackViewer:
    def __init__(self, viewer: napari.Viewer):
        self._viewer = viewer

        self._start_point = None
        self._end_point = None

        self._layer = None
        self._half_width = 10
        self._grid_spacing = (1, 1, 1)
        self._sampling_coordinates = None

    @property
    def parameters_set(self) -> bool:
        return (
            (self.layer is not None)
            and (self.start_point is not None)
            and (self.end_point is not None)
        )

    @property
    def normal_vector(self):
        if self._viewer.dims.ndisplay != 2:
            raise ValueError("Viewer must be in 2D mode")

        if self._viewer.dims.ndim != 3:
            raise ValueError("Data must be 3D")

        if 0 not in self._viewer.dims.displayed:
            return np.array([1, 0, 0])
        elif 1 not in self._viewer.dims.displayed:
            return np.array([0, 1, 0])
        elif 2 not in self._viewer.dims.displayed:
            return np.array([0, 0, 1])
        else:
            raise ValueError("image must be 3D and viewed in 2D")

    @property
    def layer(self) -> Optional[napari.layers.Image]:
        return self._layer

    @property
    def start_point(self) -> Optional[np.ndarray]:
        return self._start_point

    @start_point.setter
    def start_point(self, start_point: np.ndarray) -> None:
        self._start_point = start_point

    @property
    def end_point(self) -> Optional[np.ndarray]:
        return self._end_point

    @end_point.setter
    def end_point(self, end_point: np.ndarray) -> None:
        self._end_point = end_point

    @property
    def half_width(self) -> int:
        """Half width of the edge of the sample volume.

        The full width will be (2 * half_width) + 1
        """

        return self._half_width

    def grid_spacing(self) -> Tuple[int, int, int]:
        """Spacing between the sample grid points in layer data units."""
        return self._grid_spacing

    @property
    def sample_points(self) -> Optional[np.ndarray]:
        """Points in layer data coordinates the subvolume is sampled from"""
        return self._sample_points

    def set_sample_parameters(
        self,
        layer: Optional[napari.layers.Image] = None,
        half_width: int = 10,
    ) -> None:
        self._layer = layer
        self._half_width = half_width

    def sample_subvolume_from_line_segment(self) -> Optional[np.ndarray]:
        if self.parameters_set is False:
            # if we have incomplete parameters, just return
            return
        # Set the grid shape
        line_segment_vector = self.end_point - self.start_point
        length_of_line_segment = np.linalg.norm(line_segment_vector)

        grid_shape = (
            int(length_of_line_segment) + 1,
            ((2 * self.half_width) + 1),
            ((2 * self.half_width) + 1),
        )
        print(grid_shape)

        # Compute the shift as the coords of the midpoint of the drawn line segment
        # note that we are truncating by converting to int
        grid_center_point = ((self.start_point + self.end_point) / 2).astype(int)

        # Compute the rotation
        # To do this I want to compute the rotation matrix that maps the cartesian 3D basis
        # to the normal vector of the plane defined by line_segment.
        # But that is is simply the matrix whose columns are the axis in the new coordinate
        # system. Therefore, I need to compute:
        # - The vector identifying the direction of the line_segment
        # - The vector risulting from the cross-prod of line_segment_vector and normal_vector

        line_segment_unit_vector = line_segment_vector / length_of_line_segment
        third_vector = np.cross(line_segment_unit_vector, self.normal_vector)

        rot_matrix = np.column_stack(
            [line_segment_unit_vector, self.normal_vector, third_vector]
        )

        rotations = [Rotation.from_matrix(rot_matrix)]

        # If asked by the users, generate a grid and place it in order to check for potential errors
        grid = generate_3d_grid(grid_shape)
        self._sampling_coordinates = place_sampling_grids(
            grid, grid_center_point, rotations
        )

        if isinstance(self.layer, napari.layers.Image):
            interpolation_order = 1
        else:
            interpolation_order = 0

        return sample_volume_at_coordinates(
            self.layer.data,
            self._sampling_coordinates,
            interpolation_order=interpolation_order,
        )
