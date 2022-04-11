import numpy as np
import trimesh

from morphometrics.utils.surface_utils import (
    closed_surfaces_to_label_image,
    voxelize_closed_surface,
)


def _make_cuboid_mesh(origin: np.ndarray, extents: np.ndarray) -> trimesh.Trimesh:
    max_point = origin + extents

    vertices = np.array(
        [
            [origin[0], origin[1], origin[2]],
            [origin[0], origin[1], max_point[2]],
            [origin[0], max_point[1], origin[2]],
            [origin[0], max_point[1], max_point[2]],
            [max_point[0], origin[1], origin[2]],
            [max_point[0], origin[1], max_point[2]],
            [max_point[0], max_point[1], max_point[2]],
            [max_point[0], max_point[1], origin[2]],
        ]
    )

    faces = np.array(
        [
            [0, 1, 2],
            [1, 2, 3],
            [2, 3, 6],
            [2, 6, 7],
            [0, 2, 7],
            [0, 4, 7],
            [0, 1, 5],
            [0, 5, 4],
            [4, 5, 7],
            [5, 6, 7],
            [1, 3, 5],
            [3, 5, 6],
        ]
    )

    return trimesh.Trimesh(vertices=vertices, faces=faces)


def test_voxelize_closed_surface():
    origin = np.array([10, 10, 20])
    cube_extents = np.array([30, 10, 10])
    pitch = 0.5
    mesh = _make_cuboid_mesh(origin=origin, extents=cube_extents)

    voxelized, image_origin = voxelize_closed_surface(
        mesh, pitch=pitch, repair_mesh=True
    )

    np.testing.assert_allclose(image_origin, [9.5, -0.5, 9.5])
    np.testing.assert_allclose([63, 63, 63], voxelized.shape)


def test_closed_surfaces_to_label_image_no_crop():

    mesh_0 = _make_cuboid_mesh(np.array([10, 10, 10]), np.array([20, 20, 20]))
    mesh_1 = _make_cuboid_mesh(np.array([30, 10, 10]), np.array([10, 10, 30]))
    pitch = 0.5

    label_image, image_origin = closed_surfaces_to_label_image(
        [mesh_0, mesh_1],
        pitch=pitch,
        crop_around_mesh=False,
        repair_mesh=True,
    )

    np.testing.assert_allclose(image_origin, [0, 0, 0])
    np.testing.assert_allclose(label_image.shape, [81, 61, 81])

    assert set(np.unique(label_image)) == {0, 1, 2}
