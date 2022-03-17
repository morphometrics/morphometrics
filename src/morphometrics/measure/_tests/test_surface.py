import numpy as np
import pytest
import trimesh

from morphometrics.measure.surface import distance_between_surfaces


@pytest.mark.parametrize("fill_value", [np.nan, 0])
def test_distance_between_surfaces(fill_value):
    source_vertices = np.array([[0, 0, 0], [0, 10, 0], [0, 10, 10]])
    source_faces = np.array([[0, 1, 2]])
    vertex_normals = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]])
    source_mesh = trimesh.Trimesh(
        vertices=source_vertices, faces=source_faces, vertex_normals=vertex_normals
    )

    destination_vertices = np.array([[5, 5, 10], [5, 15, 5], [5, 15, 15]])
    destination_faces = np.array([[0, 1, 2]])
    destination_mesh = trimesh.Trimesh(
        vertices=destination_vertices,
        faces=destination_faces,
        vertex_normals=vertex_normals,
    )

    distances = distance_between_surfaces(
        source_surface=source_mesh,
        destination_surface=destination_mesh,
        fill_value=fill_value,
    )
    np.testing.assert_allclose(distances, [fill_value, fill_value, 5])


@pytest.mark.parametrize("fill_value", [np.nan, 0])
def test_distance_between_surfaces_flip_normal(fill_value):
    source_vertices = np.array([[5, 0, 0], [5, 10, 0], [5, 10, 10]])
    source_faces = np.array([[0, 1, 2]])
    vertex_normals = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]])
    source_mesh = trimesh.Trimesh(
        vertices=source_vertices, faces=source_faces, vertex_normals=vertex_normals
    )

    destination_vertices = np.array([[0, 5, 10], [0, 15, 5], [0, 15, 15]])
    destination_faces = np.array([[0, 1, 2]])
    destination_mesh = trimesh.Trimesh(
        vertices=destination_vertices,
        faces=destination_faces,
        vertex_normals=vertex_normals,
    )

    distances = distance_between_surfaces(
        source_surface=source_mesh,
        destination_surface=destination_mesh,
        fill_value=fill_value,
        flip_normals=True,
    )
    np.testing.assert_allclose(distances, [fill_value, fill_value, 5])
