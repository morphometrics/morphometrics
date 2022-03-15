import pymeshfix
from skimage.measure import marching_cubes
from trimesh import Trimesh
from trimesh.smoothing import filter_laplacian

from ..types import BinaryImage


def mesh_surface(
    object_mask: BinaryImage, n_mesh_smoothing_interations: int = 500
) -> Trimesh:
    vertices, faces, _, _ = marching_cubes(object_mask, 0)

    vertices_clean, faces_clean = pymeshfix.clean_from_arrays(vertices, faces)

    # create the mesh object
    mesh = Trimesh(vertices=vertices_clean, faces=faces_clean)

    # optionally clean up the mesh
    if n_mesh_smoothing_interations > 0:
        filter_laplacian(mesh, iterations=n_mesh_smoothing_interations)

    return mesh
