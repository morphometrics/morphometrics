import numpy as np
import pymeshfix
import pyvista as pv
from skimage.measure import marching_cubes

from ..types import BinaryImage


def mesh_surface(
    object_mask: BinaryImage, n_mesh_smoothing_interations: int = 500
) -> pv.PolyData:
    vertices, faces, _, _ = marching_cubes(object_mask, 0)

    vertices_clean, faces_clean = pymeshfix.clean_from_arrays(vertices, faces)

    # pyvista PolyData expects the faces to be
    # [n_vertices, vert_0, vert_1,..., vert_n]
    faces_poly_number = 3 * np.ones((faces_clean.shape[0],))
    faces_poly_data = np.column_stack((faces_poly_number, faces_clean)).astype(int)
    mesh = pv.PolyData(vertices_clean, faces_poly_data)

    # optionally clean up the mesh
    if n_mesh_smoothing_interations > 0:
        return mesh.smooth(n_mesh_smoothing_interations)
    else:
        return mesh
