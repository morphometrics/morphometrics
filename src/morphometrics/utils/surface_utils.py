import pymeshfix
from skimage.measure import marching_cubes
from trimesh import Trimesh
from trimesh.smoothing import filter_taubin

from ..types import BinaryImage


def binary_mask_to_surface(
    object_mask: BinaryImage, n_mesh_smoothing_interations: int = 50
) -> Trimesh:
    """Convert surface of a 3D binary mask (segmented object) into a watertight mesh.

    Parameters
    ----------
    object_mask  : BinaryMask
        A 3D binary image corresponding to the object you want to mesh.
    n_mesh_smoothing_interations : int
        The number of interations of smooting to perform. Smoothing is
        done by the trimesh taubin filter:
        https://trimsh.org/trimesh.smoothing.html#trimesh.smoothing.filter_taubin
        Default value is 50.

    Returns
    -------
    mesh : trimesh.Trimesh
        The resulting mesh as a trimesh.Trimesh object.
        https://trimsh.org/trimesh.base.html#github-com-mikedh-trimesh
    """
    vertices, faces, _, _ = marching_cubes(object_mask, 0)

    vertices_clean, faces_clean = pymeshfix.clean_from_arrays(vertices, faces)

    # create the mesh object
    mesh = Trimesh(vertices=vertices_clean, faces=faces_clean)

    # optionally clean up the mesh
    if n_mesh_smoothing_interations > 0:
        filter_taubin(mesh, iterations=n_mesh_smoothing_interations)

    return mesh
