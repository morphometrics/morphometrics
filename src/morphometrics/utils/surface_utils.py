from typing import List, Tuple

import numpy as np
import pymeshfix
import trimesh.voxel.creation
from skimage.measure import marching_cubes
from trimesh import Trimesh
from trimesh.smoothing import filter_taubin

from ..types import BinaryImage, LabelImage


def _round_to_pitch(coordinate: np.ndarray, pitch: float) -> np.ndarray:
    """Round a point to the nearest point on a grid that starts at the origin
     with a specified pitch.

    Parameters
    ----------
    coordinate : np.ndarray
        The coordinate to round
    pitch : float
        The pitch of the grid. Assumed to the be same in all directions.

    Returns
    -------
    rounded_point : np.ndarray
        The point after rounding to the nearest grid point.
    """
    return pitch * np.round(coordinate / pitch, decimals=0)


def repair_mesh(mesh: Trimesh) -> Trimesh:
    """Repair a mesh using pymeshfix.

    Parameters
    ----------
    mesh : Trimesh
        The mesh to be repaired
    """
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)

    vertices_clean, faces_clean = pymeshfix.clean_from_arrays(vertices, faces)

    # create the mesh object
    repaired_mesh = Trimesh(vertices=vertices_clean, faces=faces_clean)

    assert repaired_mesh.is_watertight, "Mesh was unable to be repaired"

    return repaired_mesh


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


def voxelize_closed_surface(
    mesh: Trimesh, pitch: float, repair_mesh: bool = True
) -> Tuple[BinaryImage, np.ndarray]:
    """Voxelize a closed surface mesh.

    Parameters
    ----------
    mesh : Trimesh
        The surface to voxelize
    pitch : float
        The voxel width in mesh units. Voxels have the
        same width in each dimension (i.e., are cubes).
    repair_mesh : bool
        Flag to attept to repair the mesh if set to True.
        Default value is True.

    Returns
    -------
    image : BinaryImage
        The binary mask created from the
    image_origin : np.ndarray
        The upper left hand corner of the voxelized image in mesh units
        (i.e., minimun of the axis aligned bounding box)
    """
    bounding_box = mesh.bounds
    centroid = np.mean(bounding_box, axis=0)

    # convert the centroid to the nearest integer multiple of the pitch
    rounded_centroid = _round_to_pitch(coordinate=centroid, pitch=pitch)

    # find the minimum cube half-width that encompases the full mesh
    cube_half_width = np.max(bounding_box - rounded_centroid)

    # get the number of voxels for the cube half-width
    n_voxels_cube_half_width = int(np.ceil(cube_half_width / pitch))

    # pad with one voxel on each side to make sure the full mesh is in range
    n_voxels_cube_half_width += 1

    # get the upper left hand (i.e., minimum) corner of the voxelized image in mesh coordinates
    image_origin = rounded_centroid - (n_voxels_cube_half_width * pitch)

    # if and (not mesh.is_watertight):
    #     mesh = repair_mesh(mesh)

    voxel_grid = trimesh.voxel.creation.local_voxelize(
        mesh=mesh,
        point=rounded_centroid,
        pitch=pitch,
        radius=n_voxels_cube_half_width,
        fill=True,
    )

    return voxel_grid.matrix.astype(bool), image_origin


def closed_surfaces_to_label_image(
    meshes: List[Trimesh],
    pitch: float,
    crop_around_mesh: bool = False,
    repair_mesh: bool = False,
) -> Tuple[LabelImage, np.ndarray]:
    """Create a label image from a set of meshes with closed surfaces.

    Notes:
        - meshes must be water tight for accurate voxelization.
        - Labels are assigned in the order the meshes appear in the list.
        - all meshes must be in the same coordinate system and scale.

    Parameters
    ----------
    meshes : List[Trimesh]
        The meshes to convert to a label image.
    pitch : float
        The width of a voxel in mesh units. Voxels are assumed to be cubes.
    crop_around_mesh : bool
        When set to True, the image is cropped around the axis aligned bounding box
        of the set of meshes with a one voxel pad in each direction.
        The default value is False
    repair_mesh : bool
        When set to True, will attempt to repair meshes with PyMeshFix.
        Default value is False.

    Returns
    -------
    label_image : LabelImage
        The label image generated from the meshes.
    image_origin : np.ndarray
        The coordinate of the upper left hand corner (i.e., minimum) of the
        label_image in mesh coordinates.

    """
    # get the bounding box around the meshes
    bounding_boxes = [mesh.bounds for mesh in meshes]

    # get the bounding box around all of them
    all_corners = np.concatenate(bounding_boxes, axis=0)
    min_corner = np.min(all_corners, axis=0)
    max_corner = np.max(all_corners, axis=0)

    # round the corners to the nearest voxel (in mesh coordinates)
    min_corner_rounded = _round_to_pitch(coordinate=min_corner, pitch=pitch)
    max_corner_rounded = _round_to_pitch(coordinate=max_corner, pitch=pitch)

    # pad the bounding box to make sure everything is accounted for
    min_corner_rounded -= pitch
    max_corner_rounded += pitch

    if crop_around_mesh is True:
        image_origin = min_corner_rounded
    else:
        image_origin = np.array([0, 0, 0])

    # determine the size of the image in pixels
    image_shape_mesh_units = max_corner_rounded - image_origin
    image_shape_voxels = np.round(image_shape_mesh_units / pitch, decimals=0).astype(
        int
    )

    # create the blank label image
    label_image = np.zeros(image_shape_voxels, dtype=np.uint16)

    for i, mesh in enumerate(meshes):
        voxelized, origin = voxelize_closed_surface(
            mesh, pitch=pitch, repair_mesh=repair_mesh
        )

        # get the coordinates of the voxels inside of the mesh
        filled_voxel_coordinates = np.argwhere(voxelized)

        # get the offset between the label image indices and the voxelized mesh indices
        mesh_offset = np.round((origin - image_origin) / pitch, decimals=0)

        # offset the voxel coordinates
        filled_voxel_indices = np.round(
            filled_voxel_coordinates + mesh_offset, decimals=0
        ).astype(int)

        # set the label value
        label_value = i + 1
        label_image[
            filled_voxel_indices[:, 0],
            filled_voxel_indices[:, 1],
            filled_voxel_indices[:, 2],
        ] = label_value

    return label_image, image_origin
