import napari
import numpy as np
import trimesh


def add_surface_normals_to_viewer(
    viewer: napari.Viewer,
    surface: trimesh.Trimesh,
    vector_width: float = 0.25,
    vector_length: float = 3,
) -> napari.layers.Vectors:
    """Add the surface normals of a mesh to a napari viewer as a Vectors layer.

    Parameters
    ----------
    viewer : napari.Viewer
        The viewer to add the surface normals to.
    surface : trimesh.Trimesh
        The mesh to extract the surface normals from. The surface normals
        will be from each vertex in the mesh.
    vector_width : float
        The width of the displayed vectors in data units.
        The default value is 0.25
    vector_length : float
        The length of the displayed vectors in data units.
        The default value is 3

    Returns
    -------
    vector_layer : napari.layers.Vectors
        The Vectors layer that is displaying the surface normals.
    """
    # get the surface normals
    normal_vectors = surface.vertex_normals
    vertices = surface.vertices

    # create the vectors data
    vectors_data = np.zeros((len(vertices), 2, 3))
    vectors_data[:, 0, :] = vertices
    vectors_data[:, 1, :] = normal_vectors

    return viewer.add_vectors(
        vectors_data,
        edge_width=vector_width,
        length=vector_length,
    )
