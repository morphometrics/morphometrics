from typing import Union

import numpy as np
import pandas as pd
import pyvista as pv
import trimesh


def measure_surface_properties(mesh: pv.PolyData) -> pd.DataFrame:
    surface_area = mesh.area
    curvatures = mesh.curvature()
    curvature_percentiles = np.percentile(curvatures, np.arange(0, 110, 10))

    return pd.DataFrame(
        [
            {
                "surface_area": surface_area,
                "curvature_mean": curvatures.mean(),
                "curvature_stdev": curvatures.std(),
                "curvature_0": curvature_percentiles[0],
                "curvature_10": curvature_percentiles[1],
                "curvature_20": curvature_percentiles[2],
                "curvature_30": curvature_percentiles[3],
                "curvature_40": curvature_percentiles[4],
                "curvature_50": curvature_percentiles[5],
                "curvature_60": curvature_percentiles[6],
                "curvature_70": curvature_percentiles[7],
                "curvature_80": curvature_percentiles[8],
                "curvature_90": curvature_percentiles[9],
                "curvature_100": curvature_percentiles[10],
            }
        ]
    )


def distance_between_surfaces(
    source_surface: trimesh.Trimesh,
    destination_surface: trimesh.Trimesh,
    fill_value: Union[float] = 0,
    flip_normals: bool = False,
) -> np.ndarray:
    """Calculate the per-vertex distance between two surfaces.

    Parameters
    ----------
    source_surface : trimesh.Trimesh
        The surface to calculate the distance from.
    destination_surface : trimesh.Trimesh
        The surface to calcuate the distance to.
    fill_value : Union[float]
        Value to fill in for the distances where the ray doesn't
        intersect the distination_surface.
    flip_normals : bool
        If True, normal vectors will be flipped.
        Default value is False.

    Returns
    -------
    distances : np.ndarray
        Distance from each vertex in source_surface to the
        destination_surface along the vertex normals of source_surface.
    """
    ray_origins = source_surface.vertices
    ray_directions = source_surface.vertex_normals
    if flip_normals is True:
        ray_directions = -1 * ray_directions

    # find where the rays intersect
    locations, index_ray, _ = destination_surface.ray.intersects_location(
        ray_origins=ray_origins, ray_directions=ray_directions
    )

    # calculate the distances
    distances = np.repeat(fill_value, len(ray_origins))
    distances[index_ray] = np.linalg.norm(locations - ray_origins[index_ray])

    return distances
