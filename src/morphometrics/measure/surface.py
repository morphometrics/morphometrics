from typing import Union

import numpy as np
import pandas as pd
import trimesh
from trimesh.curvature import discrete_mean_curvature_measure


def measure_surface_properties(
    surface: trimesh.Trimesh, curvature_radius: float = 5
) -> pd.DataFrame:
    """Measure the surface properties of a watertight mesh.

    This measures the following properties:
        surface_area: surface area of the mesh
        curvature_mean: the curvature averaged over all vertices
        curvature_stdev: the standard deviation of curvature over all vertices
        curvature_0: minimum curvature value (across all vertices)
        curvature_10: 10th percentile curvature value (across all vertices)
        curvature_20: 20th percentile curvature value (across all vertices)
        curvature_30: 30th percentile curvature value (across all vertices)
        curvature_40: 40th percentile curvature value (across all vertices)
        curvature_50: 50th percentile curvature value (across all vertices)
        curvature_60: 60th percentile curvature value (across all vertices)
        curvature_70: 70th percentile curvature value (across all vertices)
        curvature_80: 80th percentile curvature value (across all vertices)
        curvature_90: 90th percentile curvature value (across all vertices)
        curvature_100: max curvature value (across all vertices)

    Parameters
    ----------
    surface : trimesh.Trimesh
        The surface from which to measure the surface properties. The mesh should contain
        a single object and the mesh should be watertight.
    curvature_radius : float
        The radius to use for calculating

    Returns
    -------
    measurement_table : pd.DataFrame
        The measurements in a pandas DataFrame. Each measurement is
        in its own column. All measurements in a single row.
    """
    surface_area = surface.area
    curvatures = discrete_mean_curvature_measure(
        surface, surface.vertices, radius=curvature_radius
    )
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
        ray_origins=ray_origins, ray_directions=ray_directions, multiple_hits=False
    )

    # calculate the distances
    distances = np.repeat(fill_value, len(ray_origins))
    distances[index_ray] = np.linalg.norm(locations - ray_origins[index_ray], axis=1)

    return distances
