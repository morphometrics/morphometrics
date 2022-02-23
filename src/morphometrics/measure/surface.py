import numpy as np
import pandas as pd
import pyvista as pv


def measure_surface_properties(mesh: pv.PolyData, object_index: int) -> pd.DataFrame:
    surface_area = mesh.area
    curvatures = mesh.curvature()
    curvature_percentiles = np.percentile(curvatures, np.arange(0, 110, 10))

    return pd.DataFrame(
        [
            {
                "object_index": object_index,
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
    ).set_index("object_index")
