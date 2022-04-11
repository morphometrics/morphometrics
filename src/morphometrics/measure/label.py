import math
import warnings

import numpy as np
import pandas as pd
from skimage.measure import regionprops_table
from tqdm.autonotebook import tqdm

from ..types import IntensityImage, LabelImage, LabelMeasurementTable
from ..utils.surface_utils import binary_mask_to_surface
from . import register_measurement, register_measurement_set
from .surface import measure_surface_properties


@register_measurement_set(
    name="regionprops",
    choices=["size", "perimeter", "shape", "position", "moments"],
    uses_intensity_image=True,
)
def regionprops(
    intensity_image: IntensityImage,
    label_image: LabelImage,
    size: bool = True,
    intensity: bool = True,
    perimeter: bool = False,
    shape: bool = False,
    position: bool = False,
    moments: bool = False,
) -> LabelMeasurementTable:
    """
    Adds a table widget to a given napari viewer with quantitative analysis results derived from an image-label/image pair.
    """

    properties = ["label"]
    extra_properties = []

    if size:
        properties = properties + [
            "area",
            "bbox_area",
            "convex_area",
            "equivalent_diameter",
        ]

    if intensity:
        properties = properties + ["max_intensity", "mean_intensity", "min_intensity"]

        # arguments must be in the specified order, matching regionprops
        def standard_deviation_intensity(region, intensities):
            return np.std(intensities[region])

        extra_properties.append(standard_deviation_intensity)

    if perimeter:
        if len(label_image.shape) == 2:
            properties = properties + ["perimeter", "perimeter_crofton"]
        else:
            warnings.warn("Perimeter measurements are not supported in 3D")

    if shape:
        properties = properties + [
            "solidity",
            "extent",
            "feret_diameter_max",
            "local_centroid",
        ]
        if len(label_image.shape) == 2:
            properties = properties + [
                "major_axis_length",
                "minor_axis_length",
                "orientation",
                "eccentricity",
            ]
        else:
            properties = properties + ["moments_central"]
        # euler_number,

    if position:
        properties = properties + ["centroid", "bbox", "weighted_centroid"]

    if moments:
        properties = properties + ["moments", "moments_normalized"]
        if "moments_central" not in properties:
            properties = properties + ["moments_central"]
        if len(label_image.shape) == 2:
            properties = properties + ["moments_hu"]

    # todo:
    # weighted_local_centroid
    # weighted_moments
    # weighted_moments_central
    # weighted_moments_hu
    # weighted_moments_normalized

    # quantitative analysis using scikit-image's regionprops
    table = regionprops_table(
        np.asarray(label_image).astype(int),
        intensity_image=np.asarray(intensity_image),
        properties=properties,
        extra_properties=extra_properties,
    )

    if shape:
        if len(label_image.shape) == 3:
            axis_lengths_0 = []
            axis_lengths_1 = []
            axis_lengths_2 = []
            for i in range(len(table["moments_central-0-0-0"])):
                table_temp = {  # ugh
                    "moments_central-0-0-0": table["moments_central-0-0-0"][i],
                    "moments_central-2-0-0": table["moments_central-2-0-0"][i],
                    "moments_central-0-2-0": table["moments_central-0-2-0"][i],
                    "moments_central-0-0-2": table["moments_central-0-0-2"][i],
                    "moments_central-1-1-0": table["moments_central-1-1-0"][i],
                    "moments_central-1-0-1": table["moments_central-1-0-1"][i],
                    "moments_central-0-1-1": table["moments_central-0-1-1"][i],
                }
                axis_lengths = ellipsoid_axis_lengths(table_temp)
                axis_lengths_0.append(axis_lengths[0])  # ugh
                axis_lengths_1.append(axis_lengths[1])
                axis_lengths_2.append(axis_lengths[2])

            table["minor_axis_length"] = axis_lengths_2
            table["intermediate_axis_length"] = axis_lengths_1
            table["major_axis_length"] = axis_lengths_0

            if not moments:
                # remove moment from table as we didn't ask for them
                table = {k: v for k, v in table.items() if "moments_central" not in k}

    return pd.DataFrame(table).set_index("label")


def ellipsoid_axis_lengths(table):
    """Compute ellipsoid major, intermediate and minor axis length.
    Adapted from https://forum.image.sc/t/scikit-image-regionprops-minor-axis-length-in-3d-gives-first-minor-radius-regardless-of-whether-it-is-actually-the-shortest/59273/2
    Parameters
    ----------
    table from regionprops containing moments_central
    Returns
    -------
    axis_lengths: tuple of float
        The ellipsoid axis lengths in descending order.
    """

    m0 = table["moments_central-0-0-0"]
    sxx = table["moments_central-2-0-0"] / m0
    syy = table["moments_central-0-2-0"] / m0
    szz = table["moments_central-0-0-2"] / m0
    sxy = table["moments_central-1-1-0"] / m0
    sxz = table["moments_central-1-0-1"] / m0
    syz = table["moments_central-0-1-1"] / m0
    S = np.asarray([[sxx, sxy, sxz], [sxy, syy, syz], [sxz, syz, szz]])
    # determine eigenvalues in descending order
    eigvals = np.sort(np.linalg.eigvalsh(S))[::-1]
    return tuple(math.sqrt(20.0 * e) for e in eigvals)


@register_measurement(name="surface_properties_from_labels", uses_intensity_image=False)
def measure_surface_properties_from_labels(
    label_image: LabelImage,
    n_mesh_smoothing_interations: int = 10,
    verbose: bool = True,
) -> LabelMeasurementTable:
    """Measure the surface properties of all objects in a label image by
    applying morphometrics.measure.surface.measure_surface_properties to
    all objects.

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
    label_image : LabelImage
        The label image from which to measure the surface properties.
    n_mesh_smoothing_interations : int
        The number of interations of smooting to perform. Smoothing is
        done by the trimesh laplacian filter:
        https://trimsh.org/trimesh.smoothing.html#trimesh.smoothing.filter_laplacian
        Default value is 50.

    Returns
    -------
    surface_properties_table : LabelMeasurementTable
        The measured surface properties. Each measurement is a column and each
        object is a row.
    """
    if label_image.ndim != 3:
        raise ValueError(
            "measure_surface_properties_from_labels requires a 3D label image"
        )

    all_label_indices = np.unique(label_image)
    mesh_records = []
    for label_index in tqdm(all_label_indices, disable=not (verbose)):
        if label_index == 0:
            # background is assumed to be label 0
            continue
        label_mask = label_image == label_index
        smoothed_mesh = binary_mask_to_surface(
            label_mask, n_mesh_smoothing_interations=n_mesh_smoothing_interations
        )
        object_measurements = measure_surface_properties(surface=smoothed_mesh)

        # label measurement tables have index "label"
        object_measurements["label"] = label_index
        mesh_records.append(object_measurements.set_index("label"))
    surface_properties_table = pd.concat(mesh_records)

    return surface_properties_table
