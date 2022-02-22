import math
import warnings

import numpy as np
import pandas as pd
from skimage.measure import regionprops_table

from ..types import IntensityImage, LabelImage, LabelMeasurementTable
from . import register_measurement, register_measurement_set


@register_measurement(name="volume", uses_intensity_image=False)
def volume(label_image: LabelImage) -> LabelMeasurementTable:
    rp_table = regionprops_table(label_image, properties=("label", "area"))

    return pd.DataFrame(rp_table).set_index("label")


@register_measurement(name="centroid", uses_intensity_image=False)
def centroid(label_image: LabelImage) -> LabelMeasurementTable:
    rp_table = regionprops_table(label_image, properties=("label", "centroid"))

    return pd.DataFrame(rp_table).set_index("label")


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
