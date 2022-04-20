import numpy as np
import pandas as pd
from skimage.measure import regionprops_table

from ..types import IntensityImage, LabelImage, LabelMeasurementTable
from ..utils.image_utils import make_boundary_mask
from ..utils.math_utils import safe_divide
from . import register_measurement


@register_measurement(name="boundary_intensity", uses_intensity_image=True)
def measure_boundary_intensity(
    label_image: LabelImage,
    intensity_image: IntensityImage,
    boundary_dilation_size: int = 0,
) -> LabelMeasurementTable:
    """Measure the intensity at the boundary of the segmented objects.

    Parameters
    ----------
    label_image : LabelImage
        The label image containing the segmented objects.
    intensity_image : IntensityImage
        The image from which to measure the intensities
    boundary_dilation_size : int
        The size of the morphological dilation to apply to expand the edge
        size. If set to 0, no dilation is performed (i.e., boundaries are
        one pixel wide). The default value is 0.

    Returns
    -------
    boundary_intensity_measurements : LabelMeasurementTable
        The resulting intensity measurements in a LabelMeasurementTable.
        Table contains the following columns:
            boundary_intensity_mean: mean intensity of the boundary pixels
            boundary_intensity_max: maximum intensity of the boundary pixels
            boundary_intensity_min: minimum intensity of the boundary pixels
    """
    # set non-edge labels to zero (background)
    boundary_mask = make_boundary_mask(
        label_image=label_image, boundary_dilation_size=boundary_dilation_size
    )
    boundary_labels = label_image.copy()
    boundary_labels[np.logical_not(boundary_mask)] = 0

    # measure edge intensity
    intensity_measurements = regionprops_table(
        label_image=boundary_labels,
        intensity_image=intensity_image,
        properties=("label", "intensity_mean", "intensity_min", "intensity_max"),
    )

    # update the names
    intensity_measurements["boundary_intensity_mean"] = intensity_measurements.pop(
        "intensity_mean"
    )
    intensity_measurements["boundary_intensity_max"] = intensity_measurements.pop(
        "intensity_max"
    )
    intensity_measurements["boundary_intensity_min"] = intensity_measurements.pop(
        "intensity_min"
    )

    return pd.DataFrame(intensity_measurements).set_index("label")


@register_measurement(name="internal_intensity", uses_intensity_image=True)
def measure_internal_intensity(
    label_image: LabelImage,
    intensity_image: IntensityImage,
    boundary_dilation_size: int = 0,
) -> LabelMeasurementTable:
    """Measure the intensity in the inside (i.e., excluding the borders) of the segmented objects.

    Parameters
    ----------
    label_image : LabelImage
        The label image containing the segmented objects.
    intensity_image : IntensityImage
        The image from which to measure the intensities
    boundary_dilation_size : int
        The size of the morphological dilation to apply to expand the boundary
        pixels. If set to 0, no dilation is performed (i.e., boundaries are
        one pixel wide). The default value is 0.

    Returns
    -------
    internal_intensity_measurements : LabelMeasurementTable
        The resulting intensity measurements in a LabelMeasurementTable.
        Table contains the following columns:
            internal_intensity_mean: mean intensity of the internal pixels
            internal_intensity_max: maximum intensity of the internal pixels
            internal_intensity_min: minimum intensity of the internal pixels
    """
    # set edge labels to zero (background)
    boundary_mask = make_boundary_mask(
        label_image=label_image, boundary_dilation_size=boundary_dilation_size
    )
    internal_labels = label_image.copy()
    internal_labels[boundary_mask] = 0

    # measure edge intensity
    intensity_measurements = regionprops_table(
        label_image=internal_labels,
        intensity_image=intensity_image,
        properties=("label", "intensity_mean", "intensity_min", "intensity_max"),
    )

    # update the names
    intensity_measurements["internal_intensity_mean"] = intensity_measurements.pop(
        "intensity_mean"
    )
    intensity_measurements["internal_intensity_max"] = intensity_measurements.pop(
        "intensity_max"
    )
    intensity_measurements["internal_intensity_min"] = intensity_measurements.pop(
        "intensity_min"
    )

    internal_intensity_measurements = pd.DataFrame(intensity_measurements).set_index(
        "label"
    )

    # determine if any labels are not represented because they do not have
    # internal pixels
    original_label_values = set(np.unique(label_image))
    internal_label_values = set(np.unique(internal_labels))
    no_internal_pixel_label_values = original_label_values - internal_label_values

    n_columns = internal_intensity_measurements.shape[1]
    for label_value in no_internal_pixel_label_values:
        internal_intensity_measurements.loc[label_value] = np.zeros((n_columns,))

    return internal_intensity_measurements


def measure_intensity_features(
    label_image: LabelImage,
    intensity_image: IntensityImage,
    boundary_dilation_size: int = 0,
) -> LabelMeasurementTable:
    """Measure intensity features from a label image and an intensity image.

    Measures the following features:
            boundary_intensity_mean: mean intensity of the boundary pixels
            boundary_intensity_max: maximum intensity of the boundary pixels
            boundary_intensity_min: minimum intensity of the boundary pixels
            internal_intensity_mean: mean intensity of the internal pixels
            internal_intensity_max: maximum intensity of the internal pixels
            internal_intensity_min: minimum intensity of the internal pixels

    Parameters
    ----------
    label_image : LabelImage
        The label image containing the segmented objects.
    intensity_image : IntensityImage
        The image from which to measure the intensities
    boundary_dilation_size : int
        The size of the morphological dilation to apply to expand the boundary
        pixels. If set to 0, no dilation is performed (i.e., boundaries are
        one pixel wide). The default value is 0.

    Returns
    -------
    intensity_measurements : LabelMeasurementTable
        Resulting measurements where the index is the label index
        and columns are the measured features.
    """
    boundary_intensity_features = measure_boundary_intensity(
        label_image=label_image,
        intensity_image=intensity_image,
        boundary_dilation_size=boundary_dilation_size,
    )
    internal_intensity_features = measure_internal_intensity(
        label_image=label_image,
        intensity_image=intensity_image,
        boundary_dilation_size=boundary_dilation_size,
    )

    intensity_measurements = pd.concat(
        [boundary_intensity_features, internal_intensity_features], axis=1
    )

    # add the boundary internal intensity ratio
    intensity_measurements["boundary_to_internal_intensity_ratio"] = safe_divide(
        intensity_measurements["boundary_intensity_mean"],
        intensity_measurements["internal_intensity_mean"],
    )

    return intensity_measurements
