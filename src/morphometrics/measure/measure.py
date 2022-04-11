from typing import Any, Dict, List, Optional, Union

import pandas as pd
from tqdm.autonotebook import tqdm

from ..types import IntensityImage, LabelImage, LabelMeasurementTable
from . import _measurements, available_measurments


def _make_measurement_from_string_choice(
    intensity_image: IntensityImage, label_image: LabelImage, measurement_selection: str
) -> LabelMeasurementTable:
    measurement_configuration = _measurements[measurement_selection]
    measurement_function = measurement_configuration["callable"]

    if measurement_configuration["intensity_image"] is True:
        return measurement_function(
            label_image=label_image, intensity_image=intensity_image
        )
    else:
        return measurement_function(label_image=label_image)


def _make_measurement_from_dict_choice(
    intensity_image: IntensityImage,
    label_image: LabelImage,
    measurement_selection: Dict[str, Any],
) -> LabelMeasurementTable:
    measurement_name = measurement_selection["name"]
    measurement_configuration = _measurements[measurement_name]
    measurement_type = measurement_configuration["type"]

    if measurement_type == "single":
        return _make_measurement_from_string_choice(
            intensity_image=intensity_image,
            label_image=label_image,
            measurement_selection=measurement_name,
        )
    elif measurement_type == "set":
        measurement_choices = measurement_selection["choices"]
        measurement_function = measurement_configuration["callable"]

        if measurement_configuration["intensity_image"] is True:
            return measurement_function(
                intensity_image=intensity_image,
                label_image=label_image,
                **measurement_choices
            )
        else:
            return measurement_function(label_image=label_image, **measurement_choices)


def make_measurement(
    intensity_image: IntensityImage, label_image: LabelImage, measurement_selection: str
) -> LabelMeasurementTable:
    if isinstance(measurement_selection, str):
        return _make_measurement_from_string_choice(
            intensity_image=intensity_image,
            label_image=label_image,
            measurement_selection=measurement_selection,
        )
    elif isinstance(measurement_selection, dict):
        return _make_measurement_from_dict_choice(
            intensity_image=intensity_image,
            label_image=label_image,
            measurement_selection=measurement_selection,
        )

    else:
        raise ValueError("Unknown measurement_type")


def measure_selected(
    label_image: LabelImage,
    measurement_selection: List[Union[str, Dict[str, Any]]],
    intensity_image: Optional[IntensityImage] = None,
    verbose: bool = False,
) -> LabelMeasurementTable:

    if verbose is True:
        measurement_tables = []
        for selection in tqdm(measurement_selection):
            measurement_tables.append(
                make_measurement(
                    intensity_image=intensity_image,
                    label_image=label_image,
                    measurement_selection=selection,
                )
            )
    else:
        measurement_tables = [
            make_measurement(
                intensity_image=intensity_image,
                label_image=label_image,
                measurement_selection=selection,
            )
            for selection in measurement_selection
        ]

    return pd.concat(measurement_tables, axis=1)


def measure_all_with_defaults(
    label_image: LabelImage,
    intensity_image: Optional[IntensityImage] = None,
    verbose: bool = False,
) -> LabelMeasurementTable:
    measurements = available_measurments()

    return measure_selected(
        label_image=label_image,
        intensity_image=intensity_image,
        measurement_selection=measurements,
        verbose=verbose,
    )
