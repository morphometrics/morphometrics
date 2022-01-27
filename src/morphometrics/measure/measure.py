from typing import List

import pandas as pd

from ..types import LabelImage, LabelMeasurementTable
from . import _measurements, available_measurments


def measure_selected(
    label_im: LabelImage, measurements: List[str]
) -> LabelMeasurementTable:
    measure_funcs = [_measurements[name] for name in measurements]
    measurement_tables = [func(label_im) for func in measure_funcs]

    return pd.concat(measurement_tables, axis=1)


def measure_all(label_im: LabelImage) -> LabelMeasurementTable:
    measurements = available_measurments()

    return measure_selected(label_im=label_im, measurements=measurements)
