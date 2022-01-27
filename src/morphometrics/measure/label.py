import pandas as pd
from skimage.measure import regionprops_table

from ..types import LabelImage, LabelMeasurementTable
from . import register_measurement


@register_measurement(name="volume")
def volume(label_im: LabelImage) -> LabelMeasurementTable:
    rp_table = regionprops_table(label_im, properties=("label", "area"))

    return pd.DataFrame(rp_table).set_index("label")


@register_measurement(name="centroid")
def centroid(label_im: LabelImage) -> LabelMeasurementTable:
    rp_table = regionprops_table(label_im, properties=("label", "centroid"))

    return pd.DataFrame(rp_table).set_index("label")
