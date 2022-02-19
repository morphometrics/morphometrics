import numpy as np

from morphometrics.data import simple_labeled_cube
from morphometrics.measure import (
    available_measurments,
    measure_all_with_defaults,
    measure_selected,
)
from morphometrics.measure.label import regionprops

print(f"available_measurements: {available_measurments()}")

label_image = simple_labeled_cube()
intensity_image = np.random.random(label_image.shape)

measurement_table = measure_all_with_defaults(
    label_image=label_image, intensity_image=intensity_image
)
print(measurement_table)

measurements_table_2 = regionprops(
    intensity_image=intensity_image, label_image=label_image
)
print(measurements_table_2)

measurement_selection = [
    "volume",
    {
        "name": "regionprops",
        "choices": {
            "size": True,
            "intensity": False,
            "perimeter": False,
            "shape": True,
        },
    },
]
measurements_table_3 = measure_selected(
    label_image=label_image,
    intensity_image=intensity_image,
    measurement_selection=measurement_selection,
)
print(measurements_table_3)
