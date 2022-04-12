from typing import Tuple

import numpy as np
import pandas as pd
from skimage import draw

from ..types import IntensityImage, LabelImage


def simple_labeled_cube() -> LabelImage:
    label_im = np.zeros((10, 10, 10), dtype=int)
    label_im[5:10, 5:10, 5:10] = 1
    label_im[5:10, 0:5, 0:5] = 2
    label_im[0:5, 0:10, 0:10] = 3
    return label_im


def _make_cylinder(rng: np.random.Generator, length: float = 15, radius: float = 3):
    length_nominal = length
    length_stdev = 1
    length = rng.normal(length_nominal, length_stdev)

    radius_nominal = radius
    radius_stdev = 1
    radius = rng.normal(radius_nominal, radius_stdev)

    rr, cc = draw.disk((10, 10), radius)
    circle_image = np.zeros((20, 20), dtype=bool)
    circle_image[rr, cc] = True

    # extrude

    return np.repeat(circle_image[np.newaxis, :, :], length, axis=0)


def _make_sphere(rng: np.random.Generator, radius: float = 5):
    radius = 5
    radius_stdev = 1
    radius = rng.normal(radius, radius_stdev)

    return draw.ellipsoid(radius, radius, radius)


_object_functions = {"cylinder": _make_cylinder, "sphere": _make_sphere}


def cylinders_and_spheres() -> Tuple[LabelImage, pd.DataFrame, IntensityImage]:
    """Make a label image with cylinders and spheres.

    Returns
    -------
    label_image : LabelImage
        The label image. It has shape (300, 300, 300)
    label_table : pd.DataFrame
        Table with the shape for each object in label_image.
        Column "label" contains the label value and column
        "shape" contains the shape as a string (cylinder or sphere).
    """
    rng = np.random.default_rng(42)
    label_image = np.zeros((300, 300, 300), dtype=int)
    intensity_image = np.zeros((300, 300, 300), dtype=float)

    label_index = 1
    label_records = []

    # embed the image
    for x_start in np.arange(0, 300, 50):
        for y_start in np.arange(0, 300, 50):
            for z_start in np.arange(0, 300, 50):
                object_index = rng.choice(["cylinder", "sphere"])
                object_func = _object_functions[object_index]
                object_mask = object_func(rng=rng)

                object_coords = np.argwhere(object_mask)

                z_coords = z_start + object_coords[:, 0]
                y_coords = y_start + object_coords[:, 1]
                x_coords = x_start + object_coords[:, 2]

                label_image[z_coords, y_coords, x_coords] = label_index
                intensity_image[z_coords, y_coords, x_coords] = np.random.rand()

                label_records.append({"label": label_index, "shape": object_index})

                label_index += 1

    label_table = pd.DataFrame(label_records)

    return label_image, label_table, intensity_image
