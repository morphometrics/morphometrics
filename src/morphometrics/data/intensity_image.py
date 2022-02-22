from typing import Tuple

import numpy as np

from ..types import IntensityImage


def random_3d_image(shape: Tuple[int, int, int] = (10, 10, 10)) -> IntensityImage:
    return np.random.random(shape)
