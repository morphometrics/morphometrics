from typing import Callable, List, Optional

from toolz import curry

_measurements = dict()


@curry
def register_measurement(
    func: Callable, name: Optional[str] = None, uses_intensity_image: bool = True
) -> Callable:
    _measurements[name] = {
        "type": "single",
        "callable": func,
        "choices": None,
        "intensity_image": uses_intensity_image,
    }
    return func


@curry
def register_measurement_set(
    func: Callable,
    choices: List[str],
    name: Optional[str] = None,
    uses_intensity_image: bool = True,
) -> Callable:
    _measurements[name] = {
        "type": "set",
        "callable": func,
        "choices": choices,
        "intensity_image": uses_intensity_image,
    }
    return func


def available_measurments() -> List[str]:
    return [k for k in _measurements]


from .intensity import measure_boundary_intensity, measure_internal_intensity
from .label import measure_surface_properties_from_labels, regionprops
from .measure import measure_all_with_defaults, measure_selected
