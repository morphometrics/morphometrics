from typing import Callable, List, Optional

from toolz import curry

_measurements = dict()


@curry
def register_measurement(func: Callable, name: Optional[str] = None) -> Callable:
    _measurements[name] = func
    return func


def available_measurments() -> List[str]:
    return [k for k in _measurements]
