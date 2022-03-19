from typing import Union

import numpy as np


def safe_divide(
    numerator: Union[float, np.ndarray],
    denominator: Union[float, np.ndarray],
    eps: float = 1e-10,
) -> Union[float, np.ndarray]:
    """Computes a safe divide which returns 0 where the denominator < eps (~zero).

    Modified from stardist.matching._safe_divide
    https://github.com/stardist/stardist/

    Parameters
    ----------
    numerator : Union[float, np.ndarray]
        The top part of the division.
    denominator : Union[float, np.ndarray]
        The bottom part of the division.

    Returns
    -------
    quotient : Union[float, np.ndarray]
        The result of numerator / denominator. Values where the
        denominator < eps are set to 0.
    """
    if np.isscalar(numerator) and np.isscalar(denominator):
        return numerator / denominator if np.abs(denominator) > eps else 0.0
    else:
        quotient = np.zeros(np.broadcast(numerator, denominator).shape, np.float32)
        np.divide(numerator, denominator, out=quotient, where=np.abs(denominator) > eps)
        return quotient
