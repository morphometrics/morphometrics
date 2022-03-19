import numpy as np

from morphometrics.utils.math_utils import safe_divide


def test_safe_divide_array():
    quotient = safe_divide(
        np.array([1, 1, 1, 0, 1]), np.array([1, 2, 0, 5, 1e-11]), eps=1e-10
    )
    expected_quotient = np.array([1, 0.5, 0, 0, 0])
    np.testing.assert_allclose(quotient, expected_quotient)


def test_save_divide_scalar():
    quotient = safe_divide(5, 5)
    np.testing.assert_allclose(quotient, 1)

    quotient = safe_divide(0, 5)
    np.testing.assert_allclose(quotient, 0)

    quotient = safe_divide(5, 0)
    np.testing.assert_allclose(quotient, 0)

    quotient = safe_divide(5, 1e-11, eps=1e-10)
    np.testing.assert_allclose(quotient, 0)


def test_safe_divide_mixed():
    quotient = safe_divide(5, np.array([5, 0]))
    np.testing.assert_allclose(quotient, [1, 0])

    quotient = safe_divide(0, np.array([5, 0]))
    np.testing.assert_allclose(quotient, [0, 0])

    quotient = safe_divide(np.array([5, 0]), 5)
    np.testing.assert_allclose(quotient, [1, 0])

    quotient = safe_divide(np.array([5, 0]), 0)
    np.testing.assert_allclose(quotient, [0, 0])
