"""Module for testing accuracy evaluation measures (RMSE, MAE...)"""



from math import sqrt

import pytest

from surprise.accuracy import mae, rmse, fcp, mse


def pred(true_r, est, u0=None):
    """Just a small tool to build a prediction with appropriate format."""
    return (u0, None, true_r, est, None)


def test_mae():
    """Tests for the MAE function."""

    predictions = [pred(0, 0), pred(1, 1), pred(2, 2), pred(100, 100)]
    assert mae(predictions) == 0

    predictions = [pred(0, 0), pred(0, 2)]
    assert mae(predictions) == abs(0 - 2) / 2

    predictions = [pred(2, 0), pred(3, 4)]
    assert mae(predictions) == (abs(2 - 0) + abs(3 - 4)) / 2

    with pytest.raises(ValueError):
        mae([])


def test_rmse():
    """Tests for the RMSE function."""

    predictions = [pred(0, 0), pred(1, 1), pred(2, 2), pred(100, 100)]
    assert rmse(predictions) == 0

    predictions = [pred(0, 0), pred(0, 2)]
    assert rmse(predictions) == sqrt((0 - 2)**2 / 2)

    predictions = [pred(2, 0), pred(3, 4)]
    assert rmse(predictions) == sqrt(((2 - 0)**2 + (3 - 4)**2) / 2)

    with pytest.raises(ValueError):
        rmse([])


def test_fcp():
    """Tests for the FCP function."""

    predictions = [pred(0, 0, u0='u1'), pred(1, 1, u0='u1'), pred(2, 2,
                   u0='u2'), pred(100, 100, u0='u2')]
    assert fcp(predictions) == 1

    predictions = [pred(0, 0, u0='u1'), pred(0, 0, u0='u1')]
    with pytest.raises(ValueError):
        fcp(predictions)

    predictions = [pred(0, 0, u0='u1')]
    with pytest.raises(ValueError):
        fcp(predictions)

    predictions = [pred(0, 1, u0='u1'), pred(1, 0, u0='u1'), pred(2, 0.5,
                   u0='u2'), pred(0, 0.6, u0='u2')]
    assert fcp(predictions) == 0

    with pytest.raises(ValueError):
        fcp([])


def test_mse():
    """Tests for the MSE function."""

    predictions = [pred(0, 0), pred(1, 1), pred(2, 2), pred(100, 100)]
    assert mse(predictions) == 0

    predictions = [pred(0, 0), pred(0, 2)]
    assert mse(predictions) == ((0 - 2) ** 2) / 2

    predictions = [pred(2, 0), pred(3, 4)]
    assert mse(predictions) == ((2 - 0) ** 2 + (3 - 4) ** 2) / 2

    with pytest.raises(ValueError):
        mse([])
