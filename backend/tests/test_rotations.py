#!/usr/bin/env python3

# This file is based on pyelastica tests/test_math/test_rotations.py

# System imports
import numpy as np
import pytest
from numpy.testing import assert_allclose
from elasticapp._PyArrays import Tensor
from elasticapp._rotations import inv_rotate, inv_rotate_scalar


@pytest.mark.parametrize("inv_rotate_func", [inv_rotate, inv_rotate_scalar])
def test_inv_rotate(inv_rotate_func):
    # A rotation of 120 degrees about x=y=z gives
    # the permutation matrix P
    # {\begin{bmatrix}0&0&1\\1&0&0\\0&1&0\end{bmatrix}}
    # Hence if we pass in I and P*I, the vector returned should be
    # along [1.0, 1.0, 1.0] / sqrt(3.0) with a rotation of angle = 120/180 * pi
    rotate_from_matrix = np.eye(3).reshape(3, 3, 1)
    # Q_new = Q_old . R^T
    rotate_to_matrix = np.eye(3) @ np.roll(np.eye(3), -1, axis=1).T
    input_director_collection = np.dstack(
        (rotate_from_matrix, rotate_to_matrix.reshape(3, 3, -1))
    )

    correct_axis_collection = np.ones((3, 1)) / np.sqrt(3.0)
    test_axis_collection = np.asarray(
        inv_rotate_func(Tensor(input_director_collection))
    )

    correct_angle = np.deg2rad(120)
    test_angle = np.linalg.norm(test_axis_collection, axis=0)  # (3,1)
    test_axis_collection /= test_angle

    assert_allclose(test_axis_collection, correct_axis_collection)
    assert_allclose(test_angle, correct_angle)
