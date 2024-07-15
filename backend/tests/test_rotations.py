#!/usr/bin/env python3

# This file is based on pyelastica tests/test_math/test_rotations.py

# System imports
import numpy as np
import pytest
from numpy.testing import assert_allclose
from elasticapp._PyArrays import Tensor, Matrix
from elasticapp._rotations import rotate, rotate_scalar, inv_rotate, inv_rotate_scalar


@pytest.mark.parametrize("rotate_func", [rotate, rotate_scalar])
def test_rotate_correctness(rotate_func):
    blocksize = 16

    def get_aligned_director_collection(theta_collection):
        sins = np.sin(theta_collection)
        coss = np.cos(theta_collection)
        # Get basic director out, then modify it as you like
        dir = np.tile(np.eye(3).reshape(3, 3, 1), blocksize)
        dir[0, 0, ...] = coss
        # Flip signs on [0,1] and [1,0] to go from our row-wise
        # representation to the more commonly used
        # columnwise representation, for similar reasons metioned
        # before
        dir[0, 1, ...] = sins
        dir[1, 0, ...] = -sins
        dir[1, 1, ...] = coss

        return dir

    base_angle = np.deg2rad(np.linspace(0.0, 90.0, blocksize))
    rotated_by = np.deg2rad(15.0) + 0.0 * base_angle
    rotated_about = np.array([0.0, 0.0, 1.0]).reshape(-1, 1)

    director_collection = Tensor(get_aligned_director_collection(base_angle))
    axis_collection = np.tile(rotated_about, blocksize)
    axis_collection *= rotated_by

    rotate_func(director_collection, Matrix(axis_collection))
    test_rotated_director_collection = np.asarray(director_collection)
    correct_rotation = rotated_by + 1.0 * base_angle
    correct_rotated_director_collection = get_aligned_director_collection(
        correct_rotation
    )

    assert test_rotated_director_collection.shape == (3, 3, blocksize)
    assert_allclose(
        test_rotated_director_collection, correct_rotated_director_collection
    )


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


@pytest.mark.parametrize("inv_rotate_func", [inv_rotate, inv_rotate_scalar])
@pytest.mark.parametrize("blocksize", [32, 128, 512])
@pytest.mark.parametrize("point_distribution", ["anticlockwise", "clockwise"])
def test_inv_rotate_correctness_on_circle_in_two_dimensions(
    inv_rotate_func, blocksize, point_distribution
):
    """Construct a unit circle, which we know has constant curvature,
    and see if inv_rotate gives us the correct axis of rotation and
    the angle of change

    Do this when d3 = z and d3= -z to cover both cases

    Parameters
    ----------
    blocksize

    Returns
    -------

    """
    # FSAL start at 0. and proceeds counter-clockwise
    if point_distribution == "anticlockwise":
        theta_collection = np.linspace(0.0, 2.0 * np.pi, blocksize)
    elif point_distribution == "clockwise":
        theta_collection = np.linspace(2.0 * np.pi, 0.0, blocksize)
    else:
        raise NotImplementedError

    # rate of change, should correspond to frame rotation angles
    dtheta_di = np.abs(theta_collection[1] - theta_collection[0])

    # +1 because last point should be same as first point
    director_collection = np.zeros((3, 3, blocksize))

    # First fill all d1 components
    # normal direction
    director_collection[0, 0, ...] = -np.cos(theta_collection)
    director_collection[0, 1, ...] = -np.sin(theta_collection)

    # Then all d2 components
    # tangential direction
    director_collection[1, 0, ...] = -np.sin(theta_collection)
    director_collection[1, 1, ...] = np.cos(theta_collection)

    # Then all d3 components
    director_collection[2, 2, ...] = -1.0

    # blocksize - 1 to account for end effects
    if point_distribution == "anticlockwise":
        axis_of_rotation = np.array([0.0, 0.0, -1.0])
    elif point_distribution == "clockwise":
        axis_of_rotation = np.array([0.0, 0.0, 1.0])
    else:
        raise NotImplementedError

    correct_axis_collection = np.tile(axis_of_rotation.reshape(3, 1), blocksize - 1)

    test_axis_collection = np.asarray(inv_rotate_func(Tensor(director_collection)))
    test_scaling = np.linalg.norm(test_axis_collection, axis=0)
    test_axis_collection /= test_scaling

    assert test_axis_collection.shape == (3, blocksize - 1)
    assert_allclose(test_axis_collection, correct_axis_collection)
    assert_allclose(test_scaling, 0.0 * test_scaling + dtheta_di)


@pytest.mark.parametrize("inv_rotate_func", [inv_rotate, inv_rotate_scalar])
@pytest.mark.parametrize("blocksize", [32, 128])
def test_inv_rotate_correctness_on_circle_in_two_dimensions_with_different_directors(
    inv_rotate_func,
    blocksize,
):
    """Construct a unit circle, which we know has constant curvature,
    and see if inv_rotate gives us the correct axis of rotation and
    the angle of change

    Here d3 is not z axis, so the `inv_rotate` formula returns the
    curvature but with components in the local axis, i.e. it gives
    [K1, K2, K3] in kappa_l = K1 . d1 + K2 . d2 + K3 . d3

    Parameters
    ----------
    blocksize

    Returns
    -------

    """
    # FSAL start at 0. and proceeds counter-clockwise
    theta_collection = np.linspace(0.0, 2.0 * np.pi, blocksize)
    # rate of change, should correspond to frame rotation angles
    dtheta_di = theta_collection[1] - theta_collection[0]

    # +1 because last point should be same as first point
    director_collection = np.zeros((3, 3, blocksize))

    # First fill all d3 components
    # tangential direction
    director_collection[2, 0, ...] = -np.sin(theta_collection)
    director_collection[2, 1, ...] = np.cos(theta_collection)

    # Then all d2 components
    # normal direction
    director_collection[1, 0, ...] = -np.cos(theta_collection)
    director_collection[1, 1, ...] = -np.sin(theta_collection)

    # Then all d1 components
    # binormal = d2 x d3
    director_collection[0, 2, ...] = -1.0

    # blocksize - 1 to account for end effects
    # returned curvature is in local coordinates!
    correct_axis_collection = np.tile(
        np.array([-1.0, 0.0, 0.0]).reshape(3, 1), blocksize - 1
    )
    test_axis_collection = np.asarray(inv_rotate_func(Tensor(director_collection)))
    test_scaling = np.linalg.norm(test_axis_collection, axis=0)
    test_axis_collection /= test_scaling

    assert test_axis_collection.shape == (3, blocksize - 1)
    assert_allclose(test_axis_collection, correct_axis_collection)
    assert_allclose(test_scaling, 0.0 * test_scaling + dtheta_di)
