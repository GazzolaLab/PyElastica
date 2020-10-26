#!/usr/bin/env python3
__doc__ = """ Test scripts for rotation kernels in Elastica Numpy implementaiton """
# System imports
import numpy as np
import pytest
from numpy.testing import assert_allclose
import sys

from elastica._elastica_numpy._rotations import (
    _generate_skew_map,  # noqa
    _get_diag_map,  # noqa
    _get_inv_skew_map,  # noqa
    _get_skew_symmetric_pair,  # noqa
    _get_skew_map,  # noqa
    _inv_skew_symmetrize,  # noqa
    _skew_symmetrize,  # noqa
    _skew_symmetrize_sq,  # noqa
    _get_rotation_matrix,  # noqa
    _rotate,
    _inv_rotate,
)

from elastica.utils import Tolerance


###############################################################################
##################### Implementation tests start ##############################
###############################################################################

# Cross products only make sense till dim = 3 (dim = 7 is an exception)
@pytest.mark.parametrize("dim", [2, 3])
def test_skewmap_integrity(dim):
    maps = _generate_skew_map(dim)
    for a_map in maps:
        src, tgt_i, tgt_j = a_map
        assert src < dim
        assert tgt_i < dim
        assert tgt_j < dim


skewmap_correctness_data = [(2, [(1, 1, 0)]), (3, [(2, 1, 0), (1, 0, 2), (0, 2, 1)])]


@pytest.mark.parametrize("dim, correct_list", skewmap_correctness_data)
def test_skewmap_correctness(dim, correct_list):
    assert _generate_skew_map(dim) == correct_list


matrix_to_vector_data = [(2, ((1, 1, 0),)), (3, ((0, 2, 1), (1, 0, 2), (2, 1, 0)))]


@pytest.mark.parametrize("dim, matrix_to_vector_map", matrix_to_vector_data)
def test_matrix_to_vector_map(dim, matrix_to_vector_map):
    assert _get_skew_map(dim) == matrix_to_vector_map


vector_to_matrix_data = [
    (2, ((1, 0, 1),)),  # Gotcha! tuple always has an extra , at the end
    (3, ((1, 0, 2), (0, 2, 1), (2, 1, 0))),
]


@pytest.mark.parametrize("dim, vector_to_matrix_map", vector_to_matrix_data)
def test_vector_to_matrix_map(dim, vector_to_matrix_map):
    assert _get_inv_skew_map(dim) == vector_to_matrix_map


matrix_diag_data = [(2, (0, 3)), (3, (0, 4, 8)), (4, (0, 5, 10, 15))]


@pytest.mark.parametrize("dim, diag_map", matrix_diag_data)
def test_matrix_diag_map(dim, diag_map):
    assert _get_diag_map(dim) == diag_map


def test_skew_symmetrize_correctness_in_two_dimensions():
    dim = 3
    vector = np.hstack((np.random.randn(2), 1))

    test_matrix = _skew_symmetrize(vector[:, np.newaxis])

    # Reshape for checking correctness and drop the last dimension
    test_matrix = test_matrix.reshape(dim, dim, -1)
    test_matrix = test_matrix[:-1, :-1, 0]

    correct_matrix = np.array([[0.0, -1.0], [1.0, 0.0]])

    assert_allclose(test_matrix, correct_matrix)


@pytest.mark.parametrize("blocksize", [1, 32, 128, 512])
def test_skew_symmetrize_correctness_in_three_dimensions(blocksize):
    dim = 3
    vector = np.random.randn(dim, blocksize)
    correct_matrix = np.zeros((dim * dim, blocksize))

    correct_matrix[1] = -vector[2]
    correct_matrix[2] = vector[1]
    correct_matrix[3] = vector[2]
    correct_matrix[5] = -vector[0]
    correct_matrix[6] = -vector[1]
    correct_matrix[7] = vector[0]

    correct_matrix = correct_matrix.reshape(dim, dim, -1)

    # reshape and squeeze because we are testing a single vector
    test_matrix = _skew_symmetrize(vector)

    assert test_matrix.shape == (3, 3, blocksize)
    assert_allclose(test_matrix, correct_matrix)


@pytest.mark.parametrize("blocksize", [1, 32, 128, 512])
def test_skew_symmetrize_sq_correctness_in_three_dimensions(blocksize):
    dim = 3
    vector = np.random.randn(dim, blocksize)
    correct_matrix = _skew_symmetrize(vector).reshape(dim, dim, -1)
    correct_matrix = np.einsum("ijk,jlk->ilk", correct_matrix, correct_matrix)

    # reshape and squeeze because we are testing a single vector
    test_matrix = _skew_symmetrize_sq(vector)

    assert test_matrix.shape == (3, 3, blocksize)
    assert_allclose(test_matrix, correct_matrix)


def test_get_skew_symmetric_pair_correctness():
    dim = 3
    blocksize = 8
    vector_collection = np.random.randn(dim, blocksize)
    u, u_sq = _get_skew_symmetric_pair(vector_collection)
    assert u_sq.shape == (3, 3, blocksize)
    assert_allclose(u_sq, _skew_symmetrize_sq(vector_collection))


@pytest.mark.parametrize("blocksize", [1, 32, 128, 512])
def test_inv_skew_symmetrize_correctness(blocksize):
    dim = 3
    vector = np.random.randn(dim, blocksize)
    input_matrix = _skew_symmetrize(vector)

    # reshape and squeeze because we are testing a single vector
    test_vector = _inv_skew_symmetrize(input_matrix)

    assert test_vector.shape == (3, blocksize)
    assert_allclose(test_vector, vector)


@pytest.mark.parametrize("zcomp", [np.random.random_sample(), 1.0])
@pytest.mark.parametrize("dt", [np.random.random_sample(), 1.0])
def test_get_rotation_matrix_correct_rotation_about_z(zcomp, dt):
    vector_collection = np.array([0.0, 0.0, zcomp]).reshape(-1, 1)
    test_rot_mat = _get_rotation_matrix(dt, vector_collection)
    test_theta = zcomp * dt
    # Notice that the correct_rot_mat seems to be a transpose
    # ie if you take a vector v, and do Rv, it seems to do a
    # rotation be -test_theta.
    # The catch in this case is that our directors (d_1, d_2, d_3)
    # are all aligned row-wise rather than columnwise. Thus we need
    # to multiply them by a R.transpose, which is equivalent to
    # multiplying by a R(-theta).
    correct_rot_mat = np.array(
        [
            [np.cos(test_theta), np.sin(test_theta), 0.0],
            [-np.sin(test_theta), np.cos(test_theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    ).reshape(3, 3, 1)

    assert test_rot_mat.shape == (3, 3, 1)
    assert_allclose(test_rot_mat, correct_rot_mat, atol=Tolerance.atol())


@pytest.mark.parametrize("ycomp", [np.random.random_sample(), 1.0])
@pytest.mark.parametrize("dt", [np.random.random_sample(), 1.0])
def test_get_rotation_matrix_correct_rotation_about_y(ycomp, dt):
    vector_collection = np.array([0.0, ycomp, 0.0]).reshape(-1, 1)
    test_rot_mat = _get_rotation_matrix(dt, vector_collection)
    test_theta = ycomp * dt
    # Transpose for similar reasons mentioned before
    correct_rot_mat = np.array(
        [
            [np.cos(test_theta), 0.0, -np.sin(test_theta)],
            [0.0, 1.0, 0.0],
            [np.sin(test_theta), 0.0, np.cos(test_theta)],
        ]
    ).reshape(3, 3, 1)

    assert test_rot_mat.shape == (3, 3, 1)
    assert_allclose(test_rot_mat, correct_rot_mat, atol=Tolerance.atol())


@pytest.mark.parametrize("xcomp", [np.random.random_sample(), 1.0])
@pytest.mark.parametrize("dt", [np.random.random_sample(), 1.0])
def test_get_rotation_matrix_correct_rotation_about_x(xcomp, dt):
    vector_collection = np.array([xcomp, 0.0, 0.0]).reshape(-1, 1)
    test_rot_mat = _get_rotation_matrix(dt, vector_collection)
    test_theta = xcomp * dt
    # Transpose for similar reasons mentioned before
    correct_rot_mat = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(test_theta), np.sin(test_theta)],
            [0.0, -np.sin(test_theta), np.cos(test_theta)],
        ]
    ).reshape(3, 3, 1)

    assert test_rot_mat.shape == (3, 3, 1)
    assert_allclose(test_rot_mat, correct_rot_mat, atol=Tolerance.atol())


def test_get_rotation_matrix_correctness_in_three_dimensions():
    # A rotation of 120 degrees about x=y=z gives
    # the permutation matrix P
    # {\begin{bmatrix}0&0&1\\1&0&0\\0&1&0\end{bmatrix}}
    # Basically x becomes y, y becomes z, z becomes x
    # For our case then (with directors aligned on the rows,
    # rather than columns)
    # we have
    # {\begin{bmatrix}0&1&0\\0&0&1\\1&0&0\end{bmatrix}}
    vector_collection = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    vector_collection = vector_collection.reshape(-1, 1)
    theta = np.deg2rad(120.0)
    test_rot_mat = _get_rotation_matrix(theta, vector_collection)
    # previous correct matrix
    # correct_rot_mat = np.roll(np.eye(3), -1, axis=1).reshape(3, 3, 1)
    correct_rot_mat = np.roll(np.eye(3), 1, axis=1).reshape(3, 3, 1)

    assert_allclose(test_rot_mat, correct_rot_mat, atol=Tolerance.atol())


def test_get_rotation_matrix_correctness_against_canned_example():
    """
    Computer test code at page 5-7 from

    "Rotation, Reflection, and Frame Changes
    Orthogonal tensors in computational engineering mechanics"
    by Rebecca Brennen, 2018, IOP

    Returns
    -------

    """
    vector_collection = np.array([1, 3.2, 7])
    vector_collection /= np.linalg.norm(vector_collection)
    vector_collection = vector_collection.reshape(-1, 1)
    theta = np.deg2rad(76.0)
    test_rot_mat = _get_rotation_matrix(theta, vector_collection)
    # Previous correct matrix, which did not have a transpose
    """
    correct_rot_mat = np.array(
        [
            [0.254506, -0.834834, 0.488138],
            [0.915374, 0.370785, 0.156873],
            [-0.311957, 0.406903, 0.858552],
        ]
    ).reshape(3, 3, 1)
    """
    # Transpose for similar reasons mentioned before
    correct_rot_mat = np.array(
        [
            [0.254506, -0.834834, 0.488138],
            [0.915374, 0.370785, 0.156873],
            [-0.311957, 0.406903, 0.858552],
        ]
    ).T.reshape(3, 3, 1)

    assert_allclose(test_rot_mat, correct_rot_mat, atol=1e-6)


@pytest.mark.parametrize("blocksize", [32, 128, 512])
def test_get_rotation_matrix_correctness_across_blocksizes(blocksize):
    dim = 3
    dt = np.random.random_sample()
    vector_collection = np.random.randn(dim).reshape(-1, 1)
    # No need for copying the vector collection here, as we now create
    # new arrays inside
    correct_rot_mat_collection = _get_rotation_matrix(dt, vector_collection)
    correct_rot_mat_collection = np.tile(correct_rot_mat_collection, blocksize)

    # Construct
    test_vector_collection = np.tile(vector_collection, blocksize)
    test_rot_mat_collection = _get_rotation_matrix(dt, test_vector_collection)

    assert test_rot_mat_collection.shape == (3, 3, blocksize)
    assert_allclose(test_rot_mat_collection, correct_rot_mat_collection)


def test_get_rotation_matrix_gives_orthonormal_matrices():
    dim = 3
    blocksize = 16
    dt = np.random.random_sample()
    rot_mat = _get_rotation_matrix(dt, np.random.randn(dim, blocksize))

    r_rt = np.einsum("ijk,ljk->ilk", rot_mat, rot_mat)
    rt_r = np.einsum("jik,jlk->ilk", rot_mat, rot_mat)

    test_mat = np.array([np.eye(dim) for _ in range(blocksize)]).T
    # We can't get there fully, but 1e-15 suffices in precision
    assert_allclose(r_rt, test_mat, atol=Tolerance.atol())
    assert_allclose(rt_r, test_mat, atol=Tolerance.atol())


def test_get_rotation_matrix_gives_unit_determinant():
    dim = 3
    blocksize = 16
    dt = np.random.random_sample()
    test_rot_mat_collection = _get_rotation_matrix(dt, np.random.randn(dim, blocksize))

    test_det_collection = np.linalg.det(test_rot_mat_collection.T)
    correct_det_collection = 1.0 + 0.0 * test_det_collection

    assert_allclose(correct_det_collection, test_det_collection)


def test_rotate_correctness():
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

    director_collection = get_aligned_director_collection(base_angle)
    axis_collection = np.tile(rotated_about, blocksize)
    axis_collection *= rotated_by
    dt = 1.0

    test_rotated_director_collection = _rotate(director_collection, dt, axis_collection)
    correct_rotation = rotated_by + 1.0 * base_angle
    correct_rotated_director_collection = get_aligned_director_collection(
        correct_rotation
    )

    assert test_rotated_director_collection.shape == (3, 3, blocksize)
    assert_allclose(
        test_rotated_director_collection,
        correct_rotated_director_collection,
        atol=Tolerance.atol(),
    )


def test_inv_rotate_correctness_simple_in_three_dimensions():
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
    test_axis_collection = _inv_rotate(input_director_collection)

    correct_angle = np.deg2rad(120)
    test_angle = np.linalg.norm(test_axis_collection, axis=0)  # (3,1)
    test_axis_collection /= test_angle

    assert_allclose(
        test_axis_collection, correct_axis_collection, atol=Tolerance.atol()
    )
    assert_allclose(test_angle, correct_angle, atol=Tolerance.atol())


# TODO Resolve ambiguity with signs. TOP PRIORITY!!!!!!!!!!!!!!!
@pytest.mark.xfail
@pytest.mark.parametrize("blocksize", [32, 128, 512])
@pytest.mark.parametrize("point_distribution", ["anticlockwise", "clockwise"])
def test_inv_rotate_correctness_on_circle_in_two_dimensions(
    blocksize, point_distribution
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

    test_axis_collection = _inv_rotate(director_collection)
    test_scaling = np.linalg.norm(test_axis_collection, axis=0)
    test_axis_collection /= test_scaling

    assert test_axis_collection.shape == (3, blocksize - 1)
    assert_allclose(test_axis_collection, correct_axis_collection)
    assert_allclose(test_scaling, 0.0 * test_scaling + dtheta_di, atol=Tolerance.atol())


# TODO Resolve ambiguity with signs. TOP PRIORITY!!!!!!!!!!!!!!!
@pytest.mark.xfail
@pytest.mark.parametrize("blocksize", [32, 128])
def test_inv_rotate_correctness_on_circle_in_two_dimensions_with_different_directors(
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
    test_axis_collection = _inv_rotate(director_collection)
    test_scaling = np.linalg.norm(test_axis_collection, axis=0)
    test_axis_collection /= test_scaling

    assert test_axis_collection.shape == (3, blocksize - 1)
    assert_allclose(test_axis_collection, correct_axis_collection)
    assert_allclose(test_scaling, 0.0 * test_scaling + dtheta_di, atol=Tolerance.atol())


###############################################################################
##################### Implementation tests finis ##############################
###############################################################################
if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main

        main([__file__])
