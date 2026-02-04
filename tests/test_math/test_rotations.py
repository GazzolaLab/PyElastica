__doc__ = """ Test scripts for rotation kernels in Elastica Numba implementation"""
# System imports
import numpy as np
import pytest
from numpy.testing import assert_allclose

from elastica._rotations import (
    _get_rotation_matrix,
    _rotate,
    _inv_rotate,
    get_relative_rotation_two_systems,
)

from elastica.utils import Tolerance


class MockRod:
    """Mock rod class with only director_collection attribute for testing."""

    def __init__(self, director_collection):
        self.director_collection = director_collection


@pytest.mark.parametrize("zcomp", [0.5, 1.0])
@pytest.mark.parametrize("dt", [0.3, 1.0])
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


@pytest.mark.parametrize("ycomp", [0.7, 1.0])
@pytest.mark.parametrize("dt", [0.4, 1.0])
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


@pytest.mark.parametrize("xcomp", [0.6, 1.0])
@pytest.mark.parametrize("dt", [0.2, 1.0])
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
def test_get_rotation_matrix_correctness_across_blocksizes(blocksize, rng):
    dim = 3
    dt = rng.random()
    vector_collection = rng.standard_normal(dim).reshape(-1, 1)
    # No need for copying the vector collection here, as we now create
    # new arrays inside
    correct_rot_mat_collection = _get_rotation_matrix(dt, vector_collection)
    correct_rot_mat_collection = np.tile(correct_rot_mat_collection, blocksize)

    # Construct
    test_vector_collection = np.tile(vector_collection, blocksize)
    test_rot_mat_collection = _get_rotation_matrix(dt, test_vector_collection)

    assert test_rot_mat_collection.shape == (3, 3, blocksize)
    assert_allclose(test_rot_mat_collection, correct_rot_mat_collection)


def test_get_rotation_matrix_gives_orthonormal_matrices(rng):
    dim = 3
    blocksize = 16
    dt = rng.random()
    rot_mat = _get_rotation_matrix(dt, rng.standard_normal((dim, blocksize)))

    r_rt = np.einsum("ijk,ljk->ilk", rot_mat, rot_mat)
    rt_r = np.einsum("jik,jlk->ilk", rot_mat, rot_mat)

    test_mat = np.array([np.eye(dim) for _ in range(blocksize)]).T
    # We can't get there fully, but 1e-15 suffices in precision
    assert_allclose(r_rt, test_mat, atol=Tolerance.atol())
    assert_allclose(rt_r, test_mat, atol=Tolerance.atol())


def test_get_rotation_matrix_gives_unit_determinant(rng):
    dim = 3
    blocksize = 16
    dt = rng.random()
    test_rot_mat_collection = _get_rotation_matrix(
        dt, rng.standard_normal((dim, blocksize))
    )

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


def test_get_relative_rotation_two_systems_identity():
    """Test that two rods with identical orientations give identity rotation matrix."""
    # Create two mock rods with same orientation (identity rotation matrices)
    n = 4
    identity_director = np.eye(3)
    director_collection = np.tile(identity_director[..., np.newaxis], (1, 1, n + 1))

    rod1 = MockRod(director_collection)
    rod2 = MockRod(director_collection.copy())

    # Test with different indices
    rel_rot = get_relative_rotation_two_systems(rod1, 0, rod2, 0)
    rel_rot_end = get_relative_rotation_two_systems(rod1, -1, rod2, -1)

    # Should be identity matrix (or very close to it)
    identity = np.eye(3)
    assert rel_rot.shape == (3, 3)
    assert_allclose(rel_rot, identity, atol=Tolerance.atol())
    assert_allclose(rel_rot_end, identity, atol=Tolerance.atol())


def test_get_relative_rotation_two_systems_known_rotation():
    """Test with rods rotated by a known angle."""
    n = 4

    # First rod: identity rotation (aligned with standard axes)
    director1 = np.eye(3)
    director_collection1 = np.tile(director1[..., np.newaxis], (1, 1, n + 1))
    rod1 = MockRod(director_collection1)

    # Second rod: rotated 90 degrees about z-axis
    # Rotation matrix for 90 degrees about z-axis
    director2 = np.array(
        [
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    director_collection2 = np.tile(director2[..., np.newaxis], (1, 1, n + 1))
    rod2 = MockRod(director_collection2)

    rel_rot = get_relative_rotation_two_systems(rod1, 0, rod2, 0)

    # Expected rotation matrix: Since rel_rot = director1 @ director2.T
    # and director1 = I, we get director2.T
    # director2 is a 90-degree rotation about z-axis, so director2.T is:
    expected_rot = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    assert rel_rot.shape == (3, 3)
    assert_allclose(rel_rot, expected_rot, atol=Tolerance.atol())


def test_get_relative_rotation_two_systems_properties():
    """Test that the relative rotation matrix has proper rotation matrix properties."""
    n = 4

    # First rod: identity rotation
    director1 = np.eye(3)
    director_collection1 = np.tile(director1[..., np.newaxis], (1, 1, n + 1))
    rod1 = MockRod(director_collection1)

    # Second rod: rotated 90 degrees about z-axis
    director2 = np.array(
        [
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    director_collection2 = np.tile(director2[..., np.newaxis], (1, 1, n + 1))
    rod2 = MockRod(director_collection2)

    rel_rot = get_relative_rotation_two_systems(rod1, 0, rod2, 0)

    # Check orthogonality: R @ R.T should be identity
    r_rt = rel_rot @ rel_rot.T
    rt_r = rel_rot.T @ rel_rot
    identity = np.eye(3)

    assert_allclose(r_rt, identity, atol=Tolerance.atol())
    assert_allclose(rt_r, identity, atol=Tolerance.atol())

    # Check determinant is 1 (for proper rotation)
    det = np.linalg.det(rel_rot)
    assert_allclose(det, 1.0, atol=Tolerance.atol())


@pytest.mark.parametrize("index1", [0, -1, 2])
@pytest.mark.parametrize("index2", [0, -1, 1])
def test_get_relative_rotation_two_systems_different_indices(index1, index2):
    """Test function with different index combinations."""
    n = 4

    # Create mock rods with identity rotations
    director = np.eye(3)
    director_collection = np.tile(director[..., np.newaxis], (1, 1, n + 1))

    rod1 = MockRod(director_collection)
    rod2 = MockRod(director_collection.copy())

    rel_rot = get_relative_rotation_two_systems(rod1, index1, rod2, index2)

    # Should always return a 3x3 matrix
    assert rel_rot.shape == (3, 3)

    # Should be a valid rotation matrix
    r_rt = rel_rot @ rel_rot.T
    assert_allclose(r_rt, np.eye(3), atol=Tolerance.atol())
    assert_allclose(np.linalg.det(rel_rot), 1.0, atol=Tolerance.atol())


def test_get_relative_rotation_two_systems_inverse_relationship():
    """Test that rel_rot(system1, system2) is the inverse of rel_rot(system2, system1)."""
    n = 4

    # First rod: identity rotation
    director1 = np.eye(3)
    director_collection1 = np.tile(director1[..., np.newaxis], (1, 1, n + 1))
    rod1 = MockRod(director_collection1)

    # Second rod: rotated 90 degrees about z-axis
    director2 = np.array(
        [
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    director_collection2 = np.tile(director2[..., np.newaxis], (1, 1, n + 1))
    rod2 = MockRod(director_collection2)

    rel_rot_12 = get_relative_rotation_two_systems(rod1, 0, rod2, 0)
    rel_rot_21 = get_relative_rotation_two_systems(rod2, 0, rod1, 0)

    # rel_rot_21 should be the transpose (inverse) of rel_rot_12
    assert_allclose(rel_rot_21, rel_rot_12.T, atol=Tolerance.atol())
    # Or equivalently, their product should be identity
    assert_allclose(rel_rot_12 @ rel_rot_21, np.eye(3), atol=Tolerance.atol())
