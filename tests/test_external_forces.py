__doc__ = """ External forcing for rod test module for Elastica implementation"""
import sys

# System imports
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from elastica.external_forces import (
    NoForces,
    GravityForces,
    EndpointForces,
    UniformTorques,
    UniformForces,
    MuscleTorques,
    inplace_addition,
    inplace_substraction,
    EndpointForcesSinusoidal,
)
from elastica.utils import Tolerance


def mock_rod_init(self):
    self.n_elems = 0.0
    self.external_forces = 0.0
    self.external_torques = 0.0
    self.director_collection = 0.0
    self.rest_lengths = 0.0


MockRod = type("MockRod", (object,), {"__init__": mock_rod_init})


class TestNoForces:
    def test_no_forces_applied(self):
        """No force on the rod. Test purely
        to improve coverage characteristics
        """
        mock_rod = MockRod()
        ext_no_forces = NoForces()

        correct_external_forces = np.random.rand(3, 20)
        mock_rod.external_forces = correct_external_forces

        ext_no_forces.apply_forces(mock_rod)

        assert_allclose(
            mock_rod.external_forces, correct_external_forces, atol=Tolerance.atol()
        )

    def test_no_torques_applied(self):
        """No torques on the rod. Test purely
        to improve coverage characteristics
        """
        mock_rod = MockRod()
        ext_no_forces = NoForces()

        correct_external_torques = np.random.rand(3, 20)
        mock_rod.external_torques = correct_external_torques

        ext_no_forces.apply_torques(mock_rod)

        assert_allclose(
            mock_rod.external_torques, correct_external_torques, atol=Tolerance.atol()
        )


# The minimum number of nodes in a system is 2
@pytest.mark.parametrize("n_elem", [2, 4, 16])
def test_gravity_forces(n_elem):
    # tests uniform gravity
    dim = 3

    mock_rod = MockRod()
    mass = np.random.randn(n_elem)
    acceleration_gravity = np.random.rand(dim)
    correct_external_forces = (
        mass * np.broadcast_to(acceleration_gravity, (n_elem, dim)).T
    )

    mock_rod.mass = mass
    mock_rod.external_forces = np.zeros((dim, n_elem))
    ext_gravity_forces = GravityForces(acceleration_gravity)
    ext_gravity_forces.apply_forces(mock_rod)

    assert_allclose(
        mock_rod.external_forces, correct_external_forces, atol=Tolerance.atol()
    )


# The minimum number of nodes in a system is 2
@pytest.mark.parametrize("n_elem", [2, 4, 16])
@pytest.mark.parametrize("rampupTime", [5, 10, 15])
@pytest.mark.parametrize("time", [0, 8, 20])
def test_endpoint_forces(n_elem, rampupTime, time):
    dim = 3

    mock_rod = MockRod()
    mock_rod.external_forces = np.zeros((dim, n_elem))

    if rampupTime > time:
        factor = time / rampupTime
    elif rampupTime <= time:
        factor = 1.0

    start_force = np.random.rand(dim)
    end_force = np.random.rand(dim)

    ext_endpt_forces = EndpointForces(start_force, end_force, rampupTime)
    ext_endpt_forces.apply_forces(mock_rod, time)

    assert_allclose(
        mock_rod.external_forces[..., 0], start_force * factor, atol=Tolerance.atol()
    )
    assert_allclose(
        mock_rod.external_forces[..., -1], end_force * factor, atol=Tolerance.atol()
    )


# The minimum number of nodes in a system is 2
@pytest.mark.parametrize("n_elem", [2, 4, 16])
@pytest.mark.parametrize("torques", [5, 10, 15])
def test_uniform_torques(n_elem, torques, time=0.0):
    dim = 3

    mock_rod = MockRod()
    mock_rod.external_torques = np.zeros((dim, n_elem))
    mock_rod.n_elems = n_elem
    mock_rod.director_collection = np.repeat(
        np.identity(3)[:, :, np.newaxis], n_elem, axis=2
    )

    torque = np.random.rand()
    direction = np.array([1.0, 0.0, 0.0])

    uniform_torques = UniformTorques(torque, direction)
    uniform_torques.apply_torques(mock_rod, time)

    assert_allclose(mock_rod.external_torques.sum(), torque, atol=Tolerance.atol())


# The minimum number of nodes in a system is 2
@pytest.mark.parametrize("n_elem", [2, 4, 16])
@pytest.mark.parametrize("forces", [5, 10, 15])
def test_uniform_forces(n_elem, forces, time=0.0):
    dim = 3

    mock_rod = MockRod()
    mock_rod.external_forces = np.zeros((dim, n_elem + 1))
    mock_rod.n_elems = n_elem

    force = np.random.rand()
    direction = np.array([0.0, 1.0, 0.0])

    uniform_forces = UniformForces(force, direction)
    uniform_forces.apply_forces(mock_rod, time)

    assert_allclose(mock_rod.external_forces.sum(), force, atol=Tolerance.atol())


# Now test muscle torques
@pytest.mark.parametrize("n_elem", [3, 4, 16])
def test_muscle_torques(n_elem):
    # tests muscle torques
    dim = 3

    mock_rod = MockRod()
    mock_rod.external_torques = np.zeros((dim, n_elem))
    mock_rod.n_elems = n_elem
    mock_rod.director_collection = np.repeat(
        np.identity(3)[:, :, np.newaxis], n_elem, axis=2
    )

    base_length = 1.0
    mock_rod.rest_lengths = np.ones(n_elem) * base_length / n_elem

    # We wont need beta coefficients, thus we are setting them to zero.
    # The reason is that, we wont use beta spline for this test case.
    # Beta spline function does not have any pytest, It is compared with
    # Elastica cpp beta spline and coefficients are optimized using CMA-ES.
    b_coeff = np.zeros((n_elem))

    # Torque = sin(2pi*t/T + 2pi*s/lambda + phi)
    # Torque = sin(pi/2 + s) = cos(s) using the below variables
    # we will make torque as function of position only nothing else!
    period = 8.0
    wave_length = 2.0 * np.pi
    wave_number = 2.0 * np.pi / (wave_length)
    phase_shift = np.pi / 4
    ramp_up_time = 0.5  # this has to be smaller than 1. Since I will set t equal to 1
    time = 1.0

    position = np.linspace(0, base_length, n_elem + 1)
    torque = np.array([np.cos(position), np.cos(position), np.cos(position)])[..., 1:-1]
    # Torque function is opposite direction in elastica cpp. Thus we need to invert the torque profile.
    torque = torque[..., ::-1]
    correct_torque = np.zeros((dim, n_elem))
    correct_torque[..., :-1] -= torque
    correct_torque[..., 1:] += torque

    # Set an a non-physical direction to check math
    direction = np.array([1.0, 1.0, 1.0])

    # Apply torques
    muscletorques = MuscleTorques(
        base_length,
        b_coeff,
        period,
        wave_number,
        phase_shift,
        direction,
        mock_rod.rest_lengths,
        ramp_up_time,
        with_spline=False,
    )
    muscletorques.apply_torques(mock_rod, time)

    # Total torque has to be zero on the body
    assert_allclose(mock_rod.external_torques[..., :].sum(), 0.0, atol=Tolerance.atol())

    # Torques on elements
    assert_allclose(mock_rod.external_torques, correct_torque, atol=Tolerance.atol())


# The minimum number of nodes in a system is 2
@pytest.mark.parametrize("n_elem", [2, 4, 16])
@pytest.mark.parametrize("ramp_up_time", [5, 10, 15])
@pytest.mark.parametrize("time", [0, 8, 20])
def test_endpoint_forces_sinusoidal(n_elem, ramp_up_time, time):
    dim = 3

    mock_rod = MockRod()
    mock_rod.external_forces = np.zeros((dim, n_elem))
    start_force_mag = np.random.rand()
    end_force_mag = np.random.rand()

    direction = np.array([0, 0, 1])
    normal = np.array([0, 1, 0])

    if ramp_up_time > time:
        start_force = -2.0 * np.array([0, start_force_mag, 0])
        end_force = -2.0 * np.array([0, end_force_mag, 0])
    else:
        start_force = start_force_mag * np.array(
            [
                np.cos(0.5 * np.pi * (time - ramp_up_time)),
                np.sin(0.5 * np.pi * (time - ramp_up_time)),
                0,
            ]
        )
        end_force = end_force_mag * np.array(
            [
                np.cos(0.5 * np.pi * (time - ramp_up_time)),
                np.sin(0.5 * np.pi * (time - ramp_up_time)),
                0,
            ]
        )

    ext_endpt_forces = EndpointForcesSinusoidal(
        start_force_mag, end_force_mag, ramp_up_time, direction, normal
    )
    ext_endpt_forces.apply_forces(mock_rod, time)

    assert_allclose(
        mock_rod.external_forces[..., 0], start_force, atol=Tolerance.atol()
    )
    assert_allclose(mock_rod.external_forces[..., -1], end_force, atol=Tolerance.atol())


@pytest.mark.parametrize("n_elem", [33, 59, 100])
def test_inplace_addition(n_elem):
    """
    This test is for inplace addition written using Numba njit functions
    Parameters
    ----------
    n_elem

    Returns
    -------

    """

    ndim = 3

    first_input_vector = np.random.randn(ndim, n_elem)
    second_input_vector = np.random.randn(ndim, n_elem)

    correct_vector = first_input_vector + second_input_vector

    test_vector = first_input_vector.copy()
    inplace_addition(test_vector, second_input_vector)

    assert_allclose(correct_vector, test_vector, atol=Tolerance.atol())


@pytest.mark.parametrize("n_elem", [33, 59, 100])
def test_inplace_substraction(n_elem):
    """
    This test is for inplace substraction written using Numba njit functions
    Parameters
    ----------
    n_elem

    Returns
    -------

    """

    ndim = 3

    first_input_vector = np.random.randn(ndim, n_elem)
    second_input_vector = np.random.randn(ndim, n_elem)

    correct_vector = first_input_vector - second_input_vector

    test_vector = first_input_vector.copy()
    inplace_substraction(test_vector, second_input_vector)

    assert_allclose(correct_vector, test_vector, atol=Tolerance.atol())
