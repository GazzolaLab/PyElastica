__doc__ = """ External forcing for rod test module """
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
)
from elastica.utils import Tolerance


def mock_rod_init(self):
    self.n_elems = 0.0
    self.external_forces = 0.0
    self.external_torques = 0.0
    self.director_collection = 0.0


MockRod = type("MockRod", (object,), {"__init__": mock_rod_init})


class TestNoForces:
    def test_no_forces_applied(self):
        """ No force on the rod. Test purely
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
        """ No torques on the rod. Test purely
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
    mass = np.random.rand(1, n_elem)
    acceleration_gravity = np.random.rand(dim)
    correct_external_forces = (
        mass * np.broadcast_to(acceleration_gravity, (n_elem, dim)).T
    )

    mock_rod.mass = mass

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
