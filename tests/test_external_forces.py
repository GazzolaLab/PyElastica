__doc__ = """ External forcing for rod test module """
import sys

sys.path.append("..")

# System imports
import numpy as np
from test_rod import TestRod
from elastica.external_forces import NoForces, GravityForces, EndpointForces
from numpy.testing import assert_allclose, assert_array_equal
from elastica.utils import Tolerance
from elastica._linalg import _batch_matmul, _batch_matvec, _batch_cross


# tests no forces on the rod
def test_no_forces():

    test_rod = TestRod()
    no_forces = NoForces(test_rod)
    test_external_forces = np.random.rand(3, 20)
    test_rod.external_forces = test_external_forces
    no_forces.apply_forces()
    assert_allclose(
        test_external_forces, test_rod.external_forces, atol=Tolerance.atol()
    )


# tests uniform gravity
def test_gravity_forces():

    test_rod = TestRod()
    gravity = np.random.rand(3)
    mass = np.random.rand(1, 20)
    test_rod.mass = mass
    gravity_forces = GravityForces(test_rod, gravity)
    test_external_forces = mass * np.broadcast_to(gravity, (20, 3)).T
    gravity_forces.apply_forces()
    assert_allclose(
        test_external_forces, test_rod.external_forces, atol=Tolerance.atol()
    )


# tests endpoint forces
def test_endpoint_forces():

    test_rod = TestRod()
    start_force = np.random.rand(3)
    end_force = np.random.rand(3)
    test_rod.external_forces = np.zeros((3, 20))
    endpt_forces = EndpointForces(test_rod, start_force, end_force)
    endpt_forces.apply_forces()
    assert_allclose(
        start_force, test_rod.external_forces[..., 0], atol=Tolerance.atol()
    )
    assert_allclose(end_force, test_rod.external_forces[..., -1], atol=Tolerance.atol())


if __name__ == "__main__":
    from pytest import main

    main([__file__])
