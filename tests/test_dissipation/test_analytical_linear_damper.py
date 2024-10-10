__doc__ = """ Test implementation of the analytical linear damper"""

from itertools import combinations

import pytest
import numpy as np
from numpy.testing import assert_allclose

from elastica.dissipation import AnalyticalLinearDamper
from elastica.utils import Tolerance
from tests.test_rod.mock_rod import MockTestRod


@pytest.fixture
def analytical_error_message():
    message = (
        r"AnalyticalLinearDamper usage:\n"
        r"\tsimulator.dampen\(rod\).using\(\n"
        r"\t\tAnalyticalLinearDamper,\n"
        r"\t\ttranslational_damping_constant=...,\n"
        r"\t\trotational_damping_constant=...,\n"
        r"\t\ttime_step=...,\n"
        r"\t\)\n"
        r"\tor\n"
        r"\tsimulator.dampen\(rod\).using\(\n"
        r"\t\tAnalyticalLinearDamper,\n"
        r"\t\tuniform_damping_constant=...,\n"
        r"\t\ttime_step=...,\n"
        r"\t\)\n"
        r"\tor \(deprecated in 0.4.0\)\n"
        r"\tsimulator.dampen\(rod\).using\(\n"
        r"\t\tAnalyticalLinearDamper,\n"
        r"\t\tdamping_constant=...,\n"
        r"\t\ttime_step=...,\n"
        r"\t\)\n"
    )
    return message


def test_analytical_linear_damper_error(analytical_error_message):
    test_rod = MockTestRod()
    dummy = np.float64(0.0)

    kwargs = [
        "damping_constant",
        "uniform_damping_constant",
        "translational_damping_constant",
        "rotational_damping_constant",
    ]

    a = (0, 1, 2, 3)
    valid_combs = [(0,), (1,), (2, 3)]
    for i in range(5):
        combs = list(combinations(a, i))
        for c in combs:
            if c not in valid_combs:
                invalid_kwargs = dict([(kwargs[idx], dummy) for idx in c])
                with pytest.raises(ValueError, match=analytical_error_message):
                    AnalyticalLinearDamper(
                        _system=test_rod,
                        time_step=dummy,
                        **invalid_kwargs,
                    )


def test_analytical_linear_damper_deprecated():
    test_rod = MockTestRod()
    test_rod.mass[:] = 1.0
    test_dilatation = 2.0 * np.ones((3, test_rod.n_elems))
    test_inv_mass_second_moment_of_inertia = 3.0 * np.ones((3, 3, test_rod.n_elems))
    test_rod.dilatation = test_dilatation.copy()
    test_rod.inv_mass_second_moment_of_inertia = (
        test_inv_mass_second_moment_of_inertia.copy()
    )
    damping_constant = 0.25
    dt = 0.5
    exponential_damper = AnalyticalLinearDamper(
        _system=test_rod, damping_constant=damping_constant, time_step=dt
    )
    # check common prefactors
    # e ^ (-damp_coeff * dt)
    ref_translational_damping_coefficient = np.exp(-0.25 * 0.5)
    # e ^ (-damp_coeff * dt * elemental_mass * inv_mass_second_moment_of_inertia)
    ref_rotational_damping_coefficient = np.exp(-0.25 * 0.5 * 1.0 * 3.0) * np.ones(
        (3, test_rod.n_elems)
    )
    # end corrections
    ref_rotational_damping_coefficient[:, 0] = np.exp(-0.25 * 0.5 * 1.5 * 3.0)
    ref_rotational_damping_coefficient[:, -1] = np.exp(-0.25 * 0.5 * 1.5 * 3.0)

    pre_damping_velocity_collection = np.random.rand(3, test_rod.n_elems + 1)
    test_rod.velocity_collection = (
        pre_damping_velocity_collection.copy()
    )  # We need copy of the list not a reference to this array
    pre_damping_omega_collection = np.random.rand(3, test_rod.n_elems)
    test_rod.omega_collection = (
        pre_damping_omega_collection.copy()
    )  # We need copy of the list not a reference to this array
    exponential_damper.dampen_rates(test_rod, time=0)
    post_damping_velocity_collection = (
        pre_damping_velocity_collection * ref_translational_damping_coefficient
    )
    # multiplying_factor = ref_rot_coeff ^ dilation
    post_damping_omega_collection = (
        pre_damping_omega_collection * ref_rotational_damping_coefficient**2.0
    )
    assert_allclose(
        post_damping_velocity_collection,
        test_rod.velocity_collection,
        atol=Tolerance.atol(),
    )
    assert_allclose(
        post_damping_omega_collection, test_rod.omega_collection, atol=Tolerance.atol()
    )


def test_analytical_linear_damper_uniform():
    test_rod = MockTestRod()

    velocity = np.linspace(0.0, 3.0, 3 * (test_rod.n_elems + 1)).reshape(
        (3, test_rod.n_elems + 1)
    )
    omega = np.linspace(5.0, 8.0, 3 * test_rod.n_elems).reshape((3, test_rod.n_elems))
    test_rod.velocity_collection[:, :] = velocity
    test_rod.omega_collection[:, :] = omega

    test_constant = 2.0
    test_dt = np.float64(1.5)
    test_coeff = np.exp(-test_dt * test_constant)

    damper = AnalyticalLinearDamper(
        _system=test_rod, uniform_damping_constant=test_constant, time_step=test_dt
    )
    damper.dampen_rates(test_rod, time=np.float64(0.0))

    expected_velocity = velocity * test_coeff
    expected_omega = omega * test_coeff

    assert_allclose(
        expected_velocity,
        test_rod.velocity_collection,
        atol=Tolerance.atol(),
    )
    assert_allclose(expected_omega, test_rod.omega_collection, atol=Tolerance.atol())


def test_analytical_linear_damper_physical():
    test_rod = MockTestRod()

    velocity = np.linspace(0.0, 3.0, 3 * (test_rod.n_elems + 1)).reshape(
        (3, test_rod.n_elems + 1)
    )
    omega = np.linspace(5.0, 8.0, 3 * test_rod.n_elems).reshape((3, test_rod.n_elems))
    mass = np.linspace(6.0, 4.0, test_rod.n_elems + 1)
    inv_moi_full = 1.0 / np.linspace(10.0, 15.0, 9 * test_rod.n_elems).reshape(
        (3, 3, test_rod.n_elems)
    )
    inv_moi = np.diagonal(inv_moi_full).T
    dilatation = np.linspace(0.5, 1.5, test_rod.n_elems)

    test_rod.velocity_collection[:, :] = velocity
    test_rod.omega_collection[:, :] = omega
    test_rod.mass[:] = mass
    test_rod.inv_mass_second_moment_of_inertia = inv_moi_full.copy()
    test_rod.dilatation = dilatation.copy()

    test_translational_constant = 2.0
    test_rotational_constant = 3.0
    test_dt = np.float64(1.5)

    damper = AnalyticalLinearDamper(
        _system=test_rod,
        translational_damping_constant=test_translational_constant,
        rotational_damping_constant=test_rotational_constant,
        time_step=test_dt,
    )

    test_translational_coeff = np.exp(-test_dt * test_translational_constant / mass)
    test_rotational_coeff = np.exp(
        -test_dt * test_rotational_constant * inv_moi * dilatation.reshape((1, -1))
    )

    damper.dampen_rates(test_rod, time=np.float64(0.0))

    expected_velocity = velocity * test_translational_coeff
    expected_omega = omega * test_rotational_coeff

    assert_allclose(
        expected_velocity,
        test_rod.velocity_collection,
        atol=Tolerance.atol(),
    )
    assert_allclose(expected_omega, test_rod.omega_collection, atol=Tolerance.atol())
