__doc__ = """ Test Cosserat rod governing equations   """

# System imports
import numpy as np
from elastica._rod import CosseratRod
from numpy.testing import assert_allclose
from elastica.utils import Tolerance, MaxDimension
from elastica._calculus import difference_kernel
from elastica._linalg import _batch_cross, _batch_matvec
from pytest import main


def test_case_compress_straight_rod():
    """
    This test function tests
        _compute_geometry_from_state
        _compute_all_dilatations
        _compute_dilatation_rate
        _compute_shear_stretch_strains
        _compute_internal_forces

    """
    n = 10
    start = np.array([0.0, 0.0, 0.0])
    direction = np.array([0.0, 0.0, 1.0])
    normal = np.array([1.0, 0.0, 0.0])
    base_length = 1.0
    base_radius = 0.25
    density = 1
    nu = 1
    E = 1
    poisson_ratio = 0.5

    test_rod = CosseratRod.straight_rod(
        n,
        start,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        nu,
        E,
        poisson_ratio,
    )

    # Consturct position array using start and direction vectors.
    # This position array will be our reference for test cases
    end = start + direction * base_length
    position = np.zeros((MaxDimension.value(), n + 1))
    for i in range(0, MaxDimension.value()):
        position[i, ...] = np.linspace(start[i], end[i], num=n + 1)

    # Compute geometry
    # length of each element is same we dont need to use position array for calculation of lengths
    rest_lengths = np.repeat(base_length / n, n)
    rest_voronoi_lengths = np.repeat(
        base_length / n, n - 1
    )  # n-1 elements in voronoi domain
    tangents = np.repeat(direction[:, np.newaxis], n, axis=1)
    radius = np.repeat(base_radius, n)
    # Compute geometry from state
    test_rod._compute_geometry_from_state()

    assert_allclose(test_rod.lengths, rest_lengths, atol=Tolerance.atol())
    assert_allclose(test_rod.tangents, tangents, atol=Tolerance.atol())
    assert_allclose(test_rod.radius, radius, atol=Tolerance.atol())

    # Move the nodes of the rod and compress the rod to see dilatation
    coefficient = np.array([1.0, 1.0, 0.5]).reshape(3, 1)
    test_rod.position = test_rod.position * coefficient

    # Compute dilatation
    lengths = rest_lengths * coefficient[2]
    dilatation = lengths / rest_lengths
    # Compute voronoi dilatation
    voronoi_lengths = rest_voronoi_lengths * coefficient[2]
    voronoi_dilatation = voronoi_lengths / rest_voronoi_lengths

    # Compute dilatation using compute_all_dilatations
    # Compute geometry again because node positions changed.
    test_rod._compute_geometry_from_state()
    test_rod._compute_all_dilatations()

    assert_allclose(test_rod.dilatation, dilatation, atol=Tolerance.atol())
    assert_allclose(
        test_rod.voronoi_dilatation, voronoi_dilatation, atol=Tolerance.atol()
    )

    # In order to compute dilatation rate, we need to set node velocity.
    # We can compute velocity subtracting current position from the previous
    # position which is the rest_position, here take dt = 1.0 . Here we multiply
    # with tangents because velocity is a vector.
    position_rest = position.copy()  # Here take a copy before modifying position
    position = position * coefficient  # Change the position of the nodes
    velocity = position - position_rest
    velocity_difference = velocity[..., 1:] - velocity[..., :-1]
    # Hard coded, here since we know there is only velocity along the rod (d3),
    # just use those values to compute dilatation rate.
    dilatation_rate = velocity_difference[-1] / rest_lengths
    # Set velocity vector in test_rod to the computed velocity vector above,
    # since we need to initialize velocity for dilatation_rate
    test_rod.velocity = velocity
    test_rod._compute_dilatation_rate()

    assert_allclose(test_rod.dilatation_rate, dilatation_rate, atol=Tolerance.atol())

    # Compute strains. Strain is only in d3 direction.
    strain = (
        (lengths - rest_lengths) / rest_lengths * tangents
    )  # multiply with tangents to make a vector
    test_rod._compute_shear_stretch_strains()

    assert_allclose(test_rod.sigma, strain, atol=Tolerance.atol())

    # Compute Internal stress. Actually, below computation has a unit of force
    # but in RSoS 2018 paper and in _rod.py, it is called stress.
    # It is basically, shear_matrix * strain
    internal_stress = (base_radius * base_radius * np.pi) * E * strain
    test_rod._compute_internal_shear_stretch_stresses_from_model()

    assert_allclose(test_rod.internal_stress, internal_stress, atol=Tolerance.atol())

    # Compute internal forces. Here we need to set
    # velocity back to zero, because _compute_internal_forces
    # computes damping forces and forces due to stress together.
    # In order to decouple these two forces. We set velocity zero.
    # We will check damping forces in another test in this file.

    cosserat_internal_stress = (
        _batch_matvec(test_rod.directors, internal_stress) / dilatation
    )
    internal_forces = difference_kernel(cosserat_internal_stress)

    velocity = velocity[:] = np.array([0.0, 0.0, 0.0]).reshape(3, 1)
    test_rod.velocity = velocity

    test_internal_forces = test_rod._compute_internal_forces()

    assert_allclose(test_internal_forces, internal_forces, atol=Tolerance.atol())


def test_compute_damping_forces_torques():
    """
    This test function tests
        _compute_damping_forces
        _compute_damping_torques
     """
    n = 10
    start = np.array([0.0, 0.0, 0.0])
    direction = np.array([0.0, 0.0, 1.0])
    normal = np.array([0.0, 1.0, 0.0])
    base_length = 1.0
    base_radius = 0.25
    density = 1
    nu = 0.5
    E = 1
    poisson_ratio = 0.5
    test_rod = CosseratRod.straight_rod(
        n,
        start,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        nu,
        E,
        poisson_ratio,
    )

    # Construct velocity and omega
    test_rod.velocity[:] = 1.0
    test_rod.omega[:] = 1.0
    # Compute damping forces and torques
    damping_forces = (
        np.repeat(np.array([1.0, 1.0, 1.0])[:, np.newaxis], n + 1, axis=1) * nu
    )
    damping_forces[..., 0] *= 0.5
    damping_forces[..., -1] *= 0.5
    damping_torques = (
        np.repeat(np.array([1.0, 1.0, 1.0])[:, np.newaxis], n, axis=1) * nu
    )
    # Compute damping forces and torques using in class functions
    test_damping_forces = test_rod._compute_damping_forces()
    test_damping_torques = test_rod._compute_damping_torques()
    # Compare damping forces and torques computed using in class functions and above
    assert_allclose(test_damping_forces, damping_forces, atol=Tolerance.atol())
    assert_allclose(test_damping_torques, damping_torques, atol=Tolerance.atol())


def test_case_bend_straight_rod():
    """
    This test function tests
        _compute_bending_twist_strains
        _compute_internal_torques
            only bend_twist_couple terms.
    """
    n = 2
    start = np.array([0.0, 0.0, 0.0])
    direction = np.array([0.0, 0.0, 1.0])
    normal = np.array([1.0, 0.0, 0.0])
    base_length = 1.0
    base_radius = 0.25
    density = 1
    nu = 0
    E = 1
    poisson_ratio = 0.5

    test_rod = CosseratRod.straight_rod(
        n,
        start,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        nu,
        E,
        poisson_ratio,
    )

    # Compute voronoi rest length. Since elements lengths are equal
    # in this test case, rest voronoi length can be easily computed
    # dividing base length to number of elements.

    rest_voronoi_length = base_length / n

    # Consturct position array using start and direction arrays.
    # This position array will be our reference for test cases
    end = start + direction * base_length
    position = np.zeros((MaxDimension.value(), n + 1))
    for i in range(0, MaxDimension.value()):
        position[i, ...] = np.linspace(start[i], end[i], num=n + 1)

    # Now compute geometry and dilatation, which we need for curvature calculations.
    test_rod._compute_geometry_from_state()
    test_rod._compute_all_dilatations()
    test_rod._compute_dilatation_rate()
    # Change the coordinates of nodes, artificially bend the rod.
    #           /\
    # --- ==>  /  \
    #         /    \
    # Here I chose a isosceles right triangle.

    position[..., 0] = np.array([0.0, 0.0, 0.0])
    position[..., 1] = np.array([0.0, np.sqrt(2) / 4.0, np.sqrt(2) / 4.0])
    position[..., 2] = np.array([0.0, 0.0, np.sqrt(2) / 2.0])

    test_rod.position = position

    # Update directors
    directors = np.zeros((MaxDimension.value(), MaxDimension.value(), n))
    position_diff = position[..., 1:] - position[..., :-1]
    lengths = np.linalg.norm(position_diff, axis=0)
    tangents = position_diff / lengths
    normal_collection = np.repeat(normal[:, np.newaxis], n, axis=1)
    directors[0, ...] = normal_collection
    directors[1, ...] = _batch_cross(tangents, normal_collection)
    directors[2, ...] = tangents

    test_rod._compute_geometry_from_state()

    test_rod.directors = directors

    test_rod._compute_bending_twist_strains()

    # Generalized rotation per unit length is given by rest_D_i * Kappa_i.
    # Thus in order to get the angle between two elements, we need to multiply
    # kappa with rest_D_i .  We know that correct angle is 90 degrees since
    # it is isosceles triangle.

    correct_angle = np.array([90.0, 0.0, 0.0]).reshape(3, 1)
    test_angle = np.degrees(test_rod.kappa * test_rod.rest_voronoi_lengths)
    assert_allclose(test_angle, correct_angle, atol=Tolerance.atol())

    # Now lets test bending stress terms in internal torques equation.
    # Here we will test bend twist couple 2D and bend twist couple 3D terms of the
    # internal torques equation. Set the bending matrix to identity matrix for simplification.
    test_rod.bend_matrix[:] = np.repeat(np.identity(3)[:, :, np.newaxis], n - 1, axis=2)

    # We need to compute shear stress, for internal torque equation.
    # Shear stress is not used in this test case. In order to make sure shear
    # stress do not contribute to the total torque we use assert check.
    test_rod._compute_internal_shear_stretch_stresses_from_model()
    assert_allclose(
        test_rod.internal_stress, np.zeros(3 * n).reshape(3, n), atol=Tolerance.atol()
    )

    # Make sure voronoi dilatation is 1
    assert_allclose(test_rod.voronoi_dilatation, np.array([1.0]), atol=Tolerance.atol())

    # Compute correct torques, first compute correct kappa.
    correct_kappa = np.radians(correct_angle / rest_voronoi_length)
    # We only need to compute bend twist couple 2D term for comparison,
    # because bend twist couple 3D term is already zero, due to cross product.
    correct_torques = difference_kernel(correct_kappa)

    test_torques = test_rod._compute_internal_torques()

    assert_allclose(test_torques, correct_torques, atol=Tolerance.atol())


def test_case_shear_torque():
    """
        This test case is for testing shear torque term,
        in internal torques equation.
        Tested function _compute_internal_torques

    """
    n = 2
    start = np.array([0.0, 0.0, 0.0])
    direction = np.array([0.0, 0.0, 1.0])
    normal = np.array([1.0, 0.0, 0.0])
    base_length = 1.0
    base_radius = 0.25
    density = 1
    nu = 0.0
    E = 1
    poisson_ratio = 0.5

    test_rod = CosseratRod.straight_rod(
        n,
        start,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        nu,
        E,
        poisson_ratio,
    )

    # We can use compute geometry from state, because it is tested in previous tests.
    test_rod._compute_geometry_from_state()

    position = np.zeros((MaxDimension.value(), n + 1))
    position[..., 0] = np.array([0.0, 0.0, 0.0])
    position[..., 1] = np.array([0.0, 0.0, 0.5])
    position[..., 2] = np.array([0.0, -0.3, 0.9])

    test_rod.position = position

    # Compute new geometry, based on new node positions.
    test_rod._compute_geometry_from_state()

    # Simplify the computations, and chose shear matrix as identity matrix.
    test_rod.shear_matrix[:] = np.repeat(
        np.identity(3)[:, :, np.newaxis], n - 1, axis=2
    )

    # Internal shear stress function is tested previously
    test_rod._compute_internal_shear_stretch_stresses_from_model()

    correct_shear_torques = np.zeros((MaxDimension.value(), n))
    # Correct shear torques can be computed easily.
    # Procedure:
    #       1) Q = [1., 0., 0.; 0., 1., 0.; 0., 0., 1.]
    #       2) t = [0., -0.6, 0.8]
    #       3) sigma = (eQt-d3) = [0.0, -0.6, -0.2]
    #       4) Qt = [0., -0.6, 0.8]
    #       5) torque = Qt x sigma
    # Note that this is not generic, but it does not to be, it is testing the functions.
    correct_shear_torques[..., -1] = np.array([0.3, 0.0, 0.0])

    # Set bending matrix to zero matrix, because we dont want
    # any contribution from bending on total internal torques
    test_rod.bend_matrix[:] = 0.0

    test_torques = test_rod._compute_internal_torques()

    assert_allclose(test_torques, correct_shear_torques, atol=Tolerance.atol())


def test_case_lagrange_transport_unsteady_dilatation():
    """
   This test function tests
       _compute_internal_torques
       only lagrange transport and
       unsteady dilatation terms, tested numerically.
       Note that, viscous dissipation set to 0,
       since we don't want any contribution from
       damping torque.
   """
    n = 2
    start = np.array([0.0, 0.0, 0.0])
    direction = np.array([0.0, 0.0, 1.0])
    normal = np.array([1.0, 0.0, 0.0])
    base_length = 1.0
    base_radius = 0.25
    density = 1
    nu = 0
    E = 1
    poisson_ratio = 0.5

    test_rod = CosseratRod.straight_rod(
        n,
        start,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        nu,
        E,
        poisson_ratio,
    )

    # Compute geometry from state, dilatations and dilatation rate.
    # These are required for computing internal torques.
    test_rod._compute_geometry_from_state()
    test_rod._compute_all_dilatations()
    test_rod._compute_dilatation_rate()

    # Set the mass moment of inertia matrix to identity matrix for simplification.
    # When lagrangian transport tested, total torque computed by the function has
    # to be zero, because (J.w/e)xw if J=I then wxw/e = 0.

    test_rod.mass_second_moment_of_inertia[:] = np.repeat(
        np.identity(3)[:, :, np.newaxis], n, axis=2
    )

    test_rod._compute_shear_stretch_strains()
    test_rod._compute_bending_twist_strains()
    test_rod._compute_internal_forces()

    # Lets set angular velocity omega to arbitray numbers
    # Make sure shape of the random vector correct
    omega = np.zeros(3 * n).reshape(3, n)
    for i in range(0, n):
        omega[..., i] = np.random.rand(3)

    test_rod.omega = omega
    test_torques = test_rod._compute_internal_torques()

    # computed internal torques has to be zero. Internal torques created by Lagrangian
    # transport term is zero because mass moment of inertia is identity matrix and wxw=0.
    # Torques due to unsteady dilatation has to be zero because dilatation rate is zero.

    assert_allclose(test_torques, np.zeros(3 * n).reshape(3, n), atol=Tolerance.atol())

    # Now lets test torques due to unsteady dilatation. For that, lets set dilatation
    # rate to 1, it is zero before. It has to be zero before, because rod is not elongating or shortening
    assert_allclose(
        test_rod.dilatation_rate, np.zeros(n), atol=Tolerance.atol()
    )  # check if dilatation rate is 0

    # Now set velocity such that to set dilatation rate to 1.
    test_rod.velocity[..., 0] = np.ones(3) * -0.5
    test_rod.velocity[..., -1] = np.ones(3) * 0.5

    test_rod._compute_dilatation_rate()

    assert_allclose(test_rod.dilatation_rate, np.array([1.0, 1.0]))

    test_torques = test_rod._compute_internal_torques()

    # Total internal torque has to be equal to angular velocity omega.
    # All the other terms contributing total internal torque is zero,
    # other than unsteady dilatation.
    correct_torques = omega
    assert_allclose(test_torques, correct_torques, atol=Tolerance.atol())


def test_get_functions():
    """
        This test case tests,
            get_velocity
            get_angular_velocity
            get_acceleration
            get_angular_acceleration
    """

    n = 2
    start = np.array([0.0, 0.0, 0.0])
    direction = np.array([0.0, 0.0, 1.0])
    normal = np.array([1.0, 0.0, 0.0])
    base_length = 1.0
    base_radius = 0.25
    density = 1
    nu = 0
    E = 1
    poisson_ratio = 0.5

    test_rod = CosseratRod.straight_rod(
        n,
        start,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        nu,
        E,
        poisson_ratio,
    )

    mass = test_rod.mass

    velocity = np.zeros(3 * (n + 1)).reshape(3, n + 1)
    external_forces = np.zeros(3 * (n + 1)).reshape(3, n + 1)
    omega = np.zeros(3 * n).reshape(3, n)
    external_torques = np.zeros(3 * n).reshape(3, n)

    for i in range(0, n):
        omega[..., i] = np.random.rand(3)
        external_torques[..., i] = np.random.rand(3)

    for i in range(0, n + 1):
        velocity[..., i] = np.random.rand(3)
        external_forces[..., i] = np.random.rand(3)

    test_rod.velocity = velocity
    test_rod.omega = omega

    test_rod._compute_geometry_from_state()

    assert_allclose(test_rod.get_velocity(), velocity, atol=Tolerance.atol())
    assert_allclose(test_rod.get_angular_velocity(), omega, atol=Tolerance.atol())

    test_rod.external_forces = external_forces
    test_rod.external_torques = external_torques

    correct_acceleration = external_forces / mass
    assert_allclose(
        test_rod.get_acceleration(), correct_acceleration, atol=Tolerance.atol()
    )

    # No dilatation in the rods
    dilatations = np.ones(n)

    # Set angular velocity zero, so that we dont have any
    # contribution from lagrangian transport and unsteady dilatation.

    test_rod.omega[:] = 0.0
    # Set mass moment of inertia matrix to identity matrix for convenience.
    # Inverse of identity = identity
    inv_mass_moment_of_inertia = np.repeat(np.identity(3)[:, :, np.newaxis], n, axis=2)
    test_rod.inv_mass_second_moment_of_inertia = inv_mass_moment_of_inertia

    correct_angular_acceleration = (
        _batch_matvec(inv_mass_moment_of_inertia, external_torques) * dilatations
    )

    assert_allclose(
        test_rod.get_angular_acceleration(),
        correct_angular_acceleration,
        atol=Tolerance.atol(),
    )


if __name__ == "__main__":
    main([__file__])
