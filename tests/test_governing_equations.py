__doc__ = """Test Cosserat rod governing equations"""

# System imports
import numpy as np
from elastica._rod import CosseratRod
from numpy.testing import assert_allclose
from elastica.utils import Tolerance, MaxDimension
from elastica._linalg import _batch_matvec
import pytest


class BaseClass:
    def __init__(self, n_elem, nu=0.0):
        super(BaseClass, self).__init__()
        self.n_elem = n_elem
        self.start = np.array([0.0, 0.0, 0.0])
        self.direction = np.array([0.0, 0.0, 1.0])
        self.normal = np.array([1.0, 0.0, 0.0])
        self.base_length = 1.0
        self.base_radius = 0.25
        self.density = 1
        self.nu = nu
        self.E = 1
        self.poisson_ratio = 0.5


@pytest.fixture(scope="function")
def constructor(n_elem, nu=0.0):

    cls = BaseClass(n_elem, nu)
    rod = CosseratRod.straight_rod(
        cls.n_elem,
        cls.start,
        cls.direction,
        cls.normal,
        cls.base_length,
        cls.base_radius,
        cls.density,
        cls.nu,
        cls.E,
        cls.poisson_ratio,
    )
    return cls, rod


@pytest.fixture(scope="function")
def compute_geometry_analytically(n_elem):

    initial = BaseClass(n_elem)
    # Construct position array using start and direction vectors.
    # This position array will be our reference for test cases
    end = initial.start + initial.direction * initial.base_length
    position = np.zeros((MaxDimension.value(), n_elem + 1))
    for i in range(0, MaxDimension.value()):
        position[i, ...] = np.linspace(initial.start[i], end[i], num=n_elem + 1)

    # Compute geometry
    # length of each element is same we dont need to use position array for calculation of lengths
    rest_lengths = np.repeat(initial.base_length / n_elem, n_elem)

    tangents = np.repeat(initial.direction[:, np.newaxis], n_elem, axis=1)
    radius = np.repeat(initial.base_radius, n_elem)

    return position, rest_lengths, tangents, radius


@pytest.fixture(scope="function")
def compute_all_dilatations_analytically(n_elem, dilatation):

    initial = BaseClass(n_elem)
    position, rest_lengths, tangents, radius = compute_geometry_analytically(n_elem)

    rest_voronoi_lengths = np.repeat(
        initial.base_length / n_elem, n_elem - 1
    )  # n-1 elements in voronoi domain

    dilatation_collection = np.repeat(dilatation, n_elem, axis=0)

    # Compute dilatation
    lengths = rest_lengths * dilatation
    # Compute voronoi dilatation
    voronoi_lengths = rest_voronoi_lengths * dilatation
    voronoi_dilatation = voronoi_lengths / rest_voronoi_lengths

    return (
        dilatation_collection,
        voronoi_dilatation,
        lengths,
        rest_voronoi_lengths,
    )


@pytest.fixture(scope="function")
def compute_dilatation_rate_analytically(n_elem, dilatation):

    position, rest_lengths, tangents, radius = compute_geometry_analytically(n_elem)
    # In order to compute dilatation rate, we need to set node velocity.
    # We can compute velocity subtracting current position from the previous
    # position which is the rest_position, here take dt = 1.0 .
    position_rest = position.copy()  # Here take a copy before modifying position
    position *= dilatation  # Change the position of the nodes
    # TODO: Find a better way to set velocity, which we use for dilatation rate
    # v = (x[new]-x[old])/dt, dt = 1.0
    velocity = position - position_rest
    # velocity_difference = v[i+1]-v[i]
    velocity_difference = velocity[..., 1:] - velocity[..., :-1]
    # Hard coded, here since we know there is only velocity along the rod (d3),
    # just use those values to compute dilatation rate.
    dilatation_rate = velocity_difference[-1] / rest_lengths

    return dilatation_rate, velocity


@pytest.fixture(scope="function")
def compute_strain_analytically(n_elem, dilatation):
    position, rest_lengths, tangents, radius = compute_geometry_analytically(n_elem)
    (
        dilatation_collection,
        voronoi_dilatation,
        lengths,
        rest_voronoi_lengths,
    ) = compute_all_dilatations_analytically(n_elem, dilatation)
    strain = (
        (lengths - rest_lengths) / rest_lengths * tangents
    )  # multiply with tangents to make a vector

    return strain


@pytest.fixture(scope="function")
def compute_stress_analytically(n_elem, dilatation):
    initial = BaseClass(n_elem)
    strain = compute_strain_analytically(n_elem, dilatation)
    # Compute Internal stress. Actually, below computation has a unit of force
    # but in RSoS 2018 paper and in _rod.py, it is called stress.
    # It is basically, shear_matrix * strain
    stress = (initial.base_radius * initial.base_radius * np.pi) * initial.E * strain
    return stress


@pytest.fixture(scope="function")
def compute_forces_analytically(n_elem, dilatation):
    internal_stress = compute_stress_analytically(n_elem, dilatation)
    # Internal forces in between elements have to be zero, because
    # we compress every element by same amount. Thus we only need
    # to compute forces at the first and last nodes. We know that
    # forces at the first and last node have to be in opposite direction
    # thus we multiply forces on last node with -1.0.
    internal_forces = np.zeros((MaxDimension.value(), n_elem + 1))
    internal_forces[..., 0] = internal_stress[..., 0] / dilatation
    internal_forces[..., -1] = -1.0 * internal_stress[..., 0] / dilatation

    return internal_forces


class TestingClass:
    @pytest.mark.parametrize("n_elem", [2, 3, 5, 10, 20])
    def test_case_compute_geomerty_from_state(self, n_elem, nu=0):
        """
        This test case, tests compute_geometry_from_state
        function by comparing with analytical solution.
        :param n_elem:
        :param nu:
        :return:
        """

        initial, test_rod = constructor(n_elem, nu)
        position, rest_lengths, tangents, radius = compute_geometry_analytically(n_elem)
        # Compute geometry from state
        test_rod._compute_geometry_from_state()

        assert_allclose(test_rod.rest_lengths, rest_lengths, atol=Tolerance.atol())
        assert_allclose(
            test_rod.lengths, rest_lengths, atol=Tolerance.atol()
        )  # no dilatation
        assert_allclose(test_rod.tangents, tangents, atol=Tolerance.atol())
        assert_allclose(test_rod.radius, radius, atol=Tolerance.atol())

    @pytest.mark.parametrize("n_elem", [2, 3, 5, 10, 20])
    @pytest.mark.parametrize("dilatation", [0.1, 0.2, 0.3, 0.5, 1.0, 1.1])
    def test_case_compute_all_dilatations(self, n_elem, dilatation):
        """
        This test case, tests compute_all_dilatations
        function by comparing with analytical solution.
        :param n_elem:
        :param dilatation:
        :return:
        """

        initial, test_rod = constructor(n_elem)

        (
            dilatation_collection,
            voronoi_dilatation,
            lengths,
            rest_voronoi_lengths,
        ) = compute_all_dilatations_analytically(n_elem, dilatation)

        test_rod.position *= dilatation
        # Compute dilatation using compute_all_dilatations
        # Compute geometry again because node positions changed.
        # But compute geometry will be done inside compute_all_dilatations.
        test_rod._compute_all_dilatations()

        assert_allclose(test_rod.lengths, lengths, atol=Tolerance.atol())
        assert_allclose(
            test_rod.rest_voronoi_lengths, rest_voronoi_lengths, atol=Tolerance.atol()
        )
        assert_allclose(
            test_rod.dilatation, dilatation_collection, atol=Tolerance.atol()
        )
        assert_allclose(
            test_rod.voronoi_dilatation, voronoi_dilatation, atol=Tolerance.atol()
        )

    @pytest.mark.parametrize("n_elem", [2, 3, 5, 10, 20])
    @pytest.mark.parametrize("dilatation", [0.1, 0.2, 0.3, 0.5, 1.0, 1.1])
    def test_case_compute_dilatation_rate(
        self, n_elem, dilatation,
    ):
        """
        This test case tests compute_dilatation_rate
        function by comparing with analytical calculation.
        This function depends on the compute_all_dilatations.
        :param n_elem:
        :param dilatation:
        :return:
        """
        initial, test_rod = constructor(n_elem)

        dilatation_rate, velocity = compute_dilatation_rate_analytically(
            n_elem, dilatation
        )
        # Set velocity vector in test_rod to the computed velocity vector above,
        # since we need to initialize velocity for dilatation_rate
        test_rod.velocity = velocity
        test_rod._compute_all_dilatations()
        test_rod._compute_dilatation_rate()

        assert_allclose(
            test_rod.dilatation_rate, dilatation_rate, atol=Tolerance.atol()
        )

    @pytest.mark.parametrize("n_elem", [2, 3, 5, 10, 20])
    @pytest.mark.parametrize("dilatation", [0.1, 0.2, 0.3, 0.5, 1.0, 1.1])
    def test_case_compute_shear_stretch_strains(self, n_elem, dilatation):
        """
            This test case initializes a straight rod. We modify node positions
            and compress the rod numerically. By doing that we impose shear stress
            in the rod and check, computation  strains.
            This test function tests
                _compute_shear_stretch_strains
        """
        initial, test_rod = constructor(n_elem)
        test_rod.position *= dilatation

        strain = compute_strain_analytically(n_elem, dilatation)
        test_rod._compute_shear_stretch_strains()

        assert_allclose(test_rod.sigma, strain, atol=Tolerance.atol())

    @pytest.mark.parametrize("n_elem", [2, 3, 5, 10, 20])
    @pytest.mark.parametrize("dilatation", [0.1, 0.2, 0.3, 0.5, 1.0, 1.1])
    def test_case_compute_internal_shear_stretch_stresses_from_model(
        self, n_elem, dilatation
    ):
        """
            This test case initializes a straight rod. We modify node positions
            and compress the rod numerically. By doing that we impose shear stress
            in the rod and check, computation stresses.
            This test function tests
                _compute_internal_shear_stretch_stresses_from_model
        """

        initial, test_rod = constructor(n_elem)
        test_rod.position *= dilatation
        internal_stress = compute_stress_analytically(n_elem, dilatation)

        test_rod._compute_internal_shear_stretch_stresses_from_model()

        assert_allclose(
            test_rod.internal_stress, internal_stress, atol=Tolerance.atol()
        )

    @pytest.mark.parametrize("n_elem", [2, 3, 5, 10, 20])
    @pytest.mark.parametrize("dilatation", [0.1, 0.2, 0.3, 0.5, 1.0, 1.1])
    def test_case_compute_internal_forces(self, n_elem, dilatation):
        """
            This test case initializes a straight rod. We modify node positions
            and compress the rod numerically. By doing that we impose shear stress
            in the rod and check, computation stresses.
            This test function tests
                _compute_internal_shear_stretch_stresses_from_model
        """
        initial, test_rod = constructor(n_elem)
        test_rod.position *= dilatation
        internal_forces = compute_forces_analytically(n_elem, dilatation)
        test_internal_forces = test_rod._compute_internal_forces()
        assert_allclose(test_internal_forces, internal_forces, atol=Tolerance.atol())

    @pytest.mark.parametrize("n_elem", [2, 3, 5, 10, 20])
    @pytest.mark.parametrize("nu", [0.1, 0.2, 0.5, 2])
    def test_compute_damping_forces_torques(self, n_elem, nu):
        """
            In this test case, we initialize a straight rod and modify
            velocities of nodes and angular velocities of elements.
            By doing that we can test damping forces on nodes and
            damping torques on elements.
            This test function tests
                _compute_damping_forces
                _compute_damping_torques
         """
        # This is an artificial test, this part exists just to
        # keep our coverage percentage.

        initial, test_rod = constructor(n_elem, nu)
        # Construct velocity and omega
        test_rod.velocity[:] = 1.0
        test_rod.omega[:] = 1.0
        # Compute damping forces and torques
        damping_forces = (
            np.repeat(np.array([1.0, 1.0, 1.0])[:, np.newaxis], n_elem + 1, axis=1) * nu
        )
        damping_forces[..., 0] *= 0.5
        damping_forces[..., -1] *= 0.5
        damping_torques = (
            np.repeat(np.array([1.0, 1.0, 1.0])[:, np.newaxis], n_elem, axis=1) * nu
        )
        # Compute damping forces and torques using in class functions
        test_damping_forces = test_rod._compute_damping_forces()
        test_damping_torques = test_rod._compute_damping_torques()
        # Compare damping forces and torques computed using in class functions and above
        assert_allclose(test_damping_forces, damping_forces, atol=Tolerance.atol())
        assert_allclose(test_damping_torques, damping_torques, atol=Tolerance.atol())

    # alpha is base angle of isosceles triangle
    @pytest.mark.parametrize("alpha", np.radians([22.5, 30, 45, 60, 70]))
    def test_case_bend_straight_rod(self, alpha):
        """
            In this test case we initialize a straight rod with 2 elements
            and numerically bend the rod. We modify node positions and directors
            to make a isosceles triangle. Then first we compute curvature
            between two elements and compute the angle between them.
            Finally, we compute bend twist couples and compare with
            correct solution.
            This test function tests
                _compute_bending_twist_strains
                _compute_internal_torques
                only bend_twist_couple terms.
        """

        n_elem = 2
        initial, test_rod = constructor(n_elem, nu=0.0)
        base_length = initial.base_length
        # Change the coordinates of nodes, artificially bend the rod.
        #              /\
        # ------ ==>  /  \
        #            /    \
        # Here I chose a isosceles triangle.

        length = base_length / n_elem
        position = np.zeros((MaxDimension.value(), n_elem + 1))
        position[..., 0] = np.array([0.0, 0.0, 0.0])
        position[..., 1] = length * np.array([0.0, np.sin(alpha), np.cos(alpha)])
        position[..., 2] = length * np.array([0.0, 0.0, 2 * np.cos(alpha)])
        test_rod.position = position

        # Set the directors manually. This is easy since we have two elements.
        directors = np.zeros((MaxDimension.value(), MaxDimension.value(), n_elem))
        directors[..., 0] = np.array(
            (
                [1.0, 0.0, 0.0],
                [0.0, np.cos(alpha), -np.sin(alpha)],
                [0.0, np.sin(alpha), np.cos(alpha)],
            )
        )
        directors[..., -1] = np.array(
            (
                [1.0, 0.0, 0.0],
                [0.0, np.cos(alpha), np.sin(alpha)],
                [0, -np.sin(alpha), np.cos(alpha)],
            )
        )
        test_rod.directors = directors

        # Compute voronoi rest length. Since elements lengths are equal
        # in this test case, rest voronoi length can be easily computed
        # dividing base length to number of elements.

        rest_voronoi_length = base_length / n_elem

        # Now compute geometry and dilatation, which we need for curvature calculations.
        test_rod._compute_all_dilatations()
        test_rod._compute_dilatation_rate()

        test_rod._compute_bending_twist_strains()

        # Generalized rotation per unit length is given by rest_D_i * Kappa_i.
        # Thus in order to get the angle between two elements, we need to multiply
        # kappa with rest_D_i .  But this will give the exterior vertex angle of the
        # triangle. Think as, we rotate element 1 clockwise direction and align with
        # the element 2.
        #
        #               \
        #     /\         \ 1
        #  1 /  \ 2  ==>  \
        #   /    \         \
        #                   \ 2
        #                    \
        #
        # So for this transformation we use exterior vertex angle of isosceles triangle.
        # Exterior vertex angle can be computed easily, it is the sum of base angles
        # , since this is isosceles triangle it is 2*base_angle

        correct_angle = np.degrees(np.array([2 * alpha, 0.0, 0.0]).reshape(3, 1))
        test_angle = np.degrees(test_rod.kappa * test_rod.rest_voronoi_lengths)
        assert_allclose(test_angle, correct_angle, atol=Tolerance.atol())

        # Now lets test bending stress terms in internal torques equation.
        # Here we will test bend twist couple 2D and bend twist couple 3D terms of the
        # internal torques equation. Set the bending matrix to identity matrix for simplification.
        test_rod.bend_matrix[:] = np.repeat(
            np.identity(3)[:, :, np.newaxis], n_elem - 1, axis=2
        )

        # We need to compute shear stress, for internal torque equation.
        # Shear stress is not used in this test case. In order to make sure shear
        # stress do not contribute to the total torque we use assert check.
        test_rod._compute_internal_shear_stretch_stresses_from_model()
        assert_allclose(
            test_rod.internal_stress,
            np.zeros(3 * n_elem).reshape(3, n_elem),
            atol=Tolerance.atol(),
        )

        # Make sure voronoi dilatation is 1
        assert_allclose(
            test_rod.voronoi_dilatation, np.array([1.0]), atol=Tolerance.atol()
        )

        # Compute correct torques, first compute correct kappa.
        correct_kappa = np.radians(correct_angle / rest_voronoi_length)
        # We only need to compute bend twist couple 2D term for comparison,
        # because bend twist couple 3D term is already zero, due to cross product.
        # TODO: Extended this test for multiple elements more than 2.
        correct_torques = np.zeros((MaxDimension.value(), n_elem))
        correct_torques[..., 0] = correct_kappa[..., 0]
        correct_torques[..., -1] = -1.0 * correct_kappa[..., -1]

        test_torques = test_rod._compute_internal_torques()

        assert_allclose(test_torques, correct_torques, atol=Tolerance.atol())

    def test_case_shear_torque(self):
        """
            In this test case we initialize a straight rod with two elements
            and set bending matrix to zero. This gives us opportunity decouple
            shear torque from twist and bending torques in internal torques
            equation. Then we modify node positions of second element and
            introduce artificial bending. Finally, we compute shear torque
            using internal torque function and compare with analytical value.
            This test case is for testing shear torque term,
            in internal torques equation.
            Tested function
                _compute_internal_torques

        """
        n_elem = 2
        initial, test_rod = constructor(n_elem, nu=0.0)
        position = np.zeros((MaxDimension.value(), n_elem + 1))
        position[..., 0] = np.array([0.0, 0.0, 0.0])
        position[..., 1] = np.array([0.0, 0.0, 0.5])
        position[..., 2] = np.array([0.0, -0.3, 0.9])

        test_rod.position = position

        # Simplify the computations, and chose shear matrix as identity matrix.
        test_rod.shear_matrix[:] = np.repeat(
            np.identity(3)[:, :, np.newaxis], n_elem - 1, axis=2
        )

        # Internal shear stress function is tested previously
        test_rod._compute_internal_shear_stretch_stresses_from_model()

        correct_shear_torques = np.zeros((MaxDimension.value(), n_elem))
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

    def test_case_lagrange_transport_unsteady_dilatation(self):
        """
            In this test case, we initialize a straight rod. Then we modify
            angular velocity of elements and set mass moment of inertia
            to identity matrix. By doing this we need to get zero torque
            due lagrangian transport term, because of Jwxw, J=I, wxw=0.
            Next we test unsteady dilatation contribution to internal
            torques, by setting dilatation rate to 1 and recover initialized
            angular velocity back, de/dt * Jw = w , de/dt=1 J=I.

            This test function tests
                _compute_internal_torques
            only lagrange transport and
            unsteady dilatation terms, tested numerically.
            Note that, viscous dissipation set to 0,
            since we don't want any contribution from
            damping torque.
       """

        n_elem = 2
        initial, test_rod = constructor(n_elem, nu=0.0)
        # TODO: find one more test in which you dont set J=I, may be some analytical test
        # Set the mass moment of inertia matrix to identity matrix for simplification.
        # When lagrangian transport tested, total torque computed by the function has
        # to be zero, because (J.w/e)xw if J=I then wxw/e = 0.

        test_rod.mass_second_moment_of_inertia[:] = np.repeat(
            np.identity(3)[:, :, np.newaxis], n_elem, axis=2
        )

        test_rod._compute_shear_stretch_strains()
        test_rod._compute_bending_twist_strains()
        test_rod._compute_internal_forces()

        # Lets set angular velocity omega to arbitray numbers
        # Make sure shape of the random vector correct
        omega = np.zeros(3 * n_elem).reshape(3, n_elem)
        for i in range(0, n_elem):
            omega[..., i] = np.random.rand(3)

        test_rod.omega = omega
        test_torques = test_rod._compute_internal_torques()

        # computed internal torques has to be zero. Internal torques created by Lagrangian
        # transport term is zero because mass moment of inertia is identity matrix and wxw=0.
        # Torques due to unsteady dilatation has to be zero because dilatation rate is zero.

        assert_allclose(
            test_torques, np.zeros(3 * n_elem).reshape(3, n_elem), atol=Tolerance.atol()
        )

        # Now lets test torques due to unsteady dilatation. For that, lets set dilatation
        # rate to 1, it is zero before. It has to be zero before, because rod is not elongating or shortening
        assert_allclose(
            test_rod.dilatation_rate, np.zeros(n_elem), atol=Tolerance.atol()
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

    @pytest.mark.parametrize("n_elem", [2, 3, 5, 10, 20])
    def test_get_functions(self, n_elem):
        """
            In this test case, we initialize a straight rod. First
            we set velocity and angular velocity to random values,
            and we recover back initialized values using `get_velocity`
            and `get_angular_velocity` functions. Then, we set random
            external forces and torques. We compute translational
            accelerations and angular accelerations based on external
            forces and torques. Finally we use `get_acceleration` and
            `get_angular_acceleration` functions and compare returned
            translational acceleration and angular acceleration with
            analytically computed ones.
            This test case tests,
                get_velocity
                get_angular_velocity
                get_acceleration
                get_angular_acceleration
            Note that, viscous dissipation set to 0,
            since we don't want any contribution from
            damping torque.
        """

        initial, test_rod = constructor(n_elem, nu=0.0)
        mass = test_rod.mass

        velocity = np.zeros(3 * (n_elem + 1)).reshape(3, n_elem + 1)
        external_forces = np.zeros(3 * (n_elem + 1)).reshape(3, n_elem + 1)
        omega = np.zeros(3 * n_elem).reshape(3, n_elem)
        external_torques = np.zeros(3 * n_elem).reshape(3, n_elem)

        for i in range(0, n_elem):
            omega[..., i] = np.random.rand(3)
            external_torques[..., i] = np.random.rand(3)

        for i in range(0, n_elem + 1):
            velocity[..., i] = np.random.rand(3)
            external_forces[..., i] = np.random.rand(3)

        test_rod.velocity = velocity
        test_rod.omega = omega

        assert_allclose(test_rod.get_velocity(), velocity, atol=Tolerance.atol())
        assert_allclose(test_rod.get_angular_velocity(), omega, atol=Tolerance.atol())

        test_rod.external_forces = external_forces
        test_rod.external_torques = external_torques

        correct_acceleration = external_forces / mass
        assert_allclose(
            test_rod.get_acceleration(), correct_acceleration, atol=Tolerance.atol()
        )

        # No dilatation in the rods
        dilatations = np.ones(n_elem)

        # Set angular velocity zero, so that we dont have any
        # contribution from lagrangian transport and unsteady dilatation.

        test_rod.omega[:] = 0.0
        # Set mass moment of inertia matrix to identity matrix for convenience.
        # Inverse of identity = identity
        inv_mass_moment_of_inertia = np.repeat(
            np.identity(3)[:, :, np.newaxis], n_elem, axis=2
        )
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
    from pytest import main

    main([__file__])
