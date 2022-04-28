__doc__ = """Tests for rod initialisation module"""
import numpy as np
from numpy.testing import assert_allclose

from elastica.utils import MaxDimension, Tolerance

import pytest
import sys

from elastica.rod.data_structures import _RodSymplecticStepperMixin
from elastica.rod.factory_function import allocate


class MockRodForTest:
    def __init__(
        self,
        n_elements,
        position,
        velocity,
        omega,
        acceleration,
        angular_acceleration,
        directors,
        radius,
        mass_second_moment_of_inertia,
        inv_mass_second_moment_of_inertia,
        shear_matrix,
        bend_matrix,
        density,
        volume,
        mass,
        dissipation_constant_for_forces,
        dissipation_constant_for_torques,
        internal_forces,
        internal_torques,
        external_forces,
        external_torques,
        lengths,
        rest_lengths,
        tangents,
        dilatation,
        dilatation_rate,
        voronoi_dilatation,
        rest_voronoi_lengths,
        sigma,
        kappa,
        rest_sigma,
        rest_kappa,
        internal_stress,
        internal_couple,
        damping_forces,
        damping_torques,
    ):
        self.n_elems = n_elements
        self.position_collection = position
        self.velocity_collection = velocity
        self.omega_collection = omega
        self.acceleration_collection = acceleration
        self.alpha_collection = angular_acceleration
        self.director_collection = directors
        self.radius = radius
        self.mass_second_moment_of_inertia = mass_second_moment_of_inertia
        self.inv_mass_second_moment_of_inertia = inv_mass_second_moment_of_inertia
        self.shear_matrix = shear_matrix
        self.bend_matrix = bend_matrix
        self.density = density
        self.volume = volume
        self.mass = mass
        self.dissipation_constant_for_forces = dissipation_constant_for_forces
        self.dissipation_constant_for_torques = dissipation_constant_for_torques
        self.internal_forces = internal_forces
        self.internal_torques = internal_torques
        self.external_forces = external_forces
        self.external_torques = external_torques
        self.lengths = lengths
        self.rest_lengths = rest_lengths
        self.tangents = tangents
        self.dilatation = dilatation
        self.dilatation_rate = dilatation_rate
        self.voronoi_dilatation = voronoi_dilatation
        self.rest_voronoi_lengths = rest_voronoi_lengths
        self.sigma = sigma
        self.kappa = kappa
        self.rest_sigma = rest_sigma
        self.rest_kappa = rest_kappa
        self.internal_stress = internal_stress
        self.internal_couple = internal_couple
        self.damping_forces = damping_forces
        self.damping_torques = damping_torques

    @classmethod
    def straight_rod(
        cls,
        n_elements,
        start,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        nu,
        youngs_modulus,
        # poisson_ratio,
        *args,
        **kwargs
    ):

        (
            n_elements,
            position,
            velocity,
            omega,
            acceleration,
            angular_acceleration,
            directors,
            radius,
            mass_second_moment_of_inertia,
            inv_mass_second_moment_of_inertia,
            shear_matrix,
            bend_matrix,
            density,
            volume,
            mass,
            dissipation_constant_for_forces,
            dissipation_constant_for_torques,
            internal_forces,
            internal_torques,
            external_forces,
            external_torques,
            lengths,
            rest_lengths,
            tangents,
            dilatation,
            dilatation_rate,
            voronoi_dilatation,
            rest_voronoi_lengths,
            sigma,
            kappa,
            rest_sigma,
            rest_kappa,
            internal_stress,
            internal_couple,
            damping_forces,
            damping_torques,
        ) = allocate(
            n_elements,
            start,
            direction,
            normal,
            base_length,
            base_radius,
            density,
            nu,
            youngs_modulus,
            alpha_c=0.964,
            *args,
            **kwargs
        )

        return cls(
            n_elements,
            position,
            velocity,
            omega,
            acceleration,
            angular_acceleration,
            directors,
            radius,
            mass_second_moment_of_inertia,
            inv_mass_second_moment_of_inertia,
            shear_matrix,
            bend_matrix,
            density,
            volume,
            mass,
            dissipation_constant_for_forces,
            dissipation_constant_for_torques,
            internal_forces,
            internal_torques,
            external_forces,
            external_torques,
            lengths,
            rest_lengths,
            tangents,
            dilatation,
            dilatation_rate,
            voronoi_dilatation,
            rest_voronoi_lengths,
            sigma,
            kappa,
            rest_sigma,
            rest_kappa,
            internal_stress,
            internal_couple,
            damping_forces,
            damping_torques,
        )


@pytest.mark.parametrize("n_elems", [5, 10, 50])
def test_input_and_output_position_array(n_elems):
    """
    This test, tests the case if the input position array
    valid, allocate sets input position as the rod position array.
    Parameters
    ----------
    n_elems

    Returns
    -------

    """
    start = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    base_length = 1.0
    base_radius = 0.25
    density = 1000
    nu = 0.1
    youngs_modulus = 1e6
    poisson_ratio = 0.3

    # Check if the input position vector and output position vector are valid and same
    correct_position = np.zeros((3, n_elems + 1))
    correct_position[0] = np.random.randn(n_elems + 1)
    correct_position[1] = np.random.randn(n_elems + 1)
    correct_position[..., 0] = start
    shear_modulus = youngs_modulus / (poisson_ratio + 1.0)
    mockrod = MockRodForTest.straight_rod(
        n_elems,
        start,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        nu,
        youngs_modulus,
        shear_modulus=shear_modulus,
        position=correct_position,
    )
    test_position = mockrod.position_collection
    assert_allclose(correct_position, test_position, atol=Tolerance.atol())


@pytest.mark.xfail(raises=AssertionError)
@pytest.mark.parametrize("n_elems", [5, 10, 50])
def test_input_and_position_array_for_different_start(n_elems):
    """
    This function tests fail check, for which input position array
    first element is not user defined start position.
    Parameters
    ----------
    n_elems

    Returns
    -------

    """
    start = np.random.randn(3)
    direction = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    base_length = 1.0
    base_radius = 0.25
    density = 1000
    nu = 0.1
    youngs_modulus = 1e6
    poisson_ratio = 0.3
    shear_modulus = youngs_modulus / (poisson_ratio + 1.0)

    # Check if the input position vector start position is different than the user defined start position
    correct_position = np.random.randn(3, n_elems + 1)
    mockrod = MockRodForTest.straight_rod(
        n_elems,
        start,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        nu,
        youngs_modulus,
        shear_modulus=shear_modulus,
        position=correct_position,
    )
    test_position = mockrod.position_collection
    assert_allclose(correct_position, test_position, atol=Tolerance.atol())


def test_compute_position_array_using_user_inputs():
    """
    This test checks if the allocate function can compute correctly
    position vector using start, direction and base length inputs.
    Returns
    -------

    """
    n_elems = 4
    start = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    base_length = 1.0
    base_radius = 0.25
    density = 1000
    nu = 0.1
    youngs_modulus = 1e6
    poisson_ratio = 0.3
    shear_modulus = youngs_modulus / (poisson_ratio + 1.0)
    # Check if without input position vector, output position vector is valid
    mockrod = MockRodForTest.straight_rod(
        n_elems,
        start,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        nu,
        youngs_modulus,
        shear_modulus=shear_modulus,
    )
    correct_position = np.zeros((3, n_elems + 1))
    correct_position[0, :] = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    test_position = mockrod.position_collection
    assert_allclose(correct_position, test_position, atol=Tolerance.atol())


@pytest.mark.parametrize("n_elems", [5, 10, 50])
def test_compute_directors_matrix_using_user_inputs(n_elems):
    """
    This test checks the director array created by allocate function. For this
    test case we use user defined direction, normal to compute directors.
    Returns
    -------

    """
    start = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    base_length = 1.0
    base_radius = 0.25
    density = 1000
    nu = 0.1
    youngs_modulus = 1e6
    poisson_ratio = 0.3
    shear_modulus = youngs_modulus / (poisson_ratio + 1.0)
    # Check directors, if we dont input any directors, computed ones should be valid
    correct_directors = np.zeros((MaxDimension.value(), MaxDimension.value(), n_elems))
    binormal = np.cross(direction, normal)
    tangent_collection = np.repeat(direction[:, np.newaxis], n_elems, axis=1)
    normal_collection = np.repeat(normal[:, np.newaxis], n_elems, axis=1)
    binormal_collection = np.repeat(binormal[:, np.newaxis], n_elems, axis=1)

    correct_directors[0, ...] = normal_collection
    correct_directors[1, ...] = binormal_collection
    correct_directors[2, ...] = tangent_collection

    mockrod = MockRodForTest.straight_rod(
        n_elems,
        start,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        nu,
        youngs_modulus,
        shear_modulus=shear_modulus,
    )
    test_directors = mockrod.director_collection
    assert_allclose(correct_directors, test_directors, atol=Tolerance.atol())


@pytest.mark.parametrize("n_elems", [5, 10, 50])
def test_directors_using_input_position_array(n_elems):
    """
    This test is testing the case for which directors are computed
    using the input position array and user defined normal.
    Parameters
    ----------
    n_elems

    Returns
    -------

    """
    start = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    base_length = 1.0
    base_radius = 0.25
    density = 1000
    nu = 0.1
    youngs_modulus = 1e6
    poisson_ratio = 0.3
    shear_modulus = youngs_modulus / (poisson_ratio + 1.0)
    # Check directors, give position as input and let allocate function to compute directors.
    input_position = np.zeros((3, n_elems + 1))
    input_position[0, :] = np.linspace(start[0], start[0] + base_length, n_elems + 1)

    correct_directors = np.zeros((MaxDimension.value(), MaxDimension.value(), n_elems))
    binormal = np.cross(direction, normal)
    tangent_collection = np.repeat(direction[:, np.newaxis], n_elems, axis=1)
    normal_collection = np.repeat(normal[:, np.newaxis], n_elems, axis=1)
    binormal_collection = np.repeat(binormal[:, np.newaxis], n_elems, axis=1)

    correct_directors[0, ...] = normal_collection
    correct_directors[1, ...] = binormal_collection
    correct_directors[2, ...] = tangent_collection

    mockrod = MockRodForTest.straight_rod(
        n_elems,
        start,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        nu,
        youngs_modulus,
        shear_modulus=shear_modulus,
        position=input_position,
    )
    test_directors = mockrod.director_collection
    assert_allclose(correct_directors, test_directors, atol=Tolerance.atol())


@pytest.mark.parametrize("n_elems", [5, 10, 50])
def test_directors_using_input_directory_array(n_elems):
    """
    This test is testing the case for which directors are given as user input.

    Parameters
    ----------
    n_elems

    Returns
    -------

    """
    start = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    angle = np.random.uniform(0, 2 * np.pi)
    normal = np.array([0.0, np.cos(angle), np.sin(angle)])
    base_length = 1.0
    base_radius = 0.25
    density = 1000
    nu = 0.1
    youngs_modulus = 1e6
    poisson_ratio = 0.3
    shear_modulus = youngs_modulus / (poisson_ratio + 1.0)
    # Check directors, give position as input and let allocate function to compute directors.
    input_position = np.zeros((3, n_elems + 1))
    input_position[0, :] = np.linspace(start[0], start[0] + base_length, n_elems + 1)

    correct_directors = np.zeros((MaxDimension.value(), MaxDimension.value(), n_elems))
    binormal = np.cross(direction, normal)
    tangent_collection = np.repeat(direction[:, np.newaxis], n_elems, axis=1)
    normal_collection = np.repeat(normal[:, np.newaxis], n_elems, axis=1)
    binormal_collection = np.repeat(binormal[:, np.newaxis], n_elems, axis=1)

    correct_directors[0, ...] = normal_collection
    correct_directors[1, ...] = binormal_collection
    correct_directors[2, ...] = tangent_collection

    mockrod = MockRodForTest.straight_rod(
        n_elems,
        start,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        nu,
        youngs_modulus,
        shear_modulus=shear_modulus,
        position=input_position,
        directors=correct_directors,
    )
    test_directors = mockrod.director_collection
    assert_allclose(correct_directors, test_directors, atol=Tolerance.atol())


@pytest.mark.xfail(raises=AssertionError)
def test_director_if_d3_cross_d2_notequal_to_d1():
    """
    This test is checking the case if the directors, d3xd2 is not equal
    to d1 and creates an AssertionError.
    Returns
    -------

    """
    n_elems = 10
    start = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    base_length = 1.0
    base_radius = 0.25
    density = 1000
    nu = 0.1
    youngs_modulus = 1e6
    poisson_ratio = 0.3
    shear_modulus = youngs_modulus / (poisson_ratio + 1.0)
    # Check directors, give directors as input and check their validity.
    # Let the assertion fail by setting d3=d2 for the input director
    input_directors = np.zeros((MaxDimension.value(), MaxDimension.value(), n_elems))
    binormal = np.cross(direction, normal)
    normal_collection = np.repeat(normal[:, np.newaxis], n_elems, axis=1)
    binormal_collection = np.repeat(binormal[:, np.newaxis], n_elems, axis=1)

    input_directors[0, ...] = normal_collection
    input_directors[1, ...] = binormal_collection
    input_directors[2, ...] = binormal_collection

    MockRodForTest.straight_rod(
        n_elems,
        start,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        nu,
        youngs_modulus,
        shear_modulus=shear_modulus,
        directors=input_directors,
    )


@pytest.mark.xfail(raises=AssertionError)
def test_director_if_tangent_and_d3_are_not_same():
    """
    This test is checking the case if the tangent and d3 of the directors
    are not equal to each other.

    Returns
    -------

    """
    n_elems = 10
    start = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    base_length = 1.0
    base_radius = 0.25
    density = 1000
    nu = 0.1
    youngs_modulus = 1e6
    poisson_ratio = 0.3
    shear_modulus = youngs_modulus / (poisson_ratio + 1.0)

    position = np.zeros((3, n_elems + 1))
    end = start + direction * base_length
    for i in range(0, 3):
        position[i, ...] = np.linspace(start[i], end[i], n_elems + 1)

    # Set the directors such that tangent and d3 are not same.
    input_directors = np.zeros((MaxDimension.value(), MaxDimension.value(), n_elems))
    binormal = np.cross(direction, normal)
    normal_collection = np.repeat(binormal[:, np.newaxis], n_elems, axis=1)
    binormal_collection = np.repeat(normal[:, np.newaxis], n_elems, axis=1)
    new_direction = np.cross(binormal, normal)
    direction_collection = np.repeat(new_direction[:, np.newaxis], n_elems, axis=1)

    input_directors[0, ...] = normal_collection
    input_directors[1, ...] = binormal_collection
    input_directors[2, ...] = direction_collection

    MockRodForTest.straight_rod(
        n_elems,
        start,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        nu,
        youngs_modulus,
        shear_modulus=shear_modulus,
        position=position,
        directors=input_directors,
    )


@pytest.mark.parametrize("n_elems", [5, 10, 50])
def test_compute_radius_using_base_radius(n_elems):
    """
    This test is checking the case if user defined base radius
    is used to generate radius array.
    Parameters
    ----------
    n_elems

    Returns
    -------

    """
    start = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    base_length = 1.0
    base_radius = 0.25
    density = 1000
    nu = 0.1
    youngs_modulus = 1e6
    poisson_ratio = 0.3
    shear_modulus = youngs_modulus / (poisson_ratio + 1.0)

    mockrod = MockRodForTest.straight_rod(
        n_elems,
        start,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        nu,
        youngs_modulus,
        shear_modulus=shear_modulus,
    )
    correct_radius = base_radius * np.ones((n_elems))
    test_radius = mockrod.radius
    assert_allclose(correct_radius, test_radius, atol=Tolerance.atol())


@pytest.mark.parametrize("n_elems", [5, 10, 50])
def test_radius_using_user_defined_radius(n_elems):
    """
    This test is checking if user defined radius array is valid,
    and allocating radius array of rod correctly.
    Parameters
    ----------
    n_elems

    Returns
    -------

    """
    start = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    base_length = 1.0
    base_radius = np.linspace(0.1, 0.5, n_elems)
    density = 1000
    nu = 0.1
    youngs_modulus = 1e6
    poisson_ratio = 0.3
    shear_modulus = youngs_modulus / (poisson_ratio + 1.0)

    mockrod = MockRodForTest.straight_rod(
        n_elems,
        start,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        nu,
        youngs_modulus,
        shear_modulus=shear_modulus,
    )
    correct_radius = base_radius
    test_radius = mockrod.radius
    assert_allclose(correct_radius, test_radius, atol=Tolerance.atol())


@pytest.mark.xfail(raises=AssertionError)
@pytest.mark.parametrize("n_elems", [5, 10, 50])
def test_radius_not_correct_radius_shape(n_elems):
    """
    This test is checking if user gives radius array in incorrect
    format and program throws an assertion error.
    Parameters
    ----------
    n_elems

    Returns
    -------

    """
    start = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    base_length = 1.0
    base_radius = np.linspace(0.1, 0.5, n_elems).reshape(1, n_elems)
    density = 1000
    nu = 0.1
    youngs_modulus = 1e6
    poisson_ratio = 0.3
    shear_modulus = youngs_modulus / (poisson_ratio + 1.0)
    MockRodForTest.straight_rod(
        n_elems,
        start,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        nu,
        youngs_modulus,
        shear_modulus=shear_modulus,
    )


@pytest.mark.parametrize("n_elems", [5, 10, 50])
@pytest.mark.parametrize("shear_modulus", [5e3, 10e3, 50e3])
def test_shear_matrix_for_varying_shear_modulus(n_elems, shear_modulus):
    """
    This test, is checking if for user defined shear modulus and validity of shear matrix.

    Returns
    -------

    """
    start = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    base_length = 1.0
    base_radius = 0.1
    density = 1000
    nu = 0.1
    youngs_modulus = 1e6
    base_area = np.pi * base_radius ** 2

    mockrod = MockRodForTest.straight_rod(
        n_elems,
        start,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        nu,
        youngs_modulus,
        shear_modulus=shear_modulus,
    )

    test_shear_matrix = mockrod.shear_matrix

    correct_shear_matrix = np.zeros((3, 3))
    np.fill_diagonal(
        correct_shear_matrix[:],
        [
            0.964 * shear_modulus * base_area,
            0.964 * shear_modulus * base_area,
            youngs_modulus * base_area,
        ],
    )

    for k in range(n_elems):
        assert_allclose(
            correct_shear_matrix,
            test_shear_matrix[..., k],
            atol=Tolerance.atol(),
        )


@pytest.mark.parametrize("n_elems", [5, 10, 50])
@pytest.mark.parametrize("shear_modulus", [5e3, 10e3, 50e3])
def test_shear_matrix_for_varying_shear_modulus_error_message_check_if_poisson_ratio_defined(
    n_elems, shear_modulus
):
    """
    This test, is checking if for user defined shear modulus and validity of shear matrix,
    if the poisson ratio is defined. We expect if poisson ratio and shear modulus defined together then Elastica will
    raise a  UserWarning message and use the user defined shear modulus.

    Returns
    -------

    """
    start = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    base_length = 1.0
    base_radius = 0.1
    density = 1000
    nu = 0.1
    youngs_modulus = 1e6
    poisson_ratio = 0.3
    base_area = np.pi * base_radius ** 2

    with pytest.raises(NameError):
        mockrod = MockRodForTest.straight_rod(
            n_elems,
            start,
            direction,
            normal,
            base_length,
            base_radius,
            density,
            nu,
            youngs_modulus,
            shear_modulus=shear_modulus,
            poisson_ratio=poisson_ratio,
        )


def test_inertia_shear_bend_matrices_for_varying_radius():
    """
    This test, is checking if for user defined varying radius, validity
    of mass second moment of inertia, inv mass moment of inertia, shear and bend
    matrices.
    Returns
    -------

    """
    n_elems = 4
    start = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    base_length = 1.0
    base_radius = np.array([0.1, 0.2, 0.3, 0.4])
    density = 1000
    nu = 0.1
    youngs_modulus = 1e6
    poisson_ratio = 0.3
    shear_modulus = youngs_modulus / (poisson_ratio + 1.0)

    mockrod = MockRodForTest.straight_rod(
        n_elems,
        start,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        nu,
        youngs_modulus,
        shear_modulus=shear_modulus,
    )

    correct_mass_second_moment_of_inertia = np.array(
        [
            [0.019634954, 0.019634954, 0.039269908],
            [0.314159265, 0.314159265, 0.628318531],
            [1.590431281, 1.590431281, 3.180862562],
            [5.026548246, 5.026548246, 10.05309649],
        ]
    )

    correct_inv_mass_second_moment_of_inertia = np.array(
        [
            [50.92958179, 50.92958179, 25.46479089],
            [3.183098862, 3.183098862, 1.591549431],
            [0.628760269, 0.628760269, 0.314380135],
            [0.198943679, 0.198943679, 0.099471839],
        ]
    )

    correct_shear_matrix = np.array(
        [
            [23296.11783, 23296.11783, 31415.92654],
            [93184.47129, 93184.47129, 125663.7061],
            [209665.06048, 209665.06048, 282743.33882],
            [372737.88191, 372737.88191, 502654.82],
        ]
    )

    correct_bend_matrix = np.array(
        [
            [667.58844, 667.58844, 1027.05914],
            [3809.18109, 3809.18109, 5860.27860],
            [13233.95905, 13233.95905, 20359.93700],
        ]
    )

    correct_volume = np.array([0.007853982, 0.031415927, 0.070685835, 0.125663706])

    test_mass_second_moment_of_inertia = mockrod.mass_second_moment_of_inertia
    test_inv_mass_second_moment_of_inertia = mockrod.inv_mass_second_moment_of_inertia
    test_shear_matrix = mockrod.shear_matrix
    test_bend_matrix = mockrod.bend_matrix
    for k in range(n_elems):
        for i in range(3):
            assert_allclose(
                correct_mass_second_moment_of_inertia[k, i],
                test_mass_second_moment_of_inertia[i, i, k],
                atol=Tolerance.atol(),
            )

            assert_allclose(
                correct_inv_mass_second_moment_of_inertia[k, i],
                test_inv_mass_second_moment_of_inertia[i, i, k],
                atol=Tolerance.atol(),
            )

            assert_allclose(
                correct_shear_matrix[k, i],
                test_shear_matrix[i, i, k],
                atol=Tolerance.atol(),
            )

    for k in range(n_elems - 1):
        for i in range(3):
            assert_allclose(
                correct_bend_matrix[k, i],
                test_bend_matrix[i, i, k],
                atol=Tolerance.atol(),
            )

    test_volume = mockrod.volume
    assert_allclose(correct_volume, test_volume, atol=Tolerance.atol())


@pytest.mark.parametrize("n_elems", [5, 10, 50])
def test_constant_density(n_elems):
    """
    This function tests, for constant density input, validity of
    computed mass by allocated function. We check if the
    Parameters
    ----------
    n_elems

    Returns
    -------

    """
    start = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    base_length = 1.0
    base_radius = 0.25
    density = 1000
    nu = 0.1
    youngs_modulus = 1e6
    poisson_ratio = 0.3
    shear_modulus = youngs_modulus / (poisson_ratio + 1.0)

    mockrod = MockRodForTest.straight_rod(
        n_elems,
        start,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        nu,
        youngs_modulus,
        shear_modulus=shear_modulus,
    )
    correct_mass = density * np.pi * base_radius ** 2 * base_length / n_elems
    test_mass = mockrod.mass

    for i in range(1, n_elems):
        assert_allclose(correct_mass, test_mass[i], atol=Tolerance.atol())
    assert_allclose(0.5 * correct_mass, test_mass[0], atol=Tolerance.atol())
    assert_allclose(0.5 * correct_mass, test_mass[-1], atol=Tolerance.atol())


@pytest.mark.parametrize("n_elems", [5, 10, 50])
def test_varying_density(n_elems):
    """

    Parameters
    ----------
    n_elems

    Returns
    -------

    """
    start = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    base_length = 1.0
    base_radius = 0.25
    density = np.linspace(500, 1000, n_elems)
    nu = 0.1
    youngs_modulus = 1e6
    poisson_ratio = 0.3
    shear_modulus = youngs_modulus / (poisson_ratio + 1.0)

    mockrod = MockRodForTest.straight_rod(
        n_elems,
        start,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        nu,
        youngs_modulus,
        shear_modulus=shear_modulus,
    )
    volume = np.pi * base_radius ** 2 * base_length / n_elems
    correct_mass = np.zeros(n_elems + 1)
    correct_mass[:-1] += 0.5 * density * volume
    correct_mass[1:] += 0.5 * density * volume
    test_mass = mockrod.mass

    assert_allclose(correct_mass, test_mass, atol=Tolerance.atol())


@pytest.mark.xfail(raises=AssertionError)
@pytest.mark.parametrize("n_elems", [5, 10, 50])
def test_density_invalid_shape(n_elems):
    """
    This test is checking if user gives density array in incorrect
    format and program throws an assertion error.
    Parameters
    ----------
    n_elems

    Returns
    -------

    """
    start = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    base_length = 1.0
    base_radius = 0.25
    density = np.linspace(500, 1000, n_elems).reshape(1, n_elems)
    nu = 0.1
    youngs_modulus = 1e6
    poisson_ratio = 0.3
    shear_modulus = youngs_modulus / (poisson_ratio + 1.0)
    MockRodForTest.straight_rod(
        n_elems,
        start,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        nu,
        youngs_modulus,
        shear_modulus=shear_modulus,
    )


@pytest.mark.parametrize("n_elems", [5, 10, 50])
def test_constant_nu_for_forces(n_elems):
    """
    This function tests, for fix dissipation
    constant for forces, validty of dissipation constant array.

    Parameters
    ----------
    n_elems

    Returns
    -------

    """
    start = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    base_length = 1.0
    base_radius = 0.25
    density = 1000
    nu = 0.1
    youngs_modulus = 1e6
    poisson_ratio = 0.3
    shear_modulus = youngs_modulus / (poisson_ratio + 1.0)

    mockrod = MockRodForTest.straight_rod(
        n_elems,
        start,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        nu,
        youngs_modulus,
        shear_modulus=shear_modulus,
    )
    correct_nu = nu
    test_nu = mockrod.dissipation_constant_for_forces
    assert_allclose(correct_nu, test_nu, atol=Tolerance.atol())


@pytest.mark.parametrize("n_elems", [5, 10, 50])
def test_varying_nu_for_forces(n_elems):
    """
    This function tests, for varying dissipation
    constant for forces input, validty of dissipation constant array.

    Parameters
    ----------
    n_elems

    Returns
    -------

    """
    start = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    base_length = 1.0
    base_radius = 0.25
    density = 1000
    nu = np.linspace(0.1, 1.0, n_elems)
    youngs_modulus = 1e6
    poisson_ratio = 0.3
    shear_modulus = youngs_modulus / (poisson_ratio + 1.0)

    mockrod = MockRodForTest.straight_rod(
        n_elems,
        start,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        nu,
        youngs_modulus,
        shear_modulus=shear_modulus,
    )
    correct_nu = nu
    test_nu = mockrod.dissipation_constant_for_forces
    assert_allclose(correct_nu, test_nu, atol=Tolerance.atol())


@pytest.mark.xfail(raises=AssertionError)
@pytest.mark.parametrize("n_elems", [5, 10, 50])
def test_nu_for_forces_invalid_shape(n_elems):
    """
    This test is checking if user gives nu for forces array in incorrect
    format and program throws an assertion error.
    Parameters
    ----------
    n_elems

    Returns
    -------

    """
    start = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    base_length = 1.0
    base_radius = 0.25
    density = 1000
    nu = np.linspace(0.1, 1.0, n_elems).reshape(1, n_elems)
    youngs_modulus = 1e6
    poisson_ratio = 0.3
    shear_modulus = youngs_modulus / (poisson_ratio + 1.0)
    MockRodForTest.straight_rod(
        n_elems,
        start,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        nu,
        youngs_modulus,
        shear_modulus=shear_modulus,
    )


@pytest.mark.parametrize("n_elems", [5, 10, 50])
def test_constant_nu_for_torques(n_elems):
    """
    This function tests, for fix dissipation
    constant for torques input, validty of dissipation constant array.

    Parameters
    ----------
    n_elems

    Returns
    -------

    """
    start = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    base_length = 1.0
    base_radius = 0.25
    density = 1000
    nu_for_forces = 0.2
    nu_for_torques = 0.1
    youngs_modulus = 1e6
    poisson_ratio = 0.3
    shear_modulus = youngs_modulus / (poisson_ratio + 1.0)

    mockrod = MockRodForTest.straight_rod(
        n_elems,
        start,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        nu_for_forces,
        youngs_modulus,
        shear_modulus=shear_modulus,
        nu_for_torques=nu_for_torques,
    )
    correct_nu = nu_for_torques
    test_nu = mockrod.dissipation_constant_for_torques
    assert_allclose(correct_nu, test_nu, atol=Tolerance.atol())


@pytest.mark.parametrize("n_elems", [5, 10, 50])
def test_varying_nu_for_torques(n_elems):
    """
    This function tests, for varying dissipation
    constant for torques input, validty of dissipation
    constant for torques array.

    Parameters
    ----------
    n_elems

    Returns
    -------

    """
    start = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    base_length = 1.0
    base_radius = 0.25
    density = 1000
    nu = 0.1
    nu_for_torques = np.linspace(0.1, 1.0, n_elems)
    youngs_modulus = 1e6
    poisson_ratio = 0.3
    shear_modulus = youngs_modulus / (poisson_ratio + 1.0)

    mockrod = MockRodForTest.straight_rod(
        n_elems,
        start,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        nu,
        youngs_modulus,
        shear_modulus=shear_modulus,
        nu_for_torques=nu_for_torques,
    )
    correct_nu = nu_for_torques
    test_nu = mockrod.dissipation_constant_for_torques
    assert_allclose(correct_nu, test_nu, atol=Tolerance.atol())


@pytest.mark.xfail(raises=AssertionError)
@pytest.mark.parametrize("n_elems", [5, 10, 50])
def test_nu_for_torques_invalid_shape(n_elems):
    """
    This test is checking if user gives nu for torques array in incorrect
    format and program throws an assertion error.
    Parameters
    ----------
    n_elems

    Returns
    -------

    """
    start = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    base_length = 1.0
    base_radius = 0.25
    density = 1000
    nu = 0.1
    nu_for_torques = np.linspace(0.1, 1.0, n_elems).reshape(1, n_elems)
    youngs_modulus = 1e6
    poisson_ratio = 0.3
    shear_modulus = youngs_modulus / (poisson_ratio + 1.0)
    MockRodForTest.straight_rod(
        n_elems,
        start,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        nu,
        youngs_modulus,
        shear_modulus=shear_modulus,
        nu_for_torques=nu_for_torques,
    )


@pytest.mark.parametrize("n_elems", [5, 10, 50])
def test_constant_nu_for_torques_if_not_input(n_elems):
    """
    This function tests, dissipation constant for torques
    if it is not in kwargs. If dissipation constant for torques
    is not in kwargs it uses the dissipation for forces.

    Parameters
    ----------
    n_elems

    Returns
    -------

    """
    start = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    base_length = 1.0
    base_radius = 0.25
    density = 1000
    nu = 0.2
    youngs_modulus = 1e6
    poisson_ratio = 0.3
    shear_modulus = youngs_modulus / (poisson_ratio + 1.0)

    mockrod = MockRodForTest.straight_rod(
        n_elems,
        start,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        nu,
        youngs_modulus,
        shear_modulus=shear_modulus,
    )
    correct_nu = nu
    test_nu = mockrod.dissipation_constant_for_torques
    assert_allclose(correct_nu, test_nu, atol=Tolerance.atol())


@pytest.mark.parametrize("n_elems", [5, 10, 50])
def test_rest_sigma_and_kappa_user_input(n_elems):
    """
    This test is checking if user defined sigma is used to
    allocate rest sigma of the rod.
    Parameters
    ----------
    n_elems

    Returns
    -------

    """
    start = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    base_length = 1.0
    base_radius = 0.25
    density = 1000
    nu = 0.1
    youngs_modulus = 1e6
    poisson_ratio = 0.3
    shear_modulus = youngs_modulus / (poisson_ratio + 1.0)

    input_rest_sigma = np.random.randn(3, n_elems)
    input_rest_kappa = np.random.randn(3, n_elems - 1)

    mockrod = MockRodForTest.straight_rod(
        n_elems,
        start,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        nu,
        youngs_modulus,
        shear_modulus=shear_modulus,
        rest_sigma=input_rest_sigma,
        rest_kappa=input_rest_kappa,
    )

    correct_rest_sigma = input_rest_sigma
    correct_rest_kappa = input_rest_kappa
    test_rest_sigma = mockrod.rest_sigma
    test_rest_kappa = mockrod.rest_kappa

    assert_allclose(correct_rest_sigma, test_rest_sigma, atol=Tolerance.atol())
    assert_allclose(correct_rest_kappa, test_rest_kappa, atol=Tolerance.atol())


@pytest.mark.xfail(raises=AssertionError)
@pytest.mark.parametrize("n_elems", [5, 10, 50])
def test_rest_sigma_and_kappa_invalid_shape(n_elems):
    """
    This test, is checking AssertionErrors for invalid shapes for
    rest sigma and rest kappa
    Parameters
    ----------
    n_elems

    Returns
    -------

    """
    start = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    base_length = 1.0
    base_radius = 0.25
    density = 1000
    nu = 0.1
    youngs_modulus = 1e6
    poisson_ratio = 0.3
    shear_modulus = youngs_modulus / (poisson_ratio + 1.0)

    input_rest_sigma = np.random.randn(3, n_elems).reshape(n_elems, 3)
    input_rest_kappa = np.random.randn(3, n_elems - 1).reshape(n_elems - 1, 3)

    MockRodForTest.straight_rod(
        n_elems,
        start,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        nu,
        youngs_modulus,
        shear_modulus=shear_modulus,
        rest_sigma=input_rest_sigma,
        rest_kappa=input_rest_kappa,
    )


@pytest.mark.parametrize("n_elems", [5, 10, 50])
def test_validity_of_allocated(n_elems):
    """
    This test is for checking if variables are
    initialized correctly.
    Parameters
    ----------
    n_elems

    Returns
    -------

    """

    start = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    base_length = 1.0
    base_radius = 0.25
    density = 1000
    nu = 0.1
    youngs_modulus = 1e6
    poisson_ratio = 0.3
    shear_modulus = youngs_modulus / (poisson_ratio + 1.0)
    mockrod = MockRodForTest.straight_rod(
        n_elems,
        start,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        nu,
        youngs_modulus,
        shear_modulus=shear_modulus,
    )

    assert_allclose(n_elems, mockrod.n_elems, atol=Tolerance.atol())
    assert_allclose(
        mockrod.velocity_collection, np.zeros((3, n_elems + 1)), atol=Tolerance.atol()
    )
    assert_allclose(
        mockrod.acceleration_collection,
        np.zeros((3, n_elems + 1)),
        atol=Tolerance.atol(),
    )
    assert_allclose(
        mockrod.omega_collection, np.zeros((3, n_elems)), atol=Tolerance.atol()
    )
    assert_allclose(
        mockrod.alpha_collection, np.zeros((3, n_elems)), atol=Tolerance.atol()
    )
    assert_allclose(
        mockrod.internal_forces, np.zeros((3, n_elems + 1)), atol=Tolerance.atol()
    )
    assert_allclose(
        mockrod.internal_torques, np.zeros((3, n_elems)), atol=Tolerance.atol()
    )
    assert_allclose(
        mockrod.external_forces, np.zeros((3, n_elems + 1)), atol=Tolerance.atol()
    )
    assert_allclose(
        mockrod.external_torques, np.zeros((3, n_elems)), atol=Tolerance.atol()
    )
    assert_allclose(mockrod.lengths, np.zeros((n_elems)), atol=Tolerance.atol())
    assert_allclose(mockrod.tangents, np.zeros((3, n_elems)), atol=Tolerance.atol())
    assert_allclose(mockrod.dilatation, np.zeros((n_elems)), atol=Tolerance.atol())
    assert_allclose(
        mockrod.voronoi_dilatation, np.zeros((n_elems - 1)), atol=Tolerance.atol()
    )
    assert_allclose(mockrod.dilatation_rate, np.zeros((n_elems)), atol=Tolerance.atol())
    assert_allclose(mockrod.sigma, np.zeros((3, n_elems)), atol=Tolerance.atol())
    assert_allclose(mockrod.kappa, np.zeros((3, n_elems - 1)), atol=Tolerance.atol())
    assert_allclose(mockrod.rest_sigma, np.zeros((3, n_elems)), atol=Tolerance.atol())
    assert_allclose(
        mockrod.rest_kappa, np.zeros((3, n_elems - 1)), atol=Tolerance.atol()
    )
    assert_allclose(
        mockrod.internal_stress, np.zeros((3, n_elems)), atol=Tolerance.atol()
    )
    assert_allclose(
        mockrod.internal_couple, np.zeros((3, n_elems - 1)), atol=Tolerance.atol()
    )
    assert_allclose(
        mockrod.damping_forces, np.zeros((3, n_elems + 1)), atol=Tolerance.atol()
    )
    assert_allclose(
        mockrod.damping_torques, np.zeros((3, n_elems)), atol=Tolerance.atol()
    )


@pytest.mark.parametrize("n_elems", [5, 20, 50])
def test_straight_rod(n_elems):

    # setting up test params
    start = np.random.rand(3)
    direction = 5 * np.random.rand(3)
    direction_norm = np.linalg.norm(direction)
    direction /= direction_norm
    normal = np.array((direction[1], -direction[0], 0))
    base_length = 10
    base_radius = np.random.uniform(1, 10)
    density = np.random.uniform(1, 10)
    mass = density * np.pi * base_radius ** 2 * base_length / n_elems

    nu = 0.1
    # Youngs Modulus [Pa]
    E = 1e6
    # poisson ratio
    poisson_ratio = 0.5
    # Shear Modulus [Pa]
    G = E / (1.0 + poisson_ratio)
    # alpha c, constant for circular cross-sections
    # Second moment of inertia
    A0 = np.pi * base_radius * base_radius
    I0_1 = A0 * A0 / (4.0 * np.pi)
    I0_2 = I0_1
    I0_3 = 2.0 * I0_2
    I0 = np.array([I0_1, I0_2, I0_3])
    # Mass second moment of inertia for disk cross-section
    mass_second_moment_of_inertia = np.zeros((3, 3), np.float64)
    np.fill_diagonal(
        mass_second_moment_of_inertia, I0 * density * base_length / n_elems
    )
    # Inverse mass second of inertia
    inv_mass_second_moment_of_inertia = np.linalg.inv(mass_second_moment_of_inertia)
    # Shear/Stretch matrix
    shear_matrix = np.zeros((3, 3), np.float64)
    np.fill_diagonal(shear_matrix, [0.964 * G * A0, 0.964 * G * A0, E * A0])
    # Bend/Twist matrix
    bend_matrix = np.zeros((3, 3), np.float64)
    np.fill_diagonal(bend_matrix, [E * I0_1, E * I0_2, G * I0_3])

    mockrod = MockRodForTest.straight_rod(
        n_elems,
        start,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        nu,
        E,
        shear_modulus=G,
    )
    # checking origin and length of rod
    assert_allclose(mockrod.position_collection[..., 0], start, atol=Tolerance.atol())
    rod_length = np.linalg.norm(
        mockrod.position_collection[..., -1] - mockrod.position_collection[..., 0]
    )
    rest_voronoi_lengths = 0.5 * (
        base_length / n_elems + base_length / n_elems
    )  # element lengths are equal for all rod.
    # checking velocities, omegas and rest strains
    # density and mass
    assert_allclose(rod_length, base_length, atol=Tolerance.atol())
    assert_allclose(
        mockrod.velocity_collection, np.zeros((3, n_elems + 1)), atol=Tolerance.atol()
    )
    assert_allclose(
        mockrod.omega_collection, np.zeros((3, n_elems)), atol=Tolerance.atol()
    )
    assert_allclose(mockrod.rest_sigma, np.zeros((3, n_elems)), atol=Tolerance.atol())
    assert_allclose(
        mockrod.rest_kappa, np.zeros((3, n_elems - 1)), atol=Tolerance.atol()
    )
    assert_allclose(mockrod.density, density, atol=Tolerance.atol())
    assert_allclose(mockrod.dissipation_constant_for_forces, nu, atol=Tolerance.atol())
    assert_allclose(mockrod.dissipation_constant_for_torques, nu, atol=Tolerance.atol())
    assert_allclose(
        rest_voronoi_lengths, mockrod.rest_voronoi_lengths, atol=Tolerance.atol()
    )
    # Check mass at each node. Note that, node masses is
    # half of element mass at the first and last node.
    for i in range(1, n_elems):
        assert_allclose(mockrod.mass[i], mass, atol=Tolerance.atol())
    assert_allclose(mockrod.mass[0], 0.5 * mass, atol=Tolerance.atol())
    assert_allclose(mockrod.mass[-1], 0.5 * mass, atol=Tolerance.atol())
    # checking directors, rest length
    # and shear, bend matrices and moment of inertia
    for i in range(n_elems):
        assert_allclose(
            mockrod.director_collection[0, :, i], normal, atol=Tolerance.atol()
        )
        assert_allclose(
            mockrod.director_collection[1, :, i],
            np.cross(direction, normal),
            atol=Tolerance.atol(),
        )
        assert_allclose(
            mockrod.director_collection[2, :, i], direction, atol=Tolerance.atol()
        )
        assert_allclose(
            mockrod.rest_lengths, base_length / n_elems, atol=Tolerance.atol()
        )
        assert_allclose(
            mockrod.shear_matrix[..., i], shear_matrix, atol=Tolerance.atol()
        )
        assert_allclose(
            mockrod.mass_second_moment_of_inertia[..., i],
            mass_second_moment_of_inertia,
            atol=Tolerance.atol(),
        )
        assert_allclose(
            mockrod.inv_mass_second_moment_of_inertia[..., i],
            inv_mass_second_moment_of_inertia,
            atol=Tolerance.atol(),
        )
    for i in range(n_elems - 1):
        assert_allclose(mockrod.bend_matrix[..., i], bend_matrix, atol=Tolerance.atol())


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main

        main([__file__])
