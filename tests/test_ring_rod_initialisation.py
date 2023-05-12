__doc__ = """Test for ring rod initialization module"""
import numpy as np
from numpy.testing import assert_allclose
import pytest
from elastica.utils import MaxDimension, Tolerance
from elastica.rod.factory_function import allocate
import elastica as ea


class MockRingRodForTest:
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
        ring_rod_flag,
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
        self.ring_rod_flag = ring_rod_flag

    @classmethod
    def ring_rod(
        cls,
        n_elements,
        ring_center_position,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        *,
        youngs_modulus,
        **kwargs,
    ):

        # Ring rod flag is true
        ring_rod_flag = True
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
        ) = allocate(
            n_elements,
            direction,
            normal,
            base_length,
            base_radius,
            density,
            youngs_modulus,
            rod_origin_position=ring_center_position,
            ring_rod_flag=ring_rod_flag,
            **kwargs,
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
            ring_rod_flag,
        )


@pytest.mark.parametrize("n_elems", [5, 10, 50])
def test_deprecated_rod_nu_option(n_elems):
    start = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    base_length = 1.0
    base_radius = 0.25
    density = 1000
    nu = 0.1
    youngs_modulus = 1e6
    poisson_ratio = 0.3
    correct_position = np.zeros((3, n_elems + 1))
    correct_position[0] = np.random.randn(n_elems + 1)
    correct_position[1] = np.random.randn(n_elems + 1)
    correct_position[..., 0] = start
    correct_error_message = (
        "The option to set damping coefficient (nu) for the rod during rod\n"
        "initialisation is now deprecated. Instead, for adding damping to rods,\n"
        "please derive your simulation class from the add-on Damping mixin class.\n"
        "For reference see the class elastica.dissipation.AnalyticalLinearDamper(),\n"
        "and for usage check examples/axial_stretching.py"
    )
    with pytest.raises(ValueError) as exc_info:
        _ = ea.CosseratRod.ring_rod(
            n_elems,
            start,
            direction,
            normal,
            base_length,
            base_radius,
            density,
            nu=nu,
            youngs_modulus=youngs_modulus,
        )
    assert exc_info.value.args[0] == correct_error_message


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
    direction = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    base_length = 1.0
    base_radius = 0.25
    density = 1000
    youngs_modulus = 1e6

    # Check if the input position vector and output position vector are valid and same
    correct_position = np.zeros((3, n_elems))
    correct_position[0] = np.random.randn(n_elems)
    correct_position[1] = np.random.randn(n_elems)
    center_position = np.mean(correct_position, axis=1)
    mockrod = MockRingRodForTest.ring_rod(
        n_elems,
        center_position,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        youngs_modulus=youngs_modulus,
        position=correct_position,
    )
    test_position = mockrod.position_collection
    assert_allclose(correct_position, test_position, atol=Tolerance.atol())


@pytest.mark.xfail(raises=AssertionError)
@pytest.mark.parametrize("n_elems", [5, 10, 50])
def test_input_and_position_array_for_different_center_offset(n_elems):
    """
    This function tests fail check, for which input position array
    first element is not user defined center_offset position.
    Parameters
    ----------
    n_elems

    Returns
    -------

    """
    center_offset = np.random.randn(3)
    direction = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    base_length = 1.0
    base_radius = 0.25
    density = 1000
    youngs_modulus = 1e6

    # Check if the input position vector center_offset position is different than the user defined center_offset position
    correct_position = np.random.randn(3, n_elems)
    mockrod = MockRingRodForTest.ring_rod(
        n_elems,
        center_offset,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        youngs_modulus=youngs_modulus,
        position=correct_position,
    )
    test_position = mockrod.position_collection
    assert_allclose(correct_position, test_position, atol=Tolerance.atol())


def test_compute_position_array_using_user_inputs():
    """
    This test checks if the allocate function can compute correctly
    position vector using center_offset, direction and base length inputs.
    Returns
    -------

    """
    n_elems = 5
    center_offset = np.array([0.0, 0.0, 0.0])
    direction = np.array([0.0, 1.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    base_length = 1.0
    base_radius = 0.25
    density = 1000
    youngs_modulus = 1e6
    # Check if without input position vector, output position vector is valid
    mockrod = MockRingRodForTest.ring_rod(
        n_elems,
        center_offset,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        youngs_modulus=youngs_modulus,
    )
    correct_position = np.zeros((3, n_elems))
    for i in range(n_elems):
        correct_position[..., i] = (
            base_length
            / (2 * np.pi)
            * np.array(
                [np.cos(2 * np.pi / n_elems * i), np.sin(2 * np.pi / n_elems * i), 0]
            )
        )
    test_position = mockrod.position_collection
    assert_allclose(correct_position, test_position, atol=Tolerance.atol())


@pytest.mark.parametrize("n_elems", [4, 5, 10])
def test_compute_directors_matrix_using_user_inputs(n_elems):
    """
    This test checks the director array created by allocate function. For this
    test case we use user defined direction, normal to compute directors.
    Returns
    -------

    """
    center_offset = np.array([0.0, 0.0, 0.0])
    direction = np.array([0.0, 1.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    base_length = 1.0
    base_radius = 0.25
    density = 1000
    youngs_modulus = 1e6
    # Check directors, if we dont input any directors, computed ones should be valid
    correct_directors = np.zeros((MaxDimension.value(), MaxDimension.value(), n_elems))
    tangent_collection = np.zeros((MaxDimension.value(), n_elems))
    binormal_collection = np.zeros((MaxDimension.value(), n_elems))
    normal_collection = np.repeat(normal[:, np.newaxis], n_elems, axis=1)

    angle = 2 * np.pi / n_elems
    for i in range(n_elems):
        tangent_collection[..., i] = np.array(
            [
                np.cos(angle * (i + 1)) - np.cos(angle * i),
                np.sin(angle * (i + 1)) - np.sin(angle * i),
                0,
            ]
        )
        tangent_collection[..., i] /= np.linalg.norm(tangent_collection[..., i])
        binormal_collection[..., i] = np.cross(tangent_collection[..., i], normal)

    correct_directors[0, ...] = normal_collection
    correct_directors[1, ...] = binormal_collection
    correct_directors[2, ...] = tangent_collection

    mockrod = MockRingRodForTest.ring_rod(
        n_elems,
        center_offset,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        youngs_modulus=youngs_modulus,
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
    direction = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    base_length = 1.0
    base_radius = 0.25
    density = 1000
    youngs_modulus = 1e6
    # Check directors, give position as input and let allocate function to compute directors.
    input_position = np.zeros((3, n_elems))
    for i in range(n_elems):
        input_position[..., i] = (
            base_length
            / (2 * np.pi)
            * np.array(
                [np.sin(2 * np.pi / n_elems * i), np.cos(2 * np.pi / n_elems * i), 0]
            )
        )
    center_offset = np.mean(input_position, axis=1)

    correct_directors = np.zeros((MaxDimension.value(), MaxDimension.value(), n_elems))
    tangent_collection = np.zeros((MaxDimension.value(), n_elems))
    binormal_collection = np.zeros((MaxDimension.value(), n_elems))
    normal_collection = np.repeat(normal[:, np.newaxis], n_elems, axis=1)

    angle = 2 * np.pi / n_elems
    for i in range(n_elems):
        tangent_collection[..., i] = np.array(
            [
                np.sin(angle * (i + 1)) - np.sin(angle * i),
                np.cos(angle * (i + 1)) - np.cos(angle * i),
                0,
            ]
        )
        tangent_collection[..., i] /= np.linalg.norm(tangent_collection[..., i])
        binormal_collection[..., i] = np.cross(tangent_collection[..., i], normal)

    correct_directors[0, ...] = normal_collection
    correct_directors[1, ...] = binormal_collection
    correct_directors[2, ...] = tangent_collection

    mockrod = MockRingRodForTest.ring_rod(
        n_elems,
        center_offset,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        youngs_modulus=youngs_modulus,
        position=input_position,
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
    center_offset = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    base_length = 1.0
    base_radius = 0.25
    density = 1000
    youngs_modulus = 1e6
    # Check directors, give directors as input and check their validity.
    # Let the assertion fail by setting d3=d2 for the input director
    input_directors = np.zeros((MaxDimension.value(), MaxDimension.value(), n_elems))
    binormal = np.cross(direction, normal)
    normal_collection = np.repeat(normal[:, np.newaxis], n_elems, axis=1)
    binormal_collection = np.repeat(binormal[:, np.newaxis], n_elems, axis=1)

    input_directors[0, ...] = normal_collection
    input_directors[1, ...] = binormal_collection
    input_directors[2, ...] = binormal_collection

    MockRingRodForTest.ring_rod(
        n_elems,
        center_offset,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        youngs_modulus=youngs_modulus,
        directors=input_directors,
    )


@pytest.mark.parametrize("n_elems", [5, 10, 50])
def test_validity_of_user_defined_directors_matrix(n_elems):
    """
    This test checks if the user defined directors are allocated correctly by allocate function.

    Returns
    -------

    """
    center_offset = np.array([0.0, 0.0, 0.0])
    direction = np.array([0.0, 1.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    base_length = 1.0
    base_radius = 0.25
    density = 1000
    youngs_modulus = 1e6
    # Check directors, if we dont input any directors, computed ones should be valid
    correct_directors = np.zeros((MaxDimension.value(), MaxDimension.value(), n_elems))
    tangent_collection = np.zeros((MaxDimension.value(), n_elems))
    binormal_collection = np.zeros((MaxDimension.value(), n_elems))
    normal_collection = np.repeat(normal[:, np.newaxis], n_elems, axis=1)

    angle = 2 * np.pi / n_elems
    for i in range(n_elems):
        tangent_collection[..., i] = np.array(
            [
                np.cos(angle * (i + 1)) - np.cos(angle * i),
                np.sin(angle * (i + 1)) - np.sin(angle * i),
                0,
            ]
        )
        tangent_collection[..., i] /= np.linalg.norm(tangent_collection[..., i])
        binormal_collection[..., i] = np.cross(tangent_collection[..., i], normal)

    correct_directors[0, ...] = normal_collection
    correct_directors[1, ...] = binormal_collection
    correct_directors[2, ...] = tangent_collection

    mockrod = MockRingRodForTest.ring_rod(
        n_elems,
        center_offset,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        youngs_modulus=youngs_modulus,
        directors=correct_directors,
    )
    test_directors = mockrod.director_collection
    assert_allclose(correct_directors, test_directors, atol=Tolerance.atol())


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
    center_offset = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    base_length = 1.0
    base_radius = 0.25
    density = 1000
    youngs_modulus = 1e6

    mockrod = MockRingRodForTest.ring_rod(
        n_elems,
        center_offset,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        youngs_modulus=youngs_modulus,
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
    center_offset = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    base_length = 1.0
    base_radius = np.linspace(0.1, 0.5, n_elems)
    density = 1000
    youngs_modulus = 1e6

    mockrod = MockRingRodForTest.ring_rod(
        n_elems,
        center_offset,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        youngs_modulus=youngs_modulus,
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
    center_offset = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    base_length = 1.0
    base_radius = np.linspace(0.1, 0.5, n_elems).reshape(1, n_elems)
    density = 1000
    youngs_modulus = 1e6
    MockRingRodForTest.ring_rod(
        n_elems,
        center_offset,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        youngs_modulus=youngs_modulus,
    )


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
    direction = np.array([0.0, 1.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    base_length = 1.0
    base_radius = 0.25
    density = 1000
    youngs_modulus = 1e6

    correct_position = np.zeros((3, n_elems + 1))
    for i in range(n_elems):
        correct_position[..., i] = (
            base_length
            / (2 * np.pi)
            * np.array(
                [np.cos(2 * np.pi / n_elems * i), np.sin(2 * np.pi / n_elems * i), 0]
            )
        )
    center_offset = np.mean(correct_position[:, :-1], axis=1)
    correct_position[..., -1] = correct_position[..., 0]
    position_diff = correct_position[..., 1:] - correct_position[..., :-1]
    correct_length = np.linalg.norm(position_diff, axis=0)

    mockrod = MockRingRodForTest.ring_rod(
        n_elems,
        center_offset,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        youngs_modulus=youngs_modulus,
    )
    correct_mass = density * np.pi * base_radius ** 2 * correct_length
    test_mass = mockrod.mass

    assert_allclose(correct_mass, test_mass, atol=Tolerance.atol())


@pytest.mark.parametrize("n_elems", [5, 10, 50])
def test_varying_density(n_elems):
    """

    Parameters
    ----------
    n_elems

    Returns
    -------

    """
    center_offset = np.array([0.0, 0.0, 0.0])
    direction = np.array([0.0, 1.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    base_length = 1.0
    base_radius = 0.25
    density = np.linspace(500, 1000, n_elems)
    youngs_modulus = 1e6

    mockrod = MockRingRodForTest.ring_rod(
        n_elems,
        center_offset,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        youngs_modulus=youngs_modulus,
    )

    correct_position = np.zeros((3, n_elems + 1))
    for i in range(n_elems):
        correct_position[..., i] = (
            base_length
            / (2 * np.pi)
            * np.array(
                [np.cos(2 * np.pi / n_elems * i), np.sin(2 * np.pi / n_elems * i), 0]
            )
        )
    correct_position[..., -1] = correct_position[..., 0]
    position_diff = correct_position[..., 1:] - correct_position[..., :-1]
    correct_length = np.linalg.norm(position_diff, axis=0)

    volume = np.pi * base_radius ** 2 * correct_length
    correct_mass = density * volume
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
    center_offset = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    base_length = 1.0
    base_radius = 0.25
    density = np.linspace(500, 1000, n_elems).reshape(1, n_elems)
    youngs_modulus = 1e6
    MockRingRodForTest.ring_rod(
        n_elems,
        center_offset,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        youngs_modulus=youngs_modulus,
    )


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
    center_offset = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    base_length = 1.0
    base_radius = 0.25
    density = 1000
    youngs_modulus = 1e6

    input_rest_sigma = np.random.randn(3, n_elems)
    input_rest_kappa = np.random.randn(3, n_elems)

    mockrod = MockRingRodForTest.ring_rod(
        n_elems,
        center_offset,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        youngs_modulus=youngs_modulus,
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
    center_offset = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    base_length = 1.0
    base_radius = 0.25
    density = 1000
    youngs_modulus = 1e6

    input_rest_sigma = np.random.randn(3, n_elems).reshape(n_elems, 3)
    input_rest_kappa = np.random.randn(3, n_elems - 1).reshape(n_elems - 1, 3)

    MockRingRodForTest.ring_rod(
        n_elems,
        center_offset,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        youngs_modulus=youngs_modulus,
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

    center_offset = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    base_length = 1.0
    base_radius = 0.25
    density = 1000
    youngs_modulus = 1e6
    mockrod = MockRingRodForTest.ring_rod(
        n_elems,
        center_offset,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        youngs_modulus=youngs_modulus,
    )

    assert_allclose(n_elems, mockrod.n_elems, atol=Tolerance.atol())
    assert_allclose(
        mockrod.velocity_collection, np.zeros((3, n_elems)), atol=Tolerance.atol()
    )
    assert_allclose(
        mockrod.acceleration_collection,
        np.zeros((3, n_elems)),
        atol=Tolerance.atol(),
    )
    assert_allclose(
        mockrod.omega_collection, np.zeros((3, n_elems)), atol=Tolerance.atol()
    )
    assert_allclose(
        mockrod.alpha_collection, np.zeros((3, n_elems)), atol=Tolerance.atol()
    )
    assert_allclose(
        mockrod.internal_forces, np.zeros((3, n_elems)), atol=Tolerance.atol()
    )
    assert_allclose(
        mockrod.internal_torques, np.zeros((3, n_elems)), atol=Tolerance.atol()
    )
    assert_allclose(
        mockrod.external_forces, np.zeros((3, n_elems)), atol=Tolerance.atol()
    )
    assert_allclose(
        mockrod.external_torques, np.zeros((3, n_elems)), atol=Tolerance.atol()
    )
    assert_allclose(mockrod.lengths, np.zeros((n_elems)), atol=Tolerance.atol())
    assert_allclose(mockrod.tangents, np.zeros((3, n_elems)), atol=Tolerance.atol())
    assert_allclose(mockrod.dilatation, np.zeros((n_elems)), atol=Tolerance.atol())
    assert_allclose(
        mockrod.voronoi_dilatation, np.zeros((n_elems)), atol=Tolerance.atol()
    )
    assert_allclose(mockrod.dilatation_rate, np.zeros((n_elems)), atol=Tolerance.atol())
    assert_allclose(mockrod.sigma, np.zeros((3, n_elems)), atol=Tolerance.atol())
    assert_allclose(mockrod.kappa, np.zeros((3, n_elems)), atol=Tolerance.atol())
    assert_allclose(mockrod.rest_sigma, np.zeros((3, n_elems)), atol=Tolerance.atol())
    assert_allclose(mockrod.rest_kappa, np.zeros((3, n_elems)), atol=Tolerance.atol())
    assert_allclose(
        mockrod.internal_stress, np.zeros((3, n_elems)), atol=Tolerance.atol()
    )
    assert_allclose(
        mockrod.internal_couple, np.zeros((3, n_elems)), atol=Tolerance.atol()
    )


@pytest.mark.parametrize("n_elems", [80])
def test_ring_rod(n_elems):

    # setting up test params
    center_offset = np.random.rand(3)
    direction = np.array([1.0, 0, 0.0])
    direction_norm = np.linalg.norm(direction)
    direction /= direction_norm
    normal = np.array((direction[1], -direction[0], 0))
    binormal = np.cross(direction, normal)
    base_length = 10
    base_radius = np.random.uniform(1, 10)
    density = np.random.uniform(1, 10)

    # Youngs Modulus [Pa]
    E = 1e6
    # poisson ratio
    poisson_ratio = 0.5
    # Shear Modulus [Pa]
    G = E / (2 * (1.0 + poisson_ratio))

    correct_position = np.zeros((3, n_elems))
    for i in range(n_elems):
        correct_position[..., i] = (
            base_length
            / (2 * np.pi)
            * (
                np.cos(2 * np.pi / n_elems * i) * binormal
                + np.sin(2 * np.pi / n_elems * i) * direction
            )
        ) + center_offset

    position_diff_temp = np.hstack(
        (correct_position, correct_position[..., 0].reshape(3, 1))
    )
    position_diff = position_diff_temp[..., 1:] - position_diff_temp[..., :-1]
    correct_length = np.linalg.norm(position_diff, axis=0)

    mass = density * np.pi * base_radius ** 2 * correct_length

    # alpha c, constant for ring cross-sections
    # Second moment of inertia
    A0 = np.pi * base_radius * base_radius
    I0_1 = A0 * A0 / (4.0 * np.pi)
    I0_2 = I0_1
    I0_3 = 2.0 * I0_2
    I0 = np.array([I0_1, I0_2, I0_3])
    # Mass second moment of inertia for disk cross-section
    mass_second_moment_of_inertia = np.zeros((3, 3, n_elems), np.float64)
    inv_mass_second_moment_of_inertia = np.zeros((3, 3, n_elems), np.float64)
    for i in range(n_elems):
        np.fill_diagonal(
            mass_second_moment_of_inertia[..., i], I0 * density * correct_length[i]
        )
        # Inverse mass second of inertia
        inv_mass_second_moment_of_inertia[..., i] = np.linalg.inv(
            mass_second_moment_of_inertia[..., i]
        )
    # Shear/Stretch matrix
    shear_matrix = np.zeros((3, 3), np.float64)
    # Value taken based on best correlation for Poisson ratio = 0.5, from
    # "On Timoshenko's correction for shear in vibrating beams" by Kaneko, 1975
    alpha_c = 27.0 / 28.0
    np.fill_diagonal(shear_matrix, [alpha_c * G * A0, alpha_c * G * A0, E * A0])
    # Bend/Twist matrix
    bend_matrix = np.zeros((3, 3), np.float64)
    np.fill_diagonal(bend_matrix, [E * I0_1, E * I0_2, G * I0_3])

    mockrod = MockRingRodForTest.ring_rod(
        n_elems,
        center_offset,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        youngs_modulus=E,
    )
    # checking origin and length of rod
    assert_allclose(
        mockrod.position_collection[..., 0],
        center_offset + (base_length / (2 * np.pi)) * binormal,
        atol=Tolerance.atol(),
    )
    rod_length = mockrod.rest_lengths.sum()
    lenghts_temp = np.hstack((correct_length, correct_length[0]))
    rest_voronoi_lengths = 0.5 * (
        lenghts_temp[1:] + lenghts_temp[:-1]
    )  # element lengths are equal for all rod.
    # checking velocities, omegas and rest strains
    # density and mass
    assert_allclose(rod_length, base_length, atol=1e-2)
    assert_allclose(
        mockrod.velocity_collection, np.zeros((3, n_elems)), atol=Tolerance.atol()
    )
    assert_allclose(
        mockrod.omega_collection, np.zeros((3, n_elems)), atol=Tolerance.atol()
    )
    assert_allclose(mockrod.rest_sigma, np.zeros((3, n_elems)), atol=Tolerance.atol())
    assert_allclose(mockrod.rest_kappa, np.zeros((3, n_elems)), atol=Tolerance.atol())
    assert_allclose(mockrod.density, density, atol=Tolerance.atol())
    assert_allclose(
        rest_voronoi_lengths, mockrod.rest_voronoi_lengths, atol=Tolerance.atol()
    )
    assert_allclose(mockrod.mass, mass, atol=Tolerance.atol())
    # checking directors, rest length
    # and shear, bend matrices and moment of inertia
    # Check directors, if we dont input any directors, computed ones should be valid
    correct_directors = np.zeros((MaxDimension.value(), MaxDimension.value(), n_elems))
    tangent_collection = np.zeros((MaxDimension.value(), n_elems))
    binormal_collection = np.zeros((MaxDimension.value(), n_elems))
    normal_collection = np.repeat(normal[:, np.newaxis], n_elems, axis=1)

    angle = 2 * np.pi / n_elems
    for i in range(n_elems):
        tangent_collection[..., i] = np.array(
            [
                np.cos(angle * (i + 1)) - np.cos(angle * i),
                np.sin(angle * (i + 1)) - np.sin(angle * i),
                0,
            ]
        )
        tangent_collection[..., i] /= np.linalg.norm(tangent_collection[..., i])
        binormal_collection[..., i] = np.cross(tangent_collection[..., i], normal)

    correct_directors[0, ...] = normal_collection
    correct_directors[1, ...] = binormal_collection
    correct_directors[2, ...] = tangent_collection

    assert_allclose(mockrod.rest_lengths, correct_length, atol=Tolerance.atol())

    for i in range(n_elems):
        assert_allclose(
            mockrod.shear_matrix[..., i], shear_matrix, atol=Tolerance.atol()
        )
        assert_allclose(mockrod.bend_matrix[..., i], bend_matrix, atol=Tolerance.atol())

    assert_allclose(
        mockrod.mass_second_moment_of_inertia,
        mass_second_moment_of_inertia,
        atol=Tolerance.atol(),
    )
    assert_allclose(
        mockrod.inv_mass_second_moment_of_inertia,
        inv_mass_second_moment_of_inertia,
        atol=Tolerance.atol(),
    )
