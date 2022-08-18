__doc__ = """ Factory function to allocate variables for Cosserat Rod"""
__all__ = ["allocate"]
from typing import Optional, Tuple
import logging
import numpy as np
from numpy.testing import assert_allclose
from elastica.utils import MaxDimension, Tolerance
from elastica._linalg import _batch_cross, _batch_norm, _batch_dot


def allocate(
    n_elements,
    start,
    direction,
    normal,
    base_length,
    base_radius,
    density,
    nu,
    youngs_modulus: float,
    nu_for_torques: Optional[float] = None,
    shear_modulus: Optional[float] = None,
    position: Optional[np.ndarray] = None,
    directors: Optional[np.ndarray] = None,
    rest_sigma: Optional[np.ndarray] = None,
    rest_kappa: Optional[np.ndarray] = None,
    *args,
    **kwargs,
):
    log = logging.getLogger()

    if "poisson_ratio" in kwargs:
        # Deprecation warning for poission_ratio
        raise NameError(
            "Poisson's ratio is deprecated for Cosserat Rod for clarity. Please provide shear_modulus instead."
        )

    # sanity checks here
    assert n_elements > 1
    assert base_length > Tolerance.atol()
    assert np.sqrt(np.dot(normal, normal)) > Tolerance.atol()
    assert np.sqrt(np.dot(direction, direction)) > Tolerance.atol()

    # check if position is given.
    if position is None:  # Generate straight and uniform rod
        # Set the position array
        position = np.zeros((MaxDimension.value(), n_elements + 1))
        end = start + direction * base_length
        for i in range(0, 3):
            position[i, ...] = np.linspace(start[i], end[i], n_elements + 1)
    _position_validity_checker(position, start, n_elements)

    # Compute rest lengths and tangents
    position_diff = position[..., 1:] - position[..., :-1]
    rest_lengths = _batch_norm(position_diff)
    tangents = position_diff / rest_lengths
    normal /= np.linalg.norm(normal)

    if directors is None:  # Generate straight uniform rod
        # Set the directors matrix
        directors = np.zeros((MaxDimension.value(), MaxDimension.value(), n_elements))
        # Construct directors using tangents and normal
        normal_collection = np.repeat(normal[:, np.newaxis], n_elements, axis=1)
        # Check if rod normal and rod tangent are perpendicular to each other otherwise
        # directors will be wrong!!
        assert_allclose(
            _batch_dot(normal_collection, tangents),
            0,
            atol=Tolerance.atol(),
            err_msg=(" Rod normal and tangent are not perpendicular to each other!"),
        )
        directors[0, ...] = normal_collection
        directors[1, ...] = _batch_cross(tangents, normal_collection)
        directors[2, ...] = tangents
    _directors_validity_checker(directors, tangents, n_elements)

    # Set radius array
    radius = np.zeros((n_elements))
    # Check if the user input radius is valid
    radius_temp = np.array(base_radius)
    _assert_dim(radius_temp, 2, "radius")
    radius[:] = radius_temp
    # Check if the elements of radius are greater than tolerance
    assert np.all(radius > Tolerance.atol()), " Radius has to be greater than 0."

    # Set density array
    density_array = np.zeros((n_elements))
    # Check if the user input density is valid
    density_temp = np.array(density)
    _assert_dim(density_temp, 2, "density")
    density_array[:] = density_temp
    # Check if the elements of density are greater than tolerance
    assert np.all(
        density_array > Tolerance.atol()
    ), " Density has to be greater than 0."

    # Second moment of inertia
    A0 = np.pi * radius * radius
    I0_1 = A0 * A0 / (4.0 * np.pi)
    I0_2 = I0_1
    I0_3 = 2.0 * I0_2
    I0 = np.array([I0_1, I0_2, I0_3]).transpose()
    # Mass second moment of inertia for disk cross-section
    mass_second_moment_of_inertia = np.zeros(
        (MaxDimension.value(), MaxDimension.value(), n_elements), np.float64
    )

    mass_second_moment_of_inertia_temp = np.einsum(
        "ij,i->ij", I0, density * rest_lengths
    )

    for i in range(n_elements):
        np.fill_diagonal(
            mass_second_moment_of_inertia[..., i],
            mass_second_moment_of_inertia_temp[i, :],
        )
    # sanity check of mass second moment of inertia
    if (mass_second_moment_of_inertia < Tolerance.atol()).all():
        message = "Mass moment of inertia matrix smaller than tolerance, please check provided radius, density and length."
        log.warning(message)

    # Inverse of second moment of inertia
    inv_mass_second_moment_of_inertia = np.zeros(
        (MaxDimension.value(), MaxDimension.value(), n_elements)
    )
    for i in range(n_elements):
        # Check rank of mass moment of inertia matrix to see if it is invertible
        assert (
            np.linalg.matrix_rank(mass_second_moment_of_inertia[..., i])
            == MaxDimension.value()
        )
        inv_mass_second_moment_of_inertia[..., i] = np.linalg.inv(
            mass_second_moment_of_inertia[..., i]
        )

    # Shear/Stretch matrix
    if not shear_modulus:
        log.info(
            """Shear modulus is not explicitly given.\n
            In such case, we compute shear_modulus assuming poisson's ratio of 0.5"""
        )
        shear_modulus = youngs_modulus / (2.0 * (1.0 + 0.5))

    # Value taken based on best correlation for Poisson ratio = 0.5, from
    # "On Timoshenko's correction for shear in vibrating beams" by Kaneko, 1975
    alpha_c = 27.0 / 28.0
    shear_matrix = np.zeros(
        (MaxDimension.value(), MaxDimension.value(), n_elements), np.float64
    )
    for i in range(n_elements):
        np.fill_diagonal(
            shear_matrix[..., i],
            [
                alpha_c * shear_modulus * A0[i],
                alpha_c * shear_modulus * A0[i],
                youngs_modulus * A0[i],
            ],
        )

    # Bend/Twist matrix
    bend_matrix = np.zeros(
        (MaxDimension.value(), MaxDimension.value(), n_elements), np.float64
    )
    for i in range(n_elements):
        np.fill_diagonal(
            bend_matrix[..., i],
            [
                youngs_modulus * I0_1[i],
                youngs_modulus * I0_2[i],
                shear_modulus * I0_3[i],
            ],
        )
    for i in range(0, MaxDimension.value()):
        assert np.all(
            bend_matrix[i, i, :] > Tolerance.atol()
        ), " Bend matrix has to be greater than 0."

    # Compute bend matrix in Voronoi Domain
    bend_matrix = (
        bend_matrix[..., 1:] * rest_lengths[1:]
        + bend_matrix[..., :-1] * rest_lengths[0:-1]
    ) / (rest_lengths[1:] + rest_lengths[:-1])

    # Compute volume of elements
    volume = np.pi * radius ** 2 * rest_lengths

    # Compute mass of elements
    mass = np.zeros(n_elements + 1)
    mass[:-1] += 0.5 * density * volume
    mass[1:] += 0.5 * density * volume

    # Set dissipation constant or nu array
    log.warning(
        # Remove warning and add error if nu provided in v0.3.1
        # Remove the option to set internal nu inside, beyond v0.4.0
        "The option to set damping coefficient (nu) for the rod during rod "
        "initialisation is now deprecated. Instead, for adding damping to rods, "
        "please derive your simulation class from the add-on Damping mixin class. "
        "For reference see the class elastica.dissipation.AnalyticalLinearDamper(), "
        "and for usage check examples/axial_stretching.py "
        "The option to set damping coefficient (nu) during rod construction "
        "will be removed in the future (v0.3.1)."
    )
    dissipation_constant_for_forces = np.zeros((n_elements + 1))
    # Check if the user input nu is valid
    nu_temp = np.array(nu)
    _assert_dim(nu_temp, 2, "dissipation constant (nu) for forces)")
    dissipation_constant_for_forces[:] = nu * mass
    # Check if the elements of dissipation constant greater than tolerance
    assert np.all(
        dissipation_constant_for_forces >= 0.0
    ), " Dissipation constant(nu) has to be equal or greater than 0."

    # Custom nu for torques
    dissipation_constant_for_torques = np.zeros((n_elements))

    if nu_for_torques is None:
        dissipation_constant_for_torques[:] = (
            dissipation_constant_for_forces[1:] + dissipation_constant_for_forces[:-1]
        ) / 2
        # Treat the end elements carefully, since end nodes only have one neighboring element we need to add all of
        # their dissipation.
        dissipation_constant_for_torques[0] += (dissipation_constant_for_forces[0]) / 2
        dissipation_constant_for_torques[-1] += (
            dissipation_constant_for_forces[-1]
        ) / 2
    else:
        elemental_mass = (mass[1:] + mass[:-1]) / 2.0
        # Treat the end elements carefully, since end nodes only have one neighboring element we need to add all of
        # their mass.
        elemental_mass[0] += mass[0] / 2.0
        elemental_mass[-1] += mass[-1] / 2.0
        dissipation_constant_for_torques[:] = (
            np.asarray(nu_for_torques) * elemental_mass
        )
    _assert_dim(
        dissipation_constant_for_torques, 2, "dissipation constant (nu) for torque)"
    )

    # Generate rest sigma and rest kappa, use user input if defined
    # set rest strains and curvature to be  zero at start
    # if found in kwargs modify (say for curved rod)
    if rest_sigma is None:
        rest_sigma = np.zeros((MaxDimension.value(), n_elements))
    _assert_shape(rest_sigma, (MaxDimension.value(), n_elements), "rest_sigma")

    if rest_kappa is None:
        rest_kappa = np.zeros((MaxDimension.value(), n_elements - 1))
    _assert_shape(rest_kappa, (MaxDimension.value(), n_elements - 1), "rest_kappa")

    # Compute rest voronoi length
    rest_voronoi_lengths = 0.5 * (rest_lengths[1:] + rest_lengths[:-1])

    # Allocate arrays for Cosserat Rod equations
    velocities = np.zeros((MaxDimension.value(), n_elements + 1))
    omegas = np.zeros((MaxDimension.value(), n_elements))
    accelerations = 0.0 * velocities
    angular_accelerations = 0.0 * omegas
    # _vector_states = np.hstack(
    #     (position, velocities, omegas, accelerations, angular_accelerations)
    # )
    # _matrix_states = directors.copy()

    internal_forces = 0.0 * accelerations
    internal_torques = 0.0 * angular_accelerations

    external_forces = 0.0 * accelerations
    external_torques = 0.0 * angular_accelerations

    lengths = np.zeros((n_elements))
    tangents = np.zeros((3, n_elements))

    dilatation = np.zeros((n_elements))
    voronoi_dilatation = np.zeros((n_elements - 1))
    dilatation_rate = np.zeros((n_elements))

    sigma = np.zeros((3, n_elements))
    kappa = np.zeros((3, n_elements - 1))

    internal_stress = np.zeros((3, n_elements))
    internal_couple = np.zeros((3, n_elements - 1))

    damping_forces = np.zeros((3, n_elements + 1))
    damping_torques = np.zeros((3, n_elements))

    return (
        n_elements,
        position,
        velocities,
        omegas,
        accelerations,
        angular_accelerations,
        directors,
        radius,
        mass_second_moment_of_inertia,
        inv_mass_second_moment_of_inertia,
        shear_matrix,
        bend_matrix,
        density_array,
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


def _assert_dim(vector, max_dim: int, name: str):
    assert vector.ndim < max_dim, (
        f"Input {name} dimension is not correct {vector.shape}"
        + f" It should be maximum {max_dim}D vector or single floating number."
    )


def _assert_shape(array: np.ndarray, expected_shape: Tuple[int], name: str):
    assert array.shape == expected_shape, (
        f"Given {name} shape is not correct, it should be "
        + str(expected_shape)
        + " but instead "
        + str(array.shape)
    )


def _position_validity_checker(position, start, n_elements):
    """Checker on user-defined position validity"""
    _assert_shape(position, (MaxDimension.value(), n_elements + 1), "position")

    # Check if the start position of the rod and first entry of position array are the same
    assert_allclose(
        position[..., 0],
        start,
        atol=Tolerance.atol(),
        err_msg=str(
            "First entry of position" + " (" + str(position[..., 0]) + " ) "
            " is different than start " + " (" + str(start) + " ) "
        ),
    )


def _directors_validity_checker(directors, tangents, n_elements):
    """Checker on user-defined directors validity"""
    _assert_shape(
        directors, (MaxDimension.value(), MaxDimension.value(), n_elements), "directors"
    )

    # Check if d1, d2, d3 are unit vectors
    d1 = directors[0, ...]
    d2 = directors[1, ...]
    d3 = directors[2, ...]
    assert_allclose(
        _batch_norm(d1),
        np.ones((n_elements)),
        atol=Tolerance.atol(),
        err_msg=(" d1 vector of input director matrix is not unit vector "),
    )
    assert_allclose(
        _batch_norm(d2),
        np.ones((n_elements)),
        atol=Tolerance.atol(),
        err_msg=(" d2 vector of input director matrix is not unit vector "),
    )
    assert_allclose(
        _batch_norm(d3),
        np.ones((n_elements)),
        atol=Tolerance.atol(),
        err_msg=(" d3 vector of input director matrix is not unit vector "),
    )

    # Check if d3xd1 = d2
    assert_allclose(
        _batch_cross(d3, d1),
        d2,
        atol=Tolerance.atol(),
        err_msg=(" d3 x d1 != d2 of input director matrix"),
    )

    # Check if computed tangents from position is the same with d3
    assert_allclose(
        tangents,
        d3,
        atol=Tolerance.atol(),
        err_msg=" Tangent vector computed using node positions is different than d3 vector of input directors",
    )
