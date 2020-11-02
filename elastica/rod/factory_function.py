__doc__ = """ Factory function to allocate variables for Cosserat Rod"""
__all__ = ["allocate"]
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
    youngs_modulus,
    poisson_ratio,
    alpha_c=4.0 / 3.0,
    *args,
    **kwargs
):

    # sanity checks here
    assert n_elements > 1
    assert base_length > Tolerance.atol()
    assert np.sqrt(np.dot(normal, normal)) > Tolerance.atol()
    assert np.sqrt(np.dot(direction, direction)) > Tolerance.atol()

    # Set the position array
    position = np.zeros((MaxDimension.value(), n_elements + 1))
    # check if position is in kwargs, if it is use user defined position otherwise generate position
    if kwargs.__contains__("position"):
        position_temp = np.array(kwargs["position"])

        # Check the shape of the input position
        assert position_temp.shape == (MaxDimension.value(), n_elements + 1), (
            "Given position  shape is not correct, it should be "
            + str(position.shape)
            + " but instead "
            + str(position_temp.shape)
        )
        # Check if the start position of the rod and first entry of position array are the same
        assert_allclose(
            position_temp[..., 0],
            start,
            atol=Tolerance.atol(),
            err_msg=str(
                "First entry of position" + " (" + str(position_temp[..., 0]) + " ) "
                " is different than start " + " (" + str(start) + " ) "
            ),
        )
        position = position_temp.copy()

    else:
        end = start + direction * base_length
        for i in range(0, 3):
            position[i, ...] = np.linspace(start[i], end[i], n_elements + 1)

    # Compute rest lengths and tangents
    position_diff = position[..., 1:] - position[..., :-1]
    rest_lengths = _batch_norm(position_diff)
    tangents = position_diff / rest_lengths
    normal /= np.linalg.norm(normal)

    # Set the directors matrix
    directors = np.zeros((MaxDimension.value(), MaxDimension.value(), n_elements))
    # check if directors is in kwargs, if it use user defined directors otherwise generate directors
    if kwargs.__contains__("directors"):
        directors_temp = np.array(kwargs["directors"])

        # Check the shape of input directors
        assert directors_temp.shape == (
            MaxDimension.value(),
            MaxDimension.value(),
            n_elements,
        ), (
            " Given directors shape is not correct, it should be "
            + str(directors.shape)
            + " but instead "
            + str(directors_temp.shape)
        )

        # Check if d1, d2, d3 are unit vectors
        d1 = directors_temp[0, ...]
        d2 = directors_temp[1, ...]
        d3 = directors_temp[2, ...]
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

    else:
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

    # Set radius array
    radius = np.zeros((n_elements))
    # Check if the user input radius is valid
    radius_temp = np.array(base_radius)
    assert radius_temp.ndim < 2, (
        "Input radius shape is not correct "
        + str(radius_temp.shape)
        + " It should be "
        + str(radius.shape)
        + " or  single floating number "
    )
    radius[:] = radius_temp
    # Check if the elements of radius are greater than tolerance
    for k in range(n_elements):
        assert radius[k] > Tolerance.atol(), (
            " Radius has to be greater than 0" + " Check you radius input!"
        )

    # Set density array
    density_array = np.zeros((n_elements))
    # Check if the user input density is valid
    density_temp = np.array(density)
    assert density_temp.ndim < 2, (
        "Input density shape is not correct "
        + str(density_temp.shape)
        + " It should be "
        + str(density_array.shape)
        + " or  single floating number "
    )
    density_array[:] = density_temp
    # Check if the elements of density are greater than tolerance
    for k in range(n_elements):
        assert density_array[k] > Tolerance.atol(), (
            " Density has to be greater than 0" + " Check you density input!"
        )

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
    for k in range(n_elements):
        for i in range(0, MaxDimension.value()):
            assert mass_second_moment_of_inertia[i, i, k] > Tolerance.atol()

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
    shear_modulus = youngs_modulus / (poisson_ratio + 1.0)
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
    for k in range(n_elements):
        for i in range(0, MaxDimension.value()):
            assert bend_matrix[i, i, k] > Tolerance.atol()
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
    dissipation_constant_for_forces = np.zeros((n_elements))
    # Check if the user input nu is valid
    nu_temp = np.array(nu)
    assert nu_temp.ndim < 2, (
        "Input dissipation constant(nu) for forces shape is not correct "
        + str(nu_temp.shape)
        + " It should be "
        + str(dissipation_constant_for_forces.shape)
        + " or  single floating number "
    )
    dissipation_constant_for_forces[:] = nu
    # Check if the elements of dissipation constant greater than tolerance
    for k in range(n_elements):
        assert dissipation_constant_for_forces[k] >= 0.0, (
            " Dissipation constant has to be equal or greater than 0 "
            + " Check your dissipation constant(nu) input!"
        )

    dissipation_constant_for_torques = np.zeros((n_elements))
    if kwargs.__contains__("nu_for_torques"):
        temp_nu_for_torques = np.array(kwargs["nu_for_torques"])
        assert temp_nu_for_torques.ndim < 2, (
            "Input dissipation constant(nu) for torques shape is not correct "
            + str(temp_nu_for_torques.shape)
            + " It should be "
            + str(dissipation_constant_for_torques.shape)
            + " or  single floating number "
        )
        dissipation_constant_for_torques[:] = temp_nu_for_torques

    else:
        dissipation_constant_for_torques[:] = dissipation_constant_for_forces

    # Generate rest sigma and rest kappa, use user input if defined
    # set rest strains and curvature to be  zero at start
    # if found in kwargs modify (say for curved rod)
    rest_sigma = np.zeros((MaxDimension.value(), n_elements))
    if kwargs.__contains__("rest_sigma"):
        temp_rest_sigma = np.array(kwargs["rest_sigma"])
        assert temp_rest_sigma.shape == rest_sigma.shape, (
            "Input rest sigma shape is not correct "
            + str(temp_rest_sigma.shape)
            + " It should be "
            + str(rest_sigma.shape)
        )
        rest_sigma[:] = temp_rest_sigma

    rest_kappa = np.zeros((MaxDimension.value(), n_elements - 1))
    if kwargs.__contains__("rest_kappa"):
        temp_rest_kappa = np.array(kwargs["rest_kappa"])
        assert temp_rest_kappa.shape == rest_kappa.shape, (
            "Input rest kappa shape is not correct "
            + str(temp_rest_kappa.shape)
            + " It should be "
            + str(rest_kappa.shape)
        )
        rest_kappa[:] = temp_rest_kappa

    # Compute rest voronoi length
    rest_voronoi_lengths = 0.5 * (rest_lengths[1:] + rest_lengths[:-1])

    # Allocate arrays for Cosserat Rod equations
    velocities = np.zeros((MaxDimension.value(), n_elements + 1))
    omegas = np.zeros((MaxDimension.value(), n_elements))
    accelerations = 0.0 * velocities
    angular_accelerations = 0.0 * omegas
    _vector_states = np.hstack(
        (position, velocities, omegas, accelerations, angular_accelerations)
    )
    _matrix_states = directors.copy()

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
        _vector_states,
        _matrix_states,
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
