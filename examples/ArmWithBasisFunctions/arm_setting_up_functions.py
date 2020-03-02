import numpy as np
import sys

sys.path.append("../../")

# TODO: this function should be a part of rod initialization, factory function and it should be removed from here
def make_tappered_arm(
    rod,
    radius_along_rod,
    base_length,
    density,
    youngs_modulus,
    poisson_ratio,
    direction,
    normal,
    position,
    alpha_c=4.0 / 3.0,
):
    """
    This function is used to reset the rod properties for a varying radius and/or not straight rod.
    User can input a rod with varying radius. If radius varying in each element mass, mass moment  of inertia,
    shear, bend matrices and volume are different. Also user can give nodepositions as 2 dimensional array and
    this function computes corresponding directors, rest curvature, rest strain, rest lengths.
    :param rod:
    :param radius_along_rod:
    :param density:
    :param youngs_modulus:
    :param poisson_ratio:
    :param direction:
    :param normal:
    :param position:
    :param alpha_c:
    :return:
    """
    from elastica.utils import MaxDimension, Tolerance

    # Use the before hand generated rod properties
    n_elements = rod.n_elems
    rest_lengths = rod.rest_lengths

    # Compute the arm properties
    radius = radius_along_rod

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
    mass_second_moment_of_inertia_temp = I0 * density * base_length / n_elements
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
    for k in range(n_elements):
        for i in range(0, MaxDimension.value()):
            assert shear_matrix[i, i, k] > Tolerance.atol()

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
    volume = np.pi * radius_along_rod ** 2 * rest_lengths

    # Compute the mass of elements
    mass = np.zeros(n_elements + 1)
    mass[:-1] += 0.5 * density * volume
    mass[1:] += 0.5 * density * volume

    rod.radius[:] = radius_along_rod
    rod.mass_second_moment_of_inertia[:] = mass_second_moment_of_inertia
    rod.inv_mass_second_moment_of_inertia[:] = inv_mass_second_moment_of_inertia
    rod.shear_matrix[:] = shear_matrix
    rod.bend_matrix[:] = bend_matrix
    rod.volume[:] = volume
    rod.mass[:] = mass

    # Compute the tangents and directors
    position_diff = position[..., 1:] - position[..., :-1]
    lengths = np.sqrt(np.einsum("ij,ij->j", position_diff, position_diff))
    tangents = position_diff / lengths

    plane_binormals = np.cross(direction, normal)

    for k in range(n_elements):
        rod.director_collection[0, :, k] = plane_binormals
        rod.director_collection[1, :, k] = np.cross(plane_binormals, tangents[..., k])
        rod.director_collection[2, :, k] = tangents[..., k]

    rod.position_collection[:] = position

    # We have to compute
    rod._compute_shear_stretch_strains()
    rod._compute_bending_twist_strains()

    # Compute rest curvature and strains and reset the sigma and kappa
    rod.rest_kappa = rod.kappa.copy()
    rod.kappa *= 0.0
    rod.rest_sigma = rod.sigma.copy()
    rod.sigma *= 0.0


def make_two_arm_from_straigth_rod(
    rod,
    beta,
    base_length,
    direction,
    normal,
    start,
    head_n_elems,
    radius_tip,
    radius_base,
    radius_head,
):
    """
    This function is used to bend a rod and make two arms and head from the rod.
    Angle between arms and head is determined by the user input beta which is in degrees.
    This function positions and radius for three segments, which are first arm, head and
    second arm. Radius here is varying so that we can get a tappered arm.
    :param rod:
    :param beta:
    :param base_length:
    :param direction:
    :param normal:
    :param start:
    :param head_n_elems:
    :param radius_tip:
    :param radius_base:
    :param radius_head:
    :return:
    position: this is the position of nodes
    radius: for tappered arm radius is varying
    """
    from elastica.utils import MaxDimension

    n_elements = rod.n_elems

    # Compute the arm number of elements
    arm_1_n_elems = int((n_elements - head_n_elems) / 2)
    arm_2_n_elems = int((n_elements - head_n_elems) / 2)

    # Compute the radius along the rod
    s = np.linspace(
        0.0, head_n_elems / 2 * base_length / n_elements, int(head_n_elems / 2)
    )
    half_head_radius = np.tanh(s) / max(np.tanh(s)) * (radius_head) + radius_tip
    other_half_head_radius = half_head_radius[::-1]

    radius_along_rod = np.linspace(radius_tip, radius_base, arm_1_n_elems)
    radius_along_rod = np.hstack(
        (radius_along_rod, half_head_radius, other_half_head_radius)
    )
    radius_along_rod = np.hstack(
        (radius_along_rod, np.linspace(radius_tip, radius_base, arm_2_n_elems)[::-1])
    )

    # radius_along_rod = np.linspace(radius_tip, radius_base, arm_1_n_elems)
    # for i in range(head_element):
    #     radius_along_rod = np.hstack((radius_along_rod, radius_head))
    # radius_along_rod = np.hstack((radius_along_rod, np.linspace(radius_tip, radius_base, arm_2_n_elems)[::-1]))

    # beta is the angle between head elements and arm
    alpha = (90 - beta / 2) / 180 * np.pi

    d3_segment1 = np.cos(alpha) * direction + np.sin(alpha) * normal
    d3_segment1 /= np.linalg.norm(d3_segment1)

    # Set the head directors of the octopus
    d3_segment2 = direction / np.linalg.norm(direction)

    d3_segment3 = np.cos(-alpha) * direction + np.sin(-alpha) * normal
    d3_segment3 /= np.linalg.norm(d3_segment3)

    # We have to compute the correct position for arm and we have to check the the sigma, and kappa as well
    segment_number_of_elements = np.array([arm_1_n_elems, head_n_elems, arm_2_n_elems])
    start_idx_1 = 0
    end_idx_1 = start_idx_1 + arm_1_n_elems

    start_idx_2 = end_idx_1
    end_idx_2 = start_idx_2 + head_n_elems

    start_idx_3 = end_idx_2
    end_idx_3 = start_idx_3 + arm_2_n_elems

    start_idx = np.hstack((start_idx_1, start_idx_2, start_idx_3))
    end_idx = np.hstack((end_idx_1, end_idx_2, end_idx_3))

    direction_of_segments = np.vstack((d3_segment1, d3_segment2, d3_segment3))
    position = np.zeros((MaxDimension.value(), n_elements + 1))

    for k in range(segment_number_of_elements.shape[0]):
        end = (
            start
            + direction_of_segments[k, ...]
            * base_length
            / n_elements
            * segment_number_of_elements[k]
        )
        for i in range(0, MaxDimension.value()):
            position[i, start_idx[k] : end_idx[k] + 1] = np.linspace(
                start[i], end[i], num=segment_number_of_elements[k] + 1
            )
        # New segments start position should be old segments end position
        start = end

    return radius_along_rod, position
