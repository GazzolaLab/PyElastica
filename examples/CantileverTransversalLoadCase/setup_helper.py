import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from tqdm import tqdm

pass
import logging
from elastica.utils import MaxDimension, Tolerance


def adjust_square_cross_section(
    rod, youngs_modulus: float, length: float, ring_rod_flag: bool = False
):
    n_elements = rod.n_elems
    n_voronoi_elements = n_elements if ring_rod_flag else n_elements - 1

    log = logging.getLogger()

    side_length = np.zeros(n_elements)
    side_length.fill(length)

    new_area = np.pi * rod.radius * rod.radius

    new_moi_1 = (side_length**4) / 12
    new_moi_2 = (side_length**4) / 12
    new_moi_3 = new_moi_2 * 2

    new_moi = np.array([new_moi_1, new_moi_2, new_moi_3]).transpose()

    mass_second_moment_of_inertia_temp = np.einsum(
        "ij,i->ij", new_moi, rod.density * rod.rest_lengths
    )

    for i in range(n_elements):
        np.fill_diagonal(
            rod.mass_second_moment_of_inertia[..., i],
            mass_second_moment_of_inertia_temp[i, :],
        )
    # sanity check of mass second moment of inertia
    if (rod.mass_second_moment_of_inertia < Tolerance.atol()).all():
        message = "Mass moment of inertia matrix smaller than tolerance, please check provided radius, density and length."
        log.warning(message)

    for i in range(n_elements):
        # Check rank of mass moment of inertia matrix to see if it is invertible
        assert (
            np.linalg.matrix_rank(rod.mass_second_moment_of_inertia[..., i])
            == MaxDimension.value()
        )
        rod.inv_mass_second_moment_of_inertia[..., i] = np.linalg.inv(
            rod.mass_second_moment_of_inertia[..., i]
        )

    # Shear/Stretch matrix
    shear_modulus = youngs_modulus / (2.0 * (1.0 + 0.5))

    # Value taken based on best correlation for Poisson ratio = 0.5, from
    # "On Timoshenko's correction for shear in vibrating beams" by Kaneko, 1975
    alpha_c = 27.0 / 28.0
    rod.shear_matrix *= 0.0
    for i in range(n_elements):
        np.fill_diagonal(
            rod.shear_matrix[..., i],
            [
                alpha_c * shear_modulus * new_area[i],
                alpha_c * shear_modulus * new_area[i],
                youngs_modulus * new_area[i],
            ],
        )

    # Bend/Twist matrix
    bend_matrix = np.zeros(
        (MaxDimension.value(), MaxDimension.value(), n_voronoi_elements + 1), np.float64
    )
    for i in range(n_elements):
        np.fill_diagonal(
            bend_matrix[..., i],
            [
                youngs_modulus * new_moi_1[i],
                youngs_modulus * new_moi_2[i],
                shear_modulus * new_moi_3[i],
            ],
        )
    if ring_rod_flag:  # wrap around the value in the last element
        bend_matrix[..., -1] = bend_matrix[..., 0]
    for i in range(0, MaxDimension.value()):
        assert np.all(
            bend_matrix[i, i, :] > Tolerance.atol()
        ), " Bend matrix has to be greater than 0."

    # Compute bend matrix in Voronoi Domain
    rest_lengths_temp_for_voronoi = (
        np.hstack((rod.rest_lengths, rod.rest_lengths[0]))
        if ring_rod_flag
        else rod.rest_lengths
    )
    rod.bend_matrix = (
        bend_matrix[..., 1:] * rest_lengths_temp_for_voronoi[1:]
        + bend_matrix[..., :-1] * rest_lengths_temp_for_voronoi[0:-1]
    ) / (rest_lengths_temp_for_voronoi[1:] + rest_lengths_temp_for_voronoi[:-1])
