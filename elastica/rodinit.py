__doc__ = """ Rod initialiser"""

import numpy as np

from ._rod import *
from ._linalg import _batch_matmul, _batch_matvec, _batch_cross


# for now writing only one function for uniform straight rod
# with no strains or curvature
# later need to think for a class version for different rod inits
def create_straight_rod(n, start, direction, normal, base_length, base_radius, density,
                        mass_second_moment_of_inertia, shear_matrix, bend_matrix):
    # n: number of elements
    # put asserts and sanity checks here
    end = start + direction * base_length
    position = np.zeros((3, n + 1))
    for i in range(0, 3):
        position[..., i] = np.linspace(start[i], end[i], num=n + 1)

    # set initial velocity and omega to be zero
    velocity = np.zeros((3, n + 1))
    omega = np.zeros((3, n))

    # compute rest lengths and tangents
    position_diff = position[..., 1:] - position[..., :-1]
    rest_lengths = np.sqrt(np.einsum("ij,ij->j", position_diff, position_diff))
    tangents = position_diff / rest_lengths

    # set directors
    # check this order once
    directors = np.zeros((3, 3, n))
    normal_collection = np.broadcast_to(normal, (n, 3)).T
    directors[0, ...] = normal_collection
    directors[1, ...] = tangents
    directors[2, ...] = _batch_cross(tangents, normal_collection)

    # compute mass
    mass = density * np.pi * (base_radius ** 2) * rest_lengths

    # set initial strain and curvature to be zero
    rest_sigma = np.zeros((3, n))
    rest_kappa = np.zeros((3, n - 1))

    # initialise moment of inertia, shear and bend matrices
    inertia_collection = np.broadcast_to(mass_second_moment_of_inertia, (n, 3, 3)).reshape(3, 3, n)
    shear_matrix_collection = np.broadcast_to(shear_matrix, (n, 3, 3)).reshape(3, 3, n)
    bend_matrix_collection = np.broadcast_to(bend_matrix, (n - 1, 3, 3)).reshape(3, 3, n - 1)

    # create rod
    rod = CosseratRod(position, velocity, omega, directors, rest_lengths, mass, density,
                      inertia_collection, rest_sigma, rest_kappa, shear_matrix_collection,
                      bend_matrix_collection)
    return rod
