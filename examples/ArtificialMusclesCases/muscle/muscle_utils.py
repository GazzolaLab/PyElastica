"""Utilty functions for muscle construction"""

import numpy as np
from numba import njit


@njit(cache=True)
def helix_arclength(k, r, helix_length):
    return helix_length * np.sqrt((2 * np.pi * r * k) ** 2 + 1)


@njit(cache=True)
def helix_length_per_coil(k, r):
    return np.sqrt((2 * np.pi * r * k) ** 2 + 1) / k


@njit(cache=True)
def conical_helix_length(start_radius, cone_slope, height, n_turns_per_length):
    k = 2 * np.pi * n_turns_per_length
    m = cone_slope
    end_radius = start_radius + m * height
    a = k * start_radius / np.sqrt((m ** 2) + 1)
    b = k * end_radius / np.sqrt((m ** 2) + 1)
    length_at_a = (a * np.sqrt(a ** 2 + 1) + np.arcsinh(a)) / 2
    length_at_b = (b * np.sqrt(b ** 2 + 1) + np.arcsinh(b)) / 2
    length = (m ** 2 + 1) * (length_at_b - length_at_a) / (m * k)
    return length


@njit(cache=True)
def gamma_func(temp, youngs_modulus_coefficients, room_temperature):
    E_initial = 0
    E_current = 0
    i = 0
    for coeff in youngs_modulus_coefficients:
        E_initial += coeff * (room_temperature ** i)
        E_current += coeff * (temp ** i)
        i += 1
    E_correction = 1  # +((temp-room_temperature)/950)*2
    gamma = E_current * E_correction / E_initial
    return gamma
