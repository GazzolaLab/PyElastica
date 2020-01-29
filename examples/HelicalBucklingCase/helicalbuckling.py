import numpy as np

# FIXME without appending sys.path make it more generic
import sys

sys.path.append("../")

from elastica.wrappers import BaseSystemCollection, Constraints, Forcing
from elastica.rod.cosserat_rod import CosseratRod
from elastica.boundary_conditions import HelicalBucklingBC
from elastica.timestepper.symplectic_steppers import PositionVerlet, PEFRL
from elastica.timestepper import integrate
from examples.HelicalBucklingCase.helicalbuckling_postprocessing import (
    plot_helicalbuckling,
)


class HelicalBucklingSimulator(BaseSystemCollection, Constraints, Forcing):
    pass


helicalbuckling_sim = HelicalBucklingSimulator()

# Options
PLOT_FIGURE = True
SAVE_FIGURE = True
SAVE_RESULTS = False

# setting up test params
n_elem = 100
start = np.zeros((3,))
direction = np.array([0.0, 0.0, 1.0])
normal = np.array([0.0, 1.0, 0.0])
base_length = 100.0
base_radius = 0.35
base_area = np.pi * base_radius ** 2
density = 1.0 / (base_area)
nu = 0.01
E = 1e6
slack = 3
number_of_rotations = 27
# For shear modulus of 1e4, nu is 99!
poisson_ratio = 99
shear_matrix = np.repeat(1e5 * np.identity((3))[:, :, np.newaxis], n_elem, axis=2)
temp_bend_matrix = np.zeros((3, 3))
np.fill_diagonal(temp_bend_matrix, [1.345, 1.345, 0.789])
bend_matrix = np.repeat(temp_bend_matrix[:, :, np.newaxis], n_elem - 1, axis=2)

shearable_rod = CosseratRod.straight_rod(
    n_elem,
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
# TODO: CosseratRod has to be able to take shear matrix as input, we should change it as done below

shearable_rod.shear_matrix = shear_matrix
shearable_rod.bend_matrix = bend_matrix


helicalbuckling_sim.append(shearable_rod)
helicalbuckling_sim.constrain(shearable_rod).using(
    HelicalBucklingBC,
    positions=(0, -1),
    directors=(0, -1),
    twisting_time=500,
    slack=slack,
    number_of_rotations=number_of_rotations,
)

helicalbuckling_sim.finalize()
timestepper = PositionVerlet()
shearable_rod.velocity_collection[..., int((n_elem) / 2)] += np.array([0, 1e-6, 0.0])
# # timestepper = PEFRL()

final_time = 10500.0
dl = base_length / n_elem
dt = 1e-3 * dl
total_steps = int(final_time / dt)
print("Total steps", total_steps)
# FIXME: remove integrate outputs, we have call back functions now, we dont need them.
positions_over_time, directors_over_time, velocities_over_time = integrate(
    timestepper, helicalbuckling_sim, final_time, total_steps
)

if PLOT_FIGURE:
    plot_helicalbuckling(shearable_rod, SAVE_FIGURE)

if SAVE_RESULTS:
    import pickle

    filename = "HelicalBuckling_data.dat"
    file = open(filename, "wb")
    pickle.dump(shearable_rod, file)
    file.close()
