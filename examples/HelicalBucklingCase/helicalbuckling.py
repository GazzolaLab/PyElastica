__doc__ = """Helical buckling validation case, for detailed explanation refer to
Gazzola et. al. R. Soc. 2018  section 3.4.1 """

import numpy as np
import elastica as ea
from examples.HelicalBucklingCase.helicalbuckling_postprocessing import (
    plot_helicalbuckling,
)


class HelicalBucklingSimulator(
    ea.BaseSystemCollection, ea.Constraints, ea.Damping, ea.Forcing
):
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
nu = 0.01 / density / base_area
E = 1e6
slack = 3
number_of_rotations = 27
# For shear modulus of 1e5, nu is 99!
poisson_ratio = 9
shear_modulus = E / (poisson_ratio + 1.0)
shear_matrix = np.repeat(
    shear_modulus * np.identity((3))[:, :, np.newaxis], n_elem, axis=2
)
temp_bend_matrix = np.zeros((3, 3))
np.fill_diagonal(temp_bend_matrix, [1.345, 1.345, 0.789])
bend_matrix = np.repeat(temp_bend_matrix[:, :, np.newaxis], n_elem - 1, axis=2)

shearable_rod = ea.CosseratRod.straight_rod(
    n_elem,
    start,
    direction,
    normal,
    base_length,
    base_radius,
    density,
    youngs_modulus=E,
    shear_modulus=shear_modulus,
)
# TODO: CosseratRod has to be able to take shear matrix as input, we should change it as done below

shearable_rod.shear_matrix = shear_matrix
shearable_rod.bend_matrix = bend_matrix


helicalbuckling_sim.append(shearable_rod)
# add damping
dl = base_length / n_elem
dt = 1e-3 * dl
helicalbuckling_sim.dampen(shearable_rod).using(
    ea.AnalyticalLinearDamper,
    damping_constant=nu,
    time_step=dt,
)

helicalbuckling_sim.constrain(shearable_rod).using(
    ea.HelicalBucklingBC,
    constrained_position_idx=(0, -1),
    constrained_director_idx=(0, -1),
    twisting_time=500,
    slack=slack,
    number_of_rotations=number_of_rotations,
)

helicalbuckling_sim.finalize()
timestepper = ea.PositionVerlet()
shearable_rod.velocity_collection[..., int((n_elem) / 2)] += np.array([0, 1e-6, 0.0])
# timestepper = PEFRL()

final_time = 10500.0
total_steps = int(final_time / dt)
print("Total steps", total_steps)
ea.integrate(timestepper, helicalbuckling_sim, final_time, total_steps)

if PLOT_FIGURE:
    plot_helicalbuckling(shearable_rod, SAVE_FIGURE)

if SAVE_RESULTS:
    import pickle

    filename = "HelicalBuckling_data.dat"
    file = open(filename, "wb")
    pickle.dump(shearable_rod, file)
    file.close()
