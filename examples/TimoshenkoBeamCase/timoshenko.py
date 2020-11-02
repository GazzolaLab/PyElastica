__doc__ = """Timoshenko beam validation case, for detailed explanation refer to 
Gazzola et. al. R. Soc. 2018  section 3.4.3 """

import numpy as np
import sys

# FIXME without appending sys.path make it more generic
sys.path.append("../../")
from elastica import *
from examples.TimoshenkoBeamCase.timoshenko_postprocessing import plot_timoshenko


class TimoshenkoBeamSimulator(BaseSystemCollection, Constraints, Forcing):
    pass


timoshenko_sim = TimoshenkoBeamSimulator()
final_time = 5000

# Options
PLOT_FIGURE = True
SAVE_FIGURE = False
SAVE_RESULTS = False
ADD_UNSHEARABLE_ROD = False

# setting up test params
n_elem = 100
start = np.zeros((3,))
direction = np.array([0.0, 0.0, 1.0])
normal = np.array([0.0, 1.0, 0.0])
base_length = 3.0
base_radius = 0.25
base_area = np.pi * base_radius ** 2
density = 5000
nu = 0.1
E = 1e6
# For shear modulus of 1e4, nu is 99!
poisson_ratio = 99

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

timoshenko_sim.append(shearable_rod)
timoshenko_sim.constrain(shearable_rod).using(
    OneEndFixedRod, constrained_position_idx=(0,), constrained_director_idx=(0,)
)

end_force = np.array([-15.0, 0.0, 0.0])
timoshenko_sim.add_forcing_to(shearable_rod).using(
    EndpointForces, 0.0 * end_force, end_force, ramp_up_time=final_time / 2.0
)


if ADD_UNSHEARABLE_ROD:
    # Start into the plane
    unshearable_start = np.array([0.0, -1.0, 0.0])
    unshearable_rod = CosseratRod.straight_rod(
        n_elem,
        unshearable_start,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        nu,
        E,
        # Unshearable rod needs G -> inf, which is achievable with -ve poisson ratio
        poisson_ratio=-0.7,
    )

    timoshenko_sim.append(unshearable_rod)
    timoshenko_sim.constrain(unshearable_rod).using(
        OneEndFixedRod, constrained_position_idx=(0,), constrained_director_idx=(0,)
    )
    timoshenko_sim.add_forcing_to(unshearable_rod).using(
        EndpointForces, 0.0 * end_force, end_force, ramp_up_time=final_time / 2.0
    )

timoshenko_sim.finalize()
timestepper = PositionVerlet()
# timestepper = PEFRL()

dl = base_length / n_elem
dt = 0.01 * dl
total_steps = int(final_time / dt)
print("Total steps", total_steps)
integrate(timestepper, timoshenko_sim, final_time, total_steps)

if PLOT_FIGURE:
    plot_timoshenko(shearable_rod, end_force, SAVE_FIGURE, ADD_UNSHEARABLE_ROD)

if SAVE_RESULTS:
    import pickle

    filename = "Timoshenko_beam_data.dat"
    file = open(filename, "wb")
    pickle.dump(shearable_rod, file)
    file.close()
