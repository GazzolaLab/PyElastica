"""
This script is tutorial for pyelastica restart functionality.
"""

import sys

sys.path.append("../../")
from elastica import *

class RestartExampleSimulator(BaseSystemCollection, Constraints, Forcing, CallBacks):
    pass


restart_example_simulator = RestartExampleSimulator()

# setting up test params
final_time = 200

n_elem = 50
start = np.zeros((3,))
direction = np.array([0.0, 0.0, 1.0])
normal = np.array([0.0, 1.0, 0.0])
base_length = 3.0
base_radius = 0.25
base_area = np.pi * base_radius ** 2
density = 5000
nu = 10
E = 1e6
# For shear modulus of 1e4, nu is 99!
poisson_ratio = 99
shear_modulus = E / (poisson_ratio + 1.0)

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
    shear_modulus=shear_modulus,
)

restart_example_simulator.append(shearable_rod)

restart_example_simulator.append(shearable_rod)
restart_example_simulator.constrain(shearable_rod).using(
    OneEndFixedRod, constrained_position_idx=(0,), constrained_director_idx=(0,)
)

end_force = np.array([-15.0, 0.0, 0.0])
restart_example_simulator.add_forcing_to(shearable_rod).using(
    EndpointForces, 0.0 * end_force, end_force, ramp_up_time=final_time / 2.0
)

# Finalize simulation
restart_example_simulator.finalize()

# After finalize you can load restart file.
LOAD_FROM_RESTART =  False
SAVE_DATA_RESTART = True
restart_file_location = "data/"

if LOAD_FROM_RESTART:
    restart_time = load_state(restart_example_simulator, restart_file_location, True)
else:
    restart_time = np.float64(0.0)

timestepper = PositionVerlet()
dl = base_length / n_elem
dt = 0.01 * dl
total_steps = int(final_time / dt)

time = integrate(timestepper, restart_example_simulator, final_time, total_steps, restart_time=restart_time)

if SAVE_DATA_RESTART:
    save_state(restart_example_simulator, restart_file_location, time, True)

