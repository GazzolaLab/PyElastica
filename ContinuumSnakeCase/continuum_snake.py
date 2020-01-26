import numpy as np

# FIXME without appending sys.path make it more generic
import sys

sys.path.append("../")

from elastica.wrappers import (
    BaseSystemCollection,
    Connections,
    Constraints,
    Forcing,
    CallBacks,
)
from elastica.rod.cosserat_rod import CosseratRod
from elastica.boundary_conditions import FreeRod
from elastica.external_forces import GravityForces, MuscleTorques
from elastica.interaction import AnistropicFrictionalPlane
from elastica.callback_functions import ContinuumSnakeCallBack
from elastica.timestepper.symplectic_steppers import PositionVerlet, PEFRL
from elastica.timestepper import integrate
from ContinuumSnakeCase.continuum_snake_postprocessing import (
    plot_snake_velocity,
    plot_video,
)


class SnakeSimulator(BaseSystemCollection, Constraints, Forcing, CallBacks):
    pass


snake_sim = SnakeSimulator()


# Options
PLOT_FIGURE = True
SAVE_FIGURE = False
SAVE_VIDEO = False
SAVE_RESULTS = False


# setting up test params
n_elem = 50
start = np.zeros((3,))
direction = np.array([0.0, 0.0, 1.0])
normal = np.array([0.0, 1.0, 0.0])
base_length = 1.0
base_radius = 0.025
base_area = np.pi * base_radius ** 2
density = 1000
nu = 5.0
E = 1e7
poisson_ratio = 0.5

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

snake_sim.append(shearable_rod)
snake_sim.constrain(shearable_rod).using(FreeRod)

# Add gravitational forces
gravitational_acc = -9.80665
snake_sim.add_forcing_to(shearable_rod).using(
    GravityForces, acc_gravity=np.array([0.0, gravitational_acc, 0.0])
)

# Add muscle forces on the rod
t_coeff_optimized = np.genfromtxt("optimized_coefficients.txt", delimiter=",")
period = 1.0
wave_length = 0.97 * base_length
snake_sim.add_forcing_to(shearable_rod).using(
    MuscleTorques,
    base_length=base_length,
    b_coeff=t_coeff_optimized,
    period=period,
    wave_number=2.0 * np.pi / (wave_length),
    phase_shift=0.0,
    rampupTime=period,
    direction=normal,
    WithSpline=True,
)


# Add friction forces
origin_plane = np.array([0.0, -base_radius, 0.0])
normal_plane = normal
slip_velocity_tol = 1e-8
Froude = 0.1
mu = base_length / (period * period * np.abs(gravitational_acc) * Froude)
static_mu_array = np.array(
    [1.0 * 2.0 * mu, 1.0 * 3.0 * mu, 4.0 * mu]
)  # [forward, backward, sideways]
kinetic_mu_array = np.array(
    [1.0 * mu, 1.0 * 1.5 * mu, 2.0 * mu]
)  # [forward, backward, sideways]

snake_sim.add_forcing_to(shearable_rod).using(
    AnistropicFrictionalPlane,
    k=1.0,
    nu=1e-6,
    plane_origin=origin_plane,
    plane_normal=normal_plane,
    slip_velocity_tol=slip_velocity_tol,
    static_mu_array=static_mu_array,
    kinetic_mu_array=kinetic_mu_array,
)

# Add call backs
pp_list = {"time": [], "step": [], "position": [], "velocity": [], "avg_velocity": []}
snake_sim.callback_of(shearable_rod).using(
    ContinuumSnakeCallBack, step_skip=200, list=pp_list
)

snake_sim.finalize()
timestepper = PositionVerlet()
# timestepper = PEFRL()

final_time = (11.0 + 0.01) * period
dt = 1.0e-5 * period
total_steps = int(final_time / dt)
print("Total steps", total_steps)
positions_over_time, directors_over_time, velocities_over_time = integrate(
    timestepper, snake_sim, final_time, total_steps
)

if PLOT_FIGURE:
    filename_plot = "continuum_snake_velocity.png"
    plot_snake_velocity(pp_list, period, filename_plot, SAVE_FIGURE)

    if SAVE_VIDEO:
        filename_video = "continuum_snake.mp4"
        plot_video(pp_list, video_name=filename_video, margin=0.2, fps=500)


if SAVE_RESULTS:
    import pickle

    filename = "continuum_snake.dat"
    file = open(filename, "wb")
    pickle.dump(pp_list, file)
    file.close()
