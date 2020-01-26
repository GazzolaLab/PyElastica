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
from elastica.interaction import SlenderBodyTheory
from elastica.callback_functions import ContinuumSnakeCallBack
from elastica.timestepper.symplectic_steppers import PositionVerlet, PEFRL
from elastica.timestepper import integrate
from ContinuumFlagellaCase.continuum_flagella_postprocessing import (
    plot_velocity,
    plot_video,
)


class FlagellaSimulator(BaseSystemCollection, Constraints, Forcing, CallBacks):
    pass


flagella_sim = FlagellaSimulator()


# Options
PLOT_FIGURE = True
SAVE_FIGURE = True
SAVE_VIDEO = True
SAVE_RESULTS = True


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

flagella_sim.append(shearable_rod)
flagella_sim.constrain(shearable_rod).using(FreeRod)

# Add muscle forces on the rod
t_coeff_optimized = np.genfromtxt("optimized_coefficients.txt", delimiter=",")
period = 1.0
wave_length = 0.3866575573648976 * base_length
flagella_sim.add_forcing_to(shearable_rod).using(
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

# Add slender body forces
fluid_density = 1.0
reynolds_number = 1e-4
dynamic_viscosity = (
    fluid_density * base_length * base_length / (period * reynolds_number)
)
flagella_sim.add_forcing_to(shearable_rod).using(
    SlenderBodyTheory, dynamic_viscosity=dynamic_viscosity
)


# Add call backs
pp_list = {"time": [], "step": [], "position": [], "velocity": [], "avg_velocity": []}
flagella_sim.callback_of(shearable_rod).using(
    ContinuumSnakeCallBack, step_skip=200, list=pp_list
)

flagella_sim.finalize()
timestepper = PositionVerlet()
# timestepper = PEFRL()

final_time = (10.0 + 0.01) * period
dt = 2.5e-5 * period
total_steps = int(final_time / dt)
print("Total steps", total_steps)
positions_over_time, directors_over_time, velocities_over_time = integrate(
    timestepper, flagella_sim, final_time, total_steps
)

if PLOT_FIGURE:
    filename_plot = "continuum_flagella_velocity.png"
    plot_velocity(pp_list, period, filename_plot, SAVE_FIGURE)

    if SAVE_VIDEO:
        filename_video = "continuum_flagella.mp4"
        plot_video(pp_list, video_name=filename_video, margin=0.2, fps=500)


if SAVE_RESULTS:
    import pickle

    filename = "continuum_flagella.dat"
    file = open(filename, "wb")
    pickle.dump(pp_list, file)
    file.close()
