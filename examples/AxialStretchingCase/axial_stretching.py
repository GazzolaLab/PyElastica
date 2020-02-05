""" Axial stretching test-case

   isort:skip_file
"""
# FIXME without appending sys.path make it more generic
import sys

sys.path.append("../../")  # isort:skip

from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt

from elastica.boundary_conditions import FreeRod, OneEndFixedRod
from elastica.callback_functions import CallBackBaseClass
from elastica.external_forces import EndpointForces
from elastica.rod.cosserat_rod import CosseratRod
from elastica.timestepper import integrate
from elastica.timestepper.symplectic_steppers import PEFRL, PositionVerlet
from elastica.wrappers import BaseSystemCollection, CallBacks, Constraints, Forcing


class StretchingBeamSimulator(BaseSystemCollection, Constraints, Forcing, CallBacks):
    pass


stretch_sim = StretchingBeamSimulator()
final_time = 2.0

# Options
PLOT_FIGURE = True
SAVE_FIGURE = True
SAVE_RESULTS = True

# setting up test params
n_elem = 19
start = np.zeros((3,))
direction = np.array([1.0, 0.0, 0.0])
normal = np.array([0.0, 1.0, 0.0])
base_length = 1.0
base_radius = 0.025
base_area = np.pi * base_radius ** 2
density = 1000
nu = 2.0
youngs_modulus = 1e4
# For shear modulus of 1e4, nu is 99!
poisson_ratio = 0.5

stretchable_rod = CosseratRod.straight_rod(
    n_elem,
    start,
    direction,
    normal,
    base_length,
    base_radius,
    density,
    nu,
    youngs_modulus,
    poisson_ratio,
)

stretch_sim.append(stretchable_rod)
stretch_sim.constrain(stretchable_rod).using(
    OneEndFixedRod, positions=(0,), directors=(0,)
)

end_force_x = 1.0
end_force = np.array([end_force_x, 0.0, 0.0])
stretch_sim.add_forcing_to(stretchable_rod).using(
    EndpointForces, 0.0 * end_force, end_force, ramp_up_time=1e-2
)

# Add call backs
class ContinuumSnakeCallBack(CallBackBaseClass):
    """
    Call back function for continuum snake
    """

    def __init__(self, step_skip: int, callback_params: dict):
        CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.callback_params = callback_params

    def make_callback(self, system, time, current_step: int):

        if current_step % self.every == 0:

            self.callback_params["time"].append(time)
            # Collect only x
            self.callback_params["position"].append(
                system.position_collection[0, -1].copy()
            )
            return


pp_list = defaultdict(list)
stretch_sim.collect_diagnostics(stretchable_rod).using(
    ContinuumSnakeCallBack, step_skip=200, callback_params=pp_list,
)

stretch_sim.finalize()
timestepper = PositionVerlet()
# timestepper = PEFRL()

dl = base_length / n_elem
dt = 0.01 * dl
total_steps = int(final_time / dt)
print("Total steps", total_steps)
integrate(timestepper, stretch_sim, final_time, total_steps)

if PLOT_FIGURE:
    # First-order theory with base-length
    expected_tip_disp = end_force_x * base_length / base_area / youngs_modulus
    # First-order theory with modified-length, gives better estimates
    expected_tip_disp_improved = (
        end_force_x * base_length / (base_area * youngs_modulus - end_force_x)
    )

    fig = plt.figure(figsize=(10, 8), frameon=True, dpi=150)
    ax = fig.add_subplot(111)
    ax.plot(pp_list["time"], pp_list["position"], lw=2.0)
    ax.hlines(base_length + expected_tip_disp, 0.0, final_time, "k", "dashdot", lw=1.0)
    ax.hlines(
        base_length + expected_tip_disp_improved, 0.0, final_time, "k", "dashed", lw=2.0
    )
    if SAVE_FIGURE:
        fig.savefig("axial_stretching.pdf")
    plt.show()

if SAVE_RESULTS:
    import pickle

    filename = "axial_stretching_data.dat"
    file = open(filename, "wb")
    pickle.dump(stretchable_rod, file)
    file.close()
