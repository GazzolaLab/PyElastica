"""
Catenary
========

This case simulates a rod hanging under its own weight, forming a catenary
curve. The rod is fixed at both ends and is allowed to settle into its
equilibrium position.
"""

from collections import defaultdict
import numpy as np

import elastica as ea

from post_processing import (
    plot_video,
    plot_catenary,
)

# %%
# Simulation Setup
# ----------------
# We define a simulator class that inherits from the necessary mixins.


class CatenarySimulator(
    ea.BaseSystemCollection, ea.Constraints, ea.Forcing, ea.Damping, ea.CallBacks
):
    pass


catenary_sim = CatenarySimulator()
final_time = 30

# %%
# Rod Setup
# ---------
# We set up the rod parameters. This rod is affected by a gravity force.

n_elem = 500
time_step = 1e-4
total_steps = int(final_time / time_step)
rendering_fps = 20
step_skip = int(1.0 / (rendering_fps * time_step))

start = np.zeros((3,))
direction = np.array([1.0, 0.0, 0.0])
normal = np.array([0.0, 0.0, 1.0])
binormal = np.cross(direction, normal)

# catenary parameters
base_length = 1.0
base_radius = 0.01
base_area = np.pi * (base_radius**2)
volume = base_area * base_length
mass = 0.2
density = mass / volume
E = 1e4
poisson_ratio = 0.5
shear_modulus = E / (poisson_ratio + 1.0)

base_rod = ea.CosseratRod.straight_rod(
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

catenary_sim.append(base_rod)

# Add gravity
catenary_sim.add_forcing_to(base_rod).using(
    ea.GravityForces, acc_gravity=-9.80665 * normal
)

# %%
# Damping is added to the system to help it reach a steady state.

# add damping
damping_constant = 0.3
catenary_sim.dampen(base_rod).using(
    ea.AnalyticalLinearDamper,
    damping_constant=damping_constant,
    time_step=time_step,
)

# %%
# Boundary Conditions
# -------------------
# We fix both ends of the rod using the `FixedConstraint`.

# fix catenary ends
catenary_sim.constrain(base_rod).using(
    ea.FixedConstraint,
    constrained_position_idx=(0, -1),
    constrained_director_idx=(0, -1),
)

# %%
# Callback
# --------
# We define a callback class to record the rod state during the simulation.


# Add call backs
class CatenaryCallBack(ea.CallBackBaseClass):
    """
    Call back function for continuum snake
    """

    def __init__(self, step_skip: int, callback_params: dict) -> None:
        super().__init__()
        self.every = step_skip
        self.callback_params = callback_params

    def make_callback(
        self, system: ea.typing.RodType, time: np.float64, current_step: int
    ) -> None:

        if current_step % self.every == 0:

            self.callback_params["time"].append(time)
            self.callback_params["step"].append(current_step)
            self.callback_params["position"].append(system.position_collection.copy())
            self.callback_params["radius"].append(system.radius.copy())
            self.callback_params["internal_force"].append(system.internal_forces.copy())

            return


recorded_history: dict[str, list] = defaultdict(list)
catenary_sim.collect_diagnostics(base_rod).using(
    CatenaryCallBack, step_skip=step_skip, callback_params=recorded_history
)

# %%
# Finalize and Run
# ----------------
# We finalize the simulator, create the time-stepper, and run.

catenary_sim.finalize()
timestepper: ea.typing.StepperProtocol = ea.PositionVerlet()

dt = final_time / total_steps
time = 0.0
for i in range(total_steps):
    time = timestepper.step(catenary_sim, time, dt)
position = np.array(recorded_history["position"])
b = np.min(position[-1][2])

# %%
# Post-processing
# ---------------
# Finally, we can save a video of the simulation and plot the final
# shape of the catenary.

# plotting the videos
filename_video = "catenary.mp4"
plot_video(
    recorded_history,
    video_name=filename_video,
    fps=rendering_fps,
    xlim=[0, base_length],
    ylim=[-0.5 * base_length, 0.5 * base_length],
)

# %%
# .. video:: ../../../examples/CatenaryCase/catenary.mp4
#    :width: 720
#    :autoplay:
#    :muted:
#    :loop:

# %%
# plotting the catenary positions after simulation.
plot_catenary(
    recorded_history,
    xlim=(0, base_length),
    ylim=(b, 0.0),
)
