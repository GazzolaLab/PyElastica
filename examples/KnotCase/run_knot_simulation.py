"""
Knot Simulation
===============

This script simulates the formation of an overhand knot in a soft rod.
It demonstrates how to create a controller to manipulate a node on the rod,
which can be used for tasks like trajectory tracing or proportional control.
"""

from typing import Any, TypeAlias
from numpy.typing import NDArray
from elastica.typing import RodType

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import elastica as ea

from knot_forcing import TargetPoseProportionalControl
from knot_visualization import plot_video3D

Position: TypeAlias = NDArray[np.float64]  # vector (3)
Orientation: TypeAlias = NDArray[np.float64]  # SO3 matrix (3, 3)
Pose: TypeAlias = tuple[Position, Orientation]

# %%
# Simulation Setup
# ----------------
# We define a simulator class that inherits from the necessary mixins.


class SoftRodSimulator(
    ea.BaseSystemCollection,
    ea.Constraints,
    ea.Forcing,
    ea.Damping,
    ea.CallBacks,
    ea.Contact,
):
    pass


simulator = SoftRodSimulator()
final_time = 5
dt = 0.0002


# %%
# Callback Setup
# --------------
# We also define a callback class to record the position of the rod during the
# simulation.


class Callback(ea.CallBackBaseClass):
    """
    Records the position of the rod
    """

    def __init__(self, callback_params: dict) -> None:
        ea.CallBackBaseClass.__init__(self)
        self.every = 200
        self.callback_params = callback_params

    def make_callback(self, system: RodType, time: float, current_step: int) -> None:
        if current_step % self.every == 0:

            self.callback_params["time"].append(time)
            self.callback_params["step"].append(current_step)
            self.callback_params["radius"].append(system.radius.copy())
            self.callback_params["position"].append(system.position_collection.copy())
            self.callback_params["orientation"].append(
                system.director_collection.copy()
            )
            return


recorded_history: dict[str, list[Any]] = defaultdict(list)

# %%
# Rod Setup
# ---------
# Next, we set up the parameters for the rod.

# setting up test params
n_elem = 50
start = np.zeros((3,))
direction = np.array([1.0, 0.0, 0.0])
normal = np.array([0.0, 1.0, 0.0])
base_length = 1.2
base_radius = 0.025
density = 2000
youngs_modulus = 1e6
poisson_ratio = 0.5
shear_modulus = youngs_modulus / (2 * (poisson_ratio + 1.0))

# We create the `CosseratRod` object and add it to the simulator.
stretchable_rod = ea.CosseratRod.straight_rod(
    n_elem,
    start,
    direction,
    normal,
    base_length,
    base_radius,
    density,
    youngs_modulus=youngs_modulus,
    shear_modulus=shear_modulus,
)
simulator.append(stretchable_rod)

simulator.collect_diagnostics(stretchable_rod).using(
    Callback, callback_params=recorded_history
)

# %%
# Controller Setup
# ----------------
# We define a function that returns the target pose (position and
# orientation) for the controller at a given time. This function creates
# the trajectory for the end of the rod to follow to tie the knot.

activation_time = 4


def base_target(t: float, rod: RodType) -> Pose:
    target_position = direction * base_length - 5 * base_radius * normal
    if t <= activation_time / 2:
        ratio = min(2 * t / activation_time, 1.0)
        angular_ratio = ratio * np.pi * 2
        position = target_position * ratio
        orientation_twist = np.array(
            [
                [0, np.cos(angular_ratio), np.sin(angular_ratio)],
                [0, -np.sin(angular_ratio), np.cos(angular_ratio)],
                [1, 0, 0],
            ],
            dtype=float,
        )
    else:
        ratio = min(2 * (t - activation_time / 2) / activation_time, 1.0)
        R = 8
        position = np.array(
            [
                target_position[0] * (1 - ratio),
                -R * base_radius * np.cos(2 * ratio * 12) * (1 - ratio),
                -R * base_radius * np.sin(2 * ratio * 12) * (1 - ratio),
            ]
        )
        angular_ratio = (1 - ratio) * np.pi * 2
        orientation_twist = np.array(
            [
                [0, np.cos(angular_ratio), -np.sin(angular_ratio)],
                [0, np.sin(angular_ratio), np.cos(angular_ratio)],
                [1, 0, 0],
            ],
            dtype=float,
        )
    return position, orientation_twist


# %%
# We add a `TargetPoseProportionalControl` forcing to the rod. This
# controller applies forces and torques to drive a specific node of the
# rod to the target pose. The class is defined in `knot_forcing.py`.

# Control point
p = 3e3
pt = 5e0
simulator.add_forcing_to(stretchable_rod).using(
    TargetPoseProportionalControl,
    elem_index=0,
    p_linear_value=p,
    p_angular_value=pt,
    target=base_target,
    ramp_up_time=1e-6,
    target_history=recorded_history["base_pose"],
)

# %%
# Boundary Conditions
# -------------------
# We apply boundary conditions to fix the other end of the rod.

# Boundary conditions
simulator.constrain(stretchable_rod).using(
    ea.FixedConstraint, constrained_position_idx=(-1, -20)
)

# %%
# Contact Setup
# -------------
# We enable self-contact detection for the rod to prevent it from passing
# through itself.

# Self contact
simulator.detect_contact_between(stretchable_rod, stretchable_rod).using(
    ea.RodSelfContact, k=1e4, nu=3
)

# %%
# Environmental Forcing and Damping
# ---------------------------------
# We add gravity and damping to the system.

# Gravity
simulator.add_forcing_to(stretchable_rod).using(
    ea.GravityForces, acc_gravity=np.array([0.0, 0.0, -9.80665])
)

# Damping
damping_constant = 5.0
simulator.dampen(stretchable_rod).using(
    ea.AnalyticalLinearDamper,
    translational_damping_constant=damping_constant,
    rotational_damping_constant=damping_constant * 0.01,
    time_step=dt,
)
simulator.dampen(stretchable_rod).using(ea.LaplaceDissipationFilter, filter_order=5)


# %%
# Finalize and Run
# ----------------
# We finalize the simulator and create the time-stepper.

# Finalize and run the simulation
simulator.finalize()
timestepper = ea.PositionVerlet()

total_steps = int(final_time / dt)
print("Total steps", total_steps)
dt = final_time / total_steps
time = 0.0
for i in range(total_steps):
    time = timestepper.step(simulator, time, dt)

# %%
# Post-Processing
# ---------------
# After the simulation, we can generate a 3D video of the knot tying
# process.

filename_video = "knot3D.mp4"
plot_video3D(recorded_history, video_name=filename_video, margin=0.2, fps=10)

# %%
# .. video:: ../../../examples/KnotCase/knot3D.mp4
#    :width: 720
#    :autoplay:
#    :muted:
#    :loop:


# %%
# We can also plot the topological quantities of the knot, such as twist,
# writhe, and link, as a function of time.

# Plot knot topological quantities
timestep = np.asarray(recorded_history["time"])
positions = np.asarray(recorded_history["position"])
orientations = np.asarray(recorded_history["orientation"])
radii = np.asarray(recorded_history["radius"])
total_twist, _ = ea.compute_twist(positions, orientations[:, 0, ...])
total_writhe = ea.compute_writhe(positions, np.float64(base_length), "next_tangent")
total_link = ea.compute_link(
    positions,
    orientations[:, 0, ...],
    radii,
    np.float64(base_length),
    "next_tangent",
)

plt.figure()
plt.plot(timestep, total_twist, label="twist")
plt.plot(timestep, total_writhe, label="writhe")
plt.plot(timestep, total_link, label="link")
plt.legend()
plt.xlabel("time")
plt.ylabel("link-writhe-twist quantity")
plt.show()
