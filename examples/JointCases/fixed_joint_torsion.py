__doc__ = """Fixed joint example, for detailed explanation refer to Zhang et. al. Nature Comm.  methods section."""

import matplotlib.pyplot as plt
import numpy as np
import sys

# FIXME without appending sys.path make it more generic
sys.path.append("../../")
from elastica import *
from examples.JointCases.joint_cases_postprocessing import (
    plot_position,
    plot_orientation,
    plot_video,
    plot_video_xy,
    plot_video_xz,
)


class FixedJointSimulator(
    BaseSystemCollection, Constraints, Connections, Forcing, Damping, CallBacks
):
    pass


fixed_joint_sim = FixedJointSimulator()

# setting up test params
n_elem = 10
direction = np.array([0.0, 0.0, 1.0])
normal = np.array([0.0, 1.0, 0.0])
roll_direction = np.cross(direction, normal)
base_length = 0.2
base_radius = 0.007
base_area = np.pi * base_radius ** 2
density = 1750
E = 3e7
poisson_ratio = 0.5
shear_modulus = E / (poisson_ratio + 1.0)

start_rod_1 = np.zeros((3,))
start_rod_2 = start_rod_1 + direction * base_length

# Create rod 1
rod1 = CosseratRod.straight_rod(
    n_elem,
    start_rod_1,
    direction,
    normal,
    base_length,
    base_radius,
    density,
    0.0,  # internal damping constant, deprecated in v0.3.0
    E,
    shear_modulus=shear_modulus,
)
fixed_joint_sim.append(rod1)
# Create rod 2
rod2 = CosseratRod.straight_rod(
    n_elem,
    start_rod_2,
    direction,
    normal,
    base_length,
    base_radius,
    density,
    0.0,  # internal damping constant, deprecated in v0.3.0
    E,
    shear_modulus=shear_modulus,
)
fixed_joint_sim.append(rod2)

# Apply boundary conditions to rod1.
fixed_joint_sim.constrain(rod1).using(
    OneEndFixedBC, constrained_position_idx=(0,), constrained_director_idx=(0,)
)

# Connect rod 1 and rod 2
fixed_joint_sim.connect(
    first_rod=rod1, second_rod=rod2, first_connect_idx=-1, second_connect_idx=0
).using(FixedJoint, k=1e5, nu=1.0, kt=1e3, nut=1e-3)

# Add forces to rod2
fixed_joint_sim.add_forcing_to(rod2).using(
    UniformTorques, torque=5e-3, direction=np.array([0.0, 0.0, 1.0])
)

# add damping
damping_constant = 0.4
dt = 1e-5
fixed_joint_sim.dampen(rod1).using(
    ExponentialDamper,
    damping_constant=damping_constant,
    time_step=dt,
)
fixed_joint_sim.dampen(rod2).using(
    ExponentialDamper,
    damping_constant=damping_constant,
    time_step=dt,
)


pp_list_rod1 = defaultdict(list)
pp_list_rod2 = defaultdict(list)


fixed_joint_sim.collect_diagnostics(rod1).using(
    MyCallBack, step_skip=1000, callback_params=pp_list_rod1
)
fixed_joint_sim.collect_diagnostics(rod2).using(
    MyCallBack, step_skip=1000, callback_params=pp_list_rod2
)

fixed_joint_sim.finalize()
timestepper = PositionVerlet()

final_time = 10
dl = base_length / n_elem
dt = 1e-5
total_steps = int(final_time / dt)
print("Total steps", total_steps)
integrate(timestepper, fixed_joint_sim, final_time, total_steps)


plot_orientation(
    "Orientation of last node of rod 1",
    pp_list_rod1["time"],
    np.array(pp_list_rod1["director"])[..., -1],
)
plot_orientation(
    "Orientation of last node of rod 2",
    pp_list_rod2["time"],
    np.array(pp_list_rod2["director"])[..., -1],
)

PLOT_FIGURE = True
SAVE_FIGURE = True
PLOT_VIDEO = True

# plotting results
if PLOT_FIGURE:
    filename = "fixed_joint_torsion_test.png"
    plot_position(pp_list_rod1, pp_list_rod2, filename, SAVE_FIGURE)

if PLOT_VIDEO:
    filename = "fixed_joint_torsion_test"
    fps = 100  # Hz
    plot_video(
        pp_list_rod1,
        pp_list_rod2,
        video_name=filename + ".mp4",
        margin=0.2,
        fps=fps,
    )
    plot_video_xy(
        pp_list_rod1,
        pp_list_rod2,
        video_name=filename + "_xy.mp4",
        margin=0.2,
        fps=fps,
    )
    plot_video_xz(
        pp_list_rod1,
        pp_list_rod2,
        video_name=filename + "_xz.mp4",
        margin=0.2,
        fps=fps,
    )
