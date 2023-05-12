__doc__ = """Fixed joint example, for detailed explanation refer to Zhang et. al. Nature Comm.  methods section."""

import numpy as np
import elastica as ea
from examples.JointCases.joint_cases_postprocessing import (
    plot_position,
    plot_video,
    plot_video_xy,
    plot_video_xz,
)


class FixedJointSimulator(
    ea.BaseSystemCollection,
    ea.Constraints,
    ea.Connections,
    ea.Forcing,
    ea.Damping,
    ea.CallBacks,
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
rod1 = ea.CosseratRod.straight_rod(
    n_elem,
    start_rod_1,
    direction,
    normal,
    base_length,
    base_radius,
    density,
    youngs_modulus=E,
    shear_modulus=shear_modulus,
)
fixed_joint_sim.append(rod1)
# Create rod 2
rod2 = ea.CosseratRod.straight_rod(
    n_elem,
    start_rod_2,
    direction,
    normal,
    base_length,
    base_radius,
    density,
    youngs_modulus=E,
    shear_modulus=shear_modulus,
)
fixed_joint_sim.append(rod2)

# Apply boundary conditions to rod1.
fixed_joint_sim.constrain(rod1).using(
    ea.OneEndFixedBC, constrained_position_idx=(0,), constrained_director_idx=(0,)
)

# Connect rod 1 and rod 2
fixed_joint_sim.connect(
    first_rod=rod1, second_rod=rod2, first_connect_idx=-1, second_connect_idx=0
).using(ea.FixedJoint, k=1e5, nu=0.0, kt=1e1, nut=0.0)

# Add forces to rod2
fixed_joint_sim.add_forcing_to(rod2).using(
    ea.EndpointForcesSinusoidal,
    start_force_mag=0,
    end_force_mag=5e-3,
    ramp_up_time=0.2,
    tangent_direction=direction,
    normal_direction=normal,
)

# add damping
damping_constant = 0.4
dt = 1e-4
fixed_joint_sim.dampen(rod1).using(
    ea.AnalyticalLinearDamper,
    damping_constant=damping_constant,
    time_step=dt,
)
fixed_joint_sim.dampen(rod2).using(
    ea.AnalyticalLinearDamper,
    damping_constant=damping_constant,
    time_step=dt,
)

pp_list_rod1 = ea.defaultdict(list)
pp_list_rod2 = ea.defaultdict(list)

fixed_joint_sim.collect_diagnostics(rod1).using(
    ea.MyCallBack, step_skip=1000, callback_params=pp_list_rod1
)
fixed_joint_sim.collect_diagnostics(rod2).using(
    ea.MyCallBack, step_skip=1000, callback_params=pp_list_rod2
)

fixed_joint_sim.finalize()
timestepper = ea.PositionVerlet()
# timestepper = PEFRL()

final_time = 10
dl = base_length / n_elem
total_steps = int(final_time / dt)
print("Total steps", total_steps)
ea.integrate(timestepper, fixed_joint_sim, final_time, total_steps)

PLOT_FIGURE = True
SAVE_FIGURE = False
PLOT_VIDEO = True

# plotting results
if PLOT_FIGURE:
    filename = "fixed_joint_test.png"
    plot_position(pp_list_rod1, pp_list_rod2, filename, SAVE_FIGURE)

if PLOT_VIDEO:
    filename = "fixed_joint_test.mp4"
    plot_video(pp_list_rod1, pp_list_rod2, video_name=filename, margin=0.2, fps=100)
    plot_video_xy(
        pp_list_rod1, pp_list_rod2, video_name=filename + "_xy.mp4", margin=0.2, fps=100
    )
    plot_video_xz(
        pp_list_rod1, pp_list_rod2, video_name=filename + "_xz.mp4", margin=0.2, fps=100
    )
