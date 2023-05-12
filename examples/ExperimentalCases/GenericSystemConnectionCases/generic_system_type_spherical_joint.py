__doc__ = """Spherical(Free) joint example, for detailed explanation refer to Zhang et. al. Nature Comm.
methods section."""

import numpy as np
import elastica as ea
from elastica.experimental.connection_contact_joint.generic_system_type_connection import (
    GenericSystemTypeFreeJoint,
)
from joint_cases_postprocessing import (
    plot_position,
    plot_video,
    plot_video_xy,
    plot_video_xz,
)


class SphericalJointSimulator(
    ea.BaseSystemCollection,
    ea.Constraints,
    ea.Connections,
    ea.Forcing,
    ea.Damping,
    ea.CallBacks,
):
    pass


spherical_joint_sim = SphericalJointSimulator()

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

# setting up time params
final_time = 10
dt = 5e-5
fps = 100  # fps of the video
step_skip = int(1 / (dt * fps))

start_rod_1 = np.zeros((3,))
start_rod_2 = start_rod_1 + direction * base_length
start_cylinder = start_rod_2 + direction * base_length

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
spherical_joint_sim.append(rod1)
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
spherical_joint_sim.append(rod2)
# Create cylinder
cylinder = ea.Cylinder(
    start=start_cylinder,
    direction=direction,
    normal=normal,
    base_length=base_length,
    base_radius=base_radius,
    density=density,
)
spherical_joint_sim.append(cylinder)

# Apply boundary conditions to rod1.
spherical_joint_sim.constrain(rod1).using(
    ea.OneEndFixedBC, constrained_position_idx=(0,), constrained_director_idx=(0,)
)

# Connect rod 1 and rod 2
spherical_joint_sim.connect(
    first_rod=rod1, second_rod=rod2, first_connect_idx=-1, second_connect_idx=0
).using(
    GenericSystemTypeFreeJoint, k=1e5, nu=0
)  # k=kg/s2 nu=kg/s 1e-2
# Connect rod 2 and cylinder
spherical_joint_sim.connect(
    first_rod=rod2, second_rod=cylinder, first_connect_idx=-1, second_connect_idx=0
).using(
    GenericSystemTypeFreeJoint,
    k=1e5,
    nu=0,
    point_system_two=np.array([0.0, 0.0, -cylinder.length / 2]),
)

# Add forces to rod2
spherical_joint_sim.add_forcing_to(rod2).using(
    ea.EndpointForcesSinusoidal,
    start_force_mag=0,
    end_force_mag=5e-3,
    ramp_up_time=0.2,
    tangent_direction=direction,
    normal_direction=normal,
)

# add damping
damping_constant = 4e-3
spherical_joint_sim.dampen(rod1).using(
    ea.AnalyticalLinearDamper,
    damping_constant=damping_constant,
    time_step=dt,
)
spherical_joint_sim.dampen(rod2).using(
    ea.AnalyticalLinearDamper,
    damping_constant=damping_constant,
    time_step=dt,
)

pp_list_rod1 = ea.defaultdict(list)
pp_list_rod2 = ea.defaultdict(list)
pp_list_cylinder = ea.defaultdict(list)

spherical_joint_sim.collect_diagnostics(rod1).using(
    ea.MyCallBack, step_skip=step_skip, callback_params=pp_list_rod1
)
spherical_joint_sim.collect_diagnostics(rod2).using(
    ea.MyCallBack, step_skip=step_skip, callback_params=pp_list_rod2
)
spherical_joint_sim.collect_diagnostics(cylinder).using(
    ea.MyCallBack, step_skip=step_skip, callback_params=pp_list_cylinder
)

spherical_joint_sim.finalize()
timestepper = ea.PositionVerlet()
# timestepper = PEFRL()

dl = base_length / n_elem
total_steps = int(final_time / dt)
print("Total steps", total_steps)
ea.integrate(timestepper, spherical_joint_sim, final_time, total_steps)

PLOT_FIGURE = True
SAVE_FIGURE = True
PLOT_VIDEO = True

# plotting results
if PLOT_FIGURE:
    filename = "generic_system_type_spherical_joint_example_last_node_pos_xy.png"
    plot_position(
        plot_params_rod1=pp_list_rod1,
        plot_params_rod2=pp_list_rod2,
        plot_params_cylinder=pp_list_cylinder,
        filename=filename,
        SAVE_FIGURE=SAVE_FIGURE,
    )

if PLOT_VIDEO:
    filename = "generic_system_type_spherical_joint_example"
    plot_video(
        plot_params_rod1=pp_list_rod1,
        plot_params_rod2=pp_list_rod2,
        plot_params_cylinder=pp_list_cylinder,
        cylinder=cylinder,
        video_name=filename + ".mp4",
        fps=fps,
    )
    plot_video_xy(
        plot_params_rod1=pp_list_rod1,
        plot_params_rod2=pp_list_rod2,
        plot_params_cylinder=pp_list_cylinder,
        cylinder=cylinder,
        video_name=filename + "_xy.mp4",
        fps=fps,
    )
    plot_video_xz(
        plot_params_rod1=pp_list_rod1,
        plot_params_rod2=pp_list_rod2,
        plot_params_cylinder=pp_list_cylinder,
        cylinder=cylinder,
        video_name=filename + "_xz.mp4",
        fps=fps,
    )
