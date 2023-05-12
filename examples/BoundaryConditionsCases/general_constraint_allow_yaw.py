__doc__ = """Fixed joint example, for detailed explanation refer to Zhang et. al. Nature Comm.  methods section."""

import numpy as np
import elastica as ea

from examples.BoundaryConditionsCases.bc_cases_postprocessing import (
    plot_position,
    plot_orientation,
    plot_video,
    plot_video_xy,
    plot_video_xz,
)


class GeneralConstraintSimulator(
    ea.BaseSystemCollection,
    ea.Constraints,
    ea.Connections,
    ea.Forcing,
    ea.Damping,
    ea.CallBacks,
):
    pass


general_constraint_sim = GeneralConstraintSimulator()

# setting up test params
n_elem = 10
direction = np.array([0.0, 0.0, 1.0])
normal = np.array([0.0, 1.0, 0.0])
base_length = 0.2
base_radius = 0.007
base_area = np.pi * base_radius ** 2
density = 1750
E = 3e7
poisson_ratio = 0.5
shear_modulus = E / (poisson_ratio + 1.0)

# setting up timestepper and video
final_time = 10
dl = base_length / n_elem
dt = 1e-5
total_steps = int(final_time / dt)
fps = 100  # frames per second of the video
diagnostic_step_skip = 1 / (fps * dt)

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
general_constraint_sim.append(rod1)

# Apply boundary conditions to rod1: only allow yaw
general_constraint_sim.constrain(rod1).using(
    ea.GeneralConstraint,
    constrained_position_idx=(0,),
    constrained_director_idx=(0,),
    translational_constraint_selector=np.array([True, True, True]),
    rotational_constraint_selector=np.array([True, True, False]),
)

# add forces to endpoint
general_constraint_sim.add_forcing_to(rod1).using(
    ea.EndpointForces,
    start_force=np.zeros((3,)),
    end_force=1e-0 * np.ones((3,)),
    ramp_up_time=0.5,
)
# add uniform torsion torque
general_constraint_sim.add_forcing_to(rod1).using(
    ea.UniformTorques, torque=1e-3, direction=np.array([0, 0, 1])
)

# add damping
damping_constant = 0.4
general_constraint_sim.dampen(rod1).using(
    ea.AnalyticalLinearDamper,
    damping_constant=damping_constant,
    time_step=dt,
)


pp_list_rod1 = ea.defaultdict(list)


general_constraint_sim.collect_diagnostics(rod1).using(
    ea.MyCallBack, step_skip=diagnostic_step_skip, callback_params=pp_list_rod1
)

general_constraint_sim.finalize()
timestepper = ea.PositionVerlet()

print("Total steps", total_steps)
ea.integrate(timestepper, general_constraint_sim, final_time, total_steps)


plot_orientation(
    "Orientation of first element of rod 1",
    pp_list_rod1["time"],
    np.array(pp_list_rod1["directors"])[..., 0],
)

PLOT_FIGURE = True
SAVE_FIGURE = True
PLOT_VIDEO = True

# plotting results
if PLOT_FIGURE:
    filename = "configurable_fixed_constraint_example_last_node_pos_xy.png"
    plot_position(pp_list_rod1, filename, SAVE_FIGURE)

if PLOT_VIDEO:
    filename = "configurable_fixed_constraint_example"
    plot_video(
        pp_list_rod1,
        video_name=filename + ".mp4",
        fps=fps,
    )
    plot_video_xy(
        pp_list_rod1,
        video_name=filename + "_xy.mp4",
        fps=fps,
    )
    plot_video_xz(
        pp_list_rod1,
        video_name=filename + "_xz.mp4",
        fps=fps,
    )
