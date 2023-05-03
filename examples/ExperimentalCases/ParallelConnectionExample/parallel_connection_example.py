__doc__ = """Parallel connection example"""

import numpy as np
import elastica as ea
from elastica.experimental.connection_contact_joint.parallel_connection import (
    get_connection_vector_straight_straight_rod,
    SurfaceJointSideBySide,
)
from elastica._calculus import difference_kernel
from examples.JointCases.joint_cases_postprocessing import (
    plot_position,
    plot_video,
    plot_video_xy,
    plot_video_xz,
)


class ParallelConnection(
    ea.BaseSystemCollection,
    ea.Constraints,
    ea.Connections,
    ea.Forcing,
    ea.Damping,
    ea.CallBacks,
):
    pass


parallel_connection_sim = ParallelConnection()

# setting up test params
n_elem = 10
direction = np.array([0.0, 0.0, 1.0])
normal = np.array([0.0, 1.0, 0.0])
binormal = np.cross(direction, normal)
base_length = 0.2
base_radius = 0.007
base_area = np.pi * base_radius ** 2
density = 1750
E = 3e4
poisson_ratio = 0.5
shear_modulus = E / (poisson_ratio + 1.0)

start_rod_1 = np.zeros((3,)) + 0.1 * direction
start_rod_2 = start_rod_1 + binormal * 2 * base_radius

# Create rod 1
rod_one = ea.CosseratRod.straight_rod(
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
parallel_connection_sim.append(rod_one)
# Create rod 2
rod_two = ea.CosseratRod.straight_rod(
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
parallel_connection_sim.append(rod_two)

# Apply boundary conditions to rod1.
parallel_connection_sim.constrain(rod_one).using(
    ea.OneEndFixedBC, constrained_position_idx=(0,), constrained_director_idx=(0,)
)

# Apply boundary conditions to rod2.
parallel_connection_sim.constrain(rod_two).using(
    ea.OneEndFixedBC, constrained_position_idx=(0,), constrained_director_idx=(0,)
)

# Apply a contraction force on rod one.
class ContractionForce(ea.NoForces):
    def __init__(
        self,
        ramp,
        force_mag,
    ):
        self.ramp = ramp
        self.force_mag = force_mag

    def apply_forces(self, system, time: np.float64 = 0.0):
        # Ramp the force
        factor = min(1.0, time / self.ramp)

        system.external_forces[:] -= factor * difference_kernel(
            self.force_mag * system.tangents
        )


parallel_connection_sim.add_forcing_to(rod_one).using(
    ContractionForce, ramp=0.5, force_mag=1.0
)

# Connect rod 1 and rod 2
(
    rod_one_direction_vec_in_material_frame,
    rod_two_direction_vec_in_material_frame,
    offset_btw_rods,
) = get_connection_vector_straight_straight_rod(
    rod_one, rod_two, (0, n_elem), (0, n_elem)
)

for i in range(n_elem):
    parallel_connection_sim.connect(
        first_rod=rod_one, second_rod=rod_two, first_connect_idx=i, second_connect_idx=i
    ).using(
        SurfaceJointSideBySide,
        k=1e2,
        nu=1e-5,
        k_repulsive=1e3,
        rod_one_direction_vec_in_material_frame=rod_one_direction_vec_in_material_frame[
            :, i
        ],
        rod_two_direction_vec_in_material_frame=rod_two_direction_vec_in_material_frame[
            :, i
        ],
        offset_btw_rods=offset_btw_rods[i],
    )  # k=kg/s2 nu=kg/s 1e-2


# add damping
damping_constant = 4e-3
dt = 1e-3
parallel_connection_sim.dampen(rod_one).using(
    ea.AnalyticalLinearDamper,
    damping_constant=damping_constant,
    time_step=dt,
)
parallel_connection_sim.dampen(rod_two).using(
    ea.AnalyticalLinearDamper,
    damping_constant=damping_constant,
    time_step=dt,
)


class ParallelConnecitonCallback(ea.CallBackBaseClass):
    """
    Call back function for parallel connection
    """

    def __init__(self, step_skip: int, callback_params: dict):
        ea.CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.callback_params = callback_params

    def make_callback(self, system, time, current_step: int):
        if current_step % self.every == 0:
            self.callback_params["time"].append(time)
            self.callback_params["step"].append(current_step)
            self.callback_params["position"].append(system.position_collection.copy())
            self.callback_params["velocity"].append(system.velocity_collection.copy())
            return


pp_list_rod1 = ea.defaultdict(list)
pp_list_rod2 = ea.defaultdict(list)


parallel_connection_sim.collect_diagnostics(rod_one).using(
    ParallelConnecitonCallback, step_skip=40, callback_params=pp_list_rod1
)
parallel_connection_sim.collect_diagnostics(rod_two).using(
    ParallelConnecitonCallback, step_skip=40, callback_params=pp_list_rod2
)


parallel_connection_sim.finalize()
timestepper = ea.PositionVerlet()

final_time = 20.0
dl = base_length / n_elem
total_steps = int(final_time / dt)
print("Total steps", total_steps)
ea.integrate(timestepper, parallel_connection_sim, final_time, total_steps)

PLOT_FIGURE = True
SAVE_FIGURE = False
PLOT_VIDEO = True

# plotting results
if PLOT_FIGURE:
    filename = "parallel_connection_test_last_node_pos_xy.png"
    plot_position(pp_list_rod1, pp_list_rod2, filename, SAVE_FIGURE)

if PLOT_VIDEO:
    filename = "parallel_connection_test.mp4"
    plot_video(pp_list_rod1, pp_list_rod2, video_name=filename, margin=0.2, fps=100)
    plot_video_xy(
        pp_list_rod1, pp_list_rod2, video_name=filename + "_xy.mp4", margin=0.2, fps=100
    )
    plot_video_xz(
        pp_list_rod1, pp_list_rod2, video_name=filename + "_xz.mp4", margin=0.2, fps=100
    )
