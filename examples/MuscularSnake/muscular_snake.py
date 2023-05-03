__doc__ = """Muscular snake example from Zhang et. al. Nature Comm 2019 paper."""
import numpy as np
import elastica as ea
from examples.MuscularSnake.post_processing import (
    plot_video_with_surface,
    plot_snake_velocity,
)
from examples.MuscularSnake.muscle_forces import MuscleForces
from elastica.experimental.connection_contact_joint.parallel_connection import (
    SurfaceJointSideBySide,
    get_connection_vector_straight_straight_rod,
)


# Set base simulator class
class MuscularSnakeSimulator(
    ea.BaseSystemCollection,
    ea.Constraints,
    ea.Connections,
    ea.Forcing,
    ea.CallBacks,
    ea.Damping,
):
    pass


muscular_snake_simulator = MuscularSnakeSimulator()

# Simulation parameters
final_time = 16.0
time_step = 5e-6
total_steps = int(final_time / time_step)
rendering_fps = 30
step_skip = int(1.0 / (rendering_fps * time_step))


rod_list = []
# Snake body
n_elem_body = 100
density_body = 1000
base_length_body = 1.0
base_radius_body = 0.025
E = 1e7
nu = 4e-3
shear_modulus = E / 2 * (0.5 + 1.0)
poisson_ratio = 0.5
nu_body = nu / density_body / (np.pi * base_radius_body ** 2)

direction = np.array([1.0, 0.0, 0.0])
normal = np.array([0.0, 0.0, 1.0])
start = np.array([0.0, 0.0, base_radius_body])

snake_body = ea.CosseratRod.straight_rod(
    n_elem_body,
    start,
    direction,
    normal,
    base_length_body,
    base_radius_body,
    density_body,
    youngs_modulus=E,
    shear_modulus=shear_modulus,
)

body_elem_length = snake_body.rest_lengths[0]

# Define muscle fibers
n_muscle_fibers = 8

# Muscle force amplitudes
muscle_force_amplitudes = (
    np.array([22.96, 22.96, 20.95, 20.95, 9.51, 9.51, 13.7, 13.7])[::-1] / 2
)

# Set connection index of first node of each muscle with body
muscle_start_connection_index = [4, 4, 33, 33, 23, 23, 61, 61]
muscle_end_connection_index = []
muscle_glue_connection_index = (
    []
)  # These are the middle node idx of muscles that are glued to body
muscle_rod_list = []
"""
The muscle density is higher than the physiological one, since
we lump many muscles (SSP-SP, LD and IC) into one actuator. These rods
also represent the two tendons on the sides of the muscle which biologically
have a higher density than the muscle itself. For these reasons,we set the
muscle density to approximately twice the biological value.
"""

density_muscle = 2000
E_muscle = 1e4
nu_muscle = nu
shear_modulus_muscle = E_muscle / 2 * (0.5 + 1.0)

# Muscle group 1 and 3, define two antagonistic muscle pairs
n_elem_muscle_group_one_to_three = 13 * 3
base_length_muscle = 0.39
"""
In our simulation, we lump many biological tendons into one computational
tendon. As a result, our computational tendon is bigger in size, set as elements other than 4-8
below.
"""
muscle_radius = np.zeros((n_elem_muscle_group_one_to_three))
muscle_radius[:] = 0.003  # First set tendon radius for whole rod.
muscle_radius[4 * 3 : 9 * 3] = 0.006  # Change the radius of muscle elements
nu_muscle /= density_muscle * np.pi * 0.003 ** 2

for i in range(int(n_muscle_fibers / 2)):

    index = muscle_start_connection_index[i]
    # Chose which side of body we are attaching the muscles. Note that these muscles are antagonistic pairs.
    # So they are at the opposite sides of the body and side_sign determines that.
    side_sign = -1 if i % 2 == 0 else 1
    start_muscle = np.array(
        [
            index * body_elem_length,
            side_sign * (base_radius_body + 0.003),
            base_radius_body,
        ]
    )

    muscle_rod = ea.CosseratRod.straight_rod(
        n_elem_muscle_group_one_to_three,
        start_muscle,
        direction,
        normal,
        base_length_muscle,
        muscle_radius,
        density_muscle,
        youngs_modulus=E_muscle,
        shear_modulus=shear_modulus_muscle,
    )

    """
    The biological tendons have a high Young's modulus E.,but are very slender.
    As a result, they resist extension (stretch) but can bend easily.
    Due to our decision to lump tendons and in order to mimic the above behavior
    of the biological tendons, we use a lower Young's
    Modulus and harden the stiffness of the shear and stretch modes only.
    Numerically, this is done by putting a pre-factor of 50000 before the
    shear/stretch matrix below. The actual value of the prefactor does not matter,
    what is important is that it is a high value to high stretch/shear stiffness.
    """

    muscle_rod.shear_matrix[..., : 4 * 3] *= 50000
    muscle_rod.shear_matrix[..., 9 * 3 :] *= 50000

    muscle_rod_list.append(muscle_rod)
    muscle_end_connection_index.append(index + n_elem_muscle_group_one_to_three)
    muscle_glue_connection_index.append(
        np.hstack(
            (
                np.arange(0, 4 * 3, 1, dtype=np.int64),
                np.arange(9 * 3, n_elem_muscle_group_one_to_three, 1, dtype=np.int64),
            )
        )
    )


# Muscle group 2 and 4, define two antagonistic muscle pairs
n_elem_muscle_group_two_to_four = 33
base_length_muscle = 0.33
"""
In our simulation, we lump many biological tendons into one computational
tendon. As a result, our computational tendon is bigger in size, set as rm_t
below.
"""
muscle_radius = np.zeros((n_elem_muscle_group_two_to_four))
muscle_radius[:] = 0.003  # First set tendon radius for whole rod.
muscle_radius[4 * 3 : 9 * 3] = 0.006  # Change the radius of muscle elements

for i in range(int(n_muscle_fibers / 2), n_muscle_fibers):

    index = muscle_start_connection_index[i]
    # Chose which side of body we are attaching the muscles. Note that these muscles are antagonistic pairs.
    # So they are at the opposite sides of the body and side_sign determines that.
    side_sign = -1 if i % 2 == 0 else 1
    start_muscle = np.array(
        [
            index * body_elem_length,
            side_sign * (base_radius_body + 0.003),
            base_radius_body,
        ]
    )

    muscle_rod = ea.CosseratRod.straight_rod(
        n_elem_muscle_group_two_to_four,
        start_muscle,
        direction,
        normal,
        base_length_muscle,
        muscle_radius,
        density_muscle,
        youngs_modulus=E_muscle,
        shear_modulus=shear_modulus_muscle,
    )

    """
    The biological tendons have a high Young's modulus E.,but are very slender.
    As a result, they resist extension (stretch) but can bend easily.
    Due to our decision to lump tendons and in order to mimic the above behavior
    of the biological tendons, we use a lower Young's
    Modulus and harden the stiffness of the shear and stretch modes only.
    Numerically, this is done by putting a pre-factor of 50000 before the
    shear/stretch matrix below. The actual value of the prefactor does not matter,
    what is important is that it is a high value to high stretch/shear stiffness.
    """

    muscle_rod.shear_matrix[..., : 4 * 3] *= 50000
    muscle_rod.shear_matrix[..., 9 * 3 :] *= 50000

    muscle_rod_list.append(muscle_rod)
    muscle_end_connection_index.append(index + n_elem_muscle_group_two_to_four)
    muscle_glue_connection_index.append(
        # np.array([0,1, 2, 3, 9, 10 ], dtype=np.int)
        np.hstack(
            (
                np.arange(0, 4 * 3, 1, dtype=np.int64),
                np.arange(9 * 3, n_elem_muscle_group_two_to_four, 1, dtype=np.int64),
            )
        )
    )


# After initializing the rods append them on to the simulation
rod_list.append(snake_body)
rod_list = rod_list + muscle_rod_list
for _, my_rod in enumerate(rod_list):
    muscular_snake_simulator.append(my_rod)

# Add dissipation to backbone
muscular_snake_simulator.dampen(snake_body).using(
    ea.AnalyticalLinearDamper,
    damping_constant=nu_body,
    time_step=time_step,
)

# Add dissipation to muscles
for rod in rod_list:
    muscular_snake_simulator.dampen(rod).using(
        ea.AnalyticalLinearDamper,
        damping_constant=nu_muscle,
        time_step=time_step,
    )

# Muscle actuation
post_processing_forces_dict_list = []

for i in range(n_muscle_fibers):
    post_processing_forces_dict_list.append(ea.defaultdict(list))
    muscle_rod = muscle_rod_list[i]
    side_of_body = 1 if i % 2 == 0 else -1

    time_delay = muscle_start_connection_index[::-1][i] * 1.0 / 101.76

    muscular_snake_simulator.add_forcing_to(muscle_rod).using(
        MuscleForces,
        amplitude=muscle_force_amplitudes[i],
        wave_number=2.0 * np.pi / 1.0,
        arm_length=(base_radius_body + 0.003),
        time_delay=time_delay,
        side_of_body=side_of_body,
        muscle_start_end_index=np.array([4 * 3, 9 * 3], np.int64),
        step=step_skip,
        post_processing=post_processing_forces_dict_list[i],
    )


straight_straight_rod_connection_list = []
straight_straight_rod_connection_post_processing_dict = ea.defaultdict(list)
for idx, rod_two in enumerate(muscle_rod_list):
    rod_one = snake_body
    (
        rod_one_direction_vec_in_material_frame,
        rod_two_direction_vec_in_material_frame,
        offset_btw_rods,
    ) = get_connection_vector_straight_straight_rod(
        rod_one,
        rod_two,
        (muscle_start_connection_index[idx], muscle_end_connection_index[idx]),
        (0, rod_two.n_elems),
    )
    straight_straight_rod_connection_list.append(
        [
            rod_one,
            rod_two,
            rod_one_direction_vec_in_material_frame.copy(),
            rod_two_direction_vec_in_material_frame.copy(),
            offset_btw_rods.copy(),
        ]
    )
    for k in range(rod_two.n_elems):
        rod_one_index = k + muscle_start_connection_index[idx]
        rod_two_index = k
        k_conn = (
            rod_one.radius[rod_one_index]
            * rod_two.radius[rod_two_index]
            / (rod_one.radius[rod_one_index] + rod_two.radius[rod_two_index])
            * body_elem_length
            * E
            / (rod_one.radius[rod_one_index] + rod_two.radius[rod_two_index])
        )

        if k < 12 or k >= 27:
            scale = 1 * 2
            scale_contact = 20
        else:
            scale = 0.01 * 5
            scale_contact = 20

        muscular_snake_simulator.connect(
            first_rod=rod_one,
            second_rod=rod_two,
            first_connect_idx=rod_one_index,
            second_connect_idx=rod_two_index,
        ).using(
            SurfaceJointSideBySide,
            k=k_conn * scale,
            nu=1e-4,
            k_repulsive=k_conn * scale_contact,
            rod_one_direction_vec_in_material_frame=rod_one_direction_vec_in_material_frame[
                ..., k
            ],
            rod_two_direction_vec_in_material_frame=rod_two_direction_vec_in_material_frame[
                ..., k
            ],
            offset_btw_rods=offset_btw_rods[k],
            post_processing_dict=straight_straight_rod_connection_post_processing_dict,
            step_skip=step_skip,
        )

# Friction forces
# Only apply to the snake body.
gravitational_acc = -9.81
muscular_snake_simulator.add_forcing_to(snake_body).using(
    ea.GravityForces, acc_gravity=np.array([0.0, 0.0, gravitational_acc])
)

origin_plane = np.array([0.0, 0.0, 0.0])
normal_plane = normal
slip_velocity_tol = 1e-8
froude = 0.1
period = 1.0
mu = base_length_body / (period * period * np.abs(gravitational_acc) * froude)
kinetic_mu_array = np.array(
    [1.0 * mu, 1.5 * mu, 2.0 * mu]
)  # [forward, backward, sideways]
static_mu_array = 2 * kinetic_mu_array
muscular_snake_simulator.add_forcing_to(snake_body).using(
    ea.AnisotropicFrictionalPlane,
    k=1e1,
    nu=40,
    plane_origin=origin_plane,
    plane_normal=normal_plane,
    slip_velocity_tol=slip_velocity_tol,
    static_mu_array=static_mu_array,
    kinetic_mu_array=kinetic_mu_array,
)


class MuscularSnakeCallBack(ea.CallBackBaseClass):
    def __init__(self, step_skip: int, callback_params: dict):
        ea.CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.callback_params = callback_params

    def make_callback(self, system, time, current_step: int):

        if current_step % self.every == 0:
            self.callback_params["time"].append(time)
            self.callback_params["step"].append(current_step)
            self.callback_params["position"].append(system.position_collection.copy())
            self.callback_params["com"].append(system.compute_position_center_of_mass())
            self.callback_params["radius"].append(system.radius.copy())
            self.callback_params["velocity"].append(system.velocity_collection.copy())
            self.callback_params["avg_velocity"].append(
                system.compute_velocity_center_of_mass()
            )

            self.callback_params["center_of_mass"].append(
                system.compute_position_center_of_mass()
            )


post_processing_dict_list = []

for idx, rod in enumerate(rod_list):
    post_processing_dict_list.append(ea.defaultdict(list))
    muscular_snake_simulator.collect_diagnostics(rod).using(
        MuscularSnakeCallBack,
        step_skip=step_skip,
        callback_params=post_processing_dict_list[idx],
    )

muscular_snake_simulator.finalize()
timestepper = ea.PositionVerlet()
ea.integrate(timestepper, muscular_snake_simulator, final_time, total_steps)


plot_video_with_surface(
    post_processing_dict_list,
    video_name="muscular_snake.mp4",
    fps=rendering_fps,
    step=1,
    # The following parameters are optional
    x_limits=(-0.1, 1.0),  # Set bounds on x-axis
    y_limits=(-0.3, 0.3),  # Set bounds on y-axis
    z_limits=(-0.3, 0.3),  # Set bounds on z-axis
    dpi=100,  # Set the quality of the image
    vis3D=True,  # Turn on 3D visualization
    vis2D=True,  # Turn on projected (2D) visualization
)

plot_snake_velocity(
    post_processing_dict_list[0], period=period, filename="muscular_snake_velocity.png"
)
