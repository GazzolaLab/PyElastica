import numpy as np
from elastica import *
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d import proj3d, Axes3D
from tqdm import tqdm
from elastica._linalg import (
    _batch_cross,
    _batch_norm,
    _batch_dot,
    _batch_matvec,
)
from typing import Dict, Sequence
from numba import njit

# from connect_straight_rods import *
from examples.ArtificialMusclesCases import *

from elastica.experimental.connection_contact_joint.parallel_connection import (
    get_connection_vector_straight_straight_rod,
)


class OctopusArmCase(
    BaseSystemCollection,
    Constraints,
    MemoryBlockConnections,
    Forcing,
    CallBacks,
    Damping,
):
    pass


def octopus_arm_simulation(
    octopus_arm_properties,
    muscle_class,
    muscles_configuration,
    muscles_activation_signal,
    sim_settings,
):

    post_processing_dict_list = []
    octopus_arm_sim = OctopusArmCase()
    muscle_height = octopus_arm_properties.length / octopus_arm_properties.n_rows
    n_turns_per_backbone_length = 1 / (octopus_arm_properties.length)

    n_elem_muscle = 200

    import os

    current_path = os.getcwd()

    dt = (0.03 * muscle_height / n_elem_muscle) * 10
    total_steps = int(sim_settings.final_time / dt)
    time_step = np.float64(sim_settings.final_time / total_steps)
    rendering_fps = 20
    step_skip = int(1.0 / (rendering_fps * time_step))

    arm_radius = octopus_arm_properties.start_radius * np.linspace(
        1, octopus_arm_properties.taper_ratio, octopus_arm_properties.n_elem
    )
    arm_volume = (
        np.pi
        * (arm_radius[0] ** 2 - arm_radius[-1] ** 2)
        * octopus_arm_properties.length
        / 3
    )

    arm_rod = CosseratRod.straight_rod(
        octopus_arm_properties.n_elem,
        octopus_arm_properties.start,
        octopus_arm_properties.direction,
        octopus_arm_properties.normal,
        octopus_arm_properties.length,
        arm_radius,
        octopus_arm_properties.arm_density,
        youngs_modulus=octopus_arm_properties.arm_youngs_modulus,
        shear_modulus=octopus_arm_properties.arm_shear_modulus,
    )

    octopus_arm_sim.append(arm_rod)

    # Add damping
    octopus_arm_sim.dampen(arm_rod).using(
        AnalyticalLinearDamper,
        damping_constant=octopus_arm_properties.arm_damping_constant,
        time_step=dt,
    )

    # fix arm bottom
    octopus_arm_sim.constrain(arm_rod).using(
        GeneralConstraint,
        constrained_position_idx=(0,),
        constrained_director_idx=(0,),
        translational_constraint_selector=np.array([True, True, True]),
        rotational_constraint_selector=np.array([True, True, True]),
    )

    arm_post_processing_dict = defaultdict(
        list
    )  # list which collected data will be append

    # set the diagnostics for rod and collect data
    octopus_arm_sim.collect_diagnostics(arm_rod).using(
        MuscleCallBack,
        step_skip=step_skip,
        callback_params=arm_post_processing_dict,
    )
    post_processing_dict_list.append(arm_post_processing_dict)

    inner_spine_radius = arm_radius[-1]
    inner_spine_volume = (
        np.pi * (inner_spine_radius ** 2) * octopus_arm_properties.length / 3
    )

    inner_spine_rod = CosseratRod.straight_rod(
        octopus_arm_properties.n_elem,
        octopus_arm_properties.start,
        octopus_arm_properties.direction,
        octopus_arm_properties.normal,
        octopus_arm_properties.length,
        inner_spine_radius,
        octopus_arm_properties.inner_spine_density,
        youngs_modulus=octopus_arm_properties.inner_spine_youngs_modulus,
        shear_modulus=octopus_arm_properties.inner_spine_shear_modulus,
    )

    octopus_arm_sim.append(inner_spine_rod)

    # Add damping
    octopus_arm_sim.dampen(inner_spine_rod).using(
        AnalyticalLinearDamper,
        damping_constant=octopus_arm_properties.inner_spine_damping_constant,
        time_step=dt,
    )

    for elem in range(octopus_arm_properties.n_elem):
        octopus_arm_sim.connect(
            first_rod=inner_spine_rod,
            second_rod=arm_rod,
            first_connect_idx=elem,
            second_connect_idx=elem,
        ).using(
            ParallelJointInterior,
            k=octopus_arm_properties.inner_spine_shear_modulus,
            nu=octopus_arm_properties.inner_spine_damping_constant,
            k_repulsive=0,
        )

    inner_spine_post_processing_dict = defaultdict(
        list
    )  # list which collected data will be append
    # set the diagnostics for rod and collect data
    octopus_arm_sim.collect_diagnostics(inner_spine_rod).using(
        MuscleCallBack,
        step_skip=step_skip,
        callback_params=inner_spine_post_processing_dict,
    )
    post_processing_dict_list.append(inner_spine_post_processing_dict)

    slope_angle = np.arctan2(
        octopus_arm_properties.length, arm_radius[0] - arm_radius[-1]
    )
    arm_slope = (arm_radius[-1] - arm_radius[0]) / octopus_arm_properties.length

    muscles = {}

    for row, theta, orientation in muscles_configuration:
        current_muscle = muscle_class()  # create muscle class instance
        if orientation == "CCW":
            CCW = (True, True)
            link_sign = -1
        else:
            CCW = (False, False)
            link_sign = 1

        muscle_position_radius = (
            arm_radius[0]
            + arm_slope * row * muscle_height
            + (arm_radius / np.sin(slope_angle))
        )

        muscle_length = conical_helix_length(
            muscle_position_radius,
            arm_slope,
            muscle_height,
            n_turns_per_backbone_length,
        )

        n_elem_per_row = int(
            muscle_height
            * octopus_arm_properties.n_elem
            / octopus_arm_properties.length
        )

        # adjust muscle properties to attach on arm
        curved_turns_per_length_list = [n_turns_per_backbone_length]

        curved_start_radius_list = [muscle_position_radius]

        # curved_offset_list = [theta]

        curved_taper_slope_list = [arm_slope]

        for radius in current_muscle.geometry.start_radius_list:
            curved_start_radius_list.append(radius)
        current_muscle.geometry.start_radius_list = curved_start_radius_list

        for turns_per_length in current_muscle.geometry.turns_per_length_list:
            n_muscle_turns_at_coil_level = (
                turns_per_length * current_muscle.geometry.muscle_length
            )
            n_muscle_turns_per_height = (
                int(n_muscle_turns_at_coil_level) / muscle_height
            )
            curved_turns_per_length_list.append(n_muscle_turns_per_height)
        current_muscle.geometry.turns_per_length_list = curved_turns_per_length_list

        # for offset in current_muscle.geometry.offset_list: curved_offset_list.append(offset)
        # current_muscle.geometry.offset_list = curved_offset_list
        current_muscle.geometry.angular_offset = theta

        for taper_slope in current_muscle.geometry.taper_slope_list:
            curved_taper_slope_list.append(taper_slope)
        current_muscle.geometry.taper_slope_list = curved_taper_slope_list

        current_muscle.geometry.start_position = (
            current_muscle.geometry.start_position
            + row * muscle_height * octopus_arm_properties.direction
        )
        current_muscle.geometry.direction = octopus_arm_properties.direction
        current_muscle.geometry.normal = octopus_arm_properties.normal

        n_turns = (
            current_muscle.geometry.turns_per_length_list[0]
            * current_muscle.geometry.muscle_length
        )
        current_muscle.sim_settings.n_elem_per_coil = (
            octopus_arm_properties.n_elem / n_turns
        )  # adjust n_elem_per_coil so n_elem turns out to be 200
        muscles[(row, theta, orientation)] = CoiledMuscle(
            current_muscle.geometry,
            current_muscle.properties,
            current_muscle.sim_settings,
        )

        # Append muscle
        muscles[(row, theta, orientation)].append_muscle_to_sim(octopus_arm_sim)

        # Muscle callback
        muscles[(row, theta, orientation)].muscle_callback(
            octopus_arm_sim,
            post_processing_dict_list,
            step_skip,
        )

        # Add damping
        muscles[(row, theta, orientation)].dampen_muscle(
            octopus_arm_sim,
            AnalyticalLinearDamper,
            damping_constant=current_muscle.sim_settings.nu,
            time_step=dt,
        )

        for muscle_rod in muscles[(row, theta, orientation)].muscle_rods:
            (
                rod_one_direction_vec_in_material_frame,
                rod_two_direction_vec_in_material_frame,
                offset_btw_rods,
            ) = get_connection_vector_straight_straight_rod(
                muscle_rod,
                arm_rod,
                (0, 1),
                (row * n_elem_per_row, (row * n_elem_per_row) + 1),
            )

            octopus_arm_sim.connect(first_rod=muscle_rod, second_rod=arm_rod,).using(
                SurfaceJointSideBySide,
                first_connect_idx=0,
                second_connect_idx=row * n_elem_per_row,
                k=muscle_rod.shear_matrix[2, 2, 0] * 100,
                nu=1e-4,
                k_repulsive=muscle_rod.shear_matrix[2, 2, n_elem_muscle - 1] * 100,
                rod_one_direction_vec_in_material_frame=rod_one_direction_vec_in_material_frame[
                    ..., 0
                ],
                rod_two_direction_vec_in_material_frame=rod_two_direction_vec_in_material_frame[
                    ..., 0
                ],
                offset_btw_rods=offset_btw_rods[0],
            )

            (
                rod_one_direction_vec_in_material_frame,
                rod_two_direction_vec_in_material_frame,
                offset_btw_rods,
            ) = get_connection_vector_straight_straight_rod(
                muscle_rod,
                arm_rod,
                (n_elem_muscle - 1, n_elem_muscle),
                (((row + 1) * n_elem_per_row) - 1, (row + 1) * n_elem_per_row),
            )

            octopus_arm_sim.connect(
                first_rod=muscle_rod,
                second_rod=arm_rod,
                first_connect_idx=n_elem_muscle - 1,
                second_connect_idx=((row + 1) * n_elem_per_row) - 1,
            ).using(
                SurfaceJointSideBySide,
                k=muscle_rod.shear_matrix[2, 2, n_elem_muscle - 1] * 100,
                nu=1e-4,
                k_repulsive=muscle_rod.shear_matrix[2, 2, n_elem_muscle - 1] * 100,
                rod_one_direction_vec_in_material_frame=rod_one_direction_vec_in_material_frame[
                    ..., 0
                ],
                rod_two_direction_vec_in_material_frame=rod_two_direction_vec_in_material_frame[
                    ..., 0
                ],
                offset_btw_rods=offset_btw_rods[0],
            )

            for element in range(1, n_elem_muscle - 1):
                arm_element = int(element * n_elem_per_row / n_elem_muscle)
                (
                    rod_one_direction_vec_in_material_frame,
                    rod_two_direction_vec_in_material_frame,
                    offset_btw_rods,
                ) = get_connection_vector_straight_straight_rod(
                    muscle_rod,
                    arm_rod,
                    (element, element + 1),
                    (arm_element, arm_element + 1),
                )

                octopus_arm_sim.connect(
                    first_rod=muscle_rod,
                    second_rod=arm_rod,
                    first_connect_idx=element,
                    second_connect_idx=arm_element,
                ).using(
                    SurfaceJointSideBySide,
                    k=muscle_rod.shear_matrix[2, 2, element] / 100,
                    nu=0,
                    k_repulsive=muscle_rod.shear_matrix[2, 2, element] / 10,
                    rod_one_direction_vec_in_material_frame=rod_one_direction_vec_in_material_frame[
                        ..., 0
                    ],
                    rod_two_direction_vec_in_material_frame=rod_two_direction_vec_in_material_frame[
                        ..., 0
                    ],
                    offset_btw_rods=offset_btw_rods[0],
                )

    assert len(muscles_activation_signal.activation_startTime_untwistTime_force) == len(
        muscles_activation_signal.activation_group
    ), "Make sure each activated muscle has start time, untwist time, and force specified"
    for muscle_coords, startTime_untwistTime_force in zip(
        muscles_activation_signal.activation_group,
        muscles_activation_signal.activation_startTime_untwistTime_force,
    ):
        start_time = startTime_untwistTime_force[0]
        untwist_time = startTime_untwistTime_force[1]
        kappa_change = startTime_untwistTime_force[2]
        row, theta, orientation = muscle_coords

        muscles[(row, theta, orientation)].actuate(
            octopus_arm_sim,
            ArtficialMuscleActuation,
            contraction_time=sim_settings.time_untwisting,
            start_time=sim_settings.untwisting_start_time,
            kappa_change=muscles[
                (row, theta, orientation)
            ].muscle_sim_settings.actuation_kappa_change,
            room_temperature=muscles[
                (row, theta, orientation)
            ].muscle_sim_settings.actuation_start_temperature,
            end_temperature=muscles[
                (row, theta, orientation)
            ].muscle_sim_settings.actuation_end_temperature,
            youngs_modulus_coefficients=muscles[
                (row, theta, orientation)
            ].muscle_properties.youngs_modulus_coefficients,
            thermal_expansion_coefficient=muscles[
                (row, theta, orientation)
            ].muscle_properties.thermal_expansion_coefficient,
        )

    # finalize simulation
    octopus_arm_sim.finalize()

    for row, theta, orientation in muscles_configuration:
        muscles[(row, theta, orientation)].fix_shape_and_store_start_properties()

    # Run the simulation
    time_stepper = PositionVerlet()
    integrate(time_stepper, octopus_arm_sim, sim_settings.final_time, total_steps)

    # plotting the videos
    filename_video = "mini_muri.mp4"
    plot_video_with_surface(
        post_processing_dict_list,
        video_name=filename_video,
        fps=rendering_fps,
        step=1,
        vis3D=True,
        vis2D=True,
        x_limits=[-(octopus_arm_properties.length), (octopus_arm_properties.length)],
        y_limits=[-(octopus_arm_properties.length), (octopus_arm_properties.length)],
        z_limits=[
            -2 * octopus_arm_properties.backbone_radius[0],
            octopus_arm_properties.length
            + 2 * octopus_arm_properties.backbone_radius[0],
        ],
    )

    if sim_settings.save_data:
        # Save data as npz file
        import os

        current_path = os.getcwd()
        save_folder = os.path.join(current_path, "data")
        os.makedirs(save_folder, exist_ok=True)
        time = np.array(post_processing_dict_list[0]["time"])

        n_muscle_rod = len(post_processing_dict_list) - 2

        muscle_rods_position_history = np.zeros(
            (n_muscle_rod, time.shape[0], 3, n_elem_muscle + 1)
        )
        muscle_rods_radius_history = np.zeros(
            (n_muscle_rod, time.shape[0], n_elem_muscle)
        )
        backbone_position_history = np.zeros((1, time.shape[0], 3, n_elem_muscle + 1))
        backbone_radius_history = np.zeros((1, time.shape[0], n_elem_muscle))
        inner_spine_position_history = np.zeros(
            (1, time.shape[0], 3, n_elem_muscle + 1)
        )
        inner_spine_radius_history = np.zeros((1, time.shape[0], n_elem_muscle))
        marker_position_history = np.zeros((1, time.shape[0], 3, n_elem_muscle))
        marker_radius_history = (
            np.ones((1, time.shape[0], n_elem_muscle - 1)) * inner_spine_radius / 2
        )

        for t in range(len(time)):
            marker_position_history[0, t, :, :] = (
                np.array(post_processing_dict_list[0]["position"])[t, :, :-1]
                + np.array(post_processing_dict_list[0]["radius"])[t, :]
                * np.array(post_processing_dict_list[0]["directors"])[t, 0, :, :]
            )

        backbone_position_history[0, :, :, :] = np.array(
            post_processing_dict_list[0]["position"]
        )
        backbone_radius_history[0, :, :] = np.array(
            post_processing_dict_list[0]["radius"]
        )

        inner_spine_position_history[0, :, :, :] = np.array(
            post_processing_dict_list[1]["position"]
        )
        inner_spine_radius_history[0, :, :] = np.array(
            post_processing_dict_list[1]["radius"]
        )
        for i in range(0, n_muscle_rod):
            muscle_rods_position_history[i, :, :, :] = np.array(
                post_processing_dict_list[i + 2]["position"]
            )
            muscle_rods_radius_history[i, :, :] = np.array(
                post_processing_dict_list[i + 2]["radius"]
            )

        np.savez(
            os.path.join(save_folder, "octopus_arm.npz"),
            time=time,
            muscle_rods_position_history=muscle_rods_position_history,
            muscle_rods_radius_history=muscle_rods_radius_history,
            backbone_position_history=backbone_position_history,
            backbone_radius_history=backbone_radius_history,
            inner_spine_position_history=inner_spine_position_history,
            inner_spine_radius_history=inner_spine_radius_history,
            marker_position_history=marker_position_history,
            marker_radius_history=marker_radius_history,
        )
