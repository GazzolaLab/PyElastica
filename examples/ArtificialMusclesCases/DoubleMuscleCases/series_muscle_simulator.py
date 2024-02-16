import numpy as np
import os
from elastica import *
from examples.ArtificialMusclesCases import *
from tqdm import tqdm


class MuscleCase(
    BaseSystemCollection,
    Constraints,
    MemoryBlockConnections,
    Forcing,
    CallBacks,
    Damping,
):
    pass


def series_muscle_contraction_simulation(
    first_input_muscle, second_input_muscle, sim_settings
):
    # create new sim
    muscle_sim = MuscleCase()
    post_processing_dict_list = []  # list which collected data will be append
    point1 = np.array([0.0, 0.0, 0.0])
    point2 = np.array([0.0, 0.0, 0.0])

    # save folder
    current_path = os.getcwd()
    save_folder = os.path.join(
        current_path,
        sim_settings.sim_name
        + "/"
        + first_input_muscle.name
        + second_input_muscle.name
        + "/data",
    )
    os.makedirs(save_folder, exist_ok=True)

    # time step calculation
    n_turns = max(
        first_input_muscle.geometry.turns_per_length_list[0]
        * first_input_muscle.geometry.muscle_length,
        second_input_muscle.geometry.turns_per_length_list[0]
        * second_input_muscle.geometry.muscle_length,
    )
    n_elem = max(
        first_input_muscle.sim_settings.n_elem_per_coil,
        second_input_muscle.sim_settings.n_elem_per_coil,
    ) * int(n_turns)
    dt = (
        0.3
        * min(
            first_input_muscle.geometry.muscle_length,
            second_input_muscle.geometry.muscle_length,
        )
        / n_elem
    )
    total_steps = int(sim_settings.final_time / dt)
    time_step = np.float64(sim_settings.final_time / total_steps)
    step_skip = int(1.0 / (sim_settings.rendering_fps * time_step))

    # create coiled muscle
    first_muscle = CoiledMuscle(
        first_input_muscle.geometry,
        first_input_muscle.properties,
        first_input_muscle.sim_settings,
    )
    second_input_muscle.geometry.start_position += (
        first_input_muscle.geometry.direction
        * first_input_muscle.geometry.muscle_length
    )
    second_muscle = CoiledMuscle(
        second_input_muscle.geometry,
        second_input_muscle.properties,
        second_input_muscle.sim_settings,
    )

    muscle_list = [first_muscle, second_muscle]
    muscle_geometry_list = [first_input_muscle.geometry, second_input_muscle.geometry]
    muscle_sim_settings_list = [
        first_input_muscle.sim_settings,
        second_input_muscle.sim_settings,
    ]
    muscle_properties_list = [
        first_input_muscle.properties,
        second_input_muscle.properties,
    ]

    for muscle, muscle_geometry, muscle_sim_settings, muscle_properties in zip(
        muscle_list,
        muscle_geometry_list,
        muscle_sim_settings_list,
        muscle_properties_list,
    ):
        muscle.append_muscle_to_sim(muscle_sim)

        # set the diagnostics for muscle and collect data
        muscle.muscle_callback(
            muscle_sim,
            post_processing_dict_list,
            step_skip,
        )

        # Add damping
        muscle.dampen_muscle(
            muscle_sim,
            AnalyticalLinearDamper,
            damping_constant=muscle_sim_settings.nu,
            time_step=dt,
        )

        # Slider constarint
        muscle.constrain_muscle(
            muscle_sim,
            GeneralConstraint,
            constrained_position_idx=(-1,),
            constrained_director_idx=(-1,),
            translational_constraint_selector=np.array([True, True, False]),
            rotational_constraint_selector=np.array([True, True, True]),
        )

        muscle.constrain_muscle(
            muscle_sim,
            GeneralConstraint,
            constrained_position_idx=(0,),
            constrained_director_idx=(0,),
            translational_constraint_selector=np.array([True, True, False]),
            rotational_constraint_selector=np.array([True, True, True]),
        )

        # Add self contact to prevent penetration
        if sim_settings.self_contact:
            muscle.apply_self_contact(muscle_sim)

        base_area = np.pi * muscle_geometry.fiber_radius ** 2
        force_scale = (
            1e-3 * sim_settings.force_mag * muscle_properties.youngs_modulus * base_area
        )
        zero_force = np.array([0.0, 0.0, 0.0])
        end_force = force_scale * np.array([0.0, 0.0, 1])
        start_force = force_scale * np.array([0.0, 0.0, -1])
        # print("Force: "+str(end_force*muscle_sim_settings.E_scale))

        # stretch muscles
        if muscle is first_muscle:
            # actuate muscle
            if sim_settings.contraction:
                muscle.actuate(
                    muscle_sim,
                    ArtficialMuscleActuation,
                    contraction_time=sim_settings.time_untwisting,
                    start_time=sim_settings.untwisting_start_time,
                    kappa_change=muscle_sim_settings.actuation_kappa_change,
                    room_temperature=muscle_sim_settings.actuation_start_temperature,
                    end_temperature=muscle_sim_settings.actuation_end_temperature,
                    youngs_modulus_coefficients=muscle_properties.youngs_modulus_coefficients,
                    thermal_expansion_coefficient=muscle_properties.thermal_expansion_coefficient,
                )
            muscle.connect_muscle_to_point(
                muscle_sim,
                nu=0,
                point=point1,
                index=-1,
            )

            if sim_settings.muscle_strain > 0.0:
                muscle.constrain_muscle(
                    muscle_sim,
                    IsometricStrainBC,
                    desired_length=(1 + sim_settings.muscle_strain)
                    * muscle_geometry.muscle_length,
                    constraint_node_idx=[0],
                    length_node_idx=[0, -1],
                    direction=muscle_geometry.direction,
                )
                muscle.add_forcing_to_muscle(
                    muscle_sim,
                    EndpointForces,
                    start_force=start_force,
                    end_force=zero_force,
                    ramp_up_time=time_step,
                )
            else:
                muscle.constrain_muscle(
                    muscle_sim,
                    IsometricBC,
                    constrain_start_time=sim_settings.untwisting_start_time,
                    constrained_nodes=[0],
                )
        else:
            muscle.connect_muscle_to_point(
                muscle_sim,
                nu=0,
                point=point2,
                index=0,
            )
            if sim_settings.muscle_strain > 0.0:
                muscle.constrain_muscle(
                    muscle_sim,
                    IsometricStrainBC,
                    desired_length=(1 + sim_settings.muscle_strain)
                    * muscle_geometry.muscle_length,
                    constraint_node_idx=[-1],
                    length_node_idx=[0, -1],
                    direction=muscle_geometry.direction,
                )
                muscle.add_forcing_to_muscle(
                    muscle_sim,
                    EndpointForces,
                    start_force=zero_force,
                    end_force=end_force,
                    ramp_up_time=time_step,
                )
            else:
                muscle.constrain_muscle(
                    muscle_sim,
                    IsometricBC,
                    constrain_start_time=sim_settings.untwisting_start_time,
                    constrained_nodes=[-1],
                )

    # finalize simulation
    muscle_sim.finalize()

    # store initial properties for actuation
    # fix muscle shape/annealing
    for muscle in muscle_list:
        muscle.fix_shape_and_store_start_properties()

    # Run the simulation
    time_stepper = PositionVerlet()

    do_step, stages_and_updates = extend_stepper_interface(time_stepper, muscle_sim)

    dt = np.float64(float(sim_settings.final_time) / total_steps)

    time = 0
    progress_bar = True
    for i in tqdm(range(total_steps), disable=(not progress_bar)):
        try:
            _point1 = 0
            _point2 = 0
            n_muscle_rods_1 = len(first_muscle.rod_ids)
            n_muscle_rods_2 = len(second_muscle.rod_ids)
            for rod_id in second_muscle.rod_ids:
                _point1 += (
                    second_muscle.muscle_rods[rod_id].position_collection[:, 0]
                    * second_input_muscle.geometry.direction
                    / n_muscle_rods_2
                )

            for rod_id in first_muscle.rod_ids:
                _point2 += (
                    first_muscle.muscle_rods[rod_id].position_collection[:, -1]
                    * first_input_muscle.geometry.direction
                    / n_muscle_rods_1
                )

            point1[:] = _point1
            point2[:] = _point2
            time = do_step(time_stepper, stages_and_updates, muscle_sim, time, dt)
        except RuntimeError:
            print("RuntimeError, Exiting Sim")

    # plotting the videos
    if sim_settings.plot_video:
        filename_video = (
            sim_settings.sim_name
            + first_input_muscle.name
            + second_input_muscle.name
            + ".mp4"
        )
        plot_video_with_surface(
            post_processing_dict_list,
            folder_name=save_folder,
            video_name=filename_video,
            fps=sim_settings.rendering_fps,
            step=1,
            vis3D=True,
            vis2D=True,
            x_limits=[
                -muscle_geometry.muscle_length / 2,
                muscle_geometry.muscle_length / 2,
            ],
            y_limits=[
                -muscle_geometry.muscle_length / 2,
                muscle_geometry.muscle_length / 2,
            ],
            z_limits=[
                -1.1 * muscle_geometry.muscle_length,
                3.1 * muscle_geometry.muscle_length,
            ],
        )

    if sim_settings.povray_viz:
        import pickle

        filename = (
            save_folder
            + "/"
            + first_input_muscle.name
            + second_input_muscle.name
            + ".dat"
        )
        file = open(filename, "wb")
        pickle.dump(post_processing_dict_list, file)
        file.close()

    if sim_settings.save_data:
        # Save data as npz file
        time = np.array(post_processing_dict_list[0]["time"])

        n_muscle_rod = len([muscle.muscle_rods])

        muscle_rods_position_history = np.zeros(
            (n_muscle_rod, time.shape[0], 3, n_elem + 1)
        )
        muscle_rods_radius_history = np.zeros((n_muscle_rod, time.shape[0], n_elem))
        muscle_rods_tangent_history = np.zeros((n_muscle_rod, time.shape[0], 3, n_elem))
        muscle_rods_internal_force_history = np.zeros(
            (n_muscle_rod, time.shape[0], 3, n_elem + 1)
        )
        muscle_rods_external_force_history = np.zeros(
            (n_muscle_rod, time.shape[0], 3, n_elem + 1)
        )

        for i in range(n_muscle_rod):
            muscle_rods_position_history[i, :, :, :] = np.array(
                post_processing_dict_list[i]["position"]
            )
            muscle_rods_radius_history[i, :, :] = np.array(
                post_processing_dict_list[i]["radius"]
            )
            directors = np.array(post_processing_dict_list[i]["directors"])
            muscle_rods_tangent_history[i, :, :, :] = np.array(directors[:, 2, :, :])
            muscle_rods_internal_force_history[i, :, :, :] = np.array(
                post_processing_dict_list[i]["internal_force"]
            )
            muscle_rods_external_force_history[i, :, :, :] = np.array(
                post_processing_dict_list[i]["external_force"]
            )

        np.savez(
            os.path.join(
                save_folder,
                sim_settings.sim_name
                + first_input_muscle.name
                + second_input_muscle.name
                + ".npz",
            ),
            time=time,
            muscle_rods_position_history=muscle_rods_position_history,
            muscle_rods_radius_history=muscle_rods_radius_history,
        )

    if sim_settings.return_data:
        return post_processing_dict_list
