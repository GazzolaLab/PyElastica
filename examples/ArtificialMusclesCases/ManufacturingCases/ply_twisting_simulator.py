import numpy as np
import os
from elastica import *
from examples.ArtificialMusclesCases import *


class PlyCase(
    BaseSystemCollection,
    Constraints,
    MemoryBlockConnections,
    Forcing,
    CallBacks,
    Damping,
):
    pass


def ply_twisting_simulation(input_ply, sim_settings):
    # create new sim
    ply_sim = PlyCase()

    # save folder
    current_path = os.getcwd()
    save_folder = os.path.join(
        current_path, sim_settings.sim_name + "/" + input_ply.name + "/data"
    )
    os.makedirs(save_folder, exist_ok=True)

    # time step calculation
    n_turns = (
        input_ply.geometry.turns_per_length_list[0] * input_ply.geometry.muscle_length
    )
    n_elem = input_ply.sim_settings.n_elem_per_coil * int(n_turns)
    dt = 0.3 * input_ply.geometry.muscle_length / n_elem
    total_steps = int(sim_settings.final_time / dt)
    time_step = np.float64(sim_settings.final_time / total_steps)
    step_skip = int(1.0 / (sim_settings.rendering_fps * time_step))

    # create ply
    ply = CoiledMuscle(input_ply.geometry, input_ply.properties, input_ply.sim_settings)

    ply.append_muscle_to_sim(ply_sim)

    ply.connect_muscle_rods(ply_sim)

    # Add damping
    ply.dampen_muscle(
        ply_sim,
        AnalyticalLinearDamper,
        damping_constant=input_ply.sim_settings.nu,
        time_step=dt,
    )

    # fix bottom
    ply.constrain_muscle(
        ply_sim,
        GeneralConstraint,
        constrained_position_idx=(0,),
        constrained_director_idx=(0,),
        translational_constraint_selector=np.array([True, True, True]),
        rotational_constraint_selector=np.array([True, True, True]),
    )

    # twist top
    ply.constrain_muscle(
        ply_sim,
        CoilTwistBC,
        constrain_start_time=sim_settings.twisting_start_time,
        coil_radius=input_ply.geometry.start_radius_list[1],
        coil_direction=input_ply.geometry.direction,
        twisting_angular_speed=sim_settings.twisting_angular_speed,
        twisted_nodes=[-1],
        dt=dt,
    )

    post_processing_dict_list = []  # list which collected data will be append
    # set the diagnostics for rod and collect data
    ply.muscle_callback(
        ply_sim,
        post_processing_dict_list,
        step_skip,
    )

    # finalize simulation
    ply_sim.finalize()

    # store initial properties for actuation
    # fix ply shape, just want to understand the affect of twisting on the geometry and topology
    ply.fix_shape_and_store_start_properties()

    # Run the simulation
    time_stepper = PositionVerlet()
    integrate(time_stepper, ply_sim, sim_settings.final_time, total_steps)

    # plotting the videos
    if sim_settings.plot_video:
        filename_video = sim_settings.sim_name + input_ply.name + ".mp4"
        plot_video_with_surface(
            post_processing_dict_list,
            folder_name=save_folder,
            video_name=filename_video,
            fps=sim_settings.rendering_fps,
            step=1,
            vis3D=True,
            vis2D=True,
            x_limits=[
                -input_ply.geometry.muscle_length / 2,
                input_ply.geometry.muscle_length / 2,
            ],
            y_limits=[
                -input_ply.geometry.muscle_length / 2,
                input_ply.geometry.muscle_length / 2,
            ],
            z_limits=[
                -1.1 * input_ply.geometry.muscle_length,
                2.1 * input_ply.geometry.muscle_length,
            ],
        )

    if sim_settings.povray_viz:
        import pickle

        filename = save_folder + "/" + sim_settings.sim_name + input_ply.name + ".dat"
        file = open(filename, "wb")
        pickle.dump(post_processing_dict_list, file)
        file.close()

    if sim_settings.save_data:
        # Save data as npz file
        time = np.array(post_processing_dict_list[0]["time"])

        n_muscle_rod = len([ply.muscle_rods])

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
            os.path.join(save_folder, sim_settings.sim_name + input_ply.name + ".npz"),
            time=time,
            muscle_rods_position_history=muscle_rods_position_history,
            muscle_rods_radius_history=muscle_rods_radius_history,
        )

    if sim_settings.return_data:
        return post_processing_dict_list
