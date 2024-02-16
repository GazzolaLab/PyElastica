import numpy as np
from elastica import *
from examples.ArtificialMusclesCases import *
from parallel_muscle_simulator import parallel_muscle_contraction_simulation

antagonistic_sim_settings_dict = {
    "sim_name": "AntgonisticRectangularBeam",
    "final_time": 10,  # seconds
    "untwisting_start_time": 0,  # seconds
    "time_untwisting": 1,  # seconds
    "rendering_fps": 20,
    "contraction": True,
    "plot_video": True,
    "save_data": False,
    "return_data": False,
    "povray_viz": True,
    "isometric_test": False,
    "isobaric_test": False,
    "self_contact": True,
    "muscle_strain": 0.0,
    "force_mag": 0.0,
}


antagonistic_sim_settings = Dict2Class(antagonistic_sim_settings_dict)
first_muscle = Liuyang_monocoil()
second_muscle = Liuyang_monocoil()


# Create rigid beam
x_scale = 40e-3
y_scale = 5e-3
z_scale = 5e-3

rigid_body_mesh = Mesh(r"cube.stl")
rigid_body_mesh.scale(np.array([x_scale, y_scale, z_scale]))
center_of_mass = np.zeros((3,), np.float64)
base_length = 2
rigid_beam_volume = x_scale * y_scale * z_scale * base_length ** 3
rigid_beam_density = 109
mass = rigid_beam_density * rigid_beam_volume
# Mass second moment of inertia for cube
mass_second_moment_of_inertia = np.zeros((3, 3), np.float64)
mass_second_moment_of_inertia[0, 0] = (
    (x_scale ** 2 + y_scale ** 2) * mass * base_length ** 2
) / 12
mass_second_moment_of_inertia[1, 1] = (
    (z_scale ** 2 + y_scale ** 2) * mass * base_length ** 2
) / 12
mass_second_moment_of_inertia[2, 2] = (
    (z_scale ** 2 + x_scale ** 2) * mass * base_length ** 2
) / 12


rigid_body_properties_dict = {
    "center_of_mass": center_of_mass,
    "density": rigid_beam_density,
    "volume": rigid_beam_volume,
    "mass_second_moment_of_inertia": mass_second_moment_of_inertia,
    "base_length": base_length,
}

connection_distance = 20e-3

direction_to_first_muscle = (
    z_scale * first_muscle.geometry.direction
    + connection_distance * first_muscle.geometry.normal
)
distance_to_first_muscle = np.linalg.norm(direction_to_first_muscle)
direction_to_first_muscle /= distance_to_first_muscle
direction_to_second_muscle = (
    z_scale * first_muscle.geometry.direction
    - connection_distance * first_muscle.geometry.normal
)
distance_to_second_muscle = np.linalg.norm(direction_to_second_muscle)
direction_to_second_muscle /= distance_to_second_muscle


muscle_rigid_body_connections_dict = {
    "direction_to_first_muscle": direction_to_first_muscle,
    "distance_to_first_muscle": distance_to_first_muscle,
    "direction_to_second_muscle": direction_to_second_muscle,
    "distance_to_second_muscle": distance_to_second_muscle,
    "distance_to_pivot_from_center": 0.0,
    "direction_to_pivot_from_center": direction_to_first_muscle,
    "pivot_position": center_of_mass,
}

rigid_body_properties = Dict2Class(rigid_body_properties_dict)
muscle_rigid_body_connections = Dict2Class(muscle_rigid_body_connections_dict)


parallel_muscle_contraction_simulation(
    first_input_muscle=first_muscle,
    second_input_muscle=second_muscle,
    rigid_body_mesh=rigid_body_mesh,
    rigid_body_properties=rigid_body_properties,
    muscle_rigid_body_connections=muscle_rigid_body_connections,
    sim_settings=antagonistic_sim_settings,
)
