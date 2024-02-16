import numpy as np
from elastica import *
from examples.ArtificialMusclesCases import *
from parallel_muscle_simulator import parallel_muscle_contraction_simulation

antagonistic_sim_settings_dict = {
    "sim_name": "AntgonisticArm",
    "final_time": 1,  # seconds
    "untwisting_start_time": 0,  # seconds
    "time_untwisting": 1,  # seconds
    "rendering_fps": 20,
    "contraction": True,
    "plot_video": True,
    "save_data": False,
    "return_data": False,
    "povray_viz": True,
    "isometric_test": True,
    "isobaric_test": False,
    "self_contact": False,
    "muscle_strain": 0.0,  # 0.05,
    "force_mag": 3.0,
}


antagonistic_sim_settings = Dict2Class(antagonistic_sim_settings_dict)

first_muscle = Liuyang_monocoil()
second_muscle = Liuyang_monocoil()

# Create rigid beam
x_scale = 1e-3
y_scale = 1e-3
z_scale = 1e-3

direction = first_muscle.geometry.direction
normal = first_muscle.geometry.normal
binormal = np.cross(direction, normal)

rigid_body_mesh = Mesh(r"lower_arm.stl")
rigid_body_mesh.translate(
    -np.array(rigid_body_mesh.mesh_center)
)  # move mesh center to 0,0,0
rigid_body_mesh.rotate(np.array([1, 0, 0]), 180)
rigid_body_mesh.scale(np.array([x_scale, y_scale, z_scale]))
model_scale = 1e-3
center_of_mass = (
    4.096e-1 * direction + 3.7692 * binormal + 3.055e-1 * normal
) * model_scale
# print(rigid_body_mesh.faces)
base_length = 1  # does not matter, just for rigid body init
rigid_beam_volume = 5.318e-6
density_scale = 1
rigid_beam_density = 1060 / density_scale  # ABS density
mass = rigid_beam_density * rigid_beam_volume
# Mass second moment of inertia for rigid arm
mass_second_moment_of_inertia = np.zeros((3, 3), np.float64)

# Moment of Inertia at Center of Mass   (kg m^2) imported from model CAD
mass_second_moment_of_inertia[0, 0] = 1.041e-07
mass_second_moment_of_inertia[0, 1] = 3.863e-08
mass_second_moment_of_inertia[0, 2] = -3.795e-08
mass_second_moment_of_inertia[1, 0] = 3.863e-08
mass_second_moment_of_inertia[1, 1] = 4.784e-06
mass_second_moment_of_inertia[1, 2] = 1.753e-10
mass_second_moment_of_inertia[2, 0] = -3.795e-08
mass_second_moment_of_inertia[2, 1] = 1.753e-10
mass_second_moment_of_inertia[2, 2] = 4.711e-06

mass_second_moment_of_inertia /= density_scale

rigid_body_properties_dict = {
    "center_of_mass": center_of_mass,
    "density": rigid_beam_density,
    "volume": rigid_beam_volume,
    "mass_second_moment_of_inertia": mass_second_moment_of_inertia,
    "base_length": base_length,
}


direction_to_first_muscle = (
    -center_of_mass + (15 * normal + 7e-1 * direction) * model_scale
)
distance_to_first_muscle = np.linalg.norm(direction_to_first_muscle)
direction_to_first_muscle /= distance_to_first_muscle
direction_to_second_muscle = (
    -center_of_mass + (45 * normal + 7e-1 * direction) * model_scale
)
distance_to_second_muscle = np.linalg.norm(direction_to_second_muscle)
direction_to_second_muscle /= distance_to_second_muscle
direction_to_pivot_from_center = (
    -center_of_mass + (30 * normal + 7e-1 * direction) * model_scale
)
distance_to_pivot_from_center = np.linalg.norm(direction_to_pivot_from_center)
direction_to_pivot_from_center /= distance_to_pivot_from_center

muscle_rigid_body_connections_dict = {
    "direction_to_first_muscle": direction_to_first_muscle,
    "distance_to_first_muscle": distance_to_first_muscle,
    "direction_to_second_muscle": direction_to_second_muscle,
    "distance_to_second_muscle": distance_to_second_muscle,
    "direction_to_pivot_from_center": direction_to_pivot_from_center,
    "distance_to_pivot_from_center": distance_to_pivot_from_center,
    "pivot_position": center_of_mass
    + distance_to_pivot_from_center * direction_to_pivot_from_center,
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
