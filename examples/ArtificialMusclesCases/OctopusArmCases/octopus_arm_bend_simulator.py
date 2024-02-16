import numpy as np
from elastica import *
from examples.ArtificialMusclesCases import *
from octopus_arm_simulator import octopus_arm_simulation


octopus_arm_properties_dict = {
    # geometry
    "start_radius": 12e-3,
    "length": 200e-3,
    "taper_ratio": 1 / 9,  # 1/12,
    "direction": np.array([0.0, 0.0, 1.0]),
    "normal": np.array([1.0, 0.0, 0.0]),
    "start": np.zeros(
        3,
    ),
    # octopus arm material properties
    "arm_youngs_modulus": 3,
    "arm_shear_modulus": 2,
    "arm_density": 920,
    # inner spine material properties
    "inner_spine_youngs_modulus": 5000,
    "inner_spine_shear_modulus": 3333,
    "inner_spine_density": 2000,
    # sim settings
    "arm_damping_constant": 1e-5,
    "inner_spine_damping_constant": 3e-3,
    "n_elem": 200,
    # muscle-arm relation
    "n_rows": 8,
    "n_muscles": 8,
}


sim_settings_dict = {
    "final_time": 1,
    "self_contact": False,
    "save_data": True,
}

octopus_arm_properties = Dict2Class(octopus_arm_properties_dict)

muscle_class = Liuyang_monocoil

sim_settings = Dict2Class(sim_settings_dict)

# muscle configuration
n_muscles = octopus_arm_properties.n_muscles
n_rows = octopus_arm_properties.n_rows
rows = range(n_rows)
muscles = range(n_muscles)


all_muscles_coords = []
front_diamonds_coords = []
back_diamonds_coords = []
right_diamonds_coords = []
left_diamonds_coords = []

for diamond in range(int(n_rows / 2)):
    front_diamonds_coords += [
        (2 * diamond, 0, "CCW"),
        (2 * diamond, 0, "CW"),
        (2 * diamond + 1, 2 * np.pi / n_rows, "CW"),
        (2 * diamond + 1, -2 * np.pi / n_rows, "CCW"),
    ]

for diamond in range(int(n_rows / 2)):
    right_diamonds_coords += [
        (2 * diamond, np.pi / 2, "CCW"),
        (2 * diamond, np.pi / 2, "CW"),
        (2 * diamond + 1, np.pi / 2 + 2 * np.pi / n_rows, "CW"),
        (2 * diamond + 1, np.pi / 2 - 2 * np.pi / n_rows, "CCW"),
    ]

for diamond in range(int(n_rows / 2)):
    back_diamonds_coords += [
        (2 * diamond, np.pi, "CCW"),
        (2 * diamond, np.pi, "CW"),
        (2 * diamond + 1, np.pi + 2 * np.pi / n_rows, "CW"),
        (2 * diamond + 1, np.pi - 2 * np.pi / n_rows, "CCW"),
    ]

for diamond in range(int(n_rows / 2)):
    left_diamonds_coords += [
        (2 * diamond, -np.pi / 2, "CCW"),
        (2 * diamond, -np.pi / 2, "CW"),
        (2 * diamond + 1, -np.pi / 2 + 2 * np.pi / n_rows, "CW"),
        (2 * diamond + 1, -np.pi / 2 - 2 * np.pi / n_rows, "CCW"),
    ]

all_muscles_coords = (
    front_diamonds_coords
    + right_diamonds_coords
    + back_diamonds_coords
    + left_diamonds_coords
)


cw_twist_coords = []
for row in rows:
    for i in range(4):
        theta = -np.pi * row / 4 + np.pi * i / 2
        cw_twist_coords.append((row, theta, "CW"))


bottom_to_top_sequence = []
for diamond in range(int(n_rows / 2)):
    if diamond == int(n_rows / 2) - 1:
        bottom_to_top_sequence += 2 * [(28, 1, 10)]
    else:
        bottom_to_top_sequence += 4 * [(7 * diamond, 1, 10)]


quick_bend_activation = ((2 * n_rows) - 2) * [(0, 3, 10)] + 2 * [(0, 3, 1)]
quick_twist_activation = 7 * [(0, 3, 10)] + [(0, 3, 1)]

spiral_diamonds_coords = (
    front_diamonds_coords[0:4]
    + right_diamonds_coords[4:8]
    + back_diamonds_coords[8:12]
    + left_diamonds_coords[12:14]
)

included_muscles = spiral_diamonds_coords
activation_group = spiral_diamonds_coords
activation_startTime_untwistTime_force = bottom_to_top_sequence

muscles_activation_signal_dict = {
    "activation_group": spiral_diamonds_coords,
    "activation_startTime_untwistTime_force": bottom_to_top_sequence,
}

muscles_activation_signal = Dict2Class(muscles_activation_signal_dict)

octopus_arm_simulation(
    octopus_arm_properties,
    muscle_class,
    muscles_configuration=all_muscles_coords,
    muscles_activation_signal=muscles_activation_signal,
    sim_settings=sim_settings,
)
