import numpy as np
import os
from elastica import *
import argparse
from examples.ArtificialMusclesCases import *
from single_muscle_simulator_cluster import muscle_contraction_simulation
import matplotlib

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from elastica._linalg import (
    _batch_product_i_ik_to_k,
    _batch_norm,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--test_type", choices=["Isometric", "Isobaric"], type=str, required=True
)
parser.add_argument("--force_mag", help="Axial Force on muscle", type=float)
parser.add_argument(
    "--muscle_strain", help="Strain for muscle to be stretched to", type=float
)
parser.add_argument(
    "--muscle_class",
    choices=["Liuyang_monocoil", "Samuel_monocoil", "Samuel_supercoil_stl"],
    required=True,
    type=str,
)


args = parser.parse_args()


test_sim_settings_dict = {
    "sim_name": "PassiveForceTest",
    "final_time": 6,  # seconds
    "untwisting_start_time": 1,  # seconds
    "time_untwisting": 0,  # seconds
    "rendering_fps": 20,
    "contraction": False,
    "plot_video": True,
    "save_data": True,
    "return_data": True,
    "povray_viz": False,
    "isometric_test": False,
    "isobaric_test": False,
    "self_contact": False,
    "muscle_strain": 1.0,
    "force_mag": 25.0,
}


test_sim_settings = Dict2Class(test_sim_settings_dict)

if args.test_type == "Isometric":
    test_sim_settings.isometric_test = True
    if args.muscle_strain == None:
        raise Exception("Please provide muscle strain")
    test_id = "_Isometric_" + str(args.muscle_strain)
elif args.test_type == "Isobaric":
    test_sim_settings.isobaric_test = True
    if args.force_mag == None:
        raise Exception("Please provide force magnitude")
    test_id = "_Isobaric_" + str(args.force_mag)

if args.muscle_class == "Liuyang_monocoil":
    test_muscle = Liuyang_monocoil()
elif args.muscle_class == "Samuel_monocoil":
    test_muscle = Samuel_monocoil()
elif args.muscle_class == "Samuel_supercoil_stl":
    test_muscle = Samuel_supercoil_stl()
temp = 20
test_muscle.properties.youngs_modulus *= gamma_func(
    temp, test_muscle.properties.youngs_modulus_coefficients, 25
)
test_muscle.properties.shear_modulus *= gamma_func(
    temp, test_muscle.properties.youngs_modulus_coefficients, 25
)

n_coils = (
    test_muscle.geometry.turns_per_length_list[0] * test_muscle.geometry.muscle_length
)
n_muscles = 1
for i in range(len(test_muscle.geometry.n_ply_per_coil_level)):
    n_muscles *= test_muscle.geometry.n_ply_per_coil_level[i]


current_path = os.getcwd()
save_folder = os.path.join(
    current_path,
    test_sim_settings.sim_name + "/" + test_muscle.name + test_id + "/data",
)

if test_sim_settings.isometric_test and not test_sim_settings.isobaric_test:
    test_sim_settings.muscle_strain = args.muscle_strain
    print("Current Strain:" + str(test_sim_settings.muscle_strain))
    data = muscle_contraction_simulation(
        input_muscle=test_muscle,
        sim_settings=test_sim_settings,
        save_folder=save_folder,
    )


elif not test_sim_settings.isometric_test and test_sim_settings.isobaric_test:
    base_area = np.pi * (test_muscle.geometry.fiber_radius ** 2)
    strain_sim = np.zeros_like(test_muscle.strain_experimental)
    coil_radius_sim = np.zeros_like(test_muscle.strain_experimental)
    initial_coil_radius = 0
    for radius in test_muscle.geometry.start_radius_list:
        initial_coil_radius += radius
    coil_radius_sim[0] = initial_coil_radius
    test_sim_settings.final_time = 1.0
    test_sim_settings.force_mag = args.force_mag / (
        1e-3
        * n_muscles
        * test_muscle.properties.youngs_modulus
        * test_muscle.sim_settings.E_scale
        * base_area
    )
    print(
        "Current Force:",
        args.force_mag,
        "Current Normalized Force Mag:",
        test_sim_settings.force_mag,
    )
    data = muscle_contraction_simulation(
        input_muscle=test_muscle,
        sim_settings=test_sim_settings,
        save_folder=save_folder,
    )
else:
    print("Please make sure one of isometric_test or isobaric_test is True, not both")
