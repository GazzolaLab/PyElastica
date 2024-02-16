import numpy as np
import os
from elastica import *
from examples.ArtificialMusclesCases import *
from single_muscle_simulator import muscle_contraction_simulation
import matplotlib

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from elastica._linalg import (
    _batch_product_i_ik_to_k,
    _batch_norm,
)


test_sim_settings_dict = {
    "sim_name": "PassiveForceTest",
    "final_time": 6,  # seconds
    "untwisting_start_time": 1,  # seconds
    "time_untwisting": 0,  # seconds
    "rendering_fps": 20,
    "contraction": False,
    "plot_video": True,
    "save_data": False,
    "return_data": True,
    "povray_viz": False,
    "isometric_test": False,
    "isobaric_test": True,
    "self_contact": False,
    "theoretical_curve": False,
    "additional_curves": False,
    "muscle_strain": 1.0,
    "force_mag": 25.0,
}


test_sim_settings = Dict2Class(test_sim_settings_dict)

# test_muscle = Samuel_supercoil_stl()
test_muscle = Liuyang_monocoil()
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

# tested_idx = range(len(test_muscle.strain_experimental)) #[0,1,2,3,4] #
tested_idx = range(len(test_muscle.passive_force_experimental))  # [0,1,2,3,4] #
# tested_strains_idx = range(len(test_muscle.strain_experimental))
# tested_strains_idx = [0,2,4,6,8]
# additional_curve_list = [(test_muscle.experimental_tensile_test_single_fiber,"Single Fiber Tensile Test")]
# additional_curve_list = [(test_muscle.experimental_tensile_test_single_fiber_times_3,"Single Fiber Tensile Test ×3"),(test_muscle.experimental_tensile_test,"Supercoil Tensile Test")]
additional_curve_list = []

if len(additional_curve_list) > 0 and test_sim_settings.additional_curves == False:
    raise (
        "You have additional curves to plot but you have set the plotting option to false"
    )
elif len(additional_curve_list) == 0 and test_sim_settings.additional_curves == True:
    raise (
        "You have no additional curves to plot but you have set the plotting option to True"
    )

passive_force_theory = np.zeros_like(test_muscle.strain_experimental)

current_path = os.getcwd()
save_folder = os.path.join(
    current_path, test_sim_settings.sim_name + "/" + test_muscle.name + "/data"
)
D = 2 * test_muscle.geometry.start_radius_list[0]
if len(test_muscle.geometry.start_radius_list) == 1:
    d = 2 * test_muscle.geometry.fiber_radius
else:
    d = 2 * test_muscle.geometry.start_radius_list[1]
G = test_muscle.sim_settings.E_scale * test_muscle.properties.shear_modulus
k1 = G * (d ** 4 / (8 * n_coils * D ** 3))
k2 = k1 * ((1 + (0.5 * (d / D) ** 2)) ** -1)

if test_sim_settings.isometric_test and not test_sim_settings.isobaric_test:
    passive_force_sim = np.zeros_like(test_muscle.strain_experimental)
    for i in tested_idx[1:]:
        test_sim_settings.muscle_strain = test_muscle.strain_experimental[i]
        print("Current Strain:" + str(test_sim_settings.muscle_strain))
        data = muscle_contraction_simulation(
            input_muscle=test_muscle, sim_settings=test_sim_settings
        )

        time = data[0]["time"]

        passive_force_measurement_time = -1  # end of simulation

        internal_force = np.zeros_like(np.array(data[0]["internal_force"]))
        for muscle_rods in data:
            internal_force += np.array(muscle_rods["internal_force"])

        passive_force_sim[i] = (
            internal_force[passive_force_measurement_time, 2, 0]
            * test_muscle.sim_settings.E_scale
        )
        # plt.plot(internal_force[:,2,0])
        # plt.show()
        # passive_force_sim[i] = np.max(internal_force[:,2,0])*test_muscle.sim_settings.E_scale

        passive_force_theory[i] = (
            k2 * test_sim_settings.muscle_strain * test_muscle.geometry.muscle_length
        )
        print("Current Passive Force: " + str(passive_force_sim[i]))

    plt.rc("font", size=8)  # controls default text sizes
    plt.plot(
        test_muscle.strain_experimental[tested_idx],
        passive_force_sim[tested_idx],
        linewidth=3,
        marker="o",
        markersize=7,
        label="Passive force (Sim,Isometric)",
    )
    plt.suptitle("Coil Strain vs Force")
    plt.xlabel("Strain (mm/mm)")
    plt.ylabel("Force (N)")
    plt.xlim([0, 1.1 * test_muscle.strain_experimental[tested_idx[-1]]])


elif not test_sim_settings.isometric_test and test_sim_settings.isobaric_test:
    base_area = np.pi * (test_muscle.geometry.fiber_radius ** 2)
    strain_sim = np.zeros_like(test_muscle.strain_experimental)
    coil_radius_sim = np.zeros_like(test_muscle.strain_experimental)
    initial_coil_radius = 0
    for radius in test_muscle.geometry.start_radius_list:
        initial_coil_radius += radius
    coil_radius_sim[0] = initial_coil_radius
    test_sim_settings.final_time = 5
    for i in tested_idx[1:]:
        test_sim_settings.force_mag = test_muscle.passive_force_experimental[i] / (
            1e-3
            * n_muscles
            * test_muscle.properties.youngs_modulus
            * test_muscle.sim_settings.E_scale
            * base_area
        )
        print(
            "Current Force:",
            test_muscle.passive_force_experimental[i],
            "Current Normalized Force Mag:",
            test_sim_settings.force_mag,
        )
        data = muscle_contraction_simulation(
            input_muscle=test_muscle, sim_settings=test_sim_settings
        )

        time = data[0]["time"]

        centerline_position = np.zeros_like(np.array(data[0]["position"]))
        for muscle_rods in data:
            centerline_position += np.array(muscle_rods["position"]) / n_muscles

        n = centerline_position.shape[-1]
        strain_sim[i] = np.dot(
            test_muscle.geometry.direction,
            (
                (centerline_position[-1, :, -1] - centerline_position[-1, :, 0])
                - (centerline_position[0, :, -1] - centerline_position[0, :, 0])
            )
            / (centerline_position[0, :, -1] - centerline_position[0, :, 0]),
        )
        coil_radius_sim[i] = np.mean(
            _batch_norm(
                centerline_position[-1, :, :]
                - _batch_product_i_ik_to_k(
                    test_muscle.geometry.direction, centerline_position[-1, :, :]
                ).reshape(1, n)
                * test_muscle.geometry.direction.reshape(3, 1)
            )
        )

        print("Current Strain: " + str(strain_sim[i]))

    plt.rc("font", size=8)  # controls default text sizes
    plt.plot(
        strain_sim[tested_idx],
        test_muscle.passive_force_experimental[tested_idx],
        linewidth=3,
        marker="o",
        markersize=7,
        label="Passive force (Sim,Isobaric)",
    )
    plt.suptitle("Coil Strain vs Force")
    plt.xlabel("Strain (mm/mm)")
    plt.ylabel("Force (N)")
    # plt.xlim([0,1.1*test_muscle.strain_experimental[tested_idx[-1]]])
else:
    print("Please make sure one of isometric_test or isobaric_test is True, not both")


# if test_sim_settings.theoretical_curve:
#     plt.plot(test_muscle.strain_experimental[tested_idx], passive_force_theory[tested_idx],linewidth=3,markersize=5,linestyle='dashed',marker='o',label = "Passive force (Theoretical)")
if test_sim_settings.additional_curves:
    for tensile_test, name in additional_curve_list:
        plt.plot(
            tensile_test[:, 0], tensile_test[:, 1], color="k", linewidth=1, label=name
        )


plt.legend()
plt.savefig(save_folder + "/plot.png", dpi=300)
plt.show()
plt.close()
plt.plot(
    strain_sim[tested_idx],
    coil_radius_sim[tested_idx] * 1000,
    linewidth=3,
    marker="o",
    markersize=7,
    label="Coil Radius (Sim,Isobaric)",
)
plt.plot(
    strain_sim[tested_idx],
    initial_coil_radius * np.ones_like(coil_radius_sim[tested_idx]) * 1000,
    linewidth=3,
    linestyle="dashed",
    markersize=7,
    label="Start Coil Radius",
)

plt.suptitle("Coil Strain vs Coil Radius")
plt.xlabel("Strain (mm/mm)")
plt.ylabel("Radius (mm)")
plt.legend()
plt.show()
