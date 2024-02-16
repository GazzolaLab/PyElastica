import numpy as np
import os
from elastica import *
from examples.ArtificialMusclesCases import *
from single_muscle_simulator import muscle_contraction_simulation
import matplotlib

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt


test_sim_settings_dict = {
    "sim_name": "IsometricTest",
    "final_time": 16,  # seconds
    "untwisting_start_time": 6,  # seconds
    "time_untwisting": 1,  # seconds
    "rendering_fps": 20,
    "contraction": True,
    "plot_video": True,
    "save_data": False,
    "return_data": True,
    "povray_viz": False,
    "isometric_test": True,
    "isobaric_test": False,
    "self_contact": False,
    "muscle_strain": 0.0,
    "force_mag": 20.0,
}


test_sim_settings = Dict2Class(test_sim_settings_dict)

test_muscle = Liuyang_monocoil()
tested_strains_idx = [0, 9, 17]

passive_force_sim = np.zeros_like(test_muscle.strain_experimental)
total_force_sim = np.zeros_like(test_muscle.strain_experimental)

current_path = os.getcwd()
save_folder = os.path.join(
    current_path, test_sim_settings.sim_name + "/" + test_muscle.name + "/data"
)

for i in tested_strains_idx:  # range(len(test_muscle.strain_experimental)):
    test_sim_settings.muscle_strain = test_muscle.strain_experimental[i]
    print("Current Strain:" + str(test_sim_settings.muscle_strain))
    data = muscle_contraction_simulation(
        input_muscle=test_muscle, sim_settings=test_sim_settings
    )
    time = data[0]["time"]
    passive_force_measurement_time = (
        int(
            (len(time))
            * test_sim_settings.untwisting_start_time
            / test_sim_settings.final_time
        )
        - 2
    )  # just right before the muscle is acuated

    internal_force = np.zeros_like(np.array(data[0]["internal_force"]))
    for muscle_rods in data:
        internal_force += np.array(muscle_rods["internal_force"])

    passive_force_sim[i] = (
        internal_force[passive_force_measurement_time, 2, 0]
        * test_muscle.sim_settings.E_scale
    )
    total_force_sim[i] = internal_force[-1, 2, 0] * test_muscle.sim_settings.E_scale
    print(
        "Current Passive Force: " + str(passive_force_sim[i]),
        "Current Total Force: " + str(total_force_sim[i]),
    )

plt.rc("font", size=8)  # controls default text sizes
plt.plot(
    test_muscle.strain_experimental,
    test_muscle.passive_force_experimental,
    marker="o",
    label="Passive force (Experiment)",
)
plt.plot(
    test_muscle.strain_experimental,
    test_muscle.total_force_experimental,
    marker="o",
    label="Total force (Experiment at "
    + str(test_muscle.sim_settings.actuation_end_temperature)
    + "°C)",
)

plt.plot(
    test_muscle.strain_experimental[tested_strains_idx],
    passive_force_sim[tested_strains_idx],
    marker="o",
    label="Passive force (Sim)",
)
plt.plot(
    test_muscle.strain_experimental[tested_strains_idx],
    total_force_sim[tested_strains_idx],
    marker="o",
    label="Total force (Sim at "
    + str(test_muscle.sim_settings.actuation_end_temperature)
    + "C)",
)

plt.suptitle("Coil Strain vs Force")
plt.xlabel("Strain (mm/mm)")
plt.ylabel("Force (N)")
plt.xlim([0, 1.1 * test_muscle.strain_experimental[-1]])
plt.legend()
plt.savefig(save_folder + "/plot.png", dpi=300)
plt.show()
