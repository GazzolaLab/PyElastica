import numpy as np
import os
from elastica import *
from examples.ArtificialMusclesCases import *
from single_muscle_simulator import muscle_contraction_simulation
import matplotlib

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt


test_sim_settings_dict = {
    "sim_name": "TensileTest",
    "final_time": 1,  # seconds
    "untwisting_start_time": 10,  # seconds
    "time_untwisting": 0,  # seconds
    "rendering_fps": 20,
    "contraction": False,
    "plot_video": True,
    "save_data": False,
    "return_data": True,
    "povray_viz": False,
    "isometric_test": False,
    "isobaric_test": False,
    "self_contact": False,
    "muscle_strain": 10,
    "force_mag": 10,
}


test_sim_settings = Dict2Class(test_sim_settings_dict)

test_muscle = Samuel_supercoil()


current_path = os.getcwd()
save_folder = os.path.join(
    current_path, test_sim_settings.sim_name + "/" + test_muscle.name + "/data"
)

data = muscle_contraction_simulation(
    input_muscle=test_muscle, sim_settings=test_sim_settings
)
time = data[0]["time"]

internal_force = np.zeros_like(np.array(data[0]["internal_force"]))
position = np.zeros_like(np.array(data[0]["position"]))

for muscle_rod in data:
    internal_force += np.array(muscle_rod["internal_force"])
    position += np.array(muscle_rod["position"]) / len(data)

muscle_center = int(len(internal_force[0, 0, :]) / 2)
passive_force_below = (
    np.sum(internal_force[:, 2, :muscle_center], axis=1)
    * test_muscle.sim_settings.E_scale
)
passive_force_above = (
    np.sum(internal_force[:, 2, muscle_center:], axis=1)
    * test_muscle.sim_settings.E_scale
)
muscle_strain = (
    (position[:, 2, -1] - position[:, 2, 0]) - test_muscle.geometry.muscle_length
) / test_muscle.geometry.muscle_length
n_fibers = 1
for n_ply in test_muscle.geometry.n_ply_per_coil_level:
    n_fibers *= n_ply
cross_sectional_area = (
    (10 ** 6) * n_fibers * np.pi * test_muscle.geometry.fiber_radius ** 2
)
print(cross_sectional_area)
muscle_stress = passive_force_below / cross_sectional_area

plt.rc("font", size=8)  # controls default text sizes
plt.plot(muscle_strain, muscle_stress, marker="o", label="Passive force (Experiment)")

plt.suptitle("Coil Strain vs Force")
plt.xlabel("Strain (mm/mm)")
plt.ylabel("Stress (MPa)")
plt.legend()
plt.savefig(save_folder + "/plot.png", dpi=300)
plt.show()
