import numpy as np
from elastica import *
from examples.ArtificialMusclesCases import *
from examples.ArtificialMusclesCases.SingleMuscleCases.single_muscle_simulator import (
    muscle_contraction_simulation,
)


tuning_sim_settings_dict = {
    "sim_name": "Tuning",
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
    "force_mag": 3.0,
}


tuning_sim_settings = Dict2Class(tuning_sim_settings_dict)

slope_difference = 10
tuning_muscle = Liuyang_monocoil()
k = 100

total_force_slope_experimental = (
    tuning_muscle.total_force_experimental[-1]
    - tuning_muscle.total_force_experimental[0]
) / (tuning_muscle.strain_experimental[-1] - tuning_muscle.strain_experimental[0])

while abs(slope_difference) > 1e-6:
    total_forces_sim = []
    for i in [0, -1]:
        tuning_sim_settings.muscle_strain = tuning_muscle.strain_experimental[i]
        print(
            "Current initial_link_per_length:"
            + str(tuning_muscle.geometry.initial_link_per_length)
        )
        data = muscle_contraction_simulation(
            input_muscle=tuning_muscle, sim_settings=tuning_sim_settings
        )

        internal_force = np.zeros_like(np.array(data[0]["internal_force"]))
        for muscle_rods in data:
            internal_force += np.array(muscle_rods["internal_force"])

        total_forces_sim.append(
            internal_force[-1, 2, 0] * tuning_muscle.sim_settings.E_scale
        )
    total_force_slope_sim = (total_forces_sim[-1] - total_forces_sim[0]) / (
        tuning_muscle.strain_experimental[-1] - tuning_muscle.strain_experimental[0]
    )
    slope_difference = total_force_slope_experimental - total_force_slope_sim
    print(
        "Current Slope Difference:" + str(slope_difference),
        total_force_slope_experimental,
        total_force_slope_sim,
    )
    tuning_muscle.geometry.initial_link_per_length -= k * slope_difference

print(
    "Tuned initial_link_per_length:"
    + str(tuning_muscle.geometry.initial_link_per_length)
)
