import numpy as np
from elastica import *
from examples.ArtificialMusclesCases import *
from examples.ArtificialMusclesCases.SingleMuscleCases.single_muscle_simulator import (
    muscle_contraction_simulation,
)


tuning_sim_settings_dict = {
    "sim_name": "Tuning",
    "final_time": 6,  # seconds
    "untwisting_start_time": 1,  # seconds
    "time_untwisting": 0,  # seconds
    "rendering_fps": 20,
    "contraction": False,
    "plot_video": True,
    "save_data": False,
    "return_data": True,
    "povray_viz": False,
    "isometric_test": True,
    "isobaric_test": False,
    "self_contact": False,
    "muscle_strain": 0.0,
    "force_mag": 5.0,
}


tuning_sim_settings = Dict2Class(tuning_sim_settings_dict)

slope_difference = 10
tuning_muscle = Liuyang_supercoil()
k = 100

passive_force_slope_experimental = (
    tuning_muscle.passive_force_experimental[-1]
    - tuning_muscle.passive_force_experimental[0]
) / (tuning_muscle.strain_experimental[-1] - tuning_muscle.strain_experimental[0])

while abs(slope_difference) > 1e-6:
    passive_forces_sim = []
    for i in [0, -1]:
        tuning_sim_settings.muscle_strain = tuning_muscle.strain_experimental[i]
        print(
            "Current n_plys/height:"
            + str(tuning_muscle.geometry.turns_per_length_list[1])
        )
        data = muscle_contraction_simulation(
            input_muscle=tuning_muscle, sim_settings=tuning_sim_settings
        )

        internal_force = np.zeros_like(np.array(data[0]["internal_force"]))
        for muscle_rods in data:
            internal_force += np.array(muscle_rods["internal_force"])

        passive_forces_sim.append(
            internal_force[-1, 2, 0] * tuning_muscle.sim_settings.E_scale
        )
    passive_force_slope_sim = (passive_forces_sim[-1] - passive_forces_sim[0]) / (
        tuning_muscle.strain_experimental[-1] - tuning_muscle.strain_experimental[0]
    )
    slope_difference = passive_force_slope_experimental - passive_force_slope_sim
    print(
        "Current Slope Difference:" + str(slope_difference),
        passive_force_slope_experimental,
        passive_force_slope_sim,
    )
    tuning_muscle.geometry.turns_per_length_list[1] -= k * slope_difference

print("Tuned n_plys/height:" + str(tuning_muscle.geometry.turns_per_length_list[1]))
