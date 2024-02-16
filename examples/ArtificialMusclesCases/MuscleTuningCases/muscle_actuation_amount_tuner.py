import numpy as np
from elastica import *
from elastica import *
from examples.ArtificialMusclesCases import *
from examples.ArtificialMusclesCases.SingleMuscleCases.single_muscle_simulator import (
    muscle_contraction_simulation,
)


tuning_sim_settings_dict = {
    "sim_name": "Tuning",
    "final_time": 10,  # seconds
    "untwisting_start_time": 0,  # seconds
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
    "force_mag": 0.0,
}


tuning_sim_settings = Dict2Class(tuning_sim_settings_dict)

force_difference = 10
tuning_muscle = Liuyang_monocoil()
k = 0.01


while abs(force_difference) > 1e-6:
    print(
        "Current Kappa Change:" + str(tuning_muscle.sim_settings.actuation_kappa_change)
    )
    data = muscle_contraction_simulation(
        input_muscle=tuning_muscle, sim_settings=tuning_sim_settings
    )

    internal_force = np.zeros_like(np.array(data[0]["internal_force"]))
    for muscle_rods in data:
        internal_force += np.array(muscle_rods["internal_force"])

    force_difference = (
        tuning_muscle.total_force_experimental[0]
        - internal_force[-1, 2, 0] * tuning_muscle.sim_settings.E_scale
    )
    print("Current Force Difference:" + str(force_difference))
    tuning_muscle.sim_settings.actuation_kappa_change += k * force_difference

print("Tuned Kappa Change: " + str(tuning_muscle.sim_settings.actuation_kappa_change))
