import numpy as np
from elastica import *
from examples.ArtificialMusclesCases import *
from single_muscle_simulator import muscle_contraction_simulation

pure_contraction_sim_settings_dict = {
    "sim_name": "PureContraction",
    "final_time": 2,  # seconds
    "untwisting_start_time": 0,  # seconds
    "time_untwisting": 1,  # seconds
    "rendering_fps": 20,
    "contraction": True,
    "plot_video": True,
    "save_data": True,
    "return_data": False,
    "povray_viz": True,
    "isometric_test": False,
    "isobaric_test": False,
    "self_contact": False,
    "muscle_strain": 0.0,
    "force_mag": 0.0,
}

pure_contraction_sim_settings = Dict2Class(pure_contraction_sim_settings_dict)

input_muscle = Samuel_supercoil()

muscle_contraction_simulation(
    input_muscle=input_muscle,
    sim_settings=pure_contraction_sim_settings,
)
