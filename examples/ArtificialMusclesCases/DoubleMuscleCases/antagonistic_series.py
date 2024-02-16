from elastica import *
from examples.ArtificialMusclesCases import *
from series_muscle_simulator import series_muscle_contraction_simulation


antagonistic_sim_settings_dict = {
    "sim_name": "AntgonisticSeries",
    "final_time": 20,  # seconds
    "untwisting_start_time": 6,  # seconds
    "time_untwisting": 1,  # seconds
    "rendering_fps": 20,
    "contraction": True,
    "plot_video": True,
    "save_data": False,
    "return_data": False,
    "povray_viz": True,
    "isometric_test": False,
    "isobaric_test": False,
    "self_contact": False,
    "muscle_strain": 0.14,
    "force_mag": 3.0,
}


antagonistic_sim_settings = Dict2Class(antagonistic_sim_settings_dict)


first_muscle = Liuyang_supercoil()
second_muscle = Liuyang_supercoil()

first_muscle.geometry.muscle_length = 40e-3
second_muscle.geometry.muscle_length = 40e-3


series_muscle_contraction_simulation(
    first_input_muscle=first_muscle,
    second_input_muscle=second_muscle,
    sim_settings=antagonistic_sim_settings,
)
