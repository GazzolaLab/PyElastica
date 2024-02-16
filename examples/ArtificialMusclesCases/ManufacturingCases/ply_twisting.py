import numpy as np
from elastica import *
from examples.ArtificialMusclesCases import *
from ply_twisting_simulator import ply_twisting_simulation

ply_twisting_sim_settings_dict = {
    "sim_name": "PlyTwisting",
    "final_time": 20,  # seconds
    "twisting_start_time": 0,  # seconds
    "twisting_angular_speed": -0.1,  # radians/s
    "rendering_fps": 20,
    "plot_video": True,
    "save_data": True,
    "return_data": True,
    "povray_viz": True,
}

ply_twisting_sim_settings = Dict2Class(ply_twisting_sim_settings_dict)

input_ply = Jeongmin_supercoil_ply()

# render
if __name__ == "__main__":
    data = ply_twisting_simulation(
        input_ply=input_ply,
        sim_settings=ply_twisting_sim_settings,
    )
    muscle_renderer(
        data,
        ply_twisting_sim_settings.sim_name,
        camera_location=[0, 0.1, input_ply.geometry.muscle_length / 2],
        look_at_location=[0, 0, input_ply.geometry.muscle_length / 2],
    )
