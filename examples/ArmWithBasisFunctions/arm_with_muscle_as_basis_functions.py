import numpy as np
import sys

sys.path.append("../../")

import os
from collections import defaultdict
from elastica.wrappers import (
    BaseSystemCollection,
    Constraints,
    Forcing,
    CallBacks,
)
from elastica.rod.cosserat_rod import CosseratRod
from elastica.external_forces import GravityForces, MuscleTorques, NoForces
from elastica.basis_function_scripts.muscle_torque_basis_functions import (
    MuscleTorquesBasisFunctions,
)
from elastica.boundary_conditions import OneEndFixedRod
from elastica.interaction import AnistropicFrictionalPlane
from elastica.callback_functions import CallBackBaseClass
from elastica.timestepper.symplectic_steppers import PositionVerlet, PEFRL
from elastica.timestepper import integrate
from examples.ArmWithBasisFunctions.arm_sim_with_basis_functions_postprocessing import (
    plot_video,
    plot_video_actiavation_muscle,
)


class ArmBasisSimulator(BaseSystemCollection, Constraints, Forcing, CallBacks):
    pass


arm_muscle_with_basis_functions_sim = ArmBasisSimulator()

# setting up test params
n_elem = 30
start = np.zeros((3,))
direction = np.array([0.0, -1.0, 0.0])  # np.array([0.0, 0.0, 1.0])
normal = np.array([0.0, 0.0, 1.0])  # np.array([0.0, 1.0, 0.0])
base_length = 1.0
base_radius = 0.025
base_area = np.pi * base_radius ** 2
density = 1000
nu = 5.0
E = 1e7
poisson_ratio = 0.5

shearable_rod = CosseratRod.straight_rod(
    n_elem,
    start,
    direction,
    normal,
    base_length,
    base_radius,
    density,
    nu,
    E,
    poisson_ratio,
)

arm_muscle_with_basis_functions_sim.append(shearable_rod)

# Add the boundary conditions
arm_muscle_with_basis_functions_sim.constrain(shearable_rod).using(
    OneEndFixedRod, constrained_position_idx=(0,), constrained_director_idx=(0,)
)

# Basis function param
number_of_basis_functions = 4
index = np.empty((number_of_basis_functions, 2), dtype=int)
filename_basis_func_params = ""  # "spline_positions.txt"
if os.path.exists(filename_basis_func_params):
    basis_func_params = np.genfromtxt(filename_basis_func_params, delimiter=",")
    # Assert checks for making sure inputs are correct
    assert n_elem == basis_func_params[-1, 1], (
        "index of last element different than number of elements,"
        "Are you sure, you divide rod properly?"
    )
    assert number_of_basis_functions == basis_func_params.shape[0], (
        "desired number of basis functions are different "
        "than given in " + filename_basis_func_params
    )
    index[:, 0] = basis_func_params[:, 0]  # start index of segment
    index[:, 1] = basis_func_params[:, 1]  # end index of segment
    scale_factor = basis_func_params[:, 2:]  # spline coefficients
else:
    index[:, 0] = np.linspace(0, n_elem, number_of_basis_functions + 1, dtype=int)[
        :-1
    ]  # start index
    index[:, 1] = np.linspace(0, n_elem, number_of_basis_functions + 1, dtype=int)[
        1:
    ]  # end index
    scale_factor = np.ones(number_of_basis_functions)

segment_length = base_length * (index[:, 1] - index[:, 0]) / n_elem

# Firing frequency of muscle torques
frequency_of_segments = np.ones(number_of_basis_functions)

# Set the list for activation function and torque profile
activation_function_list = defaultdict(list)
torque_profile_list = defaultdict(list)

# Apply torques
arm_muscle_with_basis_functions_sim.add_forcing_to(shearable_rod).using(
    MuscleTorquesBasisFunctions,
    n_elems=n_elem,
    base_length=base_length,
    segment_length=segment_length,
    segment_idx=index,
    frequency_of_segments=frequency_of_segments,
    scale_factor=scale_factor * 10,
    direction=np.array([-1.0, 0.0, 0.0]),
    activation_function_list=activation_function_list,
    torque_profile_list=torque_profile_list,
    ramp_up_time=1.0,
)

# Add call backs
class ArmMuscleBasisCallBack(CallBackBaseClass):
    """
    Call back function for continuum snake
    """

    def __init__(self, step_skip: int, callback_params: dict):
        CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.callback_params = callback_params

    def make_callback(self, system, time, current_step: int):

        if current_step % self.every == 0:

            self.callback_params["time"].append(time)
            self.callback_params["step"].append(current_step)
            self.callback_params["position"].append(system.position_collection.copy())
            self.callback_params["velocity"].append(system.velocity_collection.copy())
            self.callback_params["avg_velocity"].append(
                system.compute_velocity_center_of_mass()
            )

            self.callback_params["center_of_mass"].append(
                system.compute_position_center_of_mass()
            )

            return


pp_list = defaultdict(list)
arm_muscle_with_basis_functions_sim.collect_diagnostics(shearable_rod).using(
    ArmMuscleBasisCallBack, step_skip=200, callback_params=pp_list,
)

arm_muscle_with_basis_functions_sim.finalize()
timestepper = PositionVerlet()

final_time = 2.0  # 11.0 + 0.01)
dt = 1.0e-5
total_steps = int(final_time / dt)
print("Total steps", total_steps)
integrate(timestepper, arm_muscle_with_basis_functions_sim, final_time, total_steps)


filename_video = "arm_simulation.mp4"
plot_video(pp_list, video_name=filename_video, margin=0.2, fps=20)

filename_activation_muscle_torque_video = "arm_activation_muscle_torque.mp4"
plot_video_actiavation_muscle(
    activation_function_list,
    torque_profile_list,
    video_name=filename_activation_muscle_torque_video,
    margin=0.2,
    fps=20,
    step=1e3,
)

# Save arm position
# saved file order is (time,x,y,z)
filename_position = "position_of_arm"

time = np.array(pp_list["time"])
position_of_arm = np.array(pp_list["position"])

position_matrix = np.zeros((time.shape[0], 4, position_of_arm.shape[2]))

for k in range(time.shape[0]):
    position_matrix[k, 0, :] = time[k]
    position_matrix[k, 1, :] = position_of_arm[k, 0, :]
    position_matrix[k, 2, :] = position_of_arm[k, 1, :]
    position_matrix[k, 3, :] = position_of_arm[k, 2, :]

np.save(filename_position, position_matrix)

# Save activation function
# saved file order is (time, and basis functions)
filename_activation = "activation_function"
time = np.array(activation_function_list["time"])
activation = np.array(activation_function_list["activation_signal"])
activation_matrix = np.zeros((time.shape[0], int(activation.shape[1] + 1)))

np.save(filename_activation, activation_matrix)

for k in range(time.shape[0]):
    activation_matrix[k, 0] = time[k]
    activation_matrix[k, 1:] = activation[k, :]

# Save muscle functions
filename_muscle_function = "muscle_torque"
time = np.array(torque_profile_list["time"])
muscle_torque = np.array(torque_profile_list["torque"])
muscle_torque_matrix = np.zeros((time.shape[0], 2, muscle_torque.shape[2]))

for k in range(time.shape[0]):
    muscle_torque_matrix[k, 0, :] = time[k]
    muscle_torque_matrix[k, 1, :] = muscle_torque[k, 1, :]

np.save(filename_muscle_function, muscle_torque_matrix)
