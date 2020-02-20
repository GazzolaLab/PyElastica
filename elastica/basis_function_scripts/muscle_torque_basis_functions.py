# import sys
#
# sys.path.append("../")
# import os
import numpy as np
from elastica._linalg import _batch_matvec
from elastica.external_forces import NoForces

from matplotlib import pyplot as plt


class MuscleTorquesBasisFunctions(NoForces):
    """
    Applies muscle torques on the body. It can apply muscle torques
    as travelling wave with beta spline or only as travelling wave.
    """

    def __init__(
        self,
        n_elems,
        base_length,
        segment_length,
        segment_idx,
        frequency_of_segments,
        scale_factor,
        direction,
        activation_function_list: dict,
        torque_profile_list: dict,
        ramp_up_time=0.0,
    ):
        super(MuscleTorquesBasisFunctions, self).__init__()

        self.direction = direction.reshape(3, 1)  # Direction torque applied
        self.frequency_of_segments = frequency_of_segments
        assert ramp_up_time >= 0.0
        self.ramp_up_time = ramp_up_time

        assert (
            scale_factor.size != 0
        ), "Beta spline coefficient array (t_coeff) is empty"

        # Compute the rest rod length, here we assume elements are equally distributed.
        position_collection = np.linspace(0, base_length, n_elems + 1)
        position_diff = position_collection[1:] - position_collection[:-1]
        rest_lengths = position_diff
        self.element_position = np.round(np.cumsum(rest_lengths), 10)
        # compute the muscle torque profile along the body and save it!
        number_of_basis_functions = segment_length.shape[0]
        self.torque_profile = np.zeros((n_elems, number_of_basis_functions))
        for i in range(0, number_of_basis_functions):
            s = np.round(
                np.cumsum(rest_lengths[segment_idx[i, 0] : segment_idx[i, 1]]), 10
            )

            # Add the spline profiles
            self.torque_profile[
                segment_idx[i, 0] : segment_idx[i, 1], i
            ] = self.spline_function(s, segment_length[i], scale_factor[i])

        self.active_torque_profiles = np.zeros((n_elems))

        self.activation_function_list = activation_function_list
        self.torque_profile_list = torque_profile_list

    def spline_function(self, evaluation_pts, segment_length, scale_factor):
        return scale_factor * np.abs(np.sin(evaluation_pts / segment_length * np.pi))

    def activation_function(self, time: np.float = 0.0):
        # self.activation = np.zeros(self.frequency_of_segments.shape[0])
        # self.activation[np.where(time % (1 / self.frequency_of_segments) == 0)] = 1.0
        if time < 0.1:
            self.activation = np.array([1.0, 1.0, 0.0, 0.0])
        elif time > 0.1 and time < 0.2:
            self.activation = np.array([1.0, 1.0, 1.0, 0.0])
        elif time > 0.2:
            self.activation = np.array([1.0, 1.0, 1.0, 1.0])
        # self.activation = np.ones(self.frequency_of_segments.shape[0])

        self.activation_function_list["time"].append(time)
        self.activation_function_list["activation_signal"].append(self.activation)

    def apply_torques(self, system, time: np.float = 0.0):

        # Ramp up the muscle torque
        # factor = min(1.0, time / self.ramp_up_time)

        # Compute the torque profile for this time-step, controller might change
        # the active and deactive splines.
        self.activation_function(time)
        self.active_torque_profiles = np.einsum(
            "ij,j->i", self.torque_profile, self.activation
        )
        torque_mag = self.active_torque_profiles
        torque = np.einsum("j,ij->ij", torque_mag, self.direction)

        # TODO: Find a way without doing tow batch_matvec product
        system.external_torques[..., 1:] += _batch_matvec(
            system.director_collection, torque
        )[..., 1:]
        system.external_torques[..., :-1] -= _batch_matvec(
            system.director_collection[..., :-1], torque[..., 1:]
        )

        self.torque_profile_list["time"].append(time)
        self.torque_profile_list["torque"].append(system.external_torques.copy())
        self.torque_profile_list["element_position"].append(
            self.element_position.copy()
        )


# # Start testing
# def mock_rod_init(self):
#     self.n_elems = 0.0
#     self.external_forces = 0.0
#     self.external_torques = 0.0
#     self.director_collection = 0.0
#     self.rest_lengths = 0.0
#
#
# MockRod = type("MockRod", (object,), {"__init__": mock_rod_init})
#
# # Sim params
# n_elem = 100
# dim = 3
# mock_rod = MockRod()
# mock_rod.external_torques = np.zeros((dim, n_elem))
# mock_rod.n_elems = n_elem
# mock_rod.director_collection = np.repeat(
#     np.identity(3)[:, :, np.newaxis], n_elem, axis=2
# )
#
# base_length = 1.0
# position_collection = np.linspace(0, base_length, n_elem + 1)
# position_diff = position_collection[1:] - position_collection[:-1]
# rest_lengths = position_diff
# mock_rod.rest_lengths = rest_lengths
#
# # Basis function param
# number_of_basis_functions = 4
# index = np.empty((number_of_basis_functions, 2), dtype=int)
#
# filename_basis_func_params = "spline_positions.txt"
# if os.path.exists(filename_basis_func_params):
#     basis_func_params = np.genfromtxt(filename_basis_func_params, delimiter=",")
#     # Assert checks for making sure inputs are correct
#     assert n_elem == basis_func_params[-1, 1], (
#         "index of last element different than number of elements,"
#         "Are you sure, you divide rod properly?"
#     )
#     assert number_of_basis_functions == basis_func_params.shape[0], (
#         "desired number of basis functions are different "
#         "than given in " + filename_basis_func_params
#     )
#     index[:, 0] = basis_func_params[:, 0]  # start index of segment
#     index[:, 1] = basis_func_params[:, 1]  # end index of segment
#     scale_factor = basis_func_params[:, 2:]  # spline coefficients
# else:
#     index[:, 0] = np.linspace(0, n_elem, number_of_basis_functions + 1, dtype=int)[
#         :-1
#     ]  # start index
#     index[:, 1] = np.linspace(0, n_elem, number_of_basis_functions + 1, dtype=int)[
#         1:
#     ]  # end index
#     scale_factor = np.ones(number_of_basis_functions)
#
# segment_length = base_length * (index[:, 1] - index[:, 0]) / n_elem
#
# # Firing frequency of muscle torques
# frequency_of_segments = np.ones(number_of_basis_functions)
#
# # Set an a non-physical direction to check math
# direction = np.array([1.0, 1.0, 1.0])
#
# # Apply torques
# muscletorquesbasisfunction = MuscleTorquesBasisFunctions(
#     n_elem,
#     base_length,
#     segment_length,
#     index,
#     frequency_of_segments,
#     scale_factor,
#     direction,
#     ramp_up_time=1.0,
# )
#
# time = np.linspace(0, 20, 201)
# profile = np.zeros((n_elem, time.shape[0]))
# torque_mag = np.zeros((n_elem, time.shape[0]))
#
# for i in range(0, time.shape[0]):
#     profile[:, i], torque_mag[:, i] = muscletorquesbasisfunction.apply_torques(
#         mock_rod, time[i]
#     )
#
#
# # # Post processing
# MOVIE = False
# if MOVIE:
#     video_name = "basis_functions_for_torque.mp4"
#     import matplotlib.animation as manimation
#
#     print("plot video")
#     FFMpegWriter = manimation.writers["ffmpeg"]
#     metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
#     writer = FFMpegWriter(fps=10, metadata=metadata)
#     fig = plt.figure()
#     plt.axis("equal")
#     rod_pos = position_collection[1:]
#     with writer.saving(fig, video_name, dpi=100):
#         for i in range(0, time.shape[0]):
#             fig.clf()
#             plt.plot(rod_pos, profile[:, i], "-", label="spline")
#             plt.plot(rod_pos, torque_mag[:, i], label="torque")
#             plt.legend()
#             plt.xlim([-0.01, 1.01])
#             plt.ylim([-0.01, 2.01])
#             writer.grab_frame()
#
# time_for_single_plot = 1
# (
#     profile_for_second_plot,
#     torque_mag_for_second_plot,
# ) = muscletorquesbasisfunction.apply_torques(mock_rod, time_for_single_plot)
#
# rod_pos = position_collection[1:]
# fig = plt.figure(figsize=(10, 8), frameon=True, dpi=150)
# plt.plot(rod_pos[:], profile_for_second_plot, "-", label="spline")
# plt.legend()
# plt.show()
# fig.savefig("sine_basis_function")
