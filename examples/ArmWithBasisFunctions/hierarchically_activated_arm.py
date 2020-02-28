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
from examples.ArmWithBasisFunctions.hierarchical_muscles.hierarchical_muscle_torques import (
    HierarchicalMuscleTorques,
)
from examples.ArmWithBasisFunctions.hierarchical_muscles.hierarchical_bases import (
    SpatiallyInvariantSplineHierarchy,
    SpatiallyInvariantSplineHierarchyMapper,
    SplineHierarchySegments,
    Union,
    Gaussian,
    ScalingFilter,
)
from elastica.boundary_conditions import OneEndFixedRod
from elastica.callback_functions import CallBackBaseClass
from elastica.timestepper.symplectic_steppers import PositionVerlet, PEFRL
from elastica.timestepper import integrate
from examples.ArmWithBasisFunctions.arm_sim_with_basis_functions_postprocessing import (
    plot_video,
    plot_video_actiavation_muscle,
)


class ArmBasisSimulator(BaseSystemCollection, Constraints, Forcing, CallBacks):
    pass


def main():
    arm_muscle_with_basis_functions_sim = ArmBasisSimulator()

    # setting up test params
    n_elem = 200
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

    # Setup the profile that we want
    """
    # First pack of hierarchical muscles
    first_muscle_segment = SpatiallyInvariantSplineHierarchy(
        Union(Gaussian(0.15), Gaussian(0.05), Gaussian(0.05))
    )
    # apply filters to change magnitude
    first_muscle_segment.apply_filter(0, ScalingFilter, 16)
    first_muscle_segment.apply_filter(1, ScalingFilter, 8)
    first_muscle_segment.apply_filter(2, ScalingFilter, 4)

    second_muscle_segment = SpatiallyInvariantSplineHierarchy(
        Union(Gaussian(0.12), Gaussian(0.06), Gaussian(0.03))
    )
    # apply filters to change magnitude
    # second_muscle_segment.apply_filter(0, ScalingFilter, 16)
    # second_muscle_segment.apply_filter(1, ScalingFilter, 8)
    # second_muscle_segment.apply_filter(2, ScalingFilter, 4)
    second_muscle_segment.apply_filter(0, ScalingFilter, 20)
    second_muscle_segment.apply_filter(1, ScalingFilter, 10)
    second_muscle_segment.apply_filter(2, ScalingFilter, 5)

    first_muscle_mapper = SpatiallyInvariantSplineHierarchyMapper(
        first_muscle_segment, (0.1, 0.4)
    )
    second_muscle_mapper = SpatiallyInvariantSplineHierarchyMapper(
        # second_muscle_segment, (0.6, 0.9)
        second_muscle_segment, (0.7, 0.9)
    )
    """

    """
    # First pack of hierarchical muscles
    first_muscle_segment = SpatiallyInvariantSplineHierarchy(
        Union(Gaussian(0.15), Gaussian(0.09), Gaussian(0.05))
    )
    # apply filters to change magnitude
    first_muscle_segment.apply_filter(0, ScalingFilter, 16)
    first_muscle_segment.apply_filter(1, ScalingFilter, 8)
    first_muscle_segment.apply_filter(2, ScalingFilter, 4)

    second_muscle_segment = SpatiallyInvariantSplineHierarchy(
        Union(Gaussian(0.10), Gaussian(0.05), Gaussian(0.03))
    )
    # apply filters to change magnitude
    # second_muscle_segment.apply_filter(0, ScalingFilter, 16)
    # second_muscle_segment.apply_filter(1, ScalingFilter, 8)
    # second_muscle_segment.apply_filter(2, ScalingFilter, 4)
    second_muscle_segment.apply_filter(0, ScalingFilter, 20)
    second_muscle_segment.apply_filter(1, ScalingFilter, 10)
    second_muscle_segment.apply_filter(2, ScalingFilter, 5)

    first_muscle_mapper = SpatiallyInvariantSplineHierarchyMapper(
        first_muscle_segment, (0.35, 0.65)
    )
    second_muscle_mapper = SpatiallyInvariantSplineHierarchyMapper(
        second_muscle_segment, (0.75, 0.95)
    )
    """

    # First pack of hierarchical muscles
    first_muscle_segment = SpatiallyInvariantSplineHierarchy(
        Union(Gaussian(0.20), Gaussian(0.08), Gaussian(0.05))
    )
    # apply filters to change magnitude
    first_muscle_segment.apply_filter(0, ScalingFilter, 32)
    first_muscle_segment.apply_filter(1, ScalingFilter, 8)
    first_muscle_segment.apply_filter(2, ScalingFilter, 2)

    second_muscle_segment = SpatiallyInvariantSplineHierarchy(
        Union(Gaussian(0.10), Gaussian(0.05), Gaussian(0.03))
    )
    # apply filters to change magnitude
    # second_muscle_segment.apply_filter(0, ScalingFilter, 16)
    # second_muscle_segment.apply_filter(1, ScalingFilter, 8)
    # second_muscle_segment.apply_filter(2, ScalingFilter, 4)
    second_muscle_segment.apply_filter(0, ScalingFilter, 20)
    second_muscle_segment.apply_filter(1, ScalingFilter, 10)
    second_muscle_segment.apply_filter(2, ScalingFilter, 3)

    first_muscle_mapper = SpatiallyInvariantSplineHierarchyMapper(
        first_muscle_segment, (0.35, 0.65)
    )
    second_muscle_mapper = SpatiallyInvariantSplineHierarchyMapper(
        second_muscle_segment, (0.75, 0.95)
    )

    segments_of_muscle_hierarchies = SplineHierarchySegments(
        second_muscle_mapper, first_muscle_mapper
    )

    """
    def activation(time_v):
        # return 13
        starts_stops = segments_of_muscle_hierarchies.activation_start_stop
        for segment in [first_muscle_segment]:
            n_levels = segment.n_levels
            for level in range(n_levels):
                start = segment.basis_start_idx(level)
                stop = segment.n_bases_at_level(level)
        for i in r
    """

    def ramped_up(shifted_time, threshold=0.1):
        return (
            0.0
            if shifted_time < 0.0
            else (
                1.0
                if shifted_time > threshold
                else 0.5 * (1.0 - np.cos(np.pi * shifted_time / threshold))
            )
        )

    def single_segment_activation(time_v):
        activation_arr = np.zeros((13,))

        # top_level
        activation_arr[0] = ramped_up(time_v - 0.6, 0.1)

        # mid_levels
        activation_arr[1:4] = ramped_up(time_v - 0.3, 0.1)

        # bottom boys
        activation_arr[4:] = ramped_up(time_v, 0.1)

        return activation_arr

    def two_segment_activation(time_v):
        activation_arr = np.zeros((13 * 2,))

        # NOTE ! Activation is reversed to make the video correct

        # The first muscle segment that controls the shoulder
        # top_level
        activation_arr[13] = ramped_up(time_v - 0.6, 0.1)
        # mid_levels
        activation_arr[14:17] = ramped_up(time_v - 0.3, 0.1)
        # bottom boys
        activation_arr[17:] = ramped_up(time_v, 0.1)

        # The seconc muscle segment that controls the finger
        activation_arr[0] = ramped_up(time_v - 1.0, 0.1)
        # mid_levels
        activation_arr[1:4] = ramped_up(time_v - 0.9, 0.1)
        # bottom boys
        activation_arr[4:13] = ramped_up(time_v - 0.8, 0.1)
        return activation_arr

    # Set the list for activation function and torque profile
    activation_function_list = defaultdict(list)
    torque_profile_list = defaultdict(list)

    step_skip = 200

    # Apply torques
    arm_muscle_with_basis_functions_sim.add_forcing_to(shearable_rod).using(
        HierarchicalMuscleTorques,
        segments_of_muscle_hierarchies,
        activation_func=single_segment_activation
        if segments_of_muscle_hierarchies.n_segments == 1
        else two_segment_activation,
        direction=np.array([-1.0, 0.0, 0.0]),
        ramp_up_time=1.0,
        step_skip=step_skip,
        activation_function_recorder=activation_function_list,
        torque_profile_recorder=torque_profile_list,
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
                self.callback_params["position"].append(
                    system.position_collection.copy()
                )
                self.callback_params["velocity"].append(
                    system.velocity_collection.copy()
                )
                self.callback_params["avg_velocity"].append(
                    system.compute_velocity_center_of_mass()
                )

                self.callback_params["center_of_mass"].append(
                    system.compute_position_center_of_mass()
                )

                return

    pp_list = defaultdict(list)
    arm_muscle_with_basis_functions_sim.collect_diagnostics(shearable_rod).using(
        ArmMuscleBasisCallBack, step_skip=step_skip, callback_params=pp_list,
    )

    arm_muscle_with_basis_functions_sim.finalize()
    timestepper = PositionVerlet()

    final_time = 3.0  # 11.0 + 0.01)
    dt = 1.0e-5
    total_steps = int(final_time / dt)
    print("Total steps", total_steps)
    integrate(timestepper, arm_muscle_with_basis_functions_sim, final_time, total_steps)

    filename_video = "arm_simulation.mp4"
    plot_video(pp_list, video_name=filename_video, margin=0.2, fps=20, step=10)

    filename_activation_muscle_torque_video = "arm_activation_muscle_torque.mp4"
    plot_video_actiavation_muscle(
        activation_function_list,
        torque_profile_list,
        video_name=filename_activation_muscle_torque_video,
        margin=0.2,
        fps=20,
        step=10,
    )

    try:
        import moviepy.editor as mpy

        # We use the GIFs generated earlier to avoid recomputing the animations.
        clip_mayavi = mpy.VideoFileClip(filename_video)
        clip_mpl = mpy.VideoFileClip(filename_activation_muscle_torque_video).resize(
            height=clip_mayavi.h
        )
        animation = mpy.clips_array([[clip_mpl, clip_mayavi]])
        animation.write_videofile("combined.mp4", fps=20)
    except ImportError:
        print("Whatsup!")

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
    time = np.array(activation_function_list["time"])
    if "activation_signal" in activation_function_list:
        filename_activation = "activation_function"
        activation = np.array(activation_function_list["activation_signal"])
        # activation_matrix = np.zeros((time.shape[0], int(activation.shape[1] + 1)))
        np.save(filename_activation, activation)
    else:
        first_activation = np.array(activation_function_list["first_activation_signal"])
        filename_first_activation = "first_activation_function"
        np.save(filename_first_activation, first_activation)
        second_activation = np.array(
            activation_function_list["second_activation_signal"]
        )
        filename_second_activation = "second_activation_function"
        np.save(filename_second_activation, second_activation)

    # for k in range(time.shape[0]):
    #     activation_matrix[k, 0] = time[k]
    #     activation_matrix[k, 1:] = activation[k, :]

    # Save muscle functions
    time = np.array(torque_profile_list["time"])
    if "torque_mag" in activation_function_list:
        muscle_torque_mag = np.array(torque_profile_list["torque_mag"])
        filename_muscle_function = "muscle_torque"
        np.save(filename_muscle_function, muscle_torque_mag)
    else:
        first_muscle_torque_mag = np.array(torque_profile_list["first_torque_mag"])
        filename_first_muscle_torque_mag = "first_muscle_torque_mag_function"
        np.save(filename_first_muscle_torque_mag, first_muscle_torque_mag)
        second_muscle_torque_mag = np.array(torque_profile_list["second_torque_mag"])
        filename_second_muscle_torque_mag = "second_muscle_torque_mag_function"
        np.save(filename_second_muscle_torque_mag, second_muscle_torque_mag)


if __name__ == "__main__":
    main()
