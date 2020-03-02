import numpy as np
import sys
from tqdm import tqdm

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
from elastica.external_forces import GravityForces
from examples.ArmWithBasisFunctions.hierarchical_muscles.hierarchical_muscle_torques import (
    HierarchicalMuscleTorques,
)
from elastica.interaction import AnistropicFrictionalPlane
from examples.ArmWithBasisFunctions.hierarchical_muscles.hierarchical_bases import (
    SpatiallyInvariantSplineHierarchy,
    SpatiallyInvariantSplineHierarchyMapper,
    SplineHierarchySegments,
    Union,
    Gaussian,
    ScalingFilter,
)

from elastica.callback_functions import CallBackBaseClass
from elastica.timestepper.symplectic_steppers import PositionVerlet
from elastica.timestepper import integrate, extend_stepper_interface
from examples.ArmWithBasisFunctions.arm_sim_with_basis_functions_postprocessing import (
    plot_video,
    plot_video_actiavation_muscle,
    plot_arm_tip_sensor_values,
    plot_video_zx,
    plot_video3d,
)
from examples.ArmWithBasisFunctions.arm_setting_up_functions import (
    make_tappered_arm,
    make_two_arm_from_straigth_rod,
)

# Set base simulator class
class BaseSimulator(BaseSystemCollection, Constraints, Forcing, CallBacks):
    pass


class Environment:
    def __init__(
        self, timestepper, COLLECT_DATA=False,
    ):
        self.StatefulStepper = timestepper
        # Collect data is a boolean. If it is true callback function collects
        # rod parameters defined by user in a list.
        self.COLLECT_DATA = COLLECT_DATA

    def reset(self):
        """
        This function, creates the simulation environment.
        First, rod intialized and then rod is modified to make it tapered.
        Second, muscle segments are intialized. Muscle segment position,
        number of basis functions and applied directions are set.
        Finally, friction plane is set and simulation is finalized.
        Returns
        -------

        """
        self.simulator = BaseSimulator()

        # setting up test params
        n_elem = 120
        start = np.zeros((3,))
        direction = np.array([0.0, 0.0, 1.0])  # rod direction
        normal = np.array([0.0, 1.0, 0.0])
        binormal = np.cross(direction, normal)
        base_length = 1.0  # rod base length
        base_radius = 0.05  # rod base radius
        base_area = np.pi * base_radius ** 2
        density = 1000
        nu = 5.0  # dissipation coefficient
        E = 5e6  # Young's Modulus
        poisson_ratio = 0.5

        self.shearable_rod = CosseratRod.straight_rod(
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

        # Set the arm properties after defining rods
        # We will have one element for head only
        head_element = int(20)

        radius_tip = 0.025  # radius of the arm at the tip
        radius_base = 0.03  # radius of the arm at the base
        radius_head = 0.1  # radius of the head

        # Below function takes the rod and computes new position of nodes
        # and centerline radius for the user defined configuration.
        radius_along_rod, position = make_two_arm_from_straigth_rod(
            self.shearable_rod,
            240,
            base_length,
            direction,
            binormal,
            start,
            head_element,
            radius_tip,
            radius_base,
            radius_head,
        )

        # Below function takes previously computed node positions and centerline
        # radius and modifies the rod.
        make_tappered_arm(
            self.shearable_rod,
            radius_along_rod,
            base_length,
            density,
            E,
            poisson_ratio,
            direction,
            normal,
            position,
        )
        # Now rod is ready for simulation, append rod to simulation
        self.simulator.append(self.shearable_rod)

        ## Add the target cyclinder

        # target_cyclinder = CosseratRod.straight_rod(
        #     n_elements=10,
        #     start=np.array([-0.5, 0, 0.5]),
        #     direction=np.array([0.0, 1.0, 0.0]),
        #     normal=np.array([0.0, 0.0, 1.0]),
        #     base_length=0.25,
        #     base_radius=0.02,
        #     density=1000,
        #     nu=5,
        #     youngs_modulus=5e6,
        #     poisson_ratio = 0.5
        # )

        # arm_muscle_with_basis_functions_sim.append(target_cyclinder)

        # As basis functions Gaussian distribution is used. Gaussian function
        # takes standard deviation as an input.
        # Scaling filter magnifies the basis function.
        # Scaling factor show the growth of number of basis functions.
        # For scaling factor 2, # of basis functions 4-2-1 from bottom level to top level.
        # For scaling factor 3, # of basis functions 9-3-1 from bottom level to top level.
        """ Muscles in normal direction / 1st bending mode """
        first_muscle_segment_normal = SpatiallyInvariantSplineHierarchy(
            Union(
                ScalingFilter(Gaussian(0.40), 10),  # Top level muscle segment
                ScalingFilter(Gaussian(0.16), 8),  # Mid level muscle segment
                ScalingFilter(Gaussian(0.10), 6),  # Bottom level muscle segment
            ),
            scaling_factor=2,
        )

        second_muscle_segment_normal = SpatiallyInvariantSplineHierarchy(
            Union(
                ScalingFilter(Gaussian(0.40), 10),  # Top level muscle segment
                ScalingFilter(Gaussian(0.16), 8),  # Mid level muscle segment
                ScalingFilter(Gaussian(0.10), 6),  # Bottom level muscle segment
            ),
            scaling_factor=2,
        )
        # Using SpatiallyInVariantSplineHierarchyMapper, enter what percentage
        # to what percentage of the rod is covered by this muscle segment
        first_muscle_mapper_in_normal_dir = SpatiallyInvariantSplineHierarchyMapper(
            first_muscle_segment_normal, (0.01, 0.40)
        )
        second_muscle_mapper_in_normal_dir = SpatiallyInvariantSplineHierarchyMapper(
            second_muscle_segment_normal, (0.60, 0.99)
        )

        segments_of_muscle_hierarchies_in_normal_dir = SplineHierarchySegments(
            first_muscle_mapper_in_normal_dir, second_muscle_mapper_in_normal_dir
        )

        """ Muscles in binormal direction / 2nd bending mode """
        first_muscle_segment_binormal = SpatiallyInvariantSplineHierarchy(
            Union(
                ScalingFilter(Gaussian(0.40), 10),  # Top level muscle segment
                ScalingFilter(Gaussian(0.16), 8),  # Mid level muscle segment
                ScalingFilter(Gaussian(0.10), 6),  # Bottom level muscle segment
            ),
            scaling_factor=2,
        )

        second_muscle_segment_binormal = SpatiallyInvariantSplineHierarchy(
            Union(
                ScalingFilter(Gaussian(0.40), 10),  # Top level muscle segment
                ScalingFilter(Gaussian(0.16), 8),  # Mid level muscle segment
                ScalingFilter(Gaussian(0.10), 6),  # Bottom level muscle segment
            ),
            scaling_factor=2,
        )

        first_muscle_mapper_in_binormal_dir = SpatiallyInvariantSplineHierarchyMapper(
            first_muscle_segment_binormal, (0.05, 0.40)
        )
        second_muscle_mapper_in_binormal_dir = SpatiallyInvariantSplineHierarchyMapper(
            second_muscle_segment_binormal, (0.60, 0.95)
        )

        segments_of_muscle_hierarchies_in_binormal_dir = SplineHierarchySegments(
            first_muscle_mapper_in_binormal_dir, second_muscle_mapper_in_binormal_dir
        )

        """ Muscles in tangent direction/ twist mode """
        first_muscle_segment_tangent = SpatiallyInvariantSplineHierarchy(
            Union(
                ScalingFilter(Gaussian(0.30), 6),  # Top level muscle segment
                ScalingFilter(Gaussian(0.24), 4),  # Mid level muscle segment
                ScalingFilter(Gaussian(0.20), 2),  # Bottom level muscle segment
            ),
            scaling_factor=2,
        )

        second_muscle_segment_tangent = SpatiallyInvariantSplineHierarchy(
            Union(
                ScalingFilter(Gaussian(0.30), 6),  # Top level muscle segment
                ScalingFilter(Gaussian(0.24), 4),  # Mid level muscle segment
                ScalingFilter(Gaussian(0.20), 2),  # Bottom level muscle segment
            ),
            scaling_factor=2,
        )

        first_muscle_mapper_in_tangent_dir = SpatiallyInvariantSplineHierarchyMapper(
            first_muscle_segment_tangent, (0.01, 0.40)
        )
        second_muscle_mapper_in_tangent_dir = SpatiallyInvariantSplineHierarchyMapper(
            second_muscle_segment_tangent, (0.60, 0.99)
        )

        segments_of_muscle_hierarchies_in_tangent_dir = SplineHierarchySegments(
            first_muscle_mapper_in_tangent_dir, second_muscle_mapper_in_tangent_dir
        )

        # Set the list for activation function and torque profile
        # activation_function_list_for_muscle_in_normal_dir = defaultdict(list)
        # torque_profile_list_for_muscle_in_normal_dir = defaultdict(list)

        # step_skip = 100

        # Set the activation arrays for each direction
        self.activation_arr_in_normal_dir = []
        self.activation_arr_in_binormal_dir = []
        self.activation_arr_in_tangent_dir = []

        # Apply torques
        self.simulator.add_forcing_to(self.shearable_rod).using(
            HierarchicalMuscleTorques,
            segments_of_muscle_hierarchies_in_normal_dir,
            activation_func=self.activation_arr_in_normal_dir,
            direction=normal,
            ramp_up_time=1.0,
            # step_skip=step_skip,
            # activation_function_recorder=activation_function_list_for_muscle_in_normal_dir,
            # torque_profile_recorder=torque_profile_list_for_muscle_in_normal_dir,
        )

        # activation_function_list_for_muscle_in_binormal_dir = defaultdict(list)
        # torque_profile_list_for_muscle_in_binormal_dir = defaultdict(list)

        self.simulator.add_forcing_to(self.shearable_rod).using(
            HierarchicalMuscleTorques,
            segments_of_muscle_hierarchies_in_binormal_dir,
            activation_func=self.activation_arr_in_binormal_dir,
            direction=np.cross(direction, normal),
            ramp_up_time=1.0,
            # step_skip=step_skip,
            # activation_function_recorder=activation_function_list_for_muscle_in_binormal_dir,
            # torque_profile_recorder=torque_profile_list_for_muscle_in_binormal_dir,
        )

        # activation_function_list_for_muscle_in_tangent_dir = defaultdict(list)
        # torque_profile_list_for_muscle_in_tangent_dir = defaultdict(list)

        self.simulator.add_forcing_to(self.shearable_rod).using(
            HierarchicalMuscleTorques,
            segments_of_muscle_hierarchies_in_tangent_dir,
            activation_func=self.activation_arr_in_tangent_dir,
            direction=direction,
            ramp_up_time=1.0,
            # step_skip=step_skip,
            # activation_function_recorder=activation_function_list_for_muscle_in_tangent_dir,
            # torque_profile_recorder=torque_profile_list_for_muscle_in_tangent_dir,
        )

        # Add gravitational forces
        gravitational_acc = -9.80665
        self.simulator.add_forcing_to(self.shearable_rod).using(
            GravityForces, acc_gravity=np.array([0.0, gravitational_acc, 0.0])
        )

        # Add frictional plane in environment
        origin_plane = np.array([0.0, 0.0, 0.0])
        normal_plane = normal
        slip_velocity_tol = 1e-8
        froude = 0.1
        period = 1.0
        mu = base_length / (period * period * np.abs(gravitational_acc) * froude)
        kinetic_mu_array = np.array([mu, mu, mu])  # [forward, backward, sideways]
        static_mu_array = 2 * kinetic_mu_array
        self.simulator.add_forcing_to(self.shearable_rod).using(
            AnistropicFrictionalPlane,
            k=1.0,
            nu=1e-0,
            plane_origin=origin_plane,
            plane_normal=normal_plane,
            slip_velocity_tol=slip_velocity_tol,
            static_mu_array=static_mu_array,
            kinetic_mu_array=kinetic_mu_array,
        )

        # Add call backs
        class ArmMuscleBasisCallBack(CallBackBaseClass):
            """
            Call back function for two arm octopus
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
                    # self.callback_params["velocity"].append(
                    #     system.velocity_collection.copy()
                    # )
                    # self.callback_params["avg_velocity"].append(
                    #     system.compute_velocity_center_of_mass()
                    # )
                    #
                    # self.callback_params["center_of_mass"].append(
                    #     system.compute_position_center_of_mass()
                    # )
                    self.callback_params["radius"].append(system.radius.copy())

                    return

        if self.COLLECT_DATA:
            # Collect data using callback function for postprocessing
            step_skip = 100  # collect data every # steps
            self.pp_list = defaultdict(list)  # list which collected data will be append
            # set the diagnostics for rod and collect data
            self.simulator.collect_diagnostics(self.shearable_rod).using(
                ArmMuscleBasisCallBack,
                step_skip=step_skip,
                callback_params=self.pp_list,
            )

        # Finalize simulation environment. After finalize, you cannot add
        # any forcing, constrain or call back functions
        self.simulator.finalize()

        # do_step, stages_and_updates will be used in step function
        self.do_step, self.stages_and_updates = extend_stepper_interface(
            self.StatefulStepper, self.simulator
        )

    def step(self, activation_array_list, time, dt):

        # Activation array contains lists for activation in different directions
        # assign correct activation arrays to correct directions.
        self.activation_arr_in_normal_dir[:] = activation_array_list[0]
        self.activation_arr_in_binormal_dir[:] = activation_array_list[1]
        self.activation_arr_in_tangent_dir[:] = activation_array_list[2]

        # Do one time step of simulation
        time = self.do_step(
            self.StatefulStepper, self.stages_and_updates, self.simulator, time, dt
        )

        # Observations, what should be observations?
        # Position, velocity ??
        # Observations can be rod parameters and can be
        # accessed after every time step.
        observation = self.shearable_rod.position_collection
        # observation = self.shearable_rod.velocity_collection

        """Reward function should be here"""
        reward = 0.0
        """Reward function should be here"""

        """ Done is a boolean to reset the environment before episode is completed """
        done = False
        """ Done is a boolean to reset the environment before episode is completed """

        return time, observation, reward, done

    def post_processing(self, filename_video):
        """
        Make video 3D rod movement in time.
        Parameters
        ----------
        filename_video
        Returns
        -------

        """

        if self.COLLECT_DATA:
            plot_video3d(
                self.pp_list, video_name=filename_video, margin=0.4, fps=20, step=10,
            )

        else:
            raise RuntimeError(
                "call back function is not called anytime during simulation, "
                "change COLLECT_DATA=True"
            )


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


def segment_activation_function(time):
    """
    This function is an example activation function for users. Similar to
    this function users can write their own activation function.
    Note that it is important to set correctly activation array sizes, which
    is number of basis functions for that muscle segment. Also users has to
    pack activation arrays in correct order at the return step, thus correct
    activation array activates correct muscle segment.
    Parameters
    ----------
    time

    Returns
    -------

    """

    # Muscle segment at the first arm, acting in first bending direction or normal direction
    activation_arr_1 = np.zeros((7))
    # Top level muscle segment
    activation_arr_1[0] = ramped_up(time - 1.0, 0.1)
    # Mid level muscle segment
    activation_arr_1[1:3] = ramped_up(time - 0.9, 0.1)
    # Bottom level muscle segment
    activation_arr_1[3:7] = ramped_up(time - 0.8, 0.1)

    # Muscle segment at the second arm, acting in first bending direction or normal direction
    activation_arr_2 = np.zeros((7))
    # Top level muscle segment
    activation_arr_2[0] = ramped_up(time - 1.0, 0.1)
    # Mid level muscle segment
    activation_arr_2[1:3] = ramped_up(time - 0.9, 0.1)
    # Bottom level muscle segment
    activation_arr_2[3:7] = ramped_up(time - 0.8, 0.1)

    # Muscle segment at the first arm, acting in second bending direction or binormal direction
    activation_arr_3 = np.zeros((7))
    # Top level muscle segment
    activation_arr_3[0] = ramped_up(time - 1.0, 0.1)
    # Mid level muscle segment
    activation_arr_3[1:3] = ramped_up(time - 0.9, 0.1)
    # Bottom level muscle segment
    activation_arr_3[3:7] = ramped_up(time - 0.8, 0.1)

    # Muscle segment at the second arm, acting in second bending direction or binormal direction
    activation_arr_4 = np.zeros((7))
    # Top level muscle segment
    activation_arr_4[0] = ramped_up(time - 1.0, 0.1)
    # Mid level muscle segment
    activation_arr_4[1:3] = ramped_up(time - 0.9, 0.1)
    # Bottom level muscle segment
    activation_arr_4[3:7] = ramped_up(time - 0.8, 0.1)

    # Muscle segment at the first arm, acting in twist direction or tangent direction
    activation_arr_5 = np.zeros((7))
    # Top level muscle segment
    activation_arr_5[0] = ramped_up(time - 1.0, 0.1)
    # Mid level muscle segment
    activation_arr_5[1:3] = ramped_up(time - 0.9, 0.1)
    # Bottom level muscle segment
    activation_arr_5[3:7] = ramped_up(time - 0.8, 0.1)

    # Muscle segment at the second arm, acting in twist direction or tangent direction
    activation_arr_6 = np.zeros((7))
    # Top level muscle segment
    activation_arr_6[0] = -1.0 * ramped_up(time - 1.0, 0.1)
    # Mid level muscle segment
    activation_arr_6[1:3] = -1.0 * ramped_up(time - 0.9, 0.1)
    # Bottom level muscle segment
    activation_arr_6[3:7] = -1.0 * ramped_up(time - 0.8, 0.1)

    return [
        [activation_arr_1, activation_arr_2],  # activation in normal direction
        [activation_arr_3, activation_arr_4],  # activation in binormal direction
        [activation_arr_5, activation_arr_6],  # activation in tangent direction
    ]


def main():
    # Set simulation integrator type, final time and time step
    timestepper = PositionVerlet()
    final_time = 10.0
    time_step = 4.0e-5
    total_steps = int(final_time / time_step)
    print("Total steps", total_steps)

    # Initialize the environment
    env = Environment(timestepper, COLLECT_DATA=True)
    env.reset()

    # Do multiple simulations for learning, or control
    for i_episodes in range(1):

        # Reset the environment before the new episode
        env.reset()

        # Simulation loop starts
        dt = np.float64(float(final_time) / total_steps)
        time = np.float64(0.0)

        for _ in tqdm(range(total_steps)):
            """ Compute the activation signal and pass to environment """
            activation = segment_activation_function(time)

            time, observation, reward, done = env.step(activation, time, dt)
            if done:
                print("Episode finished after {} ".format(time + 1))
                break

        print("Final time of simulation is : ", time)
        # Simulation loop ends

        # Post-processing
        # env.post_processing(filename_video="two_arm_simulation_3d_with_target.mp4")


# filename_video = "two_arm_simulation_zy.mp4"
# plot_video(
#     pp_list,
#     video_name=filename_video,
#     margin=0.4,
#     fps=20,
#     step=10,
# )
#
# filename_activation_muscle_torque_video = "two_arm_activation_normal_dir_muscle_torque.mp4"
# plot_video_actiavation_muscle(
#     activation_function_list_for_muscle_in_normal_dir,
#     torque_profile_list_for_muscle_in_normal_dir,
#     video_name=filename_activation_muscle_torque_video,
#     margin=0.2,
#     fps=20,
#     step=10,
# )
#
# filename_activation_muscle_torque_video = "two_arm_activation_binormal_dir_muscle_torque.mp4"
# plot_video_actiavation_muscle(
#     activation_function_list_for_muscle_in_binormal_dir,
#     torque_profile_list_for_muscle_in_binormal_dir,
#     video_name=filename_activation_muscle_torque_video,
#     margin=0.2,
#     fps=20,
#     step=10,
# )
#
# filename_activation_muscle_torque_video = "two_arm_activation_tangent_dir_muscle_torque.mp4"
# plot_video_actiavation_muscle(
#     activation_function_list_for_muscle_in_tangent_dir,
#     torque_profile_list_for_muscle_in_tangent_dir,
#     video_name=filename_activation_muscle_torque_video,
#     margin=0.2,
#     fps=20,
#     step=10,
# )
#
# filename_video = "two_arm_simulation_zx.mp4"
# plot_video_zx(
#     pp_list,
#     video_name=filename_video,
#     margin=0.4,
#     fps=20,
#     step=10,
# )
#
# filename_video = "two_arm_simulation_3d_with_target.mp4"
# plot_video3d(
#     env.pp_list, video_name=filename_video, margin=0.4, fps=20, step=10,
# )


# filename = "arm_tip_sensor_values.png"
# plot_arm_tip_sensor_values(sensor_list, filename, SAVE_FIGURE=True)


if __name__ == "__main__":
    main()
