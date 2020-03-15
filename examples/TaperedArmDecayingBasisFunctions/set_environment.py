import numpy as np

from collections import defaultdict
from elastica.wrappers import (
    BaseSystemCollection,
    Constraints,
    Forcing,
    CallBacks,
    Connections,
)
from elastica.rod.cosserat_rod import CosseratRod
from elastica.rigidbody import Cylinder
from elastica.external_forces import GravityForces
from elastica.hierarchical_muscles.hierarchical_muscle_torques import (
    HierarchicalMuscleTorques,
)
from elastica.interaction import (
    AnistropicFrictionalPlane,
    AnistropicFrictionalPlaneRigidBody,
)
from elastica.boundary_conditions import OneEndFixedRod
from elastica.hierarchical_muscles.hierarchical_bases import (
    SpatiallyInvariantSplineHierarchy,
    SpatiallyInvariantSplineHierarchyMapper,
    SplineHierarchySegments,
    Union,
    Gaussian,
    ScalingFilter,
)

from elastica.joint import ExternalContact
from elastica._calculus import _isnan_check
from elastica.callback_functions import CallBackBaseClass
from elastica.timestepper.symplectic_steppers import PositionVerlet
from elastica.timestepper import extend_stepper_interface

from examples.OctopusTwoArmCase.two_arm_octopus_postprocessing import (
    plot_video_with_surface,
)
from examples.ArmWithBasisFunctions.arm_sim_with_basis_functions_postprocessing import (
    plot_video_actiavation_muscle,
)

# Set base simulator class
class BaseSimulator(BaseSystemCollection, Constraints, Connections, Forcing, CallBacks):
    pass


class Environment:
    def __init__(
        self,
        final_time,
        number_of_muscle_segment,
        alpha,
        muscle_segment_overlap,
        COLLECT_DATA_FOR_POSTPROCESSING=False,
    ):
        # Integrator type
        self.StatefulStepper = PositionVerlet()

        # Simulation parameters
        self.final_time = final_time
        time_step = 1.0e-5  # this is a stable timestep
        self.total_steps = int(self.final_time / time_step)
        self.time_step = np.float64(float(self.final_time) / self.total_steps)
        # Video speed
        self.rendering_fps = 60
        self.step_skip = int(1.0 / (self.rendering_fps * self.time_step))

        # Number of muscle segments
        self.number_of_muscle_segment = number_of_muscle_segment

        # Basis function amplitude
        self.alpha = alpha

        # Muscle basis overlap
        self.muscle_segment_overlap = muscle_segment_overlap

        # Collect data is a boolean. If it is true callback function collects
        # rod parameters defined by user in a list.
        self.COLLECT_DATA_FOR_POSTPROCESSING = COLLECT_DATA_FOR_POSTPROCESSING

    def reset(self, cylinder_start):
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

        # Rigid body cylinder start position
        self.cylinder_start = cylinder_start

        # setting up test params
        n_elem = 220
        start = np.zeros((3,))
        direction = np.array([0.0, 1.0, 0.0])  # rod direction
        normal = np.array([0.0, 0.0, 1.0])
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
        radius_tip = 0.025  # radius of the arm at the tip
        radius_base = 0.05  # radius of the arm at the base

        radius_along_rod = np.linspace(radius_base, radius_tip, n_elem)
        position = self.shearable_rod.position_collection

        # Below function takes previously computed node positions and centerline
        # radius and modifies the rod.
        make_tapered_arm(
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

        # Constrain the rod
        self.simulator.constrain(self.shearable_rod).using(
            OneEndFixedRod, constrained_position_idx=(0,), constrained_director_idx=(0,)
        )

        self.cylinder = Cylinder(
            self.cylinder_start,  # cylinder  initial position
            normal,  # cylinder direction
            direction,  # cylinder normal
            0.6,  # cylinder length
            0.07,  # cylinder radius
            2 * 106.1032953945969,  # corresponds to mass of 4kg
        )
        self.simulator.append(self.cylinder)
        # Add external contact between rod and cyclinder
        self.simulator.connect(self.shearable_rod, self.cylinder).using(
            ExternalContact, 1e2, 0.1
        )

        """ Generate muscle segments along the arm """
        # Here we are generating multiple muscle segments along the arm. Number of muscle
        # segments is user defined input. Also user has to define the amplitude of the
        # muscle function which is alpha variable. Magnitude of basis functions decreases
        # from base to tip.

        muscle_mapper = []
        # Keep the start position of the first muscle segment at constant position(0.0) and shift
        # the rest of muscle segments towards the base so we get some overlap
        # if muscle_segment_overlap is zero, then there is no overlap between
        # muscle segment. 0.25 means we shift the muscles 25% to right and there is an overlap
        # of 25% of muscle segment length. When we move muscle segments towards right, there is
        # an empty space left at the tip of the arm. Thus, in order to have an overlap and cover the
        # whole arm with muscle segments, we need to increase the length of muscle segments.
        # In order to derive the muscle segments position, with overlap, we can use below series.
        # segment id    start position      stop  position
        # 0             0                   dx
        # 1             dx-offset*dx        2dx-offset*dx
        # 2             2dx-offset*2dx      3dx-offset*2dx
        # 3             3dx-offset*3dx      4dx-offset*3dx
        # n             ndx-offset*ndx      (n+1)dx-offset*ndx
        # As can be seen from the above series expansion, nth element stop position has to be at
        # 1.0 (non-dimensional length). Thus, we can easily find the length of segments.
        # dx = 1.0 / (n - (n-1) * offset)
        length_of_segments = 1.0 / (
            self.number_of_muscle_segment
            - (self.number_of_muscle_segment - 1) * self.muscle_segment_overlap
        )

        rod_lengths = self.shearable_rod.lengths
        rod_radius = self.shearable_rod.radius
        cumulative_lengths = np.cumsum(rod_lengths)
        non_dimensional_cum_length = cumulative_lengths / cumulative_lengths[-1]

        offset = self.muscle_segment_overlap
        for i in range(0, self.number_of_muscle_segment):
            # Start and stop non dimensional positions of the muscle segments
            start = i * (1 - offset) * length_of_segments
            stop = start + length_of_segments
            # Find the index where muscle segments start and stop
            idx = np.searchsorted(non_dimensional_cum_length, (start, stop))
            # Find the index at the middle of muscle segment to find the rod
            # radius at that position
            idx_at_the_middle = int((idx[0] + idx[1]) / 2)
            # Compute the muscle segment basis function amplitude. Here the formula is
            # A*(r/r_base)^2
            factor = self.alpha * (rod_radius[idx_at_the_middle] / max(rod_radius)) ** 2
            # Generate the muscle segment. Here we are only generating one layer
            # of muscles. By changing the standard deviation of Gaussian, we can
            # overlap the muscles. Gaussian function takes standard deviation as input.
            # For scaling factor we chose 2 and we add only one layer of basis function.
            muscle_segment = SpatiallyInvariantSplineHierarchy(
                Union(ScalingFilter(Gaussian(0.35), factor),), scaling_factor=2,
            )
            # Append muscles segments in to the list
            muscle_mapper.append(
                SpatiallyInvariantSplineHierarchyMapper(muscle_segment, (start, stop))
            )

        """ Muscles in normal direction / 1st bending mode """
        segments_of_muscle_hierarchies_in_normal_dir = SplineHierarchySegments(
            *muscle_mapper
        )

        # Set the list for activation function and torque profile
        self.activation_function_list_for_muscle_in_normal_dir = defaultdict(list)
        self.torque_profile_list_for_muscle_in_normal_dir = defaultdict(list)

        step_skip = 100

        # Set the activation arrays for normal direction
        self.activation_arr_in_normal_dir = []

        # # Apply torques
        self.simulator.add_forcing_to(self.shearable_rod).using(
            HierarchicalMuscleTorques,
            segments_of_muscle_hierarchies_in_normal_dir,
            activation_func_or_array=self.activation_arr_in_normal_dir,
            direction=-1 * normal,
            ramp_up_time=1.0,
            step_skip=step_skip,
            activation_function_recorder=self.activation_function_list_for_muscle_in_normal_dir,
            torque_profile_recorder=self.torque_profile_list_for_muscle_in_normal_dir,
        )

        # Add gravitational forces
        gravitational_acc = -9.80665
        self.simulator.add_forcing_to(self.shearable_rod).using(
            GravityForces, acc_gravity=np.array([0.0, 0.0, gravitational_acc])
        )

        # Add frictional plane in environment for shearable rod
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

        # Add gravitational forces
        gravitational_acc = -9.80665
        self.simulator.add_forcing_to(self.cylinder).using(
            GravityForces, acc_gravity=np.array([0.0, 0.0, gravitational_acc])
        )

        # Add friction plane in environment for rigid body cylinder
        origin_plane = np.array([0.0, 0.0, 0.0])
        normal_plane = normal
        slip_velocity_tol = 1e-8
        mu = 0.4
        kinetic_mu_array = np.array([mu, mu, mu])  # [forward, backward, sideways]
        static_mu_array = 2 * kinetic_mu_array
        self.simulator.add_forcing_to(self.cylinder).using(
            AnistropicFrictionalPlaneRigidBody,
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
                    self.callback_params["com"].append(
                        system.compute_position_center_of_mass()
                    )

                    return

        # Add call backs
        class RigidCylinderCallBack(CallBackBaseClass):
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
                    self.callback_params["com"].append(
                        system.compute_position_center_of_mass()
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
                    # self.callback_params["radius"].append(system.radius.copy())

                    return

        if self.COLLECT_DATA_FOR_POSTPROCESSING:
            # Collect data using callback function for postprocessing
            # step_skip = 500  # collect data every # steps
            self.post_processing_dict_rod = defaultdict(
                list
            )  # list which collected data will be append
            # set the diagnostics for rod and collect data
            self.simulator.collect_diagnostics(self.shearable_rod).using(
                ArmMuscleBasisCallBack,
                step_skip=self.step_skip,
                callback_params=self.post_processing_dict_rod,
            )

            self.post_processing_dict_cylinder = defaultdict(
                list
            )  # list which collected data will be append
            self.post_processing_dict_cylinder["radius"] = self.cylinder.radius
            self.post_processing_dict_cylinder["height"] = self.cylinder.length
            self.post_processing_dict_cylinder["direction"] = self.cylinder.tangents
            # set the diagnostics for cylinder and collect data
            self.simulator.collect_diagnostics(self.cylinder).using(
                RigidCylinderCallBack,
                step_skip=self.step_skip,
                callback_params=self.post_processing_dict_cylinder,
            )

        # Finalize simulation environment. After finalize, you cannot add
        # any forcing, constrain or call back functions
        self.simulator.finalize()

        # do_step, stages_and_updates will be used in step function
        self.do_step, self.stages_and_updates = extend_stepper_interface(
            self.StatefulStepper, self.simulator
        )

        systems = [self.shearable_rod, self.cylinder]

        return self.total_steps, systems

    def step(self, activation_array_list, time):

        # Activation array contains lists for activation in different directions
        # assign correct activation arrays to correct directions.
        self.activation_arr_in_normal_dir[:] = activation_array_list

        # Do one time step of simulation
        time = self.do_step(
            self.StatefulStepper,
            self.stages_and_updates,
            self.simulator,
            time,
            self.time_step,
        )

        systems = [self.shearable_rod, self.cylinder]

        """ Done is a boolean to reset the environment before episode is completed """
        done = False
        # Position of the rod cannot be NaN, it is not valid, stop the simulation
        invalid_values_condition = _isnan_check(self.shearable_rod.position_collection)

        if invalid_values_condition == True:
            print(" Nan detected, exiting simulation now")
            done = True
        """ Done is a boolean to reset the environment before episode is completed """

        return time, systems, done

    def post_processing(self, filename_video, **kwargs):
        """
        Make video 3D rod movement in time.
        Parameters
        ----------
        filename_video
        Returns
        -------

        """

        if self.COLLECT_DATA_FOR_POSTPROCESSING:

            plot_video_actiavation_muscle(
                self.activation_function_list_for_muscle_in_normal_dir,
                self.torque_profile_list_for_muscle_in_normal_dir,
                video_name="arm_activation_muscle_torque_in_normal_dir.mp4",
                margin=0.2,
                fps=20,
                step=10,
            )

            plot_video_with_surface(
                [self.post_processing_dict_rod],
                [self.post_processing_dict_cylinder],
                video_name=filename_video,
                fps=self.rendering_fps,
                step=1,
                **kwargs
            )

        else:
            raise RuntimeError(
                "call back function is not called anytime during simulation, "
                "change COLLECT_DATA=True"
            )


# TODO: this function should be a part of rod initialization, factory function and it should be removed from here
def make_tapered_arm(
    rod,
    radius_along_rod,
    base_length,
    density,
    youngs_modulus,
    poisson_ratio,
    direction,
    normal,
    position,
    alpha_c=4.0 / 3.0,
):
    """
    This function is used to reset the rod properties for a varying radius and/or not straight rod.
    User can input a rod with varying radius. If radius varying in each element mass, mass moment  of inertia,
    shear, bend matrices and volume are different. Also user can give nodepositions as 2 dimensional array and
    this function computes corresponding directors, rest curvature, rest strain, rest lengths.
    :param rod:
    :param radius_along_rod:
    :param density:
    :param youngs_modulus:
    :param poisson_ratio:
    :param direction:
    :param normal:
    :param position:
    :param alpha_c:
    :return:
    """
    from elastica.utils import MaxDimension, Tolerance

    # Use the before hand generated rod properties
    n_elements = rod.n_elems
    rest_lengths = rod.rest_lengths

    # Compute the arm properties
    radius = radius_along_rod

    # Second moment of inertia
    A0 = np.pi * radius * radius
    I0_1 = A0 * A0 / (4.0 * np.pi)
    I0_2 = I0_1
    I0_3 = 2.0 * I0_2
    I0 = np.array([I0_1, I0_2, I0_3]).transpose()
    # Mass second moment of inertia for disk cross-section
    mass_second_moment_of_inertia = np.zeros(
        (MaxDimension.value(), MaxDimension.value(), n_elements), np.float64
    )
    mass_second_moment_of_inertia_temp = I0 * density * base_length / n_elements
    for i in range(n_elements):
        np.fill_diagonal(
            mass_second_moment_of_inertia[..., i],
            mass_second_moment_of_inertia_temp[i, :],
        )
    # sanity check of mass second moment of inertia
    for k in range(n_elements):
        for i in range(0, MaxDimension.value()):
            assert mass_second_moment_of_inertia[i, i, k] > Tolerance.atol()

    # Inverse of second moment of inertia
    inv_mass_second_moment_of_inertia = np.zeros(
        (MaxDimension.value(), MaxDimension.value(), n_elements)
    )
    for i in range(n_elements):
        # Check rank of mass moment of inertia matrix to see if it is invertible
        assert (
            np.linalg.matrix_rank(mass_second_moment_of_inertia[..., i])
            == MaxDimension.value()
        )
        inv_mass_second_moment_of_inertia[..., i] = np.linalg.inv(
            mass_second_moment_of_inertia[..., i]
        )

    # Shear/Stretch matrix
    shear_modulus = youngs_modulus / (poisson_ratio + 1.0)
    shear_matrix = np.zeros(
        (MaxDimension.value(), MaxDimension.value(), n_elements), np.float64
    )
    for i in range(n_elements):
        np.fill_diagonal(
            shear_matrix[..., i],
            [
                alpha_c * shear_modulus * A0[i],
                alpha_c * shear_modulus * A0[i],
                youngs_modulus * A0[i],
            ],
        )
    for k in range(n_elements):
        for i in range(0, MaxDimension.value()):
            assert shear_matrix[i, i, k] > Tolerance.atol()

    # Bend/Twist matrix
    bend_matrix = np.zeros(
        (MaxDimension.value(), MaxDimension.value(), n_elements), np.float64
    )
    for i in range(n_elements):
        np.fill_diagonal(
            bend_matrix[..., i],
            [
                youngs_modulus * I0_1[i],
                youngs_modulus * I0_2[i],
                shear_modulus * I0_3[i],
            ],
        )
    for k in range(n_elements):
        for i in range(0, MaxDimension.value()):
            assert bend_matrix[i, i, k] > Tolerance.atol()
    # Compute bend matrix in Voronoi Domain
    bend_matrix = (
        bend_matrix[..., 1:] * rest_lengths[1:]
        + bend_matrix[..., :-1] * rest_lengths[0:-1]
    ) / (rest_lengths[1:] + rest_lengths[:-1])

    # Compute volume of elements
    volume = np.pi * radius_along_rod ** 2 * rest_lengths

    # Compute the mass of elements
    mass = np.zeros(n_elements + 1)
    mass[:-1] += 0.5 * density * volume
    mass[1:] += 0.5 * density * volume

    rod.radius[:] = radius_along_rod
    rod.mass_second_moment_of_inertia[:] = mass_second_moment_of_inertia
    rod.inv_mass_second_moment_of_inertia[:] = inv_mass_second_moment_of_inertia
    rod.shear_matrix[:] = shear_matrix
    rod.bend_matrix[:] = bend_matrix
    rod.volume[:] = volume
    rod.mass[:] = mass

    # Compute the tangents and directors
    position_diff = position[..., 1:] - position[..., :-1]
    lengths = np.sqrt(np.einsum("ij,ij->j", position_diff, position_diff))
    tangents = position_diff / lengths

    plane_binormals = np.cross(direction, normal)

    for k in range(n_elements):
        rod.director_collection[0, :, k] = normal
        rod.director_collection[1, :, k] = plane_binormals
        rod.director_collection[2, :, k] = tangents[..., k]

    rod.position_collection[:] = position

    # We have to compute
    rod._compute_shear_stretch_strains()
    rod._compute_bending_twist_strains()

    # Compute rest curvature and strains and reset the sigma and kappa
    rod.rest_kappa = rod.kappa.copy()
    rod.kappa *= 0.0
    rod.rest_sigma = rod.sigma.copy()
    rod.sigma *= 0.0


def make_two_arm_from_straigth_rod(
    rod,
    beta,
    base_length,
    direction,
    normal,
    start,
    head_n_elems,
    radius_tip,
    radius_base,
    radius_head,
):
    """
    This function is used to bend a rod and make two arms and head from the rod.
    Angle between arms and head is determined by the user input beta which is in degrees.
    This function positions and radius for three segments, which are first arm, head and
    second arm. Radius here is varying so that we can get a tapered arm.
    :param rod:
    :param beta:
    :param base_length:
    :param direction:
    :param normal:
    :param start:
    :param head_n_elems:
    :param radius_tip:
    :param radius_base:
    :param radius_head:
    :return:
    position: this is the position of nodes
    radius: for tapered arm radius is varying
    """
    from elastica.utils import MaxDimension

    n_elements = rod.n_elems

    # Compute the arm number of elements
    arm_1_n_elems = int((n_elements - head_n_elems) / 2)
    arm_2_n_elems = int((n_elements - head_n_elems) / 2)

    # Compute the radius along the rod
    s = np.linspace(
        0.0, head_n_elems / 2 * base_length / n_elements, int(head_n_elems / 2)
    )
    half_head_radius = np.tanh(s) / max(np.tanh(s)) * (radius_head) + radius_tip
    other_half_head_radius = half_head_radius[::-1]

    radius_along_rod = np.linspace(radius_tip, radius_base, arm_1_n_elems)
    radius_along_rod = np.hstack(
        (radius_along_rod, half_head_radius, other_half_head_radius)
    )
    radius_along_rod = np.hstack(
        (radius_along_rod, np.linspace(radius_tip, radius_base, arm_2_n_elems)[::-1])
    )

    # radius_along_rod = np.linspace(radius_tip, radius_base, arm_1_n_elems)
    # for i in range(head_element):
    #     radius_along_rod = np.hstack((radius_along_rod, radius_head))
    # radius_along_rod = np.hstack((radius_along_rod, np.linspace(radius_tip, radius_base, arm_2_n_elems)[::-1]))

    # beta is the angle between head elements and arm
    alpha = (90 - beta / 2) / 180 * np.pi

    d3_segment1 = np.cos(alpha) * direction + np.sin(alpha) * normal
    d3_segment1 /= np.linalg.norm(d3_segment1)

    # Set the head directors of the octopus
    d3_segment2 = direction / np.linalg.norm(direction)

    d3_segment3 = np.cos(-alpha) * direction + np.sin(-alpha) * normal
    d3_segment3 /= np.linalg.norm(d3_segment3)

    # We have to compute the correct position for arm and we have to check the the sigma, and kappa as well
    segment_number_of_elements = np.array([arm_1_n_elems, head_n_elems, arm_2_n_elems])
    start_idx_1 = 0
    end_idx_1 = start_idx_1 + arm_1_n_elems

    start_idx_2 = end_idx_1
    end_idx_2 = start_idx_2 + head_n_elems

    start_idx_3 = end_idx_2
    end_idx_3 = start_idx_3 + arm_2_n_elems

    start_idx = np.hstack((start_idx_1, start_idx_2, start_idx_3))
    end_idx = np.hstack((end_idx_1, end_idx_2, end_idx_3))

    direction_of_segments = np.vstack((d3_segment1, d3_segment2, d3_segment3))
    position = np.zeros((MaxDimension.value(), n_elements + 1))

    for k in range(segment_number_of_elements.shape[0]):
        end = (
            start
            + direction_of_segments[k, ...]
            * base_length
            / n_elements
            * segment_number_of_elements[k]
        )
        for i in range(0, MaxDimension.value()):
            position[i, start_idx[k] : end_idx[k] + 1] = np.linspace(
                start[i], end[i], num=segment_number_of_elements[k] + 1
            )
        # New segments start position should be old segments end position
        start = end

    return radius_along_rod, position
