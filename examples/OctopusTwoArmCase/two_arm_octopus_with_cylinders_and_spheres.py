import numpy as np

from elastica.callback_functions import CallBackBaseClass
from collections import defaultdict
from elastica.wrappers import (
    BaseSystemCollection,
    Constraints,
    Forcing,
    CallBacks,
    Connections,
)
from elastica.joint import ExternalContact
from elastica.rod.cosserat_rod import CosseratRod
from elastica.rigidbody import Sphere, Cylinder
from elastica.external_forces import GravityForces
from elastica.hierarchical_muscles.hierarchical_muscle_torques import (
    HierarchicalMuscleTorques,
)
from elastica.interaction import (
    AnistropicFrictionalPlane,
    AnistropicFrictionalPlaneRigidBody,
)
from elastica.hierarchical_muscles.hierarchical_bases import (
    SpatiallyInvariantSplineHierarchy,
    SpatiallyInvariantSplineHierarchyMapper,
    SplineHierarchySegments,
    Union,
    Gaussian,
    ScalingFilter,
)
from examples.OctopusTwoArmCase.set_environment import (
    make_two_arm_from_straigth_rod,
    make_tapered_arm,
)
from elastica.timestepper.symplectic_steppers import PositionVerlet
from elastica.timestepper import integrate


# Set base simulator class
class BaseSimulator(BaseSystemCollection, Constraints, Connections, Forcing, CallBacks):
    pass


simulator = BaseSimulator()

# setting up test params
n_elem = 120
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

# Set the arm properties after defining rods
# We will have one element for head only
head_element = int(20)

radius_tip = 0.025  # radius of the arm at the tip
radius_base = 0.03  # radius of the arm at the base
radius_head = 0.1  # radius of the head

# Below function takes the rod and computes new position of nodes
# and centerline radius for the user defined configuration.
radius_along_rod, position = make_two_arm_from_straigth_rod(
    shearable_rod,
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
make_tapered_arm(
    shearable_rod,
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
simulator.append(shearable_rod)

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


def segment_activation_function_normal(time):
    """
    This function is an example activation function for users. Similar to
    this function users can write their own activation function.
    Note that it is important to set correctly activation array sizes, which
    is number of basis functions for that muscle segment. Also users has to
    pack activation arrays in correct order at the return step, thus correct
    activation array activates correct muscle segment.
    Note that, activation array values can take numbers between -1 and 1. If you
    put numbers different than these, Elastica clips the values. If activation value
    is -1, this basis function generates torque in opposite direction.
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

    return [activation_arr_1, activation_arr_2]  # activation in normal direction


def segment_activation_function_binormal(time):
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
    return [activation_arr_3, activation_arr_4]  # activation in binormal direction


def segment_activation_function_tangent(time):

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

    return [activation_arr_5, activation_arr_6]  # activation in tangent direction


# Set the activation arrays for each direction
activation_arr_in_normal_dir = []
activation_arr_in_binormal_dir = []
activation_arr_in_tangent_dir = []

# Apply torques
simulator.add_forcing_to(shearable_rod).using(
    HierarchicalMuscleTorques,
    segments_of_muscle_hierarchies_in_normal_dir,
    activation_func_or_array=segment_activation_function_normal,
    direction=normal,
)

simulator.add_forcing_to(shearable_rod).using(
    HierarchicalMuscleTorques,
    segments_of_muscle_hierarchies_in_binormal_dir,
    activation_func_or_array=segment_activation_function_binormal,
    direction=np.cross(direction, normal),
)

simulator.add_forcing_to(shearable_rod).using(
    HierarchicalMuscleTorques,
    segments_of_muscle_hierarchies_in_tangent_dir,
    activation_func_or_array=segment_activation_function_tangent,
    direction=direction,
)

# Add gravitational forces
gravitational_acc = -9.80665
simulator.add_forcing_to(shearable_rod).using(
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
simulator.add_forcing_to(shearable_rod).using(
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
            self.callback_params["position"].append(system.position_collection.copy())
            # callback_params["velocity"].append(
            #     system.velocity_collection.copy()
            # )
            # callback_params["avg_velocity"].append(
            #     system.compute_velocity_center_of_mass()
            # )
            #
            # callback_params["center_of_mass"].append(
            #     system.compute_position_center_of_mass()
            # )
            self.callback_params["radius"].append(system.radius.copy())
            self.callback_params["com"].append(system.compute_position_center_of_mass())

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
            self.callback_params["position"].append(system.position_collection.copy())
            self.callback_params["com"].append(system.compute_position_center_of_mass())
            return


# Collect data using callback function for postprocessing
step_skip = 500  # collect data every # steps
rod_history = defaultdict(list)  # list which collected data will be append
# set the diagnostics for rod and collect data
simulator.collect_diagnostics(shearable_rod).using(
    ArmMuscleBasisCallBack, step_skip=step_skip, callback_params=rod_history,
)


N_CYLINDERS = 8
cylinders = [None for _ in range(N_CYLINDERS)]
cylinder_histories = [defaultdict(list) for _ in range(N_CYLINDERS)]
cylinder_radii = [None for _ in range(N_CYLINDERS)]

# Configuration of cylinders
mean_cylinder_radius = 0.05
max_variation_cylinder_radius = 0.03
start_circle_radius = 0.3  # controls where the cylinders will be located
assert max_variation_cylinder_radius < mean_cylinder_radius

# Prepare to add friction plane in environment for rigid body cyclinder
mu = 0.4
kinetic_mu_array = np.array([mu, mu, mu])  # [forward, backward, sideways]
static_mu_array = 2 * kinetic_mu_array

com_rod = shearable_rod.compute_position_center_of_mass()
for i in range(N_CYLINDERS):
    theta = i / N_CYLINDERS * 2.0 * np.pi
    cylinder_start = com_rod + start_circle_radius * np.array(
        [np.cos(theta), np.sin(theta), 0.0]
    )
    cylinder_radius = mean_cylinder_radius + max_variation_cylinder_radius * (
        np.random.random() * 2.0 - 1.0
    )
    cylinders[i] = Cylinder(
        cylinder_start,  # cylinder  initial position
        normal,  # cylinder direction
        direction,  # cylinder normal
        0.6,  # cylinder length
        cylinder_radius,  # cylinder radius
        2 * 106.1032953945969,  # corresponds to mass of 4kg
    )
    cylinder_radii[i] = cylinder_radius
    simulator.append(cylinders[i])
    # Add external contact between rod and cyclinder
    simulator.connect(shearable_rod, cylinders[i]).using(ExternalContact, 1e2, 0.1)
    # Add gravitational forces
    simulator.add_forcing_to(cylinders[i]).using(
        GravityForces, acc_gravity=np.array([0.0, 0.0, gravitational_acc])
    )

    simulator.add_forcing_to(cylinders[i]).using(
        AnistropicFrictionalPlaneRigidBody,
        k=1.0,
        nu=1e-0,
        plane_origin=origin_plane,
        plane_normal=normal_plane,
        slip_velocity_tol=slip_velocity_tol,
        static_mu_array=static_mu_array,
        kinetic_mu_array=kinetic_mu_array,
    )
    # set the diagnostics for cyclinder and collect data
    cylinder_histories[i]["radius"] = cylinder_radius
    cylinder_histories[i]["height"] = 0.6
    cylinder_histories[i]["direction"] = normal.copy()
    simulator.collect_diagnostics(cylinders[i]).using(
        RigidCylinderCallBack,
        step_skip=step_skip,
        callback_params=cylinder_histories[i],
    )


# Finalize simulation environment. After finalize, you cannot add
# any forcing, constrain or call back functions
simulator.finalize()

timestepper = PositionVerlet()

final_time = 5.0
dt = 4.0e-5
total_steps = int(final_time / dt)
print("Total steps", total_steps)
integrate(timestepper, simulator, final_time, total_steps)

PLOT_VIDEO = False
if PLOT_VIDEO:
    from examples.OctopusTwoArmCase.two_arm_octopus_postprocessing import (
        plot_video_with_surface,
    )

    plot_video_with_surface(
        [rod_history],
        cylinder_histories,
        fps=60,
        step=1,
        video_name="two_arm_simulation_with_cylinders.mp4",
        # The following parameters are optional
        x_limits=(-1.0, 0.5),  # Set bounds on x-axis
        y_limits=(-0.025, 1.25),  # Set bounds on y-axis
        z_limits=(-0.05, 1.00),  # Set bounds on z-axis
        dpi=100,  # Set the quality of the image
        vis3D=True,  # Turn on 3D visualization
        vis2D=True,  # Turn on projected (2D) visualization
    )

SAVE_DATA_FOR_POVRAY_VIZ = True
if SAVE_DATA_FOR_POVRAY_VIZ:
    import os

    save_folder = os.path.join(os.getcwd(), "data")
    os.makedirs(save_folder, exist_ok=True)

    np.savez(
        os.path.join(save_folder, "octopus_data.npz"),
        position=np.array(rod_history["position"]),
        radii=np.array(rod_history["radius"]),
    )

    #
    for i in range(N_CYLINDERS):
        save_file_name = os.path.join(save_folder, "cylinder_data_{:04d}.npz".format(i))
        base_point = (
            np.array(cylinder_histories[i]["com"])
            - 0.5 * cylinder_histories[i]["height"] * cylinder_histories[i]["direction"]
        )
        cap_point = (
            np.array(cylinder_histories[i]["com"])
            + 0.5 * cylinder_histories[i]["height"] * cylinder_histories[i]["direction"]
        )
        radius = cylinder_histories[i]["radius"]
        np.savez(
            save_file_name, base_point=base_point, cap_point=cap_point, radius=radius
        )
