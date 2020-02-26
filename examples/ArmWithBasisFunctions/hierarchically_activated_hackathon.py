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
from elastica.interaction import AnistropicFrictionalPlane
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
    plot_arm_tip_sensor_values,
    plot_video_zx,
    plot_video3d,
)


class ArmBasisSimulator(BaseSystemCollection, Constraints, Forcing, CallBacks):
    pass


# def main():
arm_muscle_with_basis_functions_sim = ArmBasisSimulator()

# setting up test params
n_elem = 120  # 40 #220
start = np.zeros((3,))
direction = np.array(
    [0.0, 0.0, 1.0]
)  # np.array([0.0, -1.0, 0.0])  # np.array([0.0, 0.0, 1.0])
normal = np.array(
    [0.0, 1.0, 0.0]
)  # np.array([0.0, 0.0, 1.0])  # np.array([0.0, 1.0, 0.0])
binormal = np.cross(direction, normal)
base_length = 1.0
base_radius = 0.05  # 0.025
base_area = np.pi * base_radius ** 2
density = 1000
nu = 5.0
E = 5e6  # 1e7
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

radius_tip = 0.025
radius_base = 0.03
radius_head = 0.1


# TODO: this function should be a part of rod initialization, factory function and it should be removed from here
def make_tappered_arm(
    rod,
    radius_along_rod,
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
        shearable_rod.director_collection[0, :, k] = plane_binormals
        shearable_rod.director_collection[1, :, k] = np.cross(
            plane_binormals, tangents[..., k]
        )
        shearable_rod.director_collection[2, :, k] = tangents[..., k]

    shearable_rod.position_collection[:] = position

    # We have to compute
    shearable_rod._compute_shear_stretch_strains()
    shearable_rod._compute_bending_twist_strains()

    # Compute rest curvature and strains and reset the sigma and kappa
    shearable_rod.rest_kappa = shearable_rod.kappa.copy()
    shearable_rod.kappa *= 0.0
    shearable_rod.rest_sigma = shearable_rod.sigma.copy()
    shearable_rod.sigma *= 0.0


def make_two_arm_from_straigth_rod(
    rod,
    beta,
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
    second arm. Radius here is varying so that we can get a tappered arm.
    :param rod:
    :param beta:
    :param direction:
    :param normal:
    :param start:
    :param head_n_elems:
    :param radius_tip:
    :param radius_base:
    :param radius_head:
    :return:
    position: this is the position of nodes
    radius: for tappered arm radius is varying
    """
    from elastica.utils import MaxDimension

    n_elements = rod.n_elems

    # Compute the arm number of elements
    arm_1_n_elems = int((n_elem - head_n_elems) / 2)
    arm_2_n_elems = int((n_elem - head_n_elems) / 2)

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


radius_along_rod, position = make_two_arm_from_straigth_rod(
    shearable_rod,
    240,
    direction,
    binormal,
    start,
    head_element,
    radius_tip,
    radius_base,
    radius_head,
)

make_tappered_arm(
    shearable_rod,
    radius_along_rod,
    density,
    E,
    poisson_ratio,
    direction,
    normal,
    position,
)


from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

fig = plt.figure(figsize=(10, 8), frameon=True, dpi=150)
ax = plt.axes(projection="3d")
ax.scatter(
    shearable_rod.position_collection[2],
    shearable_rod.position_collection[0],
    shearable_rod.position_collection[1],
    s=shearable_rod.radius ** 2 * np.pi * 10000,
)
ax.set_xlim(0, 1.0)
ax.set_ylim(-1.00, 0.00)
ax.set_zlim(0.0, 1.0)
ax.set_xlabel("z positon")
ax.set_ylabel("x position")
ax.set_zlabel("y position")
plt.show()


arm_muscle_with_basis_functions_sim.append(shearable_rod)

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

# shearable_rod = CosseratRod.straight_rod(
#     n_elem,
#     start,
#     direction,
#     normal,
#     base_length,
#     base_radius,
#     density,
#     nu,
#     E,
#     poisson_ratio,
# )

# arm_muscle_with_basis_functions_sim.append(target_cyclinder)

# Add the boundary conditions
# arm_muscle_with_basis_functions_sim.constrain(shearable_rod).using(
#     OneEndFixedRod, constrained_position_idx=(0,), constrained_director_idx=(0,)
# )

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

# first_muscle_segment = SpatiallyInvariantSplineHierarchy(
#     Union(
#         ScalingFilter(Gaussian(0.20),10),
#         ScalingFilter(Gaussian(0.08),8),
#         ScalingFilter(Gaussian(0.05),6),
#     ),
#     scaling_factor=2
# )
#
# second_muscle_segment = SpatiallyInvariantSplineHierarchy(
#     Union(
#         ScalingFilter(Gaussian(0.20),10),
#         ScalingFilter(Gaussian(0.08),8),
#         ScalingFilter(Gaussian(0.05),6),
#     ),
#     scaling_factor=2
# )

# First pack of hierarchical muscles
# Muscles in normal direction / 1st bending mode
first_muscle_segment_normal = SpatiallyInvariantSplineHierarchy(
    Union(
        ScalingFilter(Gaussian(0.20), 10),
        ScalingFilter(Gaussian(0.08), 8),
        ScalingFilter(Gaussian(0.05), 6),
    ),
    scaling_factor=2,
)

second_muscle_segment_normal = SpatiallyInvariantSplineHierarchy(
    Union(
        ScalingFilter(Gaussian(0.20), 10),
        ScalingFilter(Gaussian(0.08), 8),
        ScalingFilter(Gaussian(0.05), 6),
    ),
    scaling_factor=2,
)

first_muscle_mapper_in_normal_dir = SpatiallyInvariantSplineHierarchyMapper(
    first_muscle_segment_normal, (0.01, 0.40)
)
second_muscle_mapper_in_normal_dir = SpatiallyInvariantSplineHierarchyMapper(
    second_muscle_segment_normal, (0.60, 0.99)
)

segments_of_muscle_hierarchies_in_normal_dir = SplineHierarchySegments(
    first_muscle_mapper_in_normal_dir, second_muscle_mapper_in_normal_dir
)

# Muscles in binormal direction /  2nd bending mode
first_muscle_segment_binormal = SpatiallyInvariantSplineHierarchy(
    Union(
        ScalingFilter(Gaussian(0.20), 10),
        ScalingFilter(Gaussian(0.08), 8),
        ScalingFilter(Gaussian(0.05), 6),
    ),
    scaling_factor=2,
)

second_muscle_segment_binormal = SpatiallyInvariantSplineHierarchy(
    Union(
        ScalingFilter(Gaussian(0.20), 10),
        ScalingFilter(Gaussian(0.08), 8),
        ScalingFilter(Gaussian(0.05), 6),
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

# Muscles in tangent direction/ this is also twist
first_muscle_segment_tangent = SpatiallyInvariantSplineHierarchy(
    Union(
        ScalingFilter(Gaussian(0.15), 6),
        ScalingFilter(Gaussian(0.12), 4),
        ScalingFilter(Gaussian(0.10), 2),
    ),
    scaling_factor=2,
)

second_muscle_segment_tangent = SpatiallyInvariantSplineHierarchy(
    Union(
        ScalingFilter(Gaussian(0.15), 6),
        ScalingFilter(Gaussian(0.12), 4),
        ScalingFilter(Gaussian(0.10), 2),
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

#
# third_muscle_mapper = SpatiallyInvariantSplineHierarchyMapper(
#     thrid_muscle_segment, (0.60, 0.75)
# )
# fourth_muscle_mapper = SpatiallyInvariantSplineHierarchyMapper(
#     fourth_muscle_segment, (0.75, 0.95)
# )

# segments_of_muscle_hierarchies = SplineHierarchySegments(
#     first_muscle_mapper
# )

# segments_of_muscle_hierarchies = SplineHierarchySegments(
#     first_muscle_mapper ,second_muscle_mapper
# )

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


# def single_segment_activation(time_v):
#     activation_arr = np.zeros((13,))
#
#     # top_level
#     activation_arr[0] = ramped_up(time_v - 0.6, 0.1)
#
#     # mid_levels
#     activation_arr[1:4] = ramped_up(time_v - 0.3, 0.1)
#
#     # bottom boys
#     activation_arr[4:] = ramped_up(time_v, 0.1)
#
#     return activation_arr
#
# def two_segment_activation(time_v):
#
#     # NOTE ! Activation is reversed to make the video correct
#
#     # The first muscle segment that controls the shoulder
#     # top_level
#     activation_arr_1 = np.zeros((13))
#
#     activation_arr_1[0] = ramped_up(time_v - 1.0, 0.1)
#     # mid_levels
#     activation_arr_1[1:4] = ramped_up(time_v - 0.9, 0.1)
#     # bottom boys
#     activation_arr_1[4:13] = ramped_up(time_v - 0.8, 0.1)
#
#     # The second muscle segment that controls the finger
#     activation_arr_2 = np.zeros((13))
#     activation_arr_2[0] = ramped_up(time_v - 1.0, 0.1)
#     # mid_levels
#     activation_arr_2[1:4] = ramped_up(time_v - 0.9, 0.1)
#     # bottom boys
#     activation_arr_2[4:13] = ramped_up(time_v - 0.8, 0.1)
#
#     return [activation_arr_1, activation_arr_2]
#
def two_segment_activation(time_v):

    # NOTE ! Activation is reversed to make the video correct

    # The first muscle segment that controls the shoulder
    # top_level
    activation_arr_1 = np.zeros((7))

    activation_arr_1[0] = ramped_up(time_v - 1.0, 0.1)
    # mid_levels
    activation_arr_1[1:3] = ramped_up(time_v - 0.9, 0.1)
    # bottom boys
    activation_arr_1[3:7] = ramped_up(time_v - 0.8, 0.1)

    # The second muscle segment that controls the finger
    activation_arr_2 = np.zeros((13))
    activation_arr_2[0] = ramped_up(time_v - 1.0, 0.1)
    # mid_levels
    activation_arr_2[1:3] = ramped_up(time_v - 0.9, 0.1)
    # bottom boys
    activation_arr_2[3:7] = ramped_up(time_v - 0.8, 0.1)

    return [activation_arr_1, activation_arr_2]


def two_segment_activation_tangent(time_v):

    # NOTE ! Activation is reversed to make the video correct

    # The first muscle segment that controls the shoulder
    # top_level
    activation_arr_1 = np.zeros((7))

    activation_arr_1[0] = ramped_up(time_v - 1.0, 0.1)
    # mid_levels
    activation_arr_1[1:3] = ramped_up(time_v - 0.9, 0.1)
    # bottom boys
    activation_arr_1[3:7] = ramped_up(time_v - 0.8, 0.1)

    # The second muscle segment that controls the finger
    activation_arr_2 = np.zeros((13))
    activation_arr_2[0] = -1.0 * ramped_up(time_v - 1.0, 0.1)
    # mid_levels
    activation_arr_2[1:3] = -1.0 * ramped_up(time_v - 0.9, 0.1)
    # bottom boys
    activation_arr_2[3:7] = -1.0 * ramped_up(time_v - 0.8, 0.1)

    return [activation_arr_1, activation_arr_2]


# Set the list for activation function and torque profile
activation_function_list_for_muscle_in_normal_dir = defaultdict(list)
torque_profile_list_for_muscle_in_normal_dir = defaultdict(list)

step_skip = 100

# Apply torques
arm_muscle_with_basis_functions_sim.add_forcing_to(shearable_rod).using(
    HierarchicalMuscleTorques,
    segments_of_muscle_hierarchies_in_normal_dir,
    activation_func=two_segment_activation,
    direction=normal,
    ramp_up_time=1.0,
    step_skip=step_skip,
    activation_function_recorder=activation_function_list_for_muscle_in_normal_dir,
    torque_profile_recorder=torque_profile_list_for_muscle_in_normal_dir,
)


activation_function_list_for_muscle_in_binormal_dir = defaultdict(list)
torque_profile_list_for_muscle_in_binormal_dir = defaultdict(list)

arm_muscle_with_basis_functions_sim.add_forcing_to(shearable_rod).using(
    HierarchicalMuscleTorques,
    segments_of_muscle_hierarchies_in_binormal_dir,
    activation_func=two_segment_activation,
    direction=np.cross(direction, normal),
    ramp_up_time=1.0,
    step_skip=step_skip,
    activation_function_recorder=activation_function_list_for_muscle_in_binormal_dir,
    torque_profile_recorder=torque_profile_list_for_muscle_in_binormal_dir,
)

activation_function_list_for_muscle_in_tangent_dir = defaultdict(list)
torque_profile_list_for_muscle_in_tangent_dir = defaultdict(list)

arm_muscle_with_basis_functions_sim.add_forcing_to(shearable_rod).using(
    HierarchicalMuscleTorques,
    segments_of_muscle_hierarchies_in_tangent_dir,
    activation_func=two_segment_activation_tangent,
    direction=direction,
    ramp_up_time=1.0,
    step_skip=step_skip,
    activation_function_recorder=activation_function_list_for_muscle_in_tangent_dir,
    torque_profile_recorder=torque_profile_list_for_muscle_in_tangent_dir,
)

# Add gravitational forces
gravitational_acc = -9.80665
arm_muscle_with_basis_functions_sim.add_forcing_to(shearable_rod).using(
    GravityForces, acc_gravity=np.array([0.0, gravitational_acc, 0.0])
)

# Add friction forces
origin_plane = np.array([0.0, 0.0, 0.0])
normal_plane = normal
slip_velocity_tol = 1e-8
froude = 0.1
period = 1.0
mu = base_length / (period * period * np.abs(gravitational_acc) * froude)
kinetic_mu_array = np.array([mu, mu, mu])  # [forward, backward, sideways]
static_mu_array = 2 * kinetic_mu_array
arm_muscle_with_basis_functions_sim.add_forcing_to(shearable_rod).using(
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
            self.callback_params["radius"].append(system.radius.copy())

            return


# class ArmTipSensor(CallBackBaseClass):
#     """
#     Sensor function for arm
#     """
#
#     def __init__(
#         self, step_skip: int, callback_params: dict, alpha, target_position
#     ):
#         CallBackBaseClass.__init__(self)
#         self.every = step_skip
#         self.callback_params = callback_params
#         self.alpha = alpha
#         self.target_position = target_position
#
#     def make_callback(self, system, time, current_step: int):
#         if current_step % self.every == 0:
#             self.callback_params["time"].append(time)
#             self.callback_params["step"].append(current_step)
#             positon_sensor_value = self.alpha * (
#                 system.position_collection[..., -1] - self.target_position
#             )
#             self.callback_params["sensor"].append(positon_sensor_value)
#
#             return

pp_list = defaultdict(list)
arm_muscle_with_basis_functions_sim.collect_diagnostics(shearable_rod).using(
    ArmMuscleBasisCallBack, step_skip=step_skip, callback_params=pp_list,
)

# sensor_list = defaultdict(list)
# alpha = 0.1
# target_position = np.array([0, -0.45940746, 0.50805374])
# arm_muscle_with_basis_functions_sim.collect_diagnostics(shearable_rod).using(
#     ArmTipSensor,
#     step_skip=step_skip,
#     callback_params=sensor_list,
#     alpha=alpha,
#     target_position=target_position,
# )

arm_muscle_with_basis_functions_sim.finalize()
timestepper = PositionVerlet()

final_time = 10.0  # 11.0 + 0.01)
dt = 4.0e-5
total_steps = int(final_time / dt)
print("Total steps", total_steps)
integrate(timestepper, arm_muscle_with_basis_functions_sim, final_time, total_steps)

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
#     pp_list,
#     video_name=filename_video,
#     margin=0.4,
#     fps=20,
#     step=10,
# )


# filename = "arm_tip_sensor_values.png"
# plot_arm_tip_sensor_values(sensor_list, filename, SAVE_FIGURE=True)
#
# try:
#     import moviepy.editor as mpy
#
#     # We use the GIFs generated earlier to avoid recomputing the animations.
#     clip_mayavi = mpy.VideoFileClip(filename_video)
#     clip_mpl = mpy.VideoFileClip(filename_activation_muscle_torque_video).resize(
#         height=clip_mayavi.h
#     )
#     animation = mpy.clips_array([[clip_mpl, clip_mayavi]])
#     animation.write_videofile("combined.mp4", fps=20)
# except ImportError:
#     print("Whatsup!")
#
# Save arm position
# saved file order is (time,x,y,z)
filename_position = "position_of_arm"

time = np.array(pp_list["time"])
position_of_arm = np.array(pp_list["position"])

position_matrix = np.zeros((time.shape[0], 4, position_of_arm.shape[2] - 1))

for k in range(time.shape[0]):
    position_matrix[k, 0, :] = time[k]
    position_matrix[k, 1, :] = 0.5 * (
        position_of_arm[k, 0, 1:] + position_of_arm[k, 0, :-1]
    )
    position_matrix[k, 2, :] = 0.5 * (
        position_of_arm[k, 1, 1:] + position_of_arm[k, 1, :-1]
    )
    position_matrix[k, 3, :] = 0.5 * (
        position_of_arm[k, 2, 1:] + position_of_arm[k, 2, :-1]
    )

np.save(filename_position, position_matrix)

filename_radius = "radius_of_arm"
radius_of_arm = np.array(pp_list["radius"])
radius_matrix = np.zeros((time.shape[0], 2, radius_of_arm.shape[1]))

for k in range(time.shape[0]):
    radius_matrix[k, 0, :] = time[k]
    radius_matrix[k, 1, :] = radius_of_arm[k, :]

np.save(filename_radius, radius_matrix)

#
# # Save activation function
# # saved file order is (time, and basis functions)
# time = np.array(activation_function_list_for_muscle_in_normal_dir["time"])
# if "activation_signal" in activation_function_list_for_muscle_in_normal_dir:
#     filename_activation = "activation_function"
#     activation = np.array(activation_function_list_for_muscle_in_normal_dir["activation_signal"])
#     # activation_matrix = np.zeros((time.shape[0], int(activation.shape[1] + 1)))
#     np.save(filename_activation, activation)
# else:
#     first_activation = np.array(activation_function_list_for_muscle_in_normal_dir["first_activation_signal"])
#     filename_first_activation = "first_activation_function_normal_dir"
#     np.save(filename_first_activation, first_activation)
#     second_activation = np.array(
#         activation_function_list_for_muscle_in_normal_dir["second_activation_signal"]
#     )
#     filename_second_activation = "second_activation_function_normal_dir"
#     np.save(filename_second_activation, second_activation)
#
# time = np.array(activation_function_list_for_muscle_in_binormal_dir["time"])
# if "activation_signal" in activation_function_list_for_muscle_in_binormal_dir:
#     filename_activation = "activation_function"
#     activation = np.array(activation_function_list_for_muscle_in_binormal_dir["activation_signal"])
#     # activation_matrix = np.zeros((time.shape[0], int(activation.shape[1] + 1)))
#     np.save(filename_activation, activation)
# else:
#     first_activation = np.array(activation_function_list_for_muscle_in_binormal_dir["first_activation_signal"])
#     filename_first_activation = "first_activation_function_binormal_dir"
#     np.save(filename_first_activation, first_activation)
#     second_activation = np.array(
#         activation_function_list_for_muscle_in_binormal_dir["second_activation_signal"]
#     )
#     filename_second_activation = "second_activation_function_binormal_dir"
#     np.save(filename_second_activation, second_activation)
#
# time = np.array(activation_function_list_for_muscle_in_tangent_dir["time"])
# if "activation_signal" in activation_function_list_for_muscle_in_tangent_dir:
#     filename_activation = "activation_function"
#     activation = np.array(activation_function_list_for_muscle_in_tangent_dir["activation_signal"])
#     # activation_matrix = np.zeros((time.shape[0], int(activation.shape[1] + 1)))
#     np.save(filename_activation, activation)
# else:
#     first_activation = np.array(activation_function_list_for_muscle_in_tangent_dir["first_activation_signal"])
#     filename_first_activation = "first_activation_function_tangent_dir"
#     np.save(filename_first_activation, first_activation)
#     second_activation = np.array(
#         activation_function_list_for_muscle_in_tangent_dir["second_activation_signal"]
#     )
#     filename_second_activation = "second_activation_function_tangent_dir"
#     np.save(filename_second_activation, second_activation)
#
# # for k in range(time.shape[0]):
# #     activation_matrix[k, 0] = time[k]
# #     activation_matrix[k, 1:] = activation[k, :]
#
# # Save muscle functions
# time = np.array(torque_profile_list_for_muscle_in_normal_dir["time"])
# if "torque_mag" in activation_function_list_for_muscle_in_normal_dir:
#     muscle_torque_mag = np.array(torque_profile_list_for_muscle_in_normal_dir["torque_mag"])
#     filename_muscle_function = "muscle_torque"
#     np.save(filename_muscle_function, muscle_torque_mag)
# else:
#     first_muscle_torque_mag = np.array(torque_profile_list_for_muscle_in_normal_dir["first_torque_mag"])
#     filename_first_muscle_torque_mag = "first_muscle_torque_mag_function_normal_dir"
#     np.save(filename_first_muscle_torque_mag, first_muscle_torque_mag)
#     second_muscle_torque_mag = np.array(torque_profile_list_for_muscle_in_normal_dir["second_torque_mag"])
#     filename_second_muscle_torque_mag = "second_muscle_torque_mag_function_normal_dir"
#     np.save(filename_second_muscle_torque_mag, second_muscle_torque_mag)
#
# time = np.array(torque_profile_list_for_muscle_in_binormal_dir["time"])
# if "torque_mag" in activation_function_list_for_muscle_in_binormal_dir:
#     muscle_torque_mag = np.array(torque_profile_list_for_muscle_in_binormal_dir["torque_mag"])
#     filename_muscle_function = "muscle_torque"
#     np.save(filename_muscle_function, muscle_torque_mag)
# else:
#     first_muscle_torque_mag = np.array(torque_profile_list_for_muscle_in_binormal_dir["first_torque_mag"])
#     filename_first_muscle_torque_mag = "first_muscle_torque_mag_function_binormal_dir"
#     np.save(filename_first_muscle_torque_mag, first_muscle_torque_mag)
#     second_muscle_torque_mag = np.array(torque_profile_list_for_muscle_in_binormal_dir["second_torque_mag"])
#     filename_second_muscle_torque_mag = "second_muscle_torque_mag_function_binormal_dir"
#     np.save(filename_second_muscle_torque_mag, second_muscle_torque_mag)
#
# time = np.array(torque_profile_list_for_muscle_in_tangent_dir["time"])
# if "torque_mag" in activation_function_list_for_muscle_in_binormal_dir:
#     muscle_torque_mag = np.array(torque_profile_list_for_muscle_in_tangent_dir["torque_mag"])
#     filename_muscle_function = "muscle_torque"
#     np.save(filename_muscle_function, muscle_torque_mag)
# else:
#     first_muscle_torque_mag = np.array(torque_profile_list_for_muscle_in_tangent_dir["first_torque_mag"])
#     filename_first_muscle_torque_mag = "first_muscle_torque_mag_function_tangent_dir"
#     np.save(filename_first_muscle_torque_mag, first_muscle_torque_mag)
#     second_muscle_torque_mag = np.array(torque_profile_list_for_muscle_in_tangent_dir["second_torque_mag"])
#     filename_second_muscle_torque_mag = "second_muscle_torque_mag_function_tangent_dir"
#     np.save(filename_second_muscle_torque_mag, second_muscle_torque_mag)

# if __name__ == "__main__":
#     main()
