import sys

# FIXME without appending sys.path make it more generic
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")
sys.path.append("../../../../")
sys.path.append("../../../../../")
from elastica import *
from examples.MagneticRodCases.post_processing import (
    plot_video_with_surface,
)
from elastica.experimental.connection_contact_joint.connect_perpendicular_rods import (
    get_connection_vector_for_perpendicular_rods,
    PerpendicularRodsConnection,
)
from elastica.experimental.connection_contact_joint.parallel_connection import (
    SurfaceJointSideBySide,
    get_connection_vector_straight_straight_rod,
)
from elastica._linalg import _batch_norm
from elastica.experimental.interaction_plane_for_rod_tips import (
    IsotropicFrictionalPlaneForRodTips,
)


class MagneticDecapotSimualtor(
    BaseSystemCollection, Constraints, Connections, Forcing, CallBacks
):
    pass


magnetic_decapot_simulator = MagneticDecapotSimualtor()

magnetic_rod_list = []
backbone_rod_list = []
rod_list = []

spacing_direction_btw_magnetic_rods_in_one_backbone = np.array([1.0, 0.0, 0.0])
direction = np.array([0.0, 0.0, 1.0])
normal = np.array([0.0, 1.0, 0.0])
spacing_direction_btw_backbone_rods = normal
n_backbone_rods = 3

# setting up test params
n_magnetic_rod_per_backbone = 10
n_elem_magnetic_rod = 20
base_length_magnetic_rods = 1.5  # m
base_radius_backbone_rod = 0.15
base_radius_magnetic_rod = 0.15  # m

spacing_btw_magnetic_rods_in_one_backbone = (
    base_length_magnetic_rods * 2
)  # following Gu2020
spacing_btw_backbone_rods = 2 * base_radius_backbone_rod


density = 2.39e3  # kg/m3
nu = 60  # 50
E = 1.85e5  # Pa
shear_modulus = 6.16e4  # Pa

for backbone_idx in range(n_backbone_rods):
    start_collection_magnetic_rods = np.zeros((n_magnetic_rod_per_backbone, 3))
    for i in range(n_magnetic_rod_per_backbone):
        start_collection_magnetic_rods[i, :] = (
            i
            * spacing_btw_magnetic_rods_in_one_backbone
            * spacing_direction_btw_magnetic_rods_in_one_backbone
            + backbone_idx
            * spacing_btw_backbone_rods
            * spacing_direction_btw_backbone_rods
        )

    # Create back_bone
    base_length_back_bone = (
        n_magnetic_rod_per_backbone
    ) * spacing_btw_magnetic_rods_in_one_backbone
    n_elem_back_bone = (n_magnetic_rod_per_backbone) * 4
    direction_back_bone = spacing_direction_btw_magnetic_rods_in_one_backbone
    normal_back_bone = direction
    start_back_bone = (
        start_collection_magnetic_rods[0, :]
        + direction * (base_length_magnetic_rods + base_radius_backbone_rod)
        - spacing_direction_btw_magnetic_rods_in_one_backbone
        * base_length_back_bone
        / 2
        / n_elem_back_bone
    )
    # In order to adjust so that magnetic rods finishes at the edges of backbone.
    # FIXME: find a better way
    base_length_back_bone -= 3 * base_length_back_bone / n_elem_back_bone
    n_elem_back_bone -= 3

    back_bone_rod = CosseratRod.straight_rod(
        n_elem_back_bone,
        start_back_bone,
        direction_back_bone,
        normal_back_bone,
        base_length_back_bone,
        base_radius_backbone_rod,
        density,
        nu,
        E,
        shear_modulus=shear_modulus,
    )
    magnetic_decapot_simulator.append(back_bone_rod)
    backbone_rod_list.append(back_bone_rod)

    # Create magnetic rods
    base_area = np.pi * base_radius_magnetic_rod ** 2
    volume = base_area * base_length_magnetic_rods
    moment_of_inertia = np.pi / 4 * base_radius_magnetic_rod ** 4

    # Parameters are from Gu2020
    angular_frequency = np.deg2rad(
        5.0
    )  # angular frequency of the rotating magnetic field
    magnetic_field_strength = 80e-3  # 80mT
    # MBAL2_EI is a non-dimensional number from Wang 2019
    MBAL2_EI = (
        3.82e-5 * magnetic_field_strength * 4e-3 / (1.85e5 * np.pi / 4 * (0.4e-3) ** 4)
    )  # Magnetization magnitude * B * Length/(EI)
    magnetization_density = (
        MBAL2_EI
        * E
        * moment_of_inertia
        / (volume * magnetic_field_strength * base_length_magnetic_rods)
    )
    carpet_length = spacing_btw_magnetic_rods_in_one_backbone * (
        n_magnetic_rod_per_backbone - 1
    )
    spatial_magnetization_wavelength = carpet_length / 1
    spatial_magnetisation_phase_diff = np.pi
    magnetization_angle = spatial_magnetisation_phase_diff + (
        2
        * np.pi
        * start_collection_magnetic_rods[..., 0]
        / spatial_magnetization_wavelength
    )

    # Magnetic rod
    if not backbone_idx % 2 == 0:
        # Magnetic rods are not attached to every backbone. We are putting two empty backbones between every
        # magnetic rod - backbone pair.
        continue
    for i in range(n_magnetic_rod_per_backbone):
        magnetization_direction = np.array(
            [np.sin(magnetization_angle[i]), 0.0, np.cos(magnetization_angle[i])]
        ).reshape(3, 1) * np.ones((n_elem_magnetic_rod))
        magnetic_rod = MagneticRod.straight_rod(
            n_elem_magnetic_rod,
            start_collection_magnetic_rods[i],
            direction,
            normal,
            base_length_magnetic_rods,
            base_radius_magnetic_rod,
            density,
            nu,
            E,
            shear_modulus=shear_modulus,
            magnetization_density=magnetization_density,
            magnetization_direction=magnetization_direction,
        )
        magnetic_decapot_simulator.append(magnetic_rod)
        magnetic_rod_list.append(magnetic_rod)

rod_list += magnetic_rod_list + backbone_rod_list


# Connections
# Connect magnetic rods with their backbone
for backbone_idx, back_bone_rod in enumerate(backbone_rod_list):
    back_bone_rod_element_position = 0.5 * (
        back_bone_rod.position_collection[:, 1:]
        + back_bone_rod.position_collection[:, :-1]
    )
    magnetic_rod_connection_index = n_elem_magnetic_rod - 1
    rod_one_direction_vec_in_material_frame_list = []
    rod_two_direction_vec_in_material_frame_list = []
    offset_btw_rods_list = []
    for idx, magnetic_rod in enumerate(magnetic_rod_list):
        magnetic_rod_tip_element_position = 0.5 * (
            magnetic_rod.position_collection[:, magnetic_rod_connection_index]
            + magnetic_rod.position_collection[:, magnetic_rod_connection_index + 1]
        ).reshape(3, 1)

        distance_btw_rods = _batch_norm(
            back_bone_rod_element_position - magnetic_rod_tip_element_position
        )
        distance_btw_rods -= magnetic_rod.rest_lengths[-1] / 2 + back_bone_rod.radius[0]
        if np.min(distance_btw_rods) > 1e-8:
            # In this case rods are far from each other so no need for connection.
            continue
        back_bone_rod_connection_index = int(np.argmin(distance_btw_rods))

        (
            rod_one_direction_vec_in_material_frame,
            rod_two_direction_vec_in_material_frame,
            offset_btw_rods,
        ) = get_connection_vector_for_perpendicular_rods(
            back_bone_rod,
            magnetic_rod,
            rod_one_index=back_bone_rod_connection_index,
            rod_two_index=magnetic_rod_connection_index,  # Since we are connecting at its tip we know the index
        )

        rod_one_direction_vec_in_material_frame_list.append(
            rod_one_direction_vec_in_material_frame.copy()
        )
        rod_two_direction_vec_in_material_frame_list.append(
            rod_two_direction_vec_in_material_frame.copy()
        )
        offset_btw_rods_list.append(offset_btw_rods)

        magnetic_decapot_simulator.connect(
            first_rod=back_bone_rod,
            second_rod=magnetic_rod,
            first_connect_idx=back_bone_rod_connection_index,
            second_connect_idx=magnetic_rod_connection_index,
        ).using(
            PerpendicularRodsConnection,
            k=1e6 / 10,
            nu=0.1,
            k_repulsive=1e4,
            kt=1e4 * 10,
            rod_one_direction_vec_in_material_frame=rod_one_direction_vec_in_material_frame,
            rod_two_direction_vec_in_material_frame=rod_two_direction_vec_in_material_frame,
            offset_btw_rods=offset_btw_rods,
        )

# Connect backbones using parallel connections.
for rod_one_idx, rod_one in enumerate(backbone_rod_list):
    for rod_two_idx in range(rod_one_idx + 1, len(backbone_rod_list[:])):
        rod_two = backbone_rod_list[rod_two_idx]

        (
            rod_one_direction_vec_in_material_frame,
            rod_two_direction_vec_in_material_frame,
            offset_btw_rods,
        ) = get_connection_vector_straight_straight_rod(rod_one, rod_two)

        if np.max(offset_btw_rods) > 1e-8:
            # In this case rods are far from each other so no need for connection.
            continue

        assert (
            rod_one.n_elems == rod_two.n_elems
        ), " Backbone rods do not have same number of elements"

        n_elem_back_bone = rod_one.n_elems

        for elem_idx in range(n_elem_back_bone):
            magnetic_decapot_simulator.connect(
                first_rod=rod_one,
                second_rod=rod_two,
                first_connect_idx=elem_idx,
                second_connect_idx=elem_idx,
            ).using(
                SurfaceJointSideBySide,
                k=1e5,
                nu=0.1,
                k_repulsive=1e6,
                rod_one_direction_vec_in_material_frame=rod_one_direction_vec_in_material_frame[
                    :, elem_idx
                ],
                rod_two_direction_vec_in_material_frame=rod_two_direction_vec_in_material_frame[
                    :, elem_idx
                ],
                offset_btw_rods=offset_btw_rods[elem_idx],
            )


# # Create magnetic field object
# magnetic_field_object = ConstantMagneticField(
#     magnetic_field_amplitude=magnetic_field_strength * np.array([0, 0, 1]),
#     ramp_interval=0.01,
#     start_time=0.0,
#     end_time=5e3,
# )
#
# # Apply magnetic forces
# for i, rod in enumerate(magnetic_rod_list):
#     magnetic_decapot_simulator.add_forcing_to(rod).using(
#         ExternalMagneticFieldForces,
#         external_magnetic_field=magnetic_field_object,
#     )


# Add gravitational forces
from elastica.magnetic_forces import compute_ramp_factor


class GravityForcesRampUp(GravityForces):
    """
    This class applies a constant gravitational force to the entire rod.

        Attributes
        ----------
        acc_gravity: numpy.ndarray
            1D (dim) array containing data with 'float' type. Gravitational acceleration vector.

    """

    def __init__(
        self,
        start_time,
        ramp_interval,
        end_time,
        acc_gravity=np.array([0.0, -9.80665, 0.0]),
    ):
        """

        Parameters
        ----------
        acc_gravity: numpy.ndarray
            1D (dim) array containing data with 'float' type. Gravitational acceleration vector.

        """
        super(GravityForcesRampUp, self).__init__()
        self.acc_gravity = acc_gravity
        self.start_time = start_time
        self.ramp_interval = ramp_interval
        self.end_time = end_time

    def apply_forces(self, system, time=0.0):
        acc_gravity = self.acc_gravity * compute_ramp_factor(
            time, self.ramp_interval, self.start_time, self.end_time
        )
        self.compute_gravity_forces(acc_gravity, system.mass, system.external_forces)


gravitational_acc = -9.80665 / 250  # FIXME: gravity is small
for rod in magnetic_rod_list:
    magnetic_decapot_simulator.add_forcing_to(rod).using(
        GravityForcesRampUp,
        acc_gravity=direction * gravitational_acc,
        start_time=0.0,
        ramp_interval=20,
        end_time=1e6,
    )

for rod in backbone_rod_list:
    magnetic_decapot_simulator.add_forcing_to(rod).using(
        GravityForcesRampUp,
        acc_gravity=direction * gravitational_acc,
        start_time=0.0,
        ramp_interval=20,
        end_time=1e6,
    )


# Add friction forces and plane
# Add friction forces
period = 1.0
origin_plane = np.array([0.0, 0.0, 0.0])
normal_plane = direction
slip_velocity_tol = 1e-8
froude = 0.1
mu = base_length_back_bone / (period * period * np.abs(9.80665) * froude)
kinetic_mu = mu
static_mu = mu * 1.5
for rod in magnetic_rod_list:
    magnetic_decapot_simulator.add_forcing_to(rod).using(
        IsotropicFrictionalPlaneForRodTips,
        k=1000,
        nu=1,
        plane_origin=origin_plane,
        plane_normal=normal_plane,
        slip_velocity_tol=slip_velocity_tol,
        static_mu=static_mu,
        kinetic_mu=kinetic_mu,
    )


# Add callbacks
class MagneticBeamCallBack(CallBackBaseClass):
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
            self.callback_params["radius"].append(system.radius.copy())
            self.callback_params["velocity"].append(system.velocity_collection.copy())
            self.callback_params["tangents"].append(system.tangents.copy())


num_cycles = 0.25 * 4  # 4#1#8
final_time = num_cycles * 2 * np.pi / angular_frequency
dl = base_length_magnetic_rods / n_elem_magnetic_rod
dt = 0.01 * dl
total_steps = int(final_time / dt)
rendering_fps = 10
step_skip = int(1.0 / (rendering_fps * dt))


# Add call back for plotting time history of the rod
rod_post_processing_list = []
for idx, rod in enumerate(rod_list):
    rod_post_processing_list.append(defaultdict(list))
    magnetic_decapot_simulator.collect_diagnostics(rod).using(
        MagneticBeamCallBack,
        step_skip=step_skip,
        callback_params=rod_post_processing_list[idx],
    )


timestepper = PositionVerlet()
magnetic_decapot_simulator.finalize()
integrate(timestepper, magnetic_decapot_simulator, final_time, total_steps)

# Plot the magnetic rod time history
# plot_video_with_surface(
#     rod_post_processing_list,
#     fps=rendering_fps,
#     step=4,
#     x_limits=(-2, carpet_length + 4),
#     y_limits=(-2, 2),
#     z_limits=(-2, 2),
# )

plot_video_with_surface(
    rod_post_processing_list,
    fps=rendering_fps,
    step=4,
    x_limits=(-5, carpet_length + 2),
    y_limits=(-2, 2),
    z_limits=(-5, 2 + carpet_length),
)
