import numpy as np
import numba
from numba import njit
from elastica.joint import FreeJoint
from elastica._linalg import _batch_norm, _batch_matvec


def get_connection_vector_for_perpendicular_rods(
    rod_one,
    rod_two,
    rod_one_index,
    rod_two_index,
):
    """
    This function computes the connection vectors in from rod one to rod two and rod two to rod one.
    Here we are assuming rod two tip is connected with rod one. Becareful with rod orders.
    Parameters
    ----------
    rod_one : rod object
    rod_two : rod object
    rod_one_index : int
    rod_two_index : int

    Returns
    -------

    """

    # Compute rod element positions
    rod_one_element_position = 0.5 * (
        rod_one.position_collection[..., 1:] + rod_one.position_collection[..., :-1]
    )
    rod_one_element_position = rod_one_element_position[:, rod_one_index]
    rod_two_element_position = 0.5 * (
        rod_two.position_collection[..., 1:] + rod_two.position_collection[..., :-1]
    )
    rod_two_element_position = rod_two_element_position[:, rod_two_index]

    # Lets get the distance between rod elements
    distance_vector_rod_one_to_rod_two = (
        rod_two_element_position - rod_one_element_position
    )
    distance_vector_rod_one_to_rod_two_norm = np.linalg.norm(
        distance_vector_rod_one_to_rod_two
    )
    distance_vector_rod_one_to_rod_two /= distance_vector_rod_one_to_rod_two_norm

    distance_vector_rod_two_to_rod_one = -distance_vector_rod_one_to_rod_two

    rod_one_direction_vec_in_material_frame = (
        rod_one.director_collection[:, :, rod_one_index]
        @ distance_vector_rod_one_to_rod_two
    )

    rod_two_direction_vec_in_material_frame = (
        rod_two.director_collection[:, :, rod_two_index]
        @ distance_vector_rod_two_to_rod_one
    )

    offset_btw_rods = distance_vector_rod_one_to_rod_two_norm - (
        rod_one.radius[rod_one_index] + rod_two.lengths[rod_two_index] / 2
    )

    return (
        rod_one_direction_vec_in_material_frame,
        rod_two_direction_vec_in_material_frame,
        offset_btw_rods,
    )


class PerpendicularRodsConnection(FreeJoint):
    """
    This is a connection class to connect two perpendicular rod elements.
    We are connecting rod two tip element with rod one.
    """

    def __init__(
        self,
        k,
        nu,
        k_repulsive,
        kt,
        rod_one_direction_vec_in_material_frame,
        rod_two_direction_vec_in_material_frame,
        offset_btw_rods,
        **kwargs,
    ):

        super().__init__(np.array(k), np.array(nu))

        self.k_repulsive = np.array(k_repulsive)
        self.kt = kt

        self.offset_btw_rods = np.array(offset_btw_rods)

        self.rod_one_direction_vec_in_material_frame = np.array(
            rod_one_direction_vec_in_material_frame
        ).T
        self.rod_two_direction_vec_in_material_frame = np.array(
            rod_two_direction_vec_in_material_frame
        ).T

    def apply_forces(self, rod_one, index_one, rod_two, index_two):
        (self.rod_one_rd2, self.rod_two_ld3, self.spring_force,) = self._apply_forces(
            self.k,
            self.nu,
            self.k_repulsive,
            index_one,
            index_two,
            self.rod_one_direction_vec_in_material_frame,
            self.rod_two_direction_vec_in_material_frame,
            self.offset_btw_rods,
            rod_one.director_collection,
            rod_two.director_collection,
            rod_one.position_collection,
            rod_two.position_collection,
            rod_one.radius,
            rod_two.lengths,
            rod_one.dilatation,
            rod_two.dilatation,
            rod_one.velocity_collection,
            rod_two.velocity_collection,
            rod_one.external_forces,
            rod_two.external_forces,
        )

    @staticmethod
    @njit(cache=True)
    def _apply_forces(
        k,
        nu,
        k_repulsive,
        index_one,
        index_two,
        rod_one_direction_vec_in_material_frame,
        rod_two_direction_vec_in_material_frame,
        rest_offset_btw_rods,
        rod_one_director_collection,
        rod_two_director_collection,
        rod_one_position_collection,
        rod_two_position_collection,
        rod_one_radius,
        rod_two_lengths,
        rod_one_dilatation,
        rod_two_dilatation,
        rod_one_velocity_collection,
        rod_two_velocity_collection,
        rod_one_external_forces,
        rod_two_external_forces,
    ):

        rod_one_to_rod_two_connection_vec = (
            rod_one_director_collection[:, :, index_one].T
            @ rod_one_direction_vec_in_material_frame
        ).reshape(3)
        rod_two_to_rod_one_connection_vec = (
            rod_two_director_collection[:, :, index_two].T
            @ rod_two_direction_vec_in_material_frame
        ).reshape(3)

        # Compute element positions
        rod_one_element_position = 0.5 * (
            rod_one_position_collection[:, index_one]
            + rod_one_position_collection[:, index_one + 1]
        )
        rod_two_element_position = 0.5 * (
            rod_two_position_collection[:, index_two]
            + rod_two_position_collection[:, index_two + 1]
        )

        # If there is an offset between rod one and rod two surface, then it should change as a function of dilatation.
        offset_rod_one = (
            0.5 * rest_offset_btw_rods / np.sqrt(rod_one_dilatation[index_one])
        )
        offset_rod_two = 0.5 * rest_offset_btw_rods * rod_two_dilatation[index_two]

        # Compute vector r*d2 (radius * connection vector) for each rod and element
        rod_one_rd2 = rod_one_to_rod_two_connection_vec * (
            rod_one_radius[index_one] + offset_rod_one
        )
        # Compute vector ld3 (radius * connection vector) for rod two to one
        rod_two_ld3 = rod_two_to_rod_one_connection_vec * (
            rod_two_lengths[index_two] / 2 + offset_rod_two
        )

        # Compute connection points on the rod surfaces
        surface_position_rod_one = rod_one_element_position + rod_one_rd2
        surface_position_rod_two = rod_two_element_position + rod_two_ld3

        # Compute spring force between two rods
        distance_vector = surface_position_rod_two - surface_position_rod_one
        np.round_(distance_vector, 12, distance_vector)
        spring_force = k * (distance_vector)

        # Damping force
        rod_one_element_velocity = 0.5 * (
            rod_one_velocity_collection[:, index_one]
            + rod_one_velocity_collection[:, index_one + 1]
        )
        rod_two_element_velocity = 0.5 * (
            rod_two_velocity_collection[:, index_two]
            + rod_two_velocity_collection[:, index_two + 1]
        )

        relative_velocity = rod_two_element_velocity - rod_one_element_velocity

        distance = np.linalg.norm(distance_vector)

        if distance >= 1e-12:
            normalized_distance_vector = distance_vector / distance
        else:
            normalized_distance_vector = np.zeros(
                3,
            )

        normal_relative_velocity_vector = (
            np.dot(relative_velocity, normalized_distance_vector)
            * normalized_distance_vector
        )

        damping_force = -nu * normal_relative_velocity_vector

        # Compute the total force
        total_force = spring_force + damping_force

        # Compute contact forces. Contact forces are applied in the case one rod penetrates to the other, in that case
        # we apply a repulsive force.
        center_distance = rod_two_element_position - rod_one_element_position
        center_distance_unit_vec = center_distance / np.linalg.norm(center_distance)
        penetration = np.linalg.norm(center_distance) - (
            rod_one_radius[index_one]
            + offset_rod_one
            + rod_two_lengths[index_two] / 2
            + offset_rod_two
        )

        round(penetration, 12)
        # Contact present only if rods penetrate to each other
        if penetration < 0:
            # Hertzian contact
            contact_force = (
                -k_repulsive * np.abs(penetration) ** (1.5) * center_distance_unit_vec
            )
        else:
            contact_force = np.zeros(
                3,
            )

        # Add contact forces
        total_force += contact_force

        # Re-distribute forces from elements to nodes.
        rod_one_external_forces[..., index_one] += 0.5 * total_force
        rod_one_external_forces[..., index_one + 1] += 0.5 * total_force
        rod_two_external_forces[..., index_two] -= 0.5 * total_force
        rod_two_external_forces[..., index_two + 1] -= 0.5 * total_force

        return (
            rod_one_rd2,
            rod_two_ld3,
            spring_force,
        )

    def apply_torques(self, rod_one, index_one, rod_two, index_two):
        self._apply_torques(
            self.spring_force,
            self.rod_one_rd2,
            self.rod_two_ld3,
            index_one,
            index_two,
            rod_one.director_collection,
            rod_two.director_collection,
            rod_one.external_torques,
            rod_two.external_torques,
            rod_one.radius,
            rod_one.position_collection,
            rod_two.position_collection,
            self.kt,
        )

    @staticmethod
    @njit(cache=True)
    def _apply_torques(
        spring_force,
        rod_one_rd2,
        rod_two_ld3,
        index_one,
        index_two,
        rod_one_director_collection,
        rod_two_director_collection,
        rod_one_external_torques,
        rod_two_external_torques,
        rod_one_radius,
        rod_one_position_collection,
        rod_two_position_collection,
        kt,
    ):
        # Compute torques due to the connection forces
        torque_on_rod_one = np.cross(rod_one_rd2, spring_force)
        torque_on_rod_two = np.cross(rod_two_ld3, -spring_force)

        # new method
        # We want to make sure rod one and rod two stays perpendicular to each other all the time. So we are
        # adding a torque spring to bring rods perpendicular to each other if they deform.
        # Compute element positions
        rod_one_element_position = 0.5 * (
            rod_one_position_collection[:, index_one]
            + rod_one_position_collection[:, index_one + 1]
        )
        rod_two_element_position = 0.5 * (
            rod_two_position_collection[:, index_two]
            + rod_two_position_collection[:, index_two + 1]
        )
        # Moment arm is in the direction for rod two tangent.
        moment_arm_dir = rod_two_ld3 / np.linalg.norm(rod_two_ld3)
        moment_arm = rod_one_radius[index_one] * moment_arm_dir + rod_two_ld3
        # Starting from rod two element center compute, the target position for rod one element.
        rod_one_target_element_position = rod_two_element_position + moment_arm
        # If rod one element position is different than target position then apply a torque to bring it back.
        distance_vector = rod_one_element_position - rod_one_target_element_position
        np.round_(distance_vector, 12, distance_vector)
        torque_spring_force = kt * (distance_vector)
        spring_torque = np.cross(moment_arm, torque_spring_force)

        torque_on_rod_one -= spring_torque
        torque_on_rod_two += spring_torque

        #
        torque_on_rod_one_material_frame = (
            rod_one_director_collection[:, :, index_one] @ torque_on_rod_one
        )
        torque_on_rod_two_material_frame = (
            rod_two_director_collection[:, :, index_two] @ torque_on_rod_two
        )

        rod_one_external_torques[..., index_one] += torque_on_rod_one_material_frame
        rod_two_external_torques[..., index_two] += torque_on_rod_two_material_frame
