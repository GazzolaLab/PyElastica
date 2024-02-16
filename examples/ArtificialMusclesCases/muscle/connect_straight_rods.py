import numpy as np
from numba import njit
from elastica.joint import FreeJoint
from elastica.typing import SystemType, RodType
from elastica._rotations import _inv_rotate


# Join the two rods
from elastica._linalg import (
    _batch_norm,
    _batch_cross,
    _batch_matvec,
    _batch_dot,
    _batch_matmul,
    _batch_matrix_transpose,
    _batch_product_k_ik_to_ik,
)


def get_connection_vector_straight_straight_rod(
    rod_one,
    rod_two,
):

    # Compute rod element positions
    rod_one_element_position = 0.5 * (
        rod_one.position_collection[..., 1:] + rod_one.position_collection[..., :-1]
    )
    rod_two_element_position = 0.5 * (
        rod_two.position_collection[..., 1:] + rod_two.position_collection[..., :-1]
    )
    # Lets get the distance between rod elements
    distance_vector_rod_one_to_rod_two = (
        rod_two_element_position - rod_one_element_position
    )
    distance_vector_rod_one_to_rod_two_norm = _batch_norm(
        distance_vector_rod_one_to_rod_two
    )

    distance_vector_rod_one_to_rod_two /= distance_vector_rod_one_to_rod_two_norm

    distance_vector_rod_two_to_rod_one = -distance_vector_rod_one_to_rod_two

    rod_one_direction_vec_in_material_frame = _batch_matvec(
        rod_one.director_collection, distance_vector_rod_one_to_rod_two
    )
    rod_two_direction_vec_in_material_frame = _batch_matvec(
        rod_two.director_collection, distance_vector_rod_two_to_rod_one
    )

    offset_btw_rods = distance_vector_rod_one_to_rod_two_norm - (
        rod_one.radius + rod_two.radius
    )

    return (
        rod_one_direction_vec_in_material_frame,
        rod_two_direction_vec_in_material_frame,
        offset_btw_rods,
    )


class SurfaceJointSideBySide(FreeJoint):
    """
    TODO: documentation
    """

    def __init__(
        self,
        k,
        nu,
        k_repulsive,
        friction_coefficient,
        velocity_damping_coefficient,
        rod_one_direction_vec_in_material_frame,
        rod_two_direction_vec_in_material_frame,
        offset_btw_rods,
        **kwargs,
    ):
        super().__init__(np.array(k), np.array(nu))
        self.k_repulsive = np.array(k_repulsive)
        self.offset_btw_rods = np.array(offset_btw_rods)
        self.friction_coefficient = np.array(friction_coefficient)
        self.velocity_damping_coefficient = np.array(velocity_damping_coefficient)

        self.rod_one_direction_vec_in_material_frame = np.array(
            rod_one_direction_vec_in_material_frame
        ).T
        self.rod_two_direction_vec_in_material_frame = np.array(
            rod_two_direction_vec_in_material_frame
        ).T

    # Apply force is same as free joint
    def apply_forces(self, rod_one, index_one, rod_two, index_two):
        # TODO: documentation

        (self.rod_one_rd2, self.rod_two_rd2, self.spring_force,) = self._apply_forces(
            self.k,
            self.nu,
            self.k_repulsive,
            self.friction_coefficient,
            self.velocity_damping_coefficient,
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
            rod_two.radius,
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
        friction_coefficient,
        velocity_damping_coefficient,
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
        rod_two_radius,
        rod_one_dilatation,
        rod_two_dilatation,
        rod_one_velocity_collection,
        rod_two_velocity_collection,
        rod_one_external_forces,
        rod_two_external_forces,
    ):

        rod_one_to_rod_two_connection_vec = _batch_matvec(
            _batch_matrix_transpose(rod_one_director_collection[:, :, index_one]),
            rod_one_direction_vec_in_material_frame,
        )
        rod_two_to_rod_one_connection_vec = _batch_matvec(
            _batch_matrix_transpose(rod_two_director_collection[:, :, index_two]),
            rod_two_direction_vec_in_material_frame,
        )

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
        offset_rod_two = (
            0.5 * rest_offset_btw_rods / np.sqrt(rod_two_dilatation[index_two])
        )

        # Compute vector r*d2 (radius * connection vector) for each rod and element
        rod_one_rd2 = rod_one_to_rod_two_connection_vec * (
            rod_one_radius[index_one] + offset_rod_one
        )
        rod_two_rd2 = rod_two_to_rod_one_connection_vec * (
            rod_two_radius[index_two] + offset_rod_two
        )

        # Compute connection points on the rod surfaces
        surface_position_rod_one = rod_one_element_position + rod_one_rd2
        surface_position_rod_two = rod_two_element_position + rod_two_rd2

        # Compute spring force between two rods
        distance_vector = surface_position_rod_two - surface_position_rod_one

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

        np.round_(distance_vector, 12, distance_vector)
        distance = _batch_norm(distance_vector)
        normalized_distance_vector = np.zeros((relative_velocity.shape))
        idx_nonzero_distance = np.where(distance >= 1e-12)[0]
        normalized_distance_vector[..., idx_nonzero_distance] = (
            distance_vector[..., idx_nonzero_distance] / distance[idx_nonzero_distance]
        )
        # print(np.min(_batch_norm(distance_vector)),np.max(_batch_norm(distance_vector)))
        val = 1e-4
        # fx = (np.maximum(distance,val)-val)
        fx = distance
        # print(np.min(distance),np.max(distance))
        spring_force = k * fx * normalized_distance_vector

        normal_relative_velocity_vector = (
            _batch_dot(relative_velocity, normalized_distance_vector)
            * normalized_distance_vector
        )

        damping_force = -nu * normal_relative_velocity_vector

        # Compute the total force
        total_force = spring_force + damping_force

        # Compute contact forces. Contact forces are applied in the case one rod penetrates to the other, in that case
        # we apply a repulsive force. Later on these repulsive forces are used to move rods apart from each other and
        # as a pressure force.
        # We assume contact forces are in plane.
        # print('------------------___+')
        center_distance = rod_two_element_position - rod_one_element_position
        center_distance_unit_vec = center_distance / _batch_norm(center_distance)
        #        penetration_strain = (
        #            _batch_norm(center_distance)
        #            / (
        #                rod_one_radius[index_one]
        #                + offset_rod_one
        #                + rod_two_radius[index_two]
        #                + offset_rod_two
        #            )
        #            - 1
        #        )
        penetration_strain = _batch_norm(center_distance) - (
            rod_one_radius[index_one]
            + offset_rod_one
            + rod_two_radius[index_two]
            + offset_rod_two
        )
        np.round_(penetration_strain, 12, penetration_strain)
        idx_penetrate = np.where(penetration_strain < 0)[0]
        k_contact = np.zeros(index_one.shape[0])
        k_contact_temp = -k_repulsive * np.abs(penetration_strain) ** 1.5
        k_contact[idx_penetrate] += k_contact_temp[idx_penetrate]
        contact_force = k_contact * center_distance_unit_vec
        # contact_force[:,idx_penetrate] = 0.0

        # Add contact forces
        total_force += contact_force

        # Compute the spring forces in plane. If there is contact spring force is also contributing to contact force
        # so we need to compute it and add to contact_force.
        spring_force_temp_for_contact = np.zeros((3, index_one.shape[0]))
        spring_force_temp_for_contact[:, idx_penetrate] += spring_force[
            :, idx_penetrate
        ]

        contact_force += spring_force_temp_for_contact

        # Friction

        # Compute friction
        slip_interpenetration_velocity = (
            relative_velocity - normal_relative_velocity_vector
        )
        slip_interpenetration_velocity_mag = _batch_norm(slip_interpenetration_velocity)
        slip_interpenetration_velocity_unitized = slip_interpenetration_velocity / (
            slip_interpenetration_velocity_mag + 1e-14
        )

        # Compute Coulombic friction
        coulombic_friction_force = friction_coefficient * _batch_norm(contact_force)

        # Compare damping force in slip direction and kinetic friction and minimum is the friction force.
        # Compute friction force in the slip direction.
        damping_force_in_slip_direction = (
            velocity_damping_coefficient * slip_interpenetration_velocity_mag
        )
        friction_force = (
            np.minimum(damping_force_in_slip_direction, coulombic_friction_force)
            * slip_interpenetration_velocity_unitized
        )
        # Update contact force
        total_force += friction_force
        # print(np.linalg.norm(friction_force))

        # Re-distribute forces from elements to nodes.
        block_size = index_one.shape[0]
        for k in range(block_size):
            rod_one_external_forces[0, index_one[k]] += 0.5 * total_force[0, k]
            rod_one_external_forces[1, index_one[k]] += 0.5 * total_force[1, k]
            rod_one_external_forces[2, index_one[k]] += 0.5 * total_force[2, k]

            rod_one_external_forces[0, index_one[k] + 1] += 0.5 * total_force[0, k]
            rod_one_external_forces[1, index_one[k] + 1] += 0.5 * total_force[1, k]
            rod_one_external_forces[2, index_one[k] + 1] += 0.5 * total_force[2, k]

            rod_two_external_forces[0, index_two[k]] -= 0.5 * total_force[0, k]
            rod_two_external_forces[1, index_two[k]] -= 0.5 * total_force[1, k]
            rod_two_external_forces[2, index_two[k]] -= 0.5 * total_force[2, k]

            rod_two_external_forces[0, index_two[k] + 1] -= 0.5 * total_force[0, k]
            rod_two_external_forces[1, index_two[k] + 1] -= 0.5 * total_force[1, k]
            rod_two_external_forces[2, index_two[k] + 1] -= 0.5 * total_force[2, k]

        return (
            rod_one_rd2,
            rod_two_rd2,
            spring_force,
        )

    def apply_torques(self, rod_one, index_one, rod_two, index_two):
        # pass

        self._apply_torques(
            self.spring_force,
            self.rod_one_rd2,
            self.rod_two_rd2,
            index_one,
            index_two,
            rod_one.director_collection,
            rod_two.director_collection,
            rod_one.external_torques,
            rod_two.external_torques,
        )

    @staticmethod
    @njit(cache=True)
    def _apply_torques(
        spring_force,
        rod_one_rd2,
        rod_two_rd2,
        index_one,
        index_two,
        rod_one_director_collection,
        rod_two_director_collection,
        rod_one_external_torques,
        rod_two_external_torques,
    ):
        # Compute torques due to the connection forces
        torque_on_rod_one = _batch_cross(rod_one_rd2, spring_force)
        torque_on_rod_two = _batch_cross(rod_two_rd2, -spring_force)

        torque_on_rod_one_material_frame = _batch_matvec(
            rod_one_director_collection[:, :, index_one], torque_on_rod_one
        )
        torque_on_rod_two_material_frame = _batch_matvec(
            rod_two_director_collection[:, :, index_two], torque_on_rod_two
        )

        blocksize = index_one.shape[0]
        for k in range(blocksize):
            rod_one_external_torques[
                0, index_one[k]
            ] += torque_on_rod_one_material_frame[0, k]
            rod_one_external_torques[
                1, index_one[k]
            ] += torque_on_rod_one_material_frame[1, k]
            rod_one_external_torques[
                2, index_one[k]
            ] += torque_on_rod_one_material_frame[2, k]

            rod_two_external_torques[
                0, index_two[k]
            ] += torque_on_rod_two_material_frame[0, k]
            rod_two_external_torques[
                1, index_two[k]
            ] += torque_on_rod_two_material_frame[1, k]
            rod_two_external_torques[
                2, index_two[k]
            ] += torque_on_rod_two_material_frame[2, k]


class ContactSurfaceJoint(FreeJoint):
    """
    TODO: documentation
    """

    def __init__(
        self,
        k,
        nu,
        k_repulsive,
        rod_one_direction_vec_in_material_frame,
        rod_two_direction_vec_in_material_frame,
        offset_btw_rods,
        **kwargs,
    ):
        super().__init__(np.array(k), np.array(nu))
        self.k_repulsive = np.array(k_repulsive)
        self.offset_btw_rods = np.array(offset_btw_rods)

        self.rod_one_direction_vec_in_material_frame = np.array(
            rod_one_direction_vec_in_material_frame
        ).T
        self.rod_two_direction_vec_in_material_frame = np.array(
            rod_two_direction_vec_in_material_frame
        ).T

    # Apply force is same as free joint
    def apply_forces(self, rod_one, index_one, rod_two, index_two):
        # TODO: documentation

        (self.rod_one_rd2, self.rod_two_rd2, self.spring_force,) = self._apply_forces(
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
            rod_two.radius,
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
        rod_two_radius,
        rod_one_dilatation,
        rod_two_dilatation,
        rod_one_velocity_collection,
        rod_two_velocity_collection,
        rod_one_external_forces,
        rod_two_external_forces,
    ):

        rod_one_to_rod_two_connection_vec = _batch_matvec(
            _batch_matrix_transpose(rod_one_director_collection[:, :, index_one]),
            rod_one_direction_vec_in_material_frame,
        )
        rod_two_to_rod_one_connection_vec = _batch_matvec(
            _batch_matrix_transpose(rod_two_director_collection[:, :, index_two]),
            rod_two_direction_vec_in_material_frame,
        )

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
        offset_rod_two = (
            0.5 * rest_offset_btw_rods / np.sqrt(rod_two_dilatation[index_two])
        )

        # Compute vector r*d2 (radius * connection vector) for each rod and element
        rod_one_rd2 = rod_one_to_rod_two_connection_vec * (
            rod_one_radius[index_one] + offset_rod_one
        )
        rod_two_rd2 = rod_two_to_rod_one_connection_vec * (
            rod_two_radius[index_two] + offset_rod_two
        )

        # Compute connection points on the rod surfaces
        surface_position_rod_one = rod_one_element_position + rod_one_rd2
        surface_position_rod_two = rod_two_element_position + rod_two_rd2

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

        distance = _batch_norm(distance_vector)

        normalized_distance_vector = np.zeros((relative_velocity.shape))

        idx_nonzero_distance = np.where(distance >= 1e-12)[0]

        normalized_distance_vector[..., idx_nonzero_distance] = (
            distance_vector[..., idx_nonzero_distance] / distance[idx_nonzero_distance]
        )

        normal_relative_velocity_vector = (
            _batch_dot(relative_velocity, normalized_distance_vector)
            * normalized_distance_vector
        )

        damping_force = -nu * normal_relative_velocity_vector

        # Compute the total force
        total_force = spring_force + damping_force

        # Compute contact forces. Contact forces are applied in the case one rod penetrates to the other, in that case
        # we apply a repulsive force. Later on these repulsive forces are used to move rods apart from each other and
        # as a pressure force.
        # We assume contact forces are in plane.
        # print('------------------___+')
        center_distance = rod_two_element_position - rod_one_element_position
        center_distance_unit_vec = center_distance / _batch_norm(center_distance)
        #        penetration_strain = (
        #            _batch_norm(center_distance)
        #            / (
        #                rod_one_radius[index_one]
        #                + offset_rod_one
        #                + rod_two_radius[index_two]
        #                + offset_rod_two
        #            )
        #            - 1
        #        )
        penetration_strain = _batch_norm(center_distance) - (
            rod_one_radius[index_one] + rod_two_radius[index_two]
        )
        np.round_(penetration_strain, 12, penetration_strain)
        idx_penetrate = np.where(penetration_strain < 0)[0]
        k_contact = np.zeros(index_one.shape[0])
        k_contact_temp = -k_repulsive * np.abs(penetration_strain)
        k_contact[idx_penetrate] += k_contact_temp[idx_penetrate]
        contact_force = k_contact * center_distance_unit_vec
        # contact_force[:,idx_penetrate] = 0.0

        # Add contact forces
        total_force += contact_force

        # Compute the spring forces in plane. If there is contact spring force is also contributing to contact force
        # so we need to compute it and add to contact_force.
        spring_force_temp_for_contact = np.zeros((3, index_one.shape[0]))
        spring_force_temp_for_contact[:, idx_penetrate] += spring_force[
            :, idx_penetrate
        ]

        contact_force += spring_force_temp_for_contact

        # Re-distribute forces from elements to nodes.
        block_size = index_one.shape[0]
        for k in range(block_size):
            rod_one_external_forces[0, index_one[k]] += 0.5 * total_force[0, k]
            rod_one_external_forces[1, index_one[k]] += 0.5 * total_force[1, k]
            rod_one_external_forces[2, index_one[k]] += 0.5 * total_force[2, k]

            rod_one_external_forces[0, index_one[k] + 1] += 0.5 * total_force[0, k]
            rod_one_external_forces[1, index_one[k] + 1] += 0.5 * total_force[1, k]
            rod_one_external_forces[2, index_one[k] + 1] += 0.5 * total_force[2, k]

            rod_two_external_forces[0, index_two[k]] -= 0.5 * total_force[0, k]
            rod_two_external_forces[1, index_two[k]] -= 0.5 * total_force[1, k]
            rod_two_external_forces[2, index_two[k]] -= 0.5 * total_force[2, k]

            rod_two_external_forces[0, index_two[k] + 1] -= 0.5 * total_force[0, k]
            rod_two_external_forces[1, index_two[k] + 1] -= 0.5 * total_force[1, k]
            rod_two_external_forces[2, index_two[k] + 1] -= 0.5 * total_force[2, k]

        return (
            rod_one_rd2,
            rod_two_rd2,
            spring_force,
        )

    def apply_torques(self, rod_one, index_one, rod_two, index_two):
        # pass

        self._apply_torques(
            self.spring_force,
            self.rod_one_rd2,
            self.rod_two_rd2,
            index_one,
            index_two,
            rod_one.director_collection,
            rod_two.director_collection,
            rod_one.external_torques,
            rod_two.external_torques,
        )

    @staticmethod
    @njit(cache=True)
    def _apply_torques(
        spring_force,
        rod_one_rd2,
        rod_two_rd2,
        index_one,
        index_two,
        rod_one_director_collection,
        rod_two_director_collection,
        rod_one_external_torques,
        rod_two_external_torques,
    ):
        # Compute torques due to the connection forces
        torque_on_rod_one = _batch_cross(rod_one_rd2, spring_force)
        torque_on_rod_two = _batch_cross(rod_two_rd2, -spring_force)

        torque_on_rod_one_material_frame = _batch_matvec(
            rod_one_director_collection[:, :, index_one], torque_on_rod_one
        )
        torque_on_rod_two_material_frame = _batch_matvec(
            rod_two_director_collection[:, :, index_two], torque_on_rod_two
        )

        blocksize = index_one.shape[0]
        for k in range(blocksize):
            rod_one_external_torques[
                0, index_one[k]
            ] += torque_on_rod_one_material_frame[0, k]
            rod_one_external_torques[
                1, index_one[k]
            ] += torque_on_rod_one_material_frame[1, k]
            rod_one_external_torques[
                2, index_one[k]
            ] += torque_on_rod_one_material_frame[2, k]

            rod_two_external_torques[
                0, index_two[k]
            ] += torque_on_rod_two_material_frame[0, k]
            rod_two_external_torques[
                1, index_two[k]
            ] += torque_on_rod_two_material_frame[1, k]
            rod_two_external_torques[
                2, index_two[k]
            ] += torque_on_rod_two_material_frame[2, k]


class ParallelJointInterior(FreeJoint):
    """
    TODO: documentation
    """

    def __init__(
        self,
        k,
        nu,
        k_repulsive,
        **kwargs,
    ):
        super().__init__(np.array(k), np.array(nu))
        self.k_repulsive = np.array(k_repulsive)

    # Apply force is same as free joint
    def apply_forces(self, rod_one, index_one, rod_two, index_two):
        # TODO: documentation

        self._apply_forces(
            self.k,
            self.nu,
            self.k_repulsive,
            index_one,
            index_two,
            rod_one.director_collection,
            rod_two.director_collection,
            rod_one.position_collection,
            rod_two.position_collection,
            rod_one.radius,
            rod_two.radius,
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
        rod_one_director_collection,
        rod_two_director_collection,
        rod_one_position_collection,
        rod_two_position_collection,
        rod_one_radius,
        rod_two_radius,
        rod_one_dilatation,
        rod_two_dilatation,
        rod_one_velocity_collection,
        rod_two_velocity_collection,
        rod_one_external_forces,
        rod_two_external_forces,
    ):

        # Compute element positions
        rod_one_element_position = 0.5 * (
            rod_one_position_collection[:, index_one]
            + rod_one_position_collection[:, index_one + 1]
        )
        rod_two_element_position = 0.5 * (
            rod_two_position_collection[:, index_two]
            + rod_two_position_collection[:, index_two + 1]
        )

        # Compute spring force between two rods
        distance_vector = rod_two_element_position - rod_one_element_position

        spring_force = k * distance_vector

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

        distance = _batch_norm(distance_vector)

        normalized_distance_vector = np.zeros((relative_velocity.shape))

        idx_nonzero_distance = np.where(distance >= 1e-12)[0]

        normalized_distance_vector[..., idx_nonzero_distance] = (
            distance_vector[..., idx_nonzero_distance] / distance[idx_nonzero_distance]
        )

        normal_relative_velocity_vector = (
            _batch_dot(relative_velocity, normalized_distance_vector)
            * normalized_distance_vector
        )

        damping_force = -nu * normal_relative_velocity_vector

        # Compute the total force
        total_force = spring_force + damping_force

        # Re-distribute forces from elements to nodes.
        block_size = index_one.shape[0]
        for k in range(block_size):
            rod_one_external_forces[0, index_one[k]] += 0.5 * total_force[0, k]
            rod_one_external_forces[1, index_one[k]] += 0.5 * total_force[1, k]
            rod_one_external_forces[2, index_one[k]] += 0.5 * total_force[2, k]

            rod_one_external_forces[0, index_one[k] + 1] += 0.5 * total_force[0, k]
            rod_one_external_forces[1, index_one[k] + 1] += 0.5 * total_force[1, k]
            rod_one_external_forces[2, index_one[k] + 1] += 0.5 * total_force[2, k]

            rod_two_external_forces[0, index_two[k]] -= 0.5 * total_force[0, k]
            rod_two_external_forces[1, index_two[k]] -= 0.5 * total_force[1, k]
            rod_two_external_forces[2, index_two[k]] -= 0.5 * total_force[2, k]

            rod_two_external_forces[0, index_two[k] + 1] -= 0.5 * total_force[0, k]
            rod_two_external_forces[1, index_two[k] + 1] -= 0.5 * total_force[1, k]
            rod_two_external_forces[2, index_two[k] + 1] -= 0.5 * total_force[2, k]

    def apply_torques(self, rod_one, index_one, rod_two, index_two):
        # pass

        self._apply_torques(
            index_one,
            index_two,
            rod_one.director_collection,
            rod_two.director_collection,
            rod_one.external_torques,
            rod_two.external_torques,
        )

    @staticmethod
    @njit(cache=True)
    def _apply_torques(
        index_one,
        index_two,
        rod_one_director_collection,
        rod_two_director_collection,
        rod_one_external_torques,
        rod_two_external_torques,
    ):

        torque_on_rod_one_material_frame = 0  # _batch_matvec(rod_one_director_collection[:, :, index_one], torque_on_rod_one)
        torque_on_rod_two_material_frame = 0  # _batch_matvec(rod_two_director_collection[:, :, index_two], torque_on_rod_two)

        # blocksize = index_one.shape[0]
        # for k in range(blocksize):
        #     rod_one_external_torques[
        #         0, index_one[k]
        #     ] += torque_on_rod_one_material_frame[0, k]
        #     rod_one_external_torques[
        #         1, index_one[k]
        #     ] += torque_on_rod_one_material_frame[1, k]
        #     rod_one_external_torques[
        #         2, index_one[k]
        #     ] += torque_on_rod_one_material_frame[2, k]

        #     rod_two_external_torques[
        #         0, index_two[k]
        #     ] += torque_on_rod_two_material_frame[0, k]
        #     rod_two_external_torques[
        #         1, index_two[k]
        #     ] += torque_on_rod_two_material_frame[1, k]
        #     rod_two_external_torques[
        #         2, index_two[k]
        #     ] += torque_on_rod_two_material_frame[2, k]


# class SurfaceJointSideBySideNew(FreeJoint):
#     """
#     TODO: documentation
#     """
#     def __init__(
#         self,
#         regularization_threshold,
#         nu,
#         k_repulsive,
#         friction_coefficient,
#         velocity_damping_coefficient,
#         rod_one_direction_vec_in_material_frame,
#         rod_two_direction_vec_in_material_frame,
#         offset_btw_rods,
#         **kwargs,
#     ):
#         super().__init__(np.zeros_like(np.array(nu)), np.array(nu))
#         self.k_repulsive = np.array(k_repulsive)
#         self.offset_btw_rods = np.array(offset_btw_rods)
#         self.friction_coefficient = np.array(friction_coefficient)
#         self.velocity_damping_coefficient = np.array(velocity_damping_coefficient)
#         self.regularization_threshold =np.array(regularization_threshold)


#         self.rod_one_direction_vec_in_material_frame = np.array(
#             rod_one_direction_vec_in_material_frame
#         ).T
#         self.rod_two_direction_vec_in_material_frame = np.array(
#             rod_two_direction_vec_in_material_frame
#         ).T


#     # Apply force is same as free joint
#     def apply_forces(self, rod_one, index_one, rod_two, index_two):
#         # TODO: documentation


#         self._apply_forces(
#             self.regularization_threshold,
#             self.nu,
#             self.k_repulsive,
#             self.friction_coefficient,
#             self.velocity_damping_coefficient,
#             index_one,
#             index_two,
#             self.rod_one_direction_vec_in_material_frame,
#             self.rod_two_direction_vec_in_material_frame,
#             self.offset_btw_rods,
#             rod_one.director_collection,
#             rod_two.director_collection,
#             rod_one.position_collection,
#             rod_two.position_collection,
#             rod_one.radius,
#             rod_two.radius,
#             rod_one.dilatation,
#             rod_two.dilatation,
#             rod_one.velocity_collection,
#             rod_two.velocity_collection,
#             rod_one.external_forces,
#             rod_two.external_forces,
#         )

#     @staticmethod
#     @njit(cache=True)
#     def _apply_forces(
#         regularization_threshold,
#         nu,
#         k_repulsive,
#         friction_coefficient,
#         velocity_damping_coefficient,
#         index_one,
#         index_two,
#         rod_one_direction_vec_in_material_frame,
#         rod_two_direction_vec_in_material_frame,
#         rest_offset_btw_rods,
#         rod_one_director_collection,
#         rod_two_director_collection,
#         rod_one_position_collection,
#         rod_two_position_collection,
#         rod_one_radius,
#         rod_two_radius,
#         rod_one_dilatation,
#         rod_two_dilatation,
#         rod_one_velocity_collection,
#         rod_two_velocity_collection,
#         rod_one_external_forces,
#         rod_two_external_forces,
#     ):

#         rod_one_to_rod_two_connection_vec = _batch_matvec(
#             _batch_matrix_transpose(rod_one_director_collection[:, :, index_one]),
#             rod_one_direction_vec_in_material_frame,
#         )
#         rod_two_to_rod_one_connection_vec = _batch_matvec(
#             _batch_matrix_transpose(rod_two_director_collection[:, :, index_two]),
#             rod_two_direction_vec_in_material_frame,
#         )

#         # Compute element positions
#         rod_one_element_position = 0.5 * (
#             rod_one_position_collection[:, index_one]
#             + rod_one_position_collection[:, index_one + 1]
#         )
#         rod_two_element_position = 0.5 * (
#             rod_two_position_collection[:, index_two]
#             + rod_two_position_collection[:, index_two + 1]
#         )

#         # If there is an offset between rod one and rod two surface, then it should change as a function of dilatation.
#         offset_rod_one = (
#             0.5 * rest_offset_btw_rods / np.sqrt(rod_one_dilatation[index_one])
#         )
#         offset_rod_two = (
#             0.5 * rest_offset_btw_rods / np.sqrt(rod_two_dilatation[index_two])
#         )

#         # Compute contact forces. Contact forces are applied in the case one rod penetrates to the other, in that case
#         # we apply a repulsive force. Later on these repulsive forces are used to move rods apart from each other and
#         # as a pressure force.
#         # We assume contact forces are in plane.
#         # print('------------------___+')
#         center_distance = rod_two_element_position - rod_one_element_position
#         center_distance_unit_vec = center_distance / _batch_norm(center_distance)

#         penetration_strain = (
#             _batch_norm(center_distance)
#             - (
#                 rod_one_radius[index_one]
#                 + offset_rod_one
#                 + rod_two_radius[index_two]
#                 + offset_rod_two
#             )
#         )
#         np.round_(penetration_strain, 12, penetration_strain)
#         idx_penetrate_above_regularization_threshold = np.where(penetration_strain < -regularization_threshold)[0]
#         idx_penetrate_below_regularization_threshold = np.where(penetration_strain < 0)[0]
#         k_contact = np.zeros(index_one.shape[0])
#         k_contact_below_regularization_threshold = -(k_repulsive * penetration_strain**2)/(2*regularization_threshold)
#         k_contact_above_regularization_threshold = k_repulsive * (penetration_strain + (0.5*regularization_threshold))

#         k_contact[idx_penetrate_below_regularization_threshold] = k_contact_below_regularization_threshold[idx_penetrate_below_regularization_threshold]
#         k_contact[idx_penetrate_above_regularization_threshold] = k_contact_above_regularization_threshold[idx_penetrate_above_regularization_threshold]
#         contact_force = k_contact * center_distance_unit_vec


#         total_force = contact_force

#         # # tangential reaction force
#         # maximum_tangential_displacement

#         # tangential_displacement =

#         # # Compute Coulombic friction
#         # tangential_reaction_force = friction_coefficient * _batch_norm(contact_force)

#         # # Compare damping force in slip direction and kinetic friction and minimum is the friction force.
#         # # Compute friction force in the slip direction.

#         # friction_force = np.minimum(damping_force_in_slip_direction,coulombic_friction_force) * slip_interpenetration_velocity_unitized
#         # # Update contact force
#         # total_force += friction_force
#         # # print(np.linalg.norm(friction_force))

#         # # Add contact forces
#         # total_force += tangential_reaction_force


#         # Re-distribute forces from elements to nodes.
#         block_size = index_one.shape[0]
#         for k in range(block_size):
#             rod_one_external_forces[0, index_one[k]] += 0.5 * total_force[0, k]
#             rod_one_external_forces[1, index_one[k]] += 0.5 * total_force[1, k]
#             rod_one_external_forces[2, index_one[k]] += 0.5 * total_force[2, k]

#             rod_one_external_forces[0, index_one[k] + 1] += 0.5 * total_force[0, k]
#             rod_one_external_forces[1, index_one[k] + 1] += 0.5 * total_force[1, k]
#             rod_one_external_forces[2, index_one[k] + 1] += 0.5 * total_force[2, k]

#             rod_two_external_forces[0, index_two[k]] -= 0.5 * total_force[0, k]
#             rod_two_external_forces[1, index_two[k]] -= 0.5 * total_force[1, k]
#             rod_two_external_forces[2, index_two[k]] -= 0.5 * total_force[2, k]

#             rod_two_external_forces[0, index_two[k] + 1] -= 0.5 * total_force[0, k]
#             rod_two_external_forces[1, index_two[k] + 1] -= 0.5 * total_force[1, k]
#             rod_two_external_forces[2, index_two[k] + 1] -= 0.5 * total_force[2, k]


#     def apply_torques(self, rod_one, index_one, rod_two, index_two):
#         # pass

#         # self._apply_torques(
#         #     self.spring_force,
#         #     self.rod_one_rd2,
#         #     self.rod_two_rd2,
#         #     index_one,
#         #     index_two,
#         #     rod_one.director_collection,
#         #     rod_two.director_collection,
#         #     rod_one.external_torques,
#         #     rod_two.external_torques,
#         # )
#         pass

#     @staticmethod
#     @njit(cache=True)
#     def _apply_torques(
#         spring_force,
#         rod_one_rd2,
#         rod_two_rd2,
#         index_one,
#         index_two,
#         rod_one_director_collection,
#         rod_two_director_collection,
#         rod_one_external_torques,
#         rod_two_external_torques,
#     ):
#         # Compute torques due to the connection forces
#         torque_on_rod_one = _batch_cross(rod_one_rd2, spring_force)
#         torque_on_rod_two = _batch_cross(rod_two_rd2, -spring_force)

#         torque_on_rod_one_material_frame = _batch_matvec(
#             rod_one_director_collection[:, :, index_one], torque_on_rod_one
#         )
#         torque_on_rod_two_material_frame = _batch_matvec(
#             rod_two_director_collection[:, :, index_two], torque_on_rod_two
#         )

#         blocksize = index_one.shape[0]
#         for k in range(blocksize):
#             rod_one_external_torques[
#                 0, index_one[k]
#             ] += torque_on_rod_one_material_frame[0, k]
#             rod_one_external_torques[
#                 1, index_one[k]
#             ] += torque_on_rod_one_material_frame[1, k]
#             rod_one_external_torques[
#                 2, index_one[k]
#             ] += torque_on_rod_one_material_frame[2, k]

#             rod_two_external_torques[
#                 0, index_two[k]
#             ] += torque_on_rod_two_material_frame[0, k]
#             rod_two_external_torques[
#                 1, index_two[k]
#             ] += torque_on_rod_two_material_frame[1, k]
#             rod_two_external_torques[
#                 2, index_two[k]
#             ] += torque_on_rod_two_material_frame[2, k]


def get_connection_vector_straight_straight_rod_with_rest_matrix(
    rod_one,
    rod_two,
    rod_one_idx,
    rod_two_idx,
):
    rod_one_start_idx, rod_one_end_idx = rod_one_idx
    rod_two_start_idx, rod_two_end_idx = rod_two_idx

    # Compute rod element positions
    rod_one_element_position = 0.5 * (
        rod_one.position_collection[..., 1:] + rod_one.position_collection[..., :-1]
    )
    rod_one_element_position = rod_one_element_position[
        :, rod_one_start_idx:rod_one_end_idx
    ]
    rod_two_element_position = 0.5 * (
        rod_two.position_collection[..., 1:] + rod_two.position_collection[..., :-1]
    )
    rod_two_element_position = rod_two_element_position[
        :, rod_two_start_idx:rod_two_end_idx
    ]

    # Lets get the distance between rod elements
    distance_vector_rod_one_to_rod_two = (
        rod_two_element_position - rod_one_element_position
    )
    distance_vector_rod_one_to_rod_two_norm = _batch_norm(
        distance_vector_rod_one_to_rod_two
    )
    distance_vector_rod_one_to_rod_two /= distance_vector_rod_one_to_rod_two_norm

    distance_vector_rod_two_to_rod_one = -distance_vector_rod_one_to_rod_two

    rod_one_direction_vec_in_material_frame = _batch_matvec(
        rod_one.director_collection[:, :, rod_one_start_idx:rod_one_end_idx],
        distance_vector_rod_one_to_rod_two,
    )
    rod_two_direction_vec_in_material_frame = _batch_matvec(
        rod_two.director_collection[:, :, rod_two_start_idx:rod_two_end_idx],
        distance_vector_rod_two_to_rod_one,
    )

    offset_btw_rods = distance_vector_rod_one_to_rod_two_norm - (
        rod_one.radius[rod_one_start_idx:rod_one_end_idx]
        + rod_two.radius[rod_two_start_idx:rod_two_end_idx]
    )

    rest_rotation_matrix = _batch_matmul(
        rod_one.director_collection.copy(),
        _batch_matrix_transpose(rod_two.director_collection.copy()),
    )

    return (
        rod_one_direction_vec_in_material_frame,
        rod_two_direction_vec_in_material_frame,
        offset_btw_rods,
        rest_rotation_matrix,
    )


class SurfaceJointSideBySideTwo(FreeJoint):
    """
    TODO: documentation
    """

    def __init__(
        self,
        k,
        nu,
        k_repulsive,
        nut,
        kt,
        friction_coefficient,
        velocity_damping_coefficient,
        rod_one_direction_vec_in_material_frame,
        rod_two_direction_vec_in_material_frame,
        offset_btw_rods,
        rest_rotation_matrix,
        **kwargs,
    ):
        super().__init__(np.array(k), np.array(nu))
        self.k_repulsive = np.array(k_repulsive)
        self.offset_btw_rods = np.array(offset_btw_rods)
        self.friction_coefficient = np.array(friction_coefficient)
        self.velocity_damping_coefficient = np.array(velocity_damping_coefficient)
        self.kt = np.array(kt)
        self.nut = np.array(nut)
        self.rest_rotation_matrix = np.array(rest_rotation_matrix).T

        self.rod_one_direction_vec_in_material_frame = np.array(
            rod_one_direction_vec_in_material_frame
        ).T
        self.rod_two_direction_vec_in_material_frame = np.array(
            rod_two_direction_vec_in_material_frame
        ).T

    # Apply force is same as free joint

    # Apply force is same as free joint
    def apply_forces(self, rod_one, index_one, rod_two, index_two):
        # TODO: documentation

        (self.rod_one_rd2, self.rod_two_rd2, self.spring_force,) = self._apply_forces(
            self.k,
            self.nu,
            self.k_repulsive,
            self.friction_coefficient,
            self.velocity_damping_coefficient,
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
            rod_two.radius,
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
        friction_coefficient,
        velocity_damping_coefficient,
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
        rod_two_radius,
        rod_one_dilatation,
        rod_two_dilatation,
        rod_one_velocity_collection,
        rod_two_velocity_collection,
        rod_one_external_forces,
        rod_two_external_forces,
    ):

        rod_one_to_rod_two_connection_vec = _batch_matvec(
            _batch_matrix_transpose(rod_one_director_collection[:, :, index_one]),
            rod_one_direction_vec_in_material_frame,
        )
        rod_two_to_rod_one_connection_vec = _batch_matvec(
            _batch_matrix_transpose(rod_two_director_collection[:, :, index_two]),
            rod_two_direction_vec_in_material_frame,
        )

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
        offset_rod_two = (
            0.5 * rest_offset_btw_rods / np.sqrt(rod_two_dilatation[index_two])
        )

        # Compute vector r*d2 (radius * connection vector) for each rod and element
        rod_one_rd2 = rod_one_to_rod_two_connection_vec * (
            rod_one_radius[index_one] + offset_rod_one
        )
        rod_two_rd2 = rod_two_to_rod_one_connection_vec * (
            rod_two_radius[index_two] + offset_rod_two
        )

        # Compute connection points on the rod surfaces
        surface_position_rod_one = rod_one_element_position + rod_one_rd2
        surface_position_rod_two = rod_two_element_position + rod_two_rd2

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

        distance = _batch_norm(distance_vector)

        normalized_distance_vector = np.zeros((relative_velocity.shape))

        idx_nonzero_distance = np.where(distance >= 1e-12)[0]

        normalized_distance_vector[..., idx_nonzero_distance] = (
            distance_vector[..., idx_nonzero_distance] / distance[idx_nonzero_distance]
        )

        normal_relative_velocity_vector = (
            _batch_dot(relative_velocity, normalized_distance_vector)
            * normalized_distance_vector
        )

        damping_force = -nu * normal_relative_velocity_vector

        # Compute the total force
        total_force = spring_force + damping_force

        # Compute contact forces. Contact forces are applied in the case one rod penetrates to the other, in that case
        # we apply a repulsive force. Later on these repulsive forces are used to move rods apart from each other and
        # as a pressure force.
        # We assume contact forces are in plane.
        # print('------------------___+')
        center_distance = rod_two_element_position - rod_one_element_position
        center_distance_unit_vec = center_distance / _batch_norm(center_distance)
        #        penetration_strain = (
        #            _batch_norm(center_distance)
        #            / (
        #                rod_one_radius[index_one]
        #                + offset_rod_one
        #                + rod_two_radius[index_two]
        #                + offset_rod_two
        #            )
        #            - 1
        #        )
        penetration_strain = _batch_norm(center_distance) - (
            rod_one_radius[index_one]
            + offset_rod_one
            + rod_two_radius[index_two]
            + offset_rod_two
        )
        np.round_(penetration_strain, 12, penetration_strain)
        idx_penetrate = np.where(penetration_strain < 0)[0]
        k_contact = np.zeros(index_one.shape[0])
        k_contact_temp = -k_repulsive * np.abs(penetration_strain) ** (1.5)
        k_contact[idx_penetrate] += k_contact_temp[idx_penetrate]
        contact_force = k_contact * center_distance_unit_vec
        # contact_force[:,idx_penetrate] = 0.0

        # Add contact forces
        total_force += contact_force

        # Compute the spring forces in plane. If there is contact spring force is also contributing to contact force
        # so we need to compute it and add to contact_force.
        spring_force_temp_for_contact = np.zeros((3, index_one.shape[0]))
        spring_force_temp_for_contact[:, idx_penetrate] += spring_force[
            :, idx_penetrate
        ]

        contact_force += spring_force_temp_for_contact

        # Friction

        # Compute friction
        slip_interpenetration_velocity = (
            relative_velocity - normal_relative_velocity_vector
        )
        slip_interpenetration_velocity_mag = _batch_norm(slip_interpenetration_velocity)
        slip_interpenetration_velocity_unitized = slip_interpenetration_velocity / (
            slip_interpenetration_velocity_mag + 1e-14
        )

        # Compute Coulombic friction
        coulombic_friction_force = friction_coefficient * _batch_norm(contact_force)

        # Compare damping force in slip direction and kinetic friction and minimum is the friction force.
        # Compute friction force in the slip direction.
        damping_force_in_slip_direction = (
            velocity_damping_coefficient * slip_interpenetration_velocity_mag
        )
        friction_force = (
            np.minimum(damping_force_in_slip_direction, coulombic_friction_force)
            * slip_interpenetration_velocity_unitized
        )
        # Update contact force
        total_force += friction_force
        # print(np.linalg.norm(friction_force))

        # Re-distribute forces from elements to nodes.
        block_size = index_one.shape[0]
        for k in range(block_size):
            rod_one_external_forces[0, index_one[k]] += 0.5 * total_force[0, k]
            rod_one_external_forces[1, index_one[k]] += 0.5 * total_force[1, k]
            rod_one_external_forces[2, index_one[k]] += 0.5 * total_force[2, k]

            rod_one_external_forces[0, index_one[k] + 1] += 0.5 * total_force[0, k]
            rod_one_external_forces[1, index_one[k] + 1] += 0.5 * total_force[1, k]
            rod_one_external_forces[2, index_one[k] + 1] += 0.5 * total_force[2, k]

            rod_two_external_forces[0, index_two[k]] -= 0.5 * total_force[0, k]
            rod_two_external_forces[1, index_two[k]] -= 0.5 * total_force[1, k]
            rod_two_external_forces[2, index_two[k]] -= 0.5 * total_force[2, k]

            rod_two_external_forces[0, index_two[k] + 1] -= 0.5 * total_force[0, k]
            rod_two_external_forces[1, index_two[k] + 1] -= 0.5 * total_force[1, k]
            rod_two_external_forces[2, index_two[k] + 1] -= 0.5 * total_force[2, k]

        return (
            rod_one_rd2,
            rod_two_rd2,
            spring_force,
        )

    def apply_torques(self, rod_one, index_one, rod_two, index_two):
        # pass

        self._apply_torques(
            self.spring_force,
            self.kt,
            self.nut,
            self.rest_rotation_matrix,
            self.rod_one_rd2,
            self.rod_two_rd2,
            index_one,
            index_two,
            rod_one.director_collection,
            rod_two.director_collection,
            rod_one.external_torques,
            rod_two.external_torques,
        )

    @staticmethod
    @njit(cache=True)
    def _apply_torques(
        spring_force,
        kt,
        nut,
        rest_rotation_matrix,
        rod_one_rd2,
        rod_two_rd2,
        index_one,
        index_two,
        rod_one_director_collection,
        rod_two_director_collection,
        rod_one_external_torques,
        rod_two_external_torques,
    ):
        # Compute torques due to the connection forces
        torque_on_rod_one = _batch_cross(rod_one_rd2, spring_force)
        torque_on_rod_two = _batch_cross(rod_two_rd2, -spring_force)

        # rel_rot: C_12 = C_1I @ C_I2
        # C_12 is relative rotation matrix from system 1 to system 2
        # C_1I is the rotation from system 1 to the inertial frame (i.e. the world frame)
        # C_I2 is the rotation from the inertial frame to system 2 frame (inverse of system_two_director)
        # rel_rot = rod_one_director_collection @ rod_two_director_collection.T
        rel_rot = _batch_matmul(
            rod_one_director_collection[:, :, index_one],
            _batch_matrix_transpose(rod_two_director_collection[:, :, index_two]),
        )
        # error_rot: C_22* = C_21 @ C_12*
        # C_22* is rotation matrix from current orientation of system 2 to desired orientation of system 2
        # C_21 is the inverse of C_12, which describes the relative (current) rotation from system 1 to system 2
        # C_12* is the desired rotation between systems one and two, which is saved in the static_rotation attribute
        dev_rot = _batch_matmul(_batch_matrix_transpose(rel_rot), rest_rotation_matrix)

        # compute rotation vectors based on C_22*
        # scipy implementation
        # rot_vec = Rotation.from_matrix(dev_rot).as_rotvec()
        #
        # implementation using custom _inv_rotate compiled with numba
        # rotation vector between identity matrix and C_22*
        n = dev_rot.shape[-1]
        rot_vec = np.zeros((3, n))
        mat = np.zeros((3, 3, 2))
        mat[:, :, 0] = np.eye(3)
        for i in range(n):
            mat[:, :, 1] = dev_rot[:, :, i].T
            # a = np.dstack([np.eye(3), dev_rot[:,:,i].T])
            b = _inv_rotate(mat)
            rot_vec[:, i] = b.reshape((3,))

        # rotate rotation vector into inertial frame
        rot_vec_inertial_frame = _batch_matvec(
            _batch_matrix_transpose(rod_two_director_collection[:, :, index_two]),
            rot_vec,
        )

        # # deviation in rotation velocity between system 1 and system 2
        # # first convert to inertial frame, then take differences
        # dev_omega = (
        #     rod_two_director_collection.T @ system_two.omega_collection[..., index_two]
        #     - rod_one_director_collection.T @ system_one.omega_collection[..., index_one]
        # )

        # we compute the constraining torque using a rotational spring - damper system in the inertial frame
        torsional_spring = kt * (rot_vec_inertial_frame) ** 3  # - self.nut * dev_omega

        torque_on_rod_one_material_frame = _batch_matvec(
            rod_one_director_collection[:, :, index_one], torque_on_rod_one
        ) - _batch_matvec(
            rod_one_director_collection[:, :, index_one], torsional_spring
        )
        torque_on_rod_two_material_frame = _batch_matvec(
            rod_two_director_collection[:, :, index_two], torque_on_rod_two
        ) + _batch_matvec(
            rod_two_director_collection[:, :, index_two], torsional_spring
        )

        blocksize = index_one.shape[0]
        for k in range(blocksize):
            rod_one_external_torques[
                0, index_one[k]
            ] += torque_on_rod_one_material_frame[0, k]
            rod_one_external_torques[
                1, index_one[k]
            ] += torque_on_rod_one_material_frame[1, k]
            rod_one_external_torques[
                2, index_one[k]
            ] += torque_on_rod_one_material_frame[2, k]

            rod_two_external_torques[
                0, index_two[k]
            ] += torque_on_rod_two_material_frame[0, k]
            rod_two_external_torques[
                1, index_two[k]
            ] += torque_on_rod_two_material_frame[1, k]
            rod_two_external_torques[
                2, index_two[k]
            ] += torque_on_rod_two_material_frame[2, k]
