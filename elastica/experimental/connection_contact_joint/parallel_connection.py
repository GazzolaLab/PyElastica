__doc__ = """Contains SurfaceJointSideBySide class which connects two parallel rods ."""
import numpy as np
from numba import njit
from elastica.joint import FreeJoint

# Join the two rods
from elastica._linalg import (
    _batch_norm,
    _batch_matvec,
    _batch_cross,
    _batch_matrix_transpose,
)


def get_connection_vector_straight_straight_rod(
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
        )
        self.rod_two_direction_vec_in_material_frame = np.array(
            rod_two_direction_vec_in_material_frame
        )

    # Apply force is same as free joint
    def apply_forces(self, system_one, index_one, system_two, index_two):
        # TODO: documentation

        (
            self.rod_one_rd2,
            self.rod_two_rd2,
            self.spring_force,
        ) = self._apply_forces(
            self.k,
            self.nu,
            self.k_repulsive,
            index_one,
            index_two,
            self.rod_one_direction_vec_in_material_frame,
            self.rod_two_direction_vec_in_material_frame,
            self.offset_btw_rods,
            system_one.director_collection,
            system_two.director_collection,
            system_one.position_collection,
            system_two.position_collection,
            system_one.radius,
            system_two.radius,
            system_one.dilatation,
            system_two.dilatation,
            system_one.velocity_collection,
            system_two.velocity_collection,
            system_one.external_forces,
            system_two.external_forces,
        )

    @staticmethod
    @njit(cache=True)  # type: ignore
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
        damping_force = nu * relative_velocity

        # Compute the total force
        total_force = spring_force + damping_force

        # Compute contact forces. Contact forces are applied in the case one rod penetrates to the other, in that case
        # we apply a repulsive force.
        center_distance = rod_two_element_position - rod_one_element_position
        center_distance_unit_vec = center_distance / np.linalg.norm(center_distance)
        penetration_strain = np.linalg.norm(center_distance) - (
            rod_one_radius[index_one]
            + offset_rod_one
            + rod_two_radius[index_two]
            + offset_rod_two
        )

        np.round_(penetration_strain, 12, penetration_strain)
        # Contact present only if rods penetrate to each other
        idx_penetrate = np.where(penetration_strain < 0)[0]
        k_contact = np.zeros(index_one.shape[0])
        k_contact_temp = -k_repulsive * np.abs(penetration_strain) ** (1.5)
        k_contact[idx_penetrate] += k_contact_temp[idx_penetrate]
        contact_force = k_contact * center_distance_unit_vec

        # Add contact forces
        total_force += contact_force

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

    def apply_torques(self, system_one, index_one, system_two, index_two):
        # pass

        self._apply_torques(
            self.spring_force,
            self.rod_one_rd2,
            self.rod_two_rd2,
            index_one,
            index_two,
            system_one.director_collection,
            system_two.director_collection,
            system_one.external_torques,
            system_two.external_torques,
        )

    @staticmethod
    @njit(cache=True)  # type: ignore
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
