__doc__ = """ Module containing joint classes to connect multiple rods together. """

from elastica._linalg import _batch_product_k_ik_to_ik
from elastica._rotations import _inv_rotate, _skew_symmetrize
from elastica.typing import SystemType, RodType
from elastica.joint import FreeJoint
from math import sqrt
import numba
import numpy as np


class MeshRigidBodyRodJoint(FreeJoint):
    def __init__(
        self,
        k,
        nu,
        distance_to_point_from_center,
        direction_to_point_from_center,
        **kwargs,
    ):
        """

        Parameters
        ----------
        k: float
           Stiffness coefficient of the joint.
        nu: float
           Damping coefficient of the joint.
        distance_to_point_from_center : float
            distance of joint point to rigid body center.
        direction_to_point_from_center : numpy array (3,)
            direction of joint point with respect to rigid body center.


        """
        self.k = k
        self.nu = nu
        self.distance_to_point_from_center = distance_to_point_from_center
        self.direction_to_point_from_center = direction_to_point_from_center
        self.block_size = len(direction_to_point_from_center)

    def apply_forces(
        self, system_one: SystemType, index_one, system_two: SystemType, index_two
    ):
        """
        Apply joint force to the connected rod objects.

        Parameters
        ----------
        system_one : object
            Rod
        index_one : int
            Index of first rod for joint.
        system_two : object
            Mesh rigid-body object

        Returns
        -------

        """

        for block in range(self.block_size):
            current_point_position = np.zeros((3,))
            current_point_velocity = np.zeros((3,))
            system_two_omega_collection_skew = _skew_symmetrize(
                system_two.omega_collection
            )
            for i in range(3):
                current_point_position[i] += system_two.position_collection[i, 0]
                current_point_velocity[i] += system_two.velocity_collection[i, 0]
                for j in range(3):
                    current_point_position[i] += (
                        self.distance_to_point_from_center[block]
                        * system_two.director_collection[i, j, 0]
                        * ((self.direction_to_point_from_center[block])[j])
                    )  # rp = rcom + dQN
                    current_point_velocity[i] += (
                        self.distance_to_point_from_center[block]
                        * system_two_omega_collection_skew[i, j, 0]
                        * ((self.direction_to_point_from_center[block])[j])
                    )  # vp = vcom + d(wxN)

            end_distance_vector = (
                current_point_position
                - system_one.position_collection[..., index_one[block]]
            )
            elastic_force = self.k[block] * end_distance_vector

            relative_velocity = (
                current_point_velocity
                - system_one.velocity_collection[..., index_one[block]]
            )
            damping_force = self.nu[block] * relative_velocity

            contact_force = elastic_force + damping_force
            system_one.external_forces[..., index_one[block]] += contact_force
            system_two.external_forces[..., 0] -= contact_force
            system_two.external_torques[..., 0] += self.distance_to_point_from_center[
                block
            ] * np.cross(
                system_two.director_collection[..., 0]
                @ self.direction_to_point_from_center[block],
                contact_force,
            )

        return

    def apply_torques(
        self, system_one: SystemType, index_one, system_two: SystemType, index_two
    ):
        """
        Apply restoring joint torques to the connected rod objects.

        In FreeJoint class, this routine simply passes.

        Parameters
        ----------
        system_one : object
            Rod or rigid-body object
        index_one : int
            Index of first rod for joint.
        system_two : object
            Rod or rigid-body object
        index_two : int
            Index of second rod for joint.

        Returns
        -------

        """
        pass
