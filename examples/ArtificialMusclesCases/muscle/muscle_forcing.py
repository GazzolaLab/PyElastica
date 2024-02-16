import numpy as np
from elastica import *
from elastica._rotations import _inv_rotate, _skew_symmetrize


class ElementwiseForcesAndTorques(NoForces):
    """ """

    def __init__(self, torques, forces):
        super(ElementwiseForcesAndTorques, self).__init__()
        self.torques = torques
        self.forces = forces

    def apply_forces(self, system, time: np.float64 = 0.0):
        system.external_torques -= self.torques
        system.external_forces -= self.forces


class PointSpring(NoForces):
    """ """

    def __init__(self, k, nu, point, index, *args, **kwargs):
        super(PointSpring, self).__init__()
        self.point = point
        self.k = k
        self.index = index
        self.nu = nu

    def apply_forces(self, system, time: np.float64 = 0.0):
        elastic_force = self.k * (
            self.point - system.position_collection[..., self.index]
        )
        damping_force = -self.nu * (system.velocity_collection[..., self.index])
        system.external_forces[..., self.index] += elastic_force + damping_force


class MeshRigidBodyPointSpring(NoForces):
    def __init__(
        self,
        k,
        nu,
        distance_to_point_from_center,
        direction_to_point_from_center,
        point,
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
            distance of point spring connection on the rigid body with respect to rigid body center.
        direction_to_point_from_center : numpy array (3,)
            direction of point spring connection on the rigid body with respect to rigid body center.
        point : numpy array (3,)
            position of the spring connection point with respect to the world frame



        """
        try:
            self.block_size = len(k)
        except:
            self.block_size = 1
        if self.block_size == 1:
            self.k = [k]
            self.nu = [nu]
            self.distance_to_point_from_center = [distance_to_point_from_center]
            self.direction_to_point_from_center = [direction_to_point_from_center]
            self.point = [point]
        else:
            self.k = k
            self.nu = nu
            self.distance_to_point_from_center = distance_to_point_from_center
            self.direction_to_point_from_center = direction_to_point_from_center
            self.point = point

    def apply_forces(self, system, time: np.float64 = 0.0):
        """
        Apply joint force to the connected rod objects.

        Parameters
        ----------
        system : object
            Mesh rigid-body object
        Returns
        -------

        """

        for block in range(self.block_size):
            current_point_position = np.zeros((3,))
            current_point_velocity = np.zeros((3,))
            system_omega_collection_skew = _skew_symmetrize(system.omega_collection)
            for i in range(3):
                current_point_position[i] += system.position_collection[i, 0]
                current_point_velocity[i] += system.velocity_collection[i, 0]
                for j in range(3):
                    current_point_position[i] += (
                        self.distance_to_point_from_center[block]
                        * system.director_collection[i, j, 0]
                        * ((self.direction_to_point_from_center[block])[j])
                    )  # rp = rcom + dQN
                    current_point_velocity[i] += (
                        self.distance_to_point_from_center[block]
                        * system_omega_collection_skew[i, j, 0]
                        * ((self.direction_to_point_from_center[block])[j])
                    )  # vp = vcom + d(wxN)

            end_distance_vector = self.point[block] - current_point_position
            elastic_force = self.k[block] * end_distance_vector
            damping_force = self.nu[block] * current_point_velocity

            contact_force = elastic_force + damping_force
            system.external_forces[..., 0] += contact_force
            system.external_torques[..., 0] -= self.distance_to_point_from_center[
                block
            ] * np.cross(
                system.director_collection[..., 0]
                @ self.direction_to_point_from_center[block],
                contact_force,
            )

        return

    def apply_torques(self, system, time: np.float64 = 0.0):
        """
        Apply restoring joint torques to the connected rod objects.

        In FreeJoint class, this routine simply passes.

        Parameters
        ----------
        system : object
            Rod or rigid-body object

        Returns
        -------

        """
        pass
