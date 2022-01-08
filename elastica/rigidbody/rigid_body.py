__doc__ = """ Rigid body abstract base class """

import numpy as np
from abc import ABC, abstractmethod
from elastica._linalg import _batch_matvec, _batch_cross
from elastica.utils import MaxDimension


class RigidBodyBase(ABC):
    def __init__(self):

        self.position_collection = NotImplementedError
        self.velocity_collection = NotImplementedError
        self.acceleration_collection = NotImplementedError
        self.omega_collection = NotImplementedError
        self.alpha_collection = NotImplementedError
        self.director_collection = NotImplementedError

        self.external_forces = NotImplementedError
        self.external_torques = NotImplementedError

        self.mass = NotImplementedError

        self.mass_second_moment_of_inertia = NotImplementedError
        self.inv_mass_second_moment_of_inertia = NotImplementedError

    # @abstractmethod
    #     # def update_accelerations(self):
    #     #     pass

    # def _compute_internal_forces_and_torques(self):
    #     """
    #     This function here is only for integrator to work properly. We do not need
    #     internal forces and torques at all.
    #     Parameters
    #     ----------
    #     time
    #
    #     Returns
    #     -------
    #
    #     """
    #     pass

    def update_accelerations(self, time):
        """TODO Do we need to make the collection members abstract?

        Parameters
        ----------
        time

        Returns
        -------

        """
        np.copyto(
            self.acceleration_collection,
            (self.external_forces) / self.mass,
        )

        # I apply common sub expression elimination here, as J w
        J_omega = _batch_matvec(
            self.mass_second_moment_of_inertia, self.omega_collection
        )

        # (J \omega_L ) x \omega_L
        lagrangian_transport = _batch_cross(J_omega, self.omega_collection)

        np.copyto(
            self.alpha_collection,
            _batch_matvec(
                self.inv_mass_second_moment_of_inertia,
                (lagrangian_transport + self.external_torques),
            ),
        )

    def zeroed_out_external_forces_and_torques(self, time):
        # Reset forces and torques
        self.external_forces *= 0.0
        self.external_torques *= 0.0

    def compute_position_center_of_mass(self):
        return self.position_collection[..., 0].copy()

    def compute_translational_energy(self):
        return (
            0.5
            * self.mass
            * np.dot(
                self.velocity_collection[..., -1], self.velocity_collection[..., -1]
            )
        )

    def compute_rotational_energy(self):
        J_omega = np.einsum(
            "ijk,jk->ik", self.mass_second_moment_of_inertia, self.omega_collection
        )
        return 0.5 * np.einsum("ik,ik->k", self.omega_collection, J_omega).sum()
