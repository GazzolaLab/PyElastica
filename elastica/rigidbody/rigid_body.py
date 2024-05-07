__doc__ = """"""

import numpy as np
from abc import ABC
from elastica._linalg import _batch_matvec, _batch_cross
from ._typing import f_arr_t, float_t

from typing import Any


class RigidBodyBase(ABC):
    """
    Base class for rigid body classes.

    Notes
    -----
    All rigid body class should inherit this base class.

    """

    REQUISITE_MODULES = []

    def __init__(self) -> None:

        self.position_collection: f_arr_t
        self.velocity_collection: f_arr_t
        self.acceleration_collection: f_arr_t
        self.omega_collection: f_arr_t
        self.alpha_collection: f_arr_t
        self.director_collection: f_arr_t

        self.external_forces: f_arr_t
        self.external_torques: f_arr_t

        self.mass: f_arr_t

        self.mass_second_moment_of_inertia: f_arr_t
        self.inv_mass_second_moment_of_inertia: f_arr_t

    def update_accelerations(self, time: float_t) -> None:
        np.copyto(
            self.acceleration_collection,
            (self.external_forces) / self.mass,
        )

        # I apply common sub expression elimination here, as J w
        j_omega = _batch_matvec(
            self.mass_second_moment_of_inertia, self.omega_collection
        )

        # (J \omega_L ) x \omega_L
        lagrangian_transport = _batch_cross(j_omega, self.omega_collection)

        np.copyto(
            self.alpha_collection,
            _batch_matvec(
                self.inv_mass_second_moment_of_inertia,
                (lagrangian_transport + self.external_torques),
            ),
        )

    def zeroed_out_external_forces_and_torques(self, time: float_t) -> None:
        # Reset forces and torques
        self.external_forces *= 0.0
        self.external_torques *= 0.0

    def compute_position_center_of_mass(self) -> f_arr_t:
        """
        Return positional center of mass
        """
        return self.position_collection[..., 0].copy()

    def compute_translational_energy(self) -> Any:
        """
        Return translational energy
        """
        return (
            0.5
            * self.mass
            * np.dot(
                self.velocity_collection[..., -1], self.velocity_collection[..., -1]
            )
        )

    def compute_rotational_energy(self) -> Any:
        """
        Return rotational energy
        """
        j_omega = np.einsum(
            "ijk,jk->ik", self.mass_second_moment_of_inertia, self.omega_collection
        )
        return 0.5 * np.einsum("ik,ik->k", self.omega_collection, j_omega).sum()
