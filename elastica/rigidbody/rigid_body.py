__doc__ = """"""

from typing import Type

from abc import ABC

import numpy as np
from numpy.typing import NDArray
from elastica._linalg import _batch_matvec, _batch_cross


class RigidBodyBase(ABC):
    """
    Base class for rigid body classes.

    Notes
    -----
    All rigid body class should inherit this base class.

    """

    REQUISITE_MODULES: list[Type] = []

    def __init__(self) -> None:

        self.position_collection: NDArray[np.floating]
        self.velocity_collection: NDArray[np.floating]
        self.acceleration_collection: NDArray[np.floating]
        self.omega_collection: NDArray[np.floating]
        self.alpha_collection: NDArray[np.floating]
        self.director_collection: NDArray[np.floating]

        self.external_forces: NDArray[np.floating]
        self.external_torques: NDArray[np.floating]

        self.internal_forces: NDArray[np.floating]
        self.internal_torques: NDArray[np.floating]

        self.mass: np.floating
        self.volume: np.floating
        self.length: np.floating
        self.tangents: NDArray[np.floating]
        self.radius: np.floating

        self.mass_second_moment_of_inertia: NDArray[np.floating]
        self.inv_mass_second_moment_of_inertia: NDArray[np.floating]

    def update_accelerations(self, time: np.floating) -> None:
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

    def zeroed_out_external_forces_and_torques(self, time: np.floating) -> None:
        # Reset forces and torques
        self.external_forces *= 0.0
        self.external_torques *= 0.0

    def compute_internal_forces_and_torques(self, time: np.floating) -> None:
        """
        For rigid body, there is no internal forces and torques
        """
        pass

    def compute_position_center_of_mass(self) -> NDArray[np.floating]:
        """
        Return positional center of mass
        """
        return self.position_collection[..., 0].copy()

    def compute_translational_energy(self) -> NDArray[np.floating]:
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

    def compute_rotational_energy(self) -> NDArray[np.floating]:
        """
        Return rotational energy
        """
        j_omega = np.einsum(
            "ijk,jk->ik", self.mass_second_moment_of_inertia, self.omega_collection
        )
        return 0.5 * np.einsum("ik,ik->k", self.omega_collection, j_omega).sum()
