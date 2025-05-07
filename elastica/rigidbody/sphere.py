__doc__ = """
Implementation of a sphere rigid body.
"""
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from elastica._linalg import _batch_cross
from elastica.utils import MaxDimension
from elastica.rigidbody.rigid_body import RigidBodyBase


class Sphere(RigidBodyBase):
    def __init__(
        self,
        center: NDArray[np.float64],
        base_radius: float,
        density: float,
    ) -> None:
        """
        Rigid body sphere initializer.

        Parameters
        ----------
        center : NDArray[np.float64]
        base_radius : float
        density : float
        """

        super().__init__()

        dim: int = MaxDimension.value()

        assert (
            center.size == dim
        ), f"center must be of size {dim}, but was {center.size}"
        assert base_radius > 0.0, "base_radius must be positive"
        assert density > 0.0, "density must be positive"

        self.radius = np.float64(base_radius)
        self.density = np.float64(density)
        self.length = np.float64(2 * base_radius)
        # This is for a rigid body cylinder
        self.volume = np.float64(4.0 / 3.0 * np.pi * base_radius**3)
        self.mass = np.float64(self.volume * self.density)
        normal = np.array([1.0, 0.0, 0.0], dtype=np.float64).reshape(dim, 1)
        tangents = np.array([0.0, 0.0, 1.0], dtype=np.float64).reshape(dim, 1)
        binormal = _batch_cross(tangents, normal)

        # Mass second moment of inertia for disk cross-section
        mass_second_moment_of_inertia = np.zeros((dim, dim), dtype=np.float64)
        np.fill_diagonal(
            mass_second_moment_of_inertia, 2.0 / 5.0 * self.mass * self.radius**2
        )

        self.mass_second_moment_of_inertia = mass_second_moment_of_inertia.reshape(
            (dim, dim, 1)
        )

        self.inv_mass_second_moment_of_inertia = (
            np.linalg.inv(mass_second_moment_of_inertia)
            .reshape((dim, dim, 1))
            .astype(np.float64)
        )

        # Allocate properties
        self.position_collection = np.zeros((dim, 1), dtype=np.float64)
        self.velocity_collection = np.zeros((dim, 1), dtype=np.float64)
        self.acceleration_collection = np.zeros((dim, 1), dtype=np.float64)
        self.omega_collection = np.zeros((dim, 1), dtype=np.float64)
        self.alpha_collection = np.zeros((dim, 1), dtype=np.float64)
        self.director_collection = np.zeros((dim, dim, 1), dtype=np.float64)

        self.external_forces = np.zeros((dim, 1), dtype=np.float64)
        self.external_torques = np.zeros((dim, 1), dtype=np.float64)

        # position is at the center
        self.position_collection[:, 0] = center

        self.director_collection[0, ...] = normal
        self.director_collection[1, ...] = binormal
        self.director_collection[2, ...] = tangents


if TYPE_CHECKING:
    from .protocol import RigidBodyProtocol

    _: RigidBodyProtocol = Sphere(
        center=np.zeros(3),
        base_radius=1.0,
        density=1.0,
    )
