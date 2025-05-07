__doc__ = """
Implementation of a rigid body cylinder.
"""
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from elastica._linalg import _batch_cross
from elastica.utils import MaxDimension
from elastica.rigidbody.rigid_body import RigidBodyBase


class Cylinder(RigidBodyBase):
    def __init__(
        self,
        start: NDArray[np.float64],
        direction: NDArray[np.float64],
        normal: NDArray[np.float64],
        base_length: float,
        base_radius: float,
        density: float,
    ) -> None:
        """
        Rigid body cylinder initializer.

        Parameters
        ----------
        start : NDArray[np.float64]
        direction : NDArray[np.float64]
        normal : NDArray[np.float64]
        base_length : float
        base_radius : float
        density : float
        """

        # FIXME: Refactor
        def assert_check_array_size(
            to_check: NDArray[np.float64], name: str, expected: int = 3
        ) -> None:
            array_size = to_check.size
            assert array_size == expected, (
                f"Invalid size of '{name}'. "
                f"Expected: {expected}, but got: {array_size}"
            )

        # FIXME: Refactor
        def assert_check_lower_bound(
            to_check: float, name: str, lower_bound: float = 0.0
        ) -> None:
            assert (
                to_check > lower_bound
            ), f"Value for '{name}' ({to_check}) must be at lease {lower_bound}. "

        assert_check_array_size(start, "start")
        assert_check_array_size(direction, "direction")
        assert_check_array_size(normal, "normal")

        assert_check_lower_bound(base_length, "base_length")
        assert_check_lower_bound(base_radius, "base_radius")
        assert_check_lower_bound(density, "density")

        super().__init__()

        normal = normal.reshape((3, 1))
        tangents = direction.reshape((3, 1))
        binormal = _batch_cross(tangents, normal)
        self.radius = np.float64(base_radius)
        self.length = np.float64(base_length)
        self.density = np.float64(density)

        dim: int = MaxDimension.value()

        # This is for a rigid body cylinder
        self.volume = np.float64(np.pi * base_radius * base_radius * base_length)
        self.mass = np.float64(self.volume * self.density)

        # Second moment of inertia
        area = np.pi * base_radius * base_radius
        smoa_span_1 = area * area / (4.0 * np.pi)
        smoa_span_2 = smoa_span_1
        smoa_axial = 2.0 * smoa_span_1
        smoa = np.array([smoa_span_1, smoa_span_2, smoa_axial])

        # Allocate properties
        self.position_collection = np.zeros((dim, 1), dtype=np.float64)
        self.velocity_collection = np.zeros((dim, 1), dtype=np.float64)
        self.acceleration_collection = np.zeros((dim, 1), dtype=np.float64)
        self.omega_collection = np.zeros((dim, 1), dtype=np.float64)
        self.alpha_collection = np.zeros((dim, 1), dtype=np.float64)
        self.director_collection = np.zeros((dim, dim, 1), dtype=np.float64)

        self.external_forces = np.zeros((dim, 1), dtype=np.float64)
        self.external_torques = np.zeros((dim, 1), dtype=np.float64)

        # Mass second moment of inertia for disk cross-section
        mass_second_moment_of_inertia = np.diag(smoa * density * base_length)

        self.mass_second_moment_of_inertia = mass_second_moment_of_inertia.reshape(
            (dim, dim, 1)
        )

        self.inv_mass_second_moment_of_inertia = (
            np.linalg.inv(mass_second_moment_of_inertia)
            .reshape((dim, dim, 1))
            .astype(np.float64)
        )

        # position is at the center
        self.position_collection[:] = (
            start.reshape(3, 1) + direction.reshape(3, 1) * base_length / 2
        )

        self.director_collection[0, ...] = normal
        self.director_collection[1, ...] = binormal
        self.director_collection[2, ...] = tangents


if TYPE_CHECKING:
    from .protocol import RigidBodyProtocol

    _: RigidBodyProtocol = Cylinder(
        start=np.zeros(3),
        direction=np.ones(3),
        normal=np.ones(3),
        base_length=1.0,
        base_radius=1.0,
        density=1.0,
    )
