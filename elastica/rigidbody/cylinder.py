__doc__ = """
Implementation of a rigid body cylinder.
"""

import numpy as np
from numpy.typing import NDArray

from elastica._linalg import _batch_cross
from elastica.utils import MaxDimension
from elastica.rigidbody.rigid_body import RigidBodyBase


class Cylinder(RigidBodyBase):
    def __init__(
        self,
        start: NDArray[np.floating],
        direction: NDArray[np.floating],
        normal: NDArray[np.floating],
        base_length: float,
        base_radius: float,
        density: float,
    ) -> None:
        """
        Rigid body cylinder initializer.

        Parameters
        ----------
        start : NDArray[np.floating]
        direction : NDArray[np.floating]
        normal : NDArray[np.floating]
        base_length : float
        base_radius : float
        density : float
        """

        def _check_array_size(
            to_check: NDArray[np.floating], name: str, expected: int = 3
        ) -> None:
            array_size = to_check.size
            assert array_size == expected, (
                f"Invalid size of '{name}'. "
                f"Expected: {expected}, but got: {array_size}"
            )

        def _check_lower_bound(
            to_check: np.floating, name: str, lower_bound: np.floating = np.float64(0.0)
        ) -> None:
            assert (
                to_check > lower_bound
            ), f"Value for '{name}' ({to_check}) must be at lease {lower_bound}. "

        _check_array_size(start, "start")
        _check_array_size(direction, "direction")
        _check_array_size(normal, "normal")

        _check_lower_bound(base_length, "base_length")
        _check_lower_bound(base_radius, "base_radius")
        _check_lower_bound(density, "density")

        super().__init__()

        # rigid body does not have elements it only has one node. We are setting n_elems to
        # zero for only make code to work. _bootstrap_from_data requires n_elems to be define
        self.n_elem: int = 1

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

        self.inv_mass_second_moment_of_inertia = np.linalg.inv(
            mass_second_moment_of_inertia
        ).reshape((dim, dim, 1))

        # position is at the center
        self.position_collection[:] = (
            start.reshape(3, 1) + direction.reshape(3, 1) * base_length / 2
        )

        self.director_collection[0, ...] = normal
        self.director_collection[1, ...] = binormal
        self.director_collection[2, ...] = tangents
