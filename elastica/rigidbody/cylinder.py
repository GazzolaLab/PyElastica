__doc__ = """"""

import numpy as np

from elastica._linalg import _batch_cross
from elastica.utils import MaxDimension
from elastica.rigidbody.rigid_body import RigidBodyBase

from ._typing import f_arr_t, float_t, int_t


class Cylinder(RigidBodyBase):
    def __init__(
        self,
        start: f_arr_t,
        direction: f_arr_t,
        normal: f_arr_t,
        base_length: float_t,
        base_radius: float_t,
        density: float_t,
    ) -> None:
        """
        Rigid body cylinder initializer.

        Parameters
        ----------
        start
        direction
        normal
        base_length
        base_radius
        density
        """

        def _check_array_size(
            to_check: f_arr_t, name: str, expected: int_t = 3
        ) -> None:
            array_size = to_check.size
            assert array_size == expected, (
                f"Invalid size of '{name}'. "
                f"Expected: {expected}, but got: {array_size}"
            )

        def _check_lower_bound(
            to_check: float_t, name: str, lower_bound: float_t = np.float64(0.0)
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
        self.n_elem: int_t = 1

        normal = normal.reshape((3, 1))
        tangents = direction.reshape((3, 1))
        binormal = _batch_cross(tangents, normal)
        self.radius = base_radius
        self.length = base_length
        self.density = density

        dim: int_t = MaxDimension.value()

        # This is for a rigid body cylinder
        self.volume = np.pi * base_radius * base_radius * base_length
        self.mass = np.array([self.volume * self.density], dtype=np.float64)

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

        self.velocity_collection = np.zeros((MaxDimension.value(), 1))
        self.omega_collection = np.zeros((MaxDimension.value(), 1))
        self.acceleration_collection = np.zeros((MaxDimension.value(), 1))
        self.alpha_collection = np.zeros((MaxDimension.value(), 1))

        self.director_collection = np.zeros(
            (MaxDimension.value(), MaxDimension.value(), 1)
        )
        self.director_collection[0, ...] = normal
        self.director_collection[1, ...] = binormal
        self.director_collection[2, ...] = tangents

        self.external_forces = np.zeros((MaxDimension.value())).reshape(
            MaxDimension.value(), 1
        )
        self.external_torques = np.zeros((MaxDimension.value())).reshape(
            MaxDimension.value(), 1
        )
        # Assemble directors
        self.director_collection[0, :] = normal
        self.director_collection[1, :] = binormal
        self.director_collection[2, :] = tangents
