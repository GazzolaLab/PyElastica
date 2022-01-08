__doc__ = """ Cylinder rigid body class """

import numpy as np

from elastica._linalg import _batch_cross
from elastica.utils import MaxDimension
from elastica.rigidbody.rigid_body import RigidBodyBase


class Cylinder(RigidBodyBase):
    def __init__(self, start, direction, normal, base_length, base_radius, density):
        # rigid body does not have elements it only have one node. We are setting n_elems to
        # zero for only make code to work. _bootstrap_from_data requires n_elems to be defined
        self.n_elems = 1

        normal = normal.reshape(3, 1)
        tangents = direction.reshape(3, 1)
        binormal = _batch_cross(tangents, normal)
        self.radius = base_radius
        self.length = base_length
        self.density = density
        # This is for a rigid body cylinder
        self.volume = np.pi * base_radius * base_radius * base_length
        self.mass = np.array([self.volume * self.density])

        # Second moment of inertia
        A0 = np.pi * base_radius * base_radius
        I0_1 = A0 * A0 / (4.0 * np.pi)
        I0_2 = I0_1
        I0_3 = 2.0 * I0_2
        I0 = np.array([I0_1, I0_2, I0_3])

        # Mass second moment of inertia for disk cross-section
        mass_second_moment_of_inertia = np.zeros(
            (MaxDimension.value(), MaxDimension.value()), np.float64
        )
        np.fill_diagonal(mass_second_moment_of_inertia, I0 * density * base_length)

        self.mass_second_moment_of_inertia = mass_second_moment_of_inertia.reshape(
            MaxDimension.value(), MaxDimension.value(), 1
        )

        self.inv_mass_second_moment_of_inertia = np.linalg.inv(
            mass_second_moment_of_inertia
        ).reshape(MaxDimension.value(), MaxDimension.value(), 1)

        # position is at the center
        self.position_collection = np.zeros((MaxDimension.value(), 1))
        self.position_collection[:] = (
            start.reshape(3, 1) + direction.reshape(3, 1) * base_length / 2
        )

        self.velocity_collection = np.zeros((MaxDimension.value(), 1))
        self.omega_collection = np.zeros((MaxDimension.value(), 1))
        self.acceleration_collection = 0.0 * self.velocity_collection
        self.alpha_collection = 0.0 * self.omega_collection

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
