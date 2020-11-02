__doc__ = """ Cylinder rigid body class """

import numpy as np

from elastica._linalg import _batch_matvec, _batch_cross
from elastica.utils import MaxDimension
from elastica.rigidbody.rigid_body import RigidBodyBase
from elastica.rigidbody.data_structures import _RigidRodSymplecticStepperMixin


class Cylinder(RigidBodyBase, _RigidRodSymplecticStepperMixin):
    def __init__(self, start, direction, normal, base_length, base_radius, density):
        # rigid body does not have elements it only have one node. We are setting n_elems to
        # zero for only make code to work. _bootstrap_from_data requires n_elems to be defined
        self.n_elems = 1

        self.normal = normal.reshape(3, 1)
        self.tangents = direction.reshape(3, 1)
        self.binormal = np.cross(direction, normal).reshape(3, 1)
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

        self.inv_mass_second_moment_of_inertia = np.linalg.inv(
            mass_second_moment_of_inertia
        ).reshape(MaxDimension.value(), MaxDimension.value(), 1)

        # position is at the center
        position = np.zeros((MaxDimension.value(), 1))
        position[:] = start.reshape(3, 1) + direction.reshape(3, 1) * base_length / 2

        velocities = np.zeros((MaxDimension.value(), 1))
        omegas = np.zeros((MaxDimension.value(), 1))
        accelerations = 0.0 * velocities
        angular_accelerations = 0.0 * omegas

        directors = np.zeros((MaxDimension.value(), MaxDimension.value(), 1))
        directors[0, ...] = self.normal
        directors[1, ...] = _batch_cross(self.tangents, self.normal)
        directors[2, ...] = self.tangents

        self._vector_states = np.hstack(
            (position, velocities, omegas, accelerations, angular_accelerations)
        )
        self._matrix_states = directors.copy()

        self.internal_forces = np.zeros((MaxDimension.value())).reshape(
            MaxDimension.value(), 1
        )
        self.internal_torques = np.zeros((MaxDimension.value())).reshape(
            MaxDimension.value(), 1
        )

        self.external_forces = np.zeros((MaxDimension.value())).reshape(
            MaxDimension.value(), 1
        )
        self.external_torques = np.zeros((MaxDimension.value())).reshape(
            MaxDimension.value(), 1
        )

        _RigidRodSymplecticStepperMixin.__init__(self)

    def _compute_internal_forces_and_torques(self, time):
        """
        This function here is only for integrator to work properly. We do not need
        internal forces and torques at all.
        Parameters
        ----------
        time

        Returns
        -------

        """
        self.internal_forces *= 0.0
        self.internal_torques *= 0.0

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
            (self.internal_forces + self.external_forces) / self.mass,
        )
        np.copyto(
            self.alpha_collection,
            _batch_matvec(
                self.inv_mass_second_moment_of_inertia,
                (self.internal_torques + self.external_torques),
            ),
        )

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
