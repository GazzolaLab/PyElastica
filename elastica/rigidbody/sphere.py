__doc__ = """ Sphere rigid body class """

import numpy as np

from elastica._linalg import _batch_matvec, _batch_cross
from elastica.utils import MaxDimension
from elastica.rigidbody.rigid_body import RigidBodyBase
from elastica.rigidbody.data_structures import _RigidRodSymplecticStepperMixin


class Sphere(RigidBodyBase, _RigidRodSymplecticStepperMixin):
    def __init__(self, center, base_radius, density):
        # rigid body does not have elements it only have one node. We are setting n_elems to
        # zero for only make code to work. _bootstrap_from_data requires n_elems to be defined
        self.n_elems = 1

        self.radius = base_radius
        self.density = density
        self.length = 2 * base_radius
        # This is for a rigid body cylinder
        self.volume = 4.0 / 3.0 * np.pi * base_radius ** 3
        self.mass = np.array([self.volume * self.density])
        self.normal = np.array([1.0, 0.0, 0.0]).reshape(3, 1)
        self.tangents = np.array([0.0, 0.0, 1.0]).reshape(3, 1)
        self.binormal = _batch_cross(self.tangents, self.normal)

        # Mass second moment of inertia for disk cross-section
        mass_second_moment_of_inertia = np.zeros(
            (MaxDimension.value(), MaxDimension.value()), np.float64
        )
        np.fill_diagonal(
            mass_second_moment_of_inertia, 2.0 / 5.0 * self.mass * self.radius ** 2
        )

        self.inv_mass_second_moment_of_inertia = np.linalg.inv(
            mass_second_moment_of_inertia
        ).reshape(MaxDimension.value(), MaxDimension.value(), 1)

        # position is at the center
        position = np.zeros((MaxDimension.value(), 1))
        position[:, 0] = center

        velocities = np.zeros((MaxDimension.value(), 1))
        omegas = np.zeros((MaxDimension.value(), 1))
        accelerations = 0.0 * velocities
        angular_accelerations = 0.0 * omegas

        directors = np.zeros((MaxDimension.value(), MaxDimension.value(), 1))
        directors[0, ...] = self.normal
        directors[1, ...] = _batch_cross(self.tangents, self.normal)
        directors[2, ...] = self.tangents
        # directors[0, ...] = [[1.0], [0.0], [0.0]]
        # directors[1, ...] = [[0.0], [1.0], [0.0]]
        # directors[2, ...] = [[0.0], [0.0], [1.0]]

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
        pass

    def update_accelerations(self, time):
        """TODO Do we need to make the collection members abstract?

        Parameters
        ----------
        time

        Returns
        -------

        """
        np.copyto(self.acceleration_collection, self.external_forces / self.mass)
        np.copyto(
            self.alpha_collection,
            _batch_matvec(
                self.inv_mass_second_moment_of_inertia, self.external_torques
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
