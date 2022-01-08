__doc__ = """Create block-structure class for collection of rigid body systems."""
import numpy as np
from typing import Sequence

from elastica.rigidbody import RigidBodyBase, Cylinder

# from elastica.rigidbody.rigid_body import RigidBody
from elastica.rigidbody.data_structures import _RigidRodSymplecticStepperMixin
from elastica.rod.data_structures import _RodSymplecticStepperMixin
from elastica.reset_functions_for_block_structure import _reset_scalar_ghost
from elastica.rod.cosserat_rod import CosseratRod
from elastica.rigidbody import sphere, cylinder
from elastica._linalg import _batch_matvec, _batch_cross


class MemoryBlockRigidBody(RigidBodyBase, _RigidRodSymplecticStepperMixin):
    def __init__(self, systems: Sequence):

        self.n_bodies = len(systems)
        self.n_elems = self.n_bodies
        self.n_nodes = self.n_elems

        # Allocate block structure using system collection.
        self.allocate_block_variables_scalars(systems)
        self.allocate_block_variables_vectors(systems)
        self.allocate_block_variables_matrix(systems)
        self.allocate_block_variables_for_symplectic_stepper(systems)

        # Initialize the mixin class for symplectic time-stepper.
        _RigidRodSymplecticStepperMixin.__init__(self)

    def allocate_block_variables_scalars(self, systems: Sequence):
        """
        This function takes system collection and allocates the variables for
        block-structure and references allocated variables back to the systems.

        Parameters
        ----------
        systems

        Returns
        -------

        """

        # Things in rigid bodies that are scalars
        #           0 ("radius", float64),
        #           1 ("length", float64),
        #           2 ("density", float64),
        #           3 ("volume" , float 64),
        #           4 ("mass, float64),

        map_scalar_dofs_in_rigid_bodies = {
            "radius": 0,
            "length": 1,
            "density": 2,
            "volume": 3,
            "mass": 4,
        }
        self.scalar_dofs_in_rigid_bodies = np.zeros(
            (len(map_scalar_dofs_in_rigid_bodies), self.n_elems)
        )

        for k, v in map_scalar_dofs_in_rigid_bodies.items():
            self.__dict__[k] = np.lib.stride_tricks.as_strided(
                self.scalar_dofs_in_rigid_bodies[v], (self.n_elems,)
            )

        for k, v in map_scalar_dofs_in_rigid_bodies.items():
            for system_idx, system in enumerate(systems):
                self.__dict__[k][..., system_idx : system_idx + 1] = system.__dict__[k]
                system.__dict__[k] = np.ndarray.view(
                    self.__dict__[k][..., system_idx : system_idx + 1]
                )

    def allocate_block_variables_vectors(self, systems: Sequence):
        """
        This function takes system collection and allocates the vector variables for
        block-structure and references allocated vector variables back to the systems.

        Parameters
        ----------
        systems

        Returns
        -------

        """

        # Things in rigid bodies that are vectors
        #           0 ("position_collection", float64[:,:])
        #           1 ("external_forces", float64[:,:])
        #           2 ("external_torques", float64[:,:])

        map_vector_dofs_in_rigid_bodies = {
            "position_collection": 0,
            "external_forces": 1,
            "external_torques": 2,
        }

        self.vector_dofs_in_rigid_bodies = np.zeros(
            (len(map_vector_dofs_in_rigid_bodies), 3 * self.n_elems)
        )

        for k, v in map_vector_dofs_in_rigid_bodies.items():
            self.__dict__[k] = np.lib.stride_tricks.as_strided(
                self.vector_dofs_in_rigid_bodies[v], (3, self.n_elems)
            )

        for k, v in map_vector_dofs_in_rigid_bodies.items():
            for system_idx, system in enumerate(systems):
                self.__dict__[k][..., system_idx : system_idx + 1] = system.__dict__[
                    k
                ].copy()
                system.__dict__[k] = np.ndarray.view(
                    self.__dict__[k][..., system_idx : system_idx + 1]
                )

    def allocate_block_variables_matrix(self, systems: Sequence):
        """
        This function takes system collection and allocates the matrix variables for
        block-structure and references allocated matrix variables back to the systems.

        Parameters
        ----------
        system

        Returns
        -------

        """

        # Things in rigid bodies that are matrix
        #           0 ("director_collection", float64[:, : ,:]
        #           1 ("mass_moment_of_inertia", float64[:, :, :]
        #           2 ("inv_mass_moment_of_inertia", float64[:, :, :]

        map_matrix_dofs_in_rigid_bodies = {
            "director_collection": 0,
            "mass_second_moment_of_inertia": 1,
            "inv_mass_second_moment_of_inertia": 2,
        }

        self.matrix_dofs_in_rigid_bodies = np.zeros(
            (len(map_matrix_dofs_in_rigid_bodies), 9 * self.n_elems)
        )

        for k, v in map_matrix_dofs_in_rigid_bodies.items():
            self.__dict__[k] = np.lib.stride_tricks.as_strided(
                self.matrix_dofs_in_rigid_bodies[v], (3, 3, self.n_elems)
            )

        for k, v in map_matrix_dofs_in_rigid_bodies.items():
            for system_idx, system in enumerate(systems):
                self.__dict__[k][..., system_idx : system_idx + 1] = system.__dict__[
                    k
                ].copy()
                system.__dict__[k] = np.ndarray.view(
                    self.__dict__[k][..., system_idx : system_idx + 1]
                )

    def allocate_block_variables_for_symplectic_stepper(self, systems: Sequence):
        """
        This function takes system collection and allocates the variables used by symplectic
        stepper for block-structure and references allocated variables back to the systems.

        Parameters
        ----------
        systems

        Returns
        -------

        """
        # These vectors are on nodes or on elements, but we stack them together for
        # better memory access. Because we use them together in time-steppers.
        #             0 ("velocity_collection", float64[:, :]),
        #             1 ("omega_collection", float64[:, :]),
        #             2 ("acceleration_collection", float64[:, :]),
        #             3 ("alpha_collection", float64[:, :]),
        # 4 in total

        map_rate_collection = {
            "velocity_collection": 0,
            "omega_collection": 1,
            "acceleration_collection": 2,
            "alpha_collection": 3,
        }
        self.rate_collection = np.zeros((len(map_rate_collection), 3 * self.n_elems))
        for k, v in map_rate_collection.items():
            self.__dict__[k] = np.lib.stride_tricks.as_strided(
                self.rate_collection[v], (3, self.n_elems)
            )

        # For Dynamic state update of position Verlet create references
        self.v_w_collection = np.lib.stride_tricks.as_strided(
            self.rate_collection[0:2], (2, 3 * self.n_elems)
        )

        self.dvdt_dwdt_collection = np.lib.stride_tricks.as_strided(
            self.rate_collection[2:-1], (2, 3 * self.n_elems)
        )

        for k, v in map_rate_collection.items():
            for system_idx, system in enumerate(systems):
                self.__dict__[k][..., system_idx : system_idx + 1] = system.__dict__[
                    k
                ].copy()
                system.__dict__[k] = np.ndarray.view(
                    self.__dict__[k][..., system_idx : system_idx + 1]
                )
