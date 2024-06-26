__doc__ = """Create block-structure class for collection of rigid body systems."""
from typing import Literal
import numpy as np
from elastica.typing import SystemIdxType, RigidBodyType

from elastica.rigidbody import RigidBodyBase
from elastica.rigidbody.data_structures import _RigidRodSymplecticStepperMixin


class MemoryBlockRigidBody(RigidBodyBase, _RigidRodSymplecticStepperMixin):
    def __init__(
        self, systems: list[RigidBodyType], system_idx_list: list[SystemIdxType]
    ) -> None:

        self.n_systems = len(systems)
        self.n_elems = self.n_systems
        self.n_nodes = self.n_elems
        self.system_idx_list = np.array(system_idx_list, dtype=np.int32)

        # Allocate block structure using system collection.
        self._allocate_block_variables_scalars(systems)
        self._allocate_block_variables_vectors(systems)
        self._allocate_block_variables_matrix(systems)
        self._allocate_block_variables_for_symplectic_stepper(systems)

        # Initialize the mixin class for symplectic time-stepper.
        _RigidRodSymplecticStepperMixin.__init__(self)

    def _allocate_block_variables_scalars(self, systems: list[RigidBodyType]) -> None:
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
        #           0 ("density", float64),
        #           1 ("volume" , float64),
        #           2 ("mass, float64),

        map_scalar_dofs_in_rigid_bodies = {
            "density": 0,
            "volume": 1,
            "mass": 2,
        }
        self.scalar_dofs_in_rigid_bodies = np.zeros(
            (len(map_scalar_dofs_in_rigid_bodies), self.n_elems)
        )

        self._map_system_properties_to_block_memory(
            mapping_dict=map_scalar_dofs_in_rigid_bodies,
            systems=systems,
            block_memory=self.scalar_dofs_in_rigid_bodies,
            value_type="scalar",
        )

    def _allocate_block_variables_vectors(self, systems: list[RigidBodyType]) -> None:
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

        self._map_system_properties_to_block_memory(
            mapping_dict=map_vector_dofs_in_rigid_bodies,
            systems=systems,
            block_memory=self.vector_dofs_in_rigid_bodies,
            value_type="vector",
        )

    def _allocate_block_variables_matrix(self, systems: list[RigidBodyType]) -> None:
        """
        This function takes system collection and allocates the matrix variables for
        block-structure and references allocated matrix variables back to the systems.

        Parameters
        ----------
        systems

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

        self._map_system_properties_to_block_memory(
            mapping_dict=map_matrix_dofs_in_rigid_bodies,
            systems=systems,
            block_memory=self.matrix_dofs_in_rigid_bodies,
            value_type="tensor",
        )

    def _allocate_block_variables_for_symplectic_stepper(
        self, systems: list[RigidBodyType]
    ) -> None:
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
        self._map_system_properties_to_block_memory(
            mapping_dict=map_rate_collection,
            systems=systems,
            block_memory=self.rate_collection,
            value_type="vector",
        )
        # For Dynamic state update of position Verlet create references
        self.v_w_collection = np.lib.stride_tricks.as_strided(
            self.rate_collection[0:2], (2, 3 * self.n_elems)
        )

        self.dvdt_dwdt_collection = np.lib.stride_tricks.as_strided(
            self.rate_collection[2:-1], (2, 3 * self.n_elems)
        )

    def _map_system_properties_to_block_memory(
        self,
        mapping_dict: dict,
        systems: list[RigidBodyType],
        block_memory: np.ndarray,
        value_type: Literal["scalar", "vector", "tensor"],
    ) -> None:
        """Map system (rigid bodies) properties to memory blocks.

        Parameters
        ----------
        mapping_dict: dict
            Dictionary with attribute names as keys and block row index as values.
        systems: list[RigidBodyType]
            A sequence containing rigid body objects to map from.
        block_memory: ndarray
            Memory block that, at the end of the method execution, contains all designated
            attributes of all systems.
        value_type: str
            A string that indicates the shape of the attribute.
            Options among "scalar", "vector", and "tensor".

        """
        if value_type == "scalar":
            view_shape: tuple[int, ...] = (self.n_elems,)

        elif value_type == "vector":
            view_shape = (3, self.n_elems)

        elif value_type == "tensor":
            view_shape = (3, 3, self.n_elems)

        else:
            raise ValueError(
                "Incorrect value type. Must be one of scalar, vector, and tensor."
            )

        for k, v in mapping_dict.items():
            self.__dict__[k] = np.lib.stride_tricks.as_strided(
                block_memory[v],
                shape=view_shape,
            )

            for system_idx, system in enumerate(systems):
                self.__dict__[k][..., system_idx : system_idx + 1] = (
                    system.__dict__[k]
                    if value_type == "scalar"
                    else system.__dict__[k].copy()
                )
                system.__dict__[k] = np.ndarray.view(
                    self.__dict__[k][..., system_idx : system_idx + 1]
                )
