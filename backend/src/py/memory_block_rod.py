"""Create block-structure class for collection of Cosserat rod systems."""
import numpy as np
from typing import Literal

from elastica.memory_block.utils import make_block_memory_metadata
from elastica.rod.cosserat_rod import (
    CosseratRod,
    _compute_sigma_kappa_for_blockstructure,
)
from elastica.rod.data_structures import _RodSymplecticStepperMixin
from elastica.reset_functions_for_block_structure import _reset_scalar_ghost
from elastica.typing import RodType, SystemIdxType

from elasticapp import BlockRodSystem  # noqa: E402

# Mapping python CosseratRod attribute to C++ tag that will be block memory allocated.
# Tags are defined in cosserat_rod_system.h
PY2CPP_VARNAMES: dict[str, str] = {
    # Node variables
    "mass": "mass",
    "position_collection": "position",
    "velocity_collection": "velocity",
    "acceleration_collection": "acceleration",
    "internal_forces": "internal_forces",
    "external_forces": "external_forces",
    # Element variables
    "omega_collection": "omega",
    "alpha_collection": "alpha",
    "director_collection": "director",
    "rest_lengths": "rest_lengths",
    "density": "density",
    "volume": "volume",
    "mass_second_moment_of_inertia": "mass_second_moment_of_inertia",
    "inv_mass_second_moment_of_inertia": "inv_mass_second_moment_of_inertia",
    "internal_torques": "internal_torques",
    "external_torques": "external_torques",
    "lengths": "lengths",
    "tangents": "tangents",
    "radius": "radius",
    "dilatation": "dilatation",
    "dilatation_rate": "dilatation_rate",
    "sigma": "sigma",
    "rest_sigma": "rest_sigma",
    "internal_stress": "internal_stress",
    "shear_matrix": "shear_matrix",
    # Voronoi variables
    "rest_voronoi_lengths": "rest_voronoi_lengths",
    "voronoi_dilatation": "voronoi_dilatation",
    "kappa": "kappa",
    "rest_kappa": "rest_kappa",
    "internal_couple": "internal_couple",
    "bend_matrix": "bend_matrix",
}


class MemoryBlockCosseratRod(CosseratRod, _RodSymplecticStepperMixin):
    """
    Memory block class for Cosserat rod equations.

    This class is derived from CosseratRod to inherit all rod methods while providing
    a memory-efficient block structure for multiple rod systems. It uses the C++ backend
    BlockRodSystem for efficient memory management.

    Parameters
    ----------
    systems : list[RodType]
        List of CosseratRod objects to be included in the memory block structure.
        Currently only straight rods are supported (ring rods are not yet implemented).
    system_idx_list : list[SystemIdxType]
        List of system indices corresponding to each rod in the `systems` list.
        These indices are used to map rods back to their original positions in
        the simulator's system collection.

    Attributes
    ----------
    n_systems : int
        Total number of rod systems in the memory block.
    n_rods : int
        Total number of rods (same as n_systems).
    n_elems : int
        Total number of elements across all rods in the block structure.
    n_nodes : int
        Total number of nodes across all rods (n_elems + 1).
    n_voronoi : int
        Total number of Voronoi points across all rods (n_elems - 1).
    system_idx_list : numpy.ndarray
        Array of system indices mapping rods to their original positions.
    ghost_nodes_idx : numpy.ndarray
        Indices of ghost nodes used for boundary conditions.
    ghost_elems_idx : numpy.ndarray
        Indices of ghost elements used for boundary conditions.
    ghost_voronoi_idx : numpy.ndarray
        Indices of ghost Voronoi points used for boundary conditions.

    Notes
    -----
    - Currently only straight rods are supported. Ring rod support is planned for future.
    - All rod data (positions, directors, velocities, etc.) is stored in contiguous
      memory blocks for efficient computation.
    """

    def __init__(
        self, systems: list[RodType], system_idx_list: list[SystemIdxType]
    ) -> None:
        self.n_systems = len(systems)

        # Sorted systems (only straight rods for now)
        self.system_idx_list = np.array(system_idx_list, dtype=np.int32)

        n_elems_straight_rods = np.array(
            [x.n_elems for x in systems], dtype=np.int32
        )

        # Create C++ block with element counts
        # BlockRodSystem accepts numpy arrays directly (as well as lists/tuples)
        self._block = BlockRodSystem(n_elems_straight_rods)

        # Compute metadata using utility function
        (
            self.n_elems,
            self.ghost_nodes_idx,
            self.ghost_elems_idx,
            self.ghost_voronoi_idx,
        ) = make_block_memory_metadata(n_elems_straight_rods)

        self.n_nodes = self.n_elems + 1
        self.n_voronoi = self.n_elems - 1
        self.n_rods = len(systems)

        # Compute start and end indices for nodes, elements, and voronoi
        self.start_idx_in_rod_nodes = np.hstack(
            (0, self.ghost_nodes_idx + 1)
        )
        self.end_idx_in_rod_nodes = np.hstack(
            (self.ghost_nodes_idx, self.n_nodes)
        )
        self.start_idx_in_rod_elems = np.hstack((0, self.ghost_elems_idx[1::2] + 1))
        self.end_idx_in_rod_elems = np.hstack((self.ghost_elems_idx[::2], self.n_elems))
        self.start_idx_in_rod_voronoi = np.hstack((0, self.ghost_voronoi_idx[2::3] + 1))
        self.end_idx_in_rod_voronoi = np.hstack(
            (self.ghost_voronoi_idx[::3], self.n_voronoi)
        )

        # Allocate block structure using system collection
        self._allocate_block_variables_in_nodes(systems)
        self._allocate_block_variables_in_elements(systems)
        self._allocate_blocks_variables_in_voronoi(systems)
        self._allocate_blocks_variables_for_symplectic_stepper(systems)

        # Reset ghosts of mass, rest length and rest voronoi length to 1
        _reset_scalar_ghost(self.mass, self.ghost_nodes_idx, 1.0)
        _reset_scalar_ghost(self.rest_lengths, self.ghost_elems_idx, 1.0)
        _reset_scalar_ghost(self.rest_voronoi_lengths, self.ghost_voronoi_idx, 1.0)

        # Compute strains for the block
        _compute_sigma_kappa_for_blockstructure(self)

        # Initialize the mixin class for symplectic time-stepper
        _RodSymplecticStepperMixin.__init__(self)

    def _allocate_block_variables_in_nodes(self, systems: list[RodType]) -> None:
        """Allocate variables on nodes for block-structure."""
        # Scalar variables on nodes
        map_scalar_dofs_in_rod_nodes = {"mass": 0}
        self.scalar_dofs_in_rod_nodes = np.zeros(
            (len(map_scalar_dofs_in_rod_nodes), self.n_nodes)
        )
        self._map_system_properties_to_block_memory(
            mapping_dict=map_scalar_dofs_in_rod_nodes,
            systems=systems,
            block_memory=self.scalar_dofs_in_rod_nodes,
            domain_type="node",
            value_type="scalar",
        )

        # Vector variables on nodes
        map_vector_dofs_in_rod_nodes = {
            "position_collection": 0,
            "internal_forces": 1,
            "external_forces": 2,
        }
        self.vector_dofs_in_rod_nodes = np.zeros(
            (len(map_vector_dofs_in_rod_nodes), 3 * self.n_nodes)
        )
        self._map_system_properties_to_block_memory(
            mapping_dict=map_vector_dofs_in_rod_nodes,
            systems=systems,
            block_memory=self.vector_dofs_in_rod_nodes,
            domain_type="node",
            value_type="vector",
        )

    def _allocate_block_variables_in_elements(self, systems: list[RodType]) -> None:
        """Allocate variables on elements for block-structure."""
        # Scalar variables on elements
        map_scalar_dofs_in_rod_elems = {
            "radius": 0,
            "volume": 1,
            "density": 2,
            "lengths": 3,
            "rest_lengths": 4,
            "dilatation": 5,
            "dilatation_rate": 6,
        }
        self.scalar_dofs_in_rod_elems = np.zeros(
            (len(map_scalar_dofs_in_rod_elems), self.n_elems)
        )
        self._map_system_properties_to_block_memory(
            mapping_dict=map_scalar_dofs_in_rod_elems,
            systems=systems,
            block_memory=self.scalar_dofs_in_rod_elems,
            domain_type="element",
            value_type="scalar",
        )

        # Vector variables on elements
        map_vector_dofs_in_rod_elems = {
            "tangents": 0,
            "sigma": 1,
            "rest_sigma": 2,
            "internal_torques": 3,
            "external_torques": 4,
            "internal_stress": 5,
        }
        self.vector_dofs_in_rod_elems = np.zeros(
            (len(map_vector_dofs_in_rod_elems), 3 * self.n_elems)
        )
        self._map_system_properties_to_block_memory(
            mapping_dict=map_vector_dofs_in_rod_elems,
            systems=systems,
            block_memory=self.vector_dofs_in_rod_elems,
            domain_type="element",
            value_type="vector",
        )

        # Matrix variables on elements
        map_matrix_dofs_in_rod_elems = {
            "director_collection": 0,
            "mass_second_moment_of_inertia": 1,
            "inv_mass_second_moment_of_inertia": 2,
            "shear_matrix": 3,
        }
        self.matrix_dofs_in_rod_elems = np.zeros(
            (len(map_matrix_dofs_in_rod_elems), 9 * self.n_elems)
        )
        self._map_system_properties_to_block_memory(
            mapping_dict=map_matrix_dofs_in_rod_elems,
            systems=systems,
            block_memory=self.matrix_dofs_in_rod_elems,
            domain_type="element",
            value_type="tensor",
        )

    def _allocate_blocks_variables_in_voronoi(self, systems: list[RodType]) -> None:
        """Allocate variables on voronoi for block-structure."""
        # Scalar variables on voronoi
        map_scalar_dofs_in_rod_voronois = {
            "voronoi_dilatation": 0,
            "rest_voronoi_lengths": 1,
        }
        self.scalar_dofs_in_rod_voronois = np.zeros(
            (len(map_scalar_dofs_in_rod_voronois), self.n_voronoi)
        )
        self._map_system_properties_to_block_memory(
            mapping_dict=map_scalar_dofs_in_rod_voronois,
            systems=systems,
            block_memory=self.scalar_dofs_in_rod_voronois,
            domain_type="voronoi",
            value_type="scalar",
        )

        # Vector variables on voronoi
        map_vector_dofs_in_rod_voronois = {
            "kappa": 0,
            "rest_kappa": 1,
            "internal_couple": 2,
        }
        self.vector_dofs_in_rod_voronois = np.zeros(
            (len(map_vector_dofs_in_rod_voronois), 3 * self.n_voronoi)
        )
        self._map_system_properties_to_block_memory(
            mapping_dict=map_vector_dofs_in_rod_voronois,
            systems=systems,
            block_memory=self.vector_dofs_in_rod_voronois,
            domain_type="voronoi",
            value_type="vector",
        )

        # Matrix variables on voronoi
        map_matrix_dofs_in_rod_voronois = {"bend_matrix": 0}
        self.matrix_dofs_in_rod_voronois = np.zeros(
            (len(map_matrix_dofs_in_rod_voronois), 9 * self.n_voronoi)
        )
        self._map_system_properties_to_block_memory(
            mapping_dict=map_matrix_dofs_in_rod_voronois,
            systems=systems,
            block_memory=self.matrix_dofs_in_rod_voronois,
            domain_type="voronoi",
            value_type="tensor",
        )

    def _allocate_blocks_variables_for_symplectic_stepper(
        self, systems: list[RodType]
    ) -> None:
        """Allocate variables used by symplectic stepper for block-structure."""
        map_rate_collection = {
            "velocity_collection": 0,
            "omega_collection": 1,
            "acceleration_collection": 2,
            "alpha_collection": 3,
        }
        self.rate_collection = np.zeros((len(map_rate_collection), 3 * self.n_nodes))

        # For Dynamic state update of position Verlet create references
        self.v_w_collection = np.lib.stride_tricks.as_strided(
            self.rate_collection[0:2], (2, 3 * self.n_nodes)
        )

        self.dvdt_dwdt_collection = np.lib.stride_tricks.as_strided(
            self.rate_collection[2:], (2, 3 * self.n_nodes)
        )

        # Copy systems variables on nodes to block structure
        map_rate_collection_dofs_in_rod_nodes = {
            "velocity_collection": 0,
            "acceleration_collection": 2,
        }
        self._map_system_properties_to_block_memory(
            mapping_dict=map_rate_collection_dofs_in_rod_nodes,
            systems=systems,
            block_memory=self.rate_collection,
            domain_type="node",
            value_type="vector",
        )

        # Copy systems variables on elements to block structure
        map_rate_collection_dofs_in_rod_elems = {
            "omega_collection": 1,
            "alpha_collection": 3,
        }
        self._map_system_properties_to_block_memory(
            mapping_dict=map_rate_collection_dofs_in_rod_elems,
            systems=systems,
            block_memory=self.rate_collection,
            domain_type="element",
            value_type="vector",
        )

    def _map_system_properties_to_block_memory(
        self,
        mapping_dict: dict,
        systems: list[RodType],
        block_memory: np.ndarray,
        domain_type: Literal["node", "element", "voronoi"],
        value_type: Literal["scalar", "vector", "tensor"],
    ) -> None:
        """Map system (Cosserat rods) properties to memory blocks.

        This method takes domain types (node, element, voronoi) and value
        types (scalar, vector, tensor) as inputs and computes internally how to
        construct the mapping properly.

        Parameters
        ----------
        mapping_dict: dict
            Dictionary with attribute names as keys and block row index as values.
        systems: list[RodType]
            A sequence containing Cosserat rod objects to map from.
        block_memory: ndarray
            Memory block that, at the end of the method execution, contains all designated
            attributes of all systems.
        domain_type: str
            A string that indicates the discretized domain where the attributes reside.
            Options among "node", "element", and "voronoi".
        value_type: str
            A string that indicates the shape of the attribute.
            Options among "scalar", "vector", and "tensor".
        """
        # Get appropriate start/end indices based on domain type
        if domain_type == "node":
            start_idx_list = self.start_idx_in_rod_nodes
            end_idx_list = self.end_idx_in_rod_nodes
        elif domain_type == "element":
            start_idx_list = self.start_idx_in_rod_elems
            end_idx_list = self.end_idx_in_rod_elems
        elif domain_type == "voronoi":
            start_idx_list = self.start_idx_in_rod_voronoi
            end_idx_list = self.end_idx_in_rod_voronoi
        else:
            raise ValueError(
                "Incorrect domain type. Must be one of node, element, and voronoi"
            )

        # Determine view shape based on value type
        if value_type == "scalar":
            view_shape = (self.n_nodes if domain_type == "node" else
                         self.n_elems if domain_type == "element" else
                         self.n_voronoi,)
        elif value_type == "vector":
            domain_num = (self.n_nodes if domain_type == "node" else
                         self.n_elems if domain_type == "element" else
                         self.n_voronoi)
            view_shape = (3, domain_num)
        elif value_type == "tensor":
            domain_num = (self.n_nodes if domain_type == "node" else
                         self.n_elems if domain_type == "element" else
                         self.n_voronoi)
            view_shape = (3, 3, domain_num)
        else:
            raise ValueError(
                "Incorrect value type. Must be one of scalar, vector, and tensor."
            )

        for k, v in mapping_dict.items():
            # Map class attributes to block memory using strided view
            self.__dict__[k] = np.lib.stride_tricks.as_strided(
                block_memory[v],
                shape=view_shape,
            )

            # Copy system attributes into block memory, then make system attributes
            # views into the block memory
            for system_idx, system in enumerate(systems):
                start_idx = start_idx_list[system_idx]
                end_idx = end_idx_list[system_idx]
                self.__dict__[k][..., start_idx:end_idx] = system.__dict__[k].copy()
                system.__dict__[k] = np.ndarray.view(
                    self.__dict__[k][..., start_idx:end_idx]
                )

    def relink_system_properties_to_block_memory(
        self,
        mapping_dict: dict,
        systems: list[RodType],
        block_memory: np.ndarray,
        domain_type: Literal["node", "element", "voronoi"],
        value_type: Literal["scalar", "vector", "tensor"],
    ) -> None:
        # Get appropriate start/end indices based on domain type
        if domain_type == "node":
            start_idx_list = self.start_idx_in_rod_nodes
            end_idx_list = self.end_idx_in_rod_nodes
        elif domain_type == "element":
            start_idx_list = self.start_idx_in_rod_elems
            end_idx_list = self.end_idx_in_rod_elems
        elif domain_type == "voronoi":
            start_idx_list = self.start_idx_in_rod_voronoi
            end_idx_list = self.end_idx_in_rod_voronoi
        else:
            raise ValueError(
                "Incorrect domain type. Must be one of node, element, and voronoi"
            )

        # Determine view shape based on value type
        if value_type == "scalar":
            view_shape = (self.n_nodes if domain_type == "node" else
                         self.n_elems if domain_type == "element" else
                         self.n_voronoi,)
        elif value_type == "vector":
            domain_num = (self.n_nodes if domain_type == "node" else
                         self.n_elems if domain_type == "element" else
                         self.n_voronoi)
            view_shape = (3, domain_num)
        elif value_type == "tensor":
            domain_num = (self.n_nodes if domain_type == "node" else
                         self.n_elems if domain_type == "element" else
                         self.n_voronoi)
            view_shape = (3, 3, domain_num)
        else:
            raise ValueError(
                "Incorrect value type. Must be one of scalar, vector, and tensor."
            )

        for k, v in mapping_dict.items():
            # Map class attributes to block memory using strided view
            self.__dict__[k] = np.lib.stride_tricks.as_strided(
                block_memory[v],
                shape=view_shape,
            )

            # Copy system attributes into block memory, then make system attributes
            # views into the block memory
            for system_idx, system in enumerate(systems):
                start_idx = start_idx_list[system_idx]
                end_idx = end_idx_list[system_idx]
                self.__dict__[k][..., start_idx:end_idx] = system.__dict__[k].copy()
                system.__dict__[k] = np.ndarray.view(
                    self.__dict__[k][..., start_idx:end_idx]
                )
