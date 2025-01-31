__doc__ = """Create block-structure class for collection of Cosserat rod systems."""
import numpy as np
from typing import Literal, Callable
from elastica.typing import SystemIdxType, RodType
from elastica.rod.data_structures import _RodSymplecticStepperMixin
from elastica.reset_functions_for_block_structure import _reset_scalar_ghost
from elastica.rod.cosserat_rod import (
    CosseratRod,
    _compute_sigma_kappa_for_blockstructure,
)
from elastica._synchronize_periodic_boundary import (
    _synchronize_periodic_boundary_of_vector_collection,
    _synchronize_periodic_boundary_of_scalar_collection,
    _synchronize_periodic_boundary_of_matrix_collection,
)

from .utils import (
    make_block_memory_metadata,
    make_block_memory_periodic_boundary_metadata,
)


class MemoryBlockCosseratRod(CosseratRod, _RodSymplecticStepperMixin):
    """
    Memory block class for Cosserat rod equations. This class is derived from Cosserat Rod class in order to inherit
    the methods of Cosserat rod class. This class takes the cosserat rod object (systems) and creates big
    arrays to store the system data and returns a reference of that data to the systems.
    Thus each system is now in contiguous memory, so it is faster to compute Cosserat rod equations.

    TODO: need more documentation!
    """

    def __init__(
        self, systems: list[RodType], system_idx_list: list[SystemIdxType]
    ) -> None:
        self.n_systems = len(systems)

        # separate straight and ring rods
        system_straight_rod = []
        system_ring_rod = []
        system_idx_list_ring_rod = []
        system_idx_list_straight_rod = []
        for k, system_to_be_added in enumerate(systems):

            if system_to_be_added.ring_rod_flag:
                system_ring_rod.append(system_to_be_added)
                system_idx_list_ring_rod.append(system_idx_list[k])
                self.ring_rod_flag = True
            else:
                system_straight_rod.append(system_to_be_added)
                system_idx_list_straight_rod.append(system_idx_list[k])

        # Sorted systems
        systems = system_straight_rod + system_ring_rod
        self.system_idx_list = np.array(
            system_idx_list_straight_rod + system_idx_list_ring_rod, dtype=np.int32
        )

        n_elems_straight_rods = np.array(
            [x.n_elems for x in system_straight_rod], dtype=np.int32
        )
        n_elems_ring_rods = np.array(
            [x.n_elems for x in system_ring_rod], dtype=np.int32
        )

        n_straight_rods: int = len(system_straight_rod)
        n_ring_rods: int = len(system_ring_rod)

        # self.n_elems_in_rods = np.array([x.n_elems for x in systems], dtype=np.int32)
        self.n_elems_in_rods = np.hstack((n_elems_straight_rods, n_elems_ring_rods + 2))
        self.n_rods = len(systems)
        (
            self.n_elems,
            self.ghost_nodes_idx,
            self.ghost_elems_idx,
            self.ghost_voronoi_idx,
        ) = make_block_memory_metadata(self.n_elems_in_rods)
        self.n_nodes = self.n_elems + 1
        self.n_voronoi = self.n_elems - 1

        # n_nodes_in_rods = self.n_elems_in_rods + 1
        # n_voronois_in_rods = self.n_elems_in_rods - 1

        self.start_idx_in_rod_nodes = np.hstack(
            (0, self.ghost_nodes_idx + 1)
        )  # Start index of subsequent rod
        self.end_idx_in_rod_nodes = np.hstack(
            (self.ghost_nodes_idx, self.n_nodes)
        )  # End index of the rod, Some max size, doesn't really matter
        self.start_idx_in_rod_elems = np.hstack((0, self.ghost_elems_idx[1::2] + 1))
        self.end_idx_in_rod_elems = np.hstack((self.ghost_elems_idx[::2], self.n_elems))
        self.start_idx_in_rod_voronoi = np.hstack((0, self.ghost_voronoi_idx[2::3] + 1))
        self.end_idx_in_rod_voronoi = np.hstack(
            (self.ghost_voronoi_idx[::3], self.n_voronoi)
        )

        # Periodic boundaries are only used if there is a ring rod present in the simulation, otherwise below
        # arrays are empty.
        (
            _,
            self.periodic_boundary_nodes_idx,
            self.periodic_boundary_elems_idx,
            self.periodic_boundary_voronoi_idx,
        ) = make_block_memory_periodic_boundary_metadata(n_elems_ring_rods)

        """
        If there are ring rods appended to simulation, then start and end idx of rod nodes, elements and voronoi
        have to be updated, to accommodate the periodic boundaries. Periodic boundaries can only accessed by
        memory block  and rods does not have access to periodic boundaries.
        """
        if n_ring_rods != 0:
            """
            Number of nodes, elements and voronoi of the ring rod is same. When user wants to access to the rod, they
            will also see that. However, in order to be compatible with block structure implementation, it has to be
            n_nodes=n_elems+1 and n_voronoi=n_elems-1. Thus, we add these periodic nodes, element and voronoi at the
            end of the rod. We need 3 periodic nodes,  2 periodic elements and 1 periodic voronoi.
            Below you will see some magic numbers such as 1, 2, and 3. These are related with the number of periodic
            nodes, element and voronoi.
            For example if user sets 50 elements for one ring rod with 50 elements then we need two periodic element
            at the start and end of the rod. So there are total of 52 elements on memory block including periodic ones.
            In our numerical method n_nodes=n_elems+1 and n_voronoi=n_elems-1 so there are 53 nodes and 51 voronoi.
            This requires 3 additional periodic nodes and 1 additional periodic voronoi. We add one of this periodic
            node at the start of the rod and two at the end of the rod.

            So below magic number 3 is coming from having extra 3 periodic nodes for each ring rod.
                self.start_idx_in_rod_nodes[:] = (
                    self.periodic_boundary_nodes_idx[0, 0::3] + 1
                )

            Same idea is used for elements and voronoi as well.
            """
            if n_straight_rods != 0:
                # Here the idea of adding ghost nodes, elems and voronoi of straight rods is that, in memory block
                # we place first straight rods, then ring rods.
                # TODO: in future consider a better implementation for packing problem.
                # +1 is because we want to start from next idx, where periodic boundary starts
                self.periodic_boundary_nodes_idx += (
                    self.ghost_nodes_idx[n_straight_rods - 1] + 1
                )
                self.periodic_boundary_elems_idx += (
                    self.ghost_elems_idx[1::2][n_straight_rods - 1] + 1
                )
                self.periodic_boundary_voronoi_idx += (
                    self.ghost_voronoi_idx[2::3][n_straight_rods - 1] + 1
                )

            # Compute the start and end of the rod nodes again. This time, boundary cells are added.
            self.start_idx_in_rod_nodes[n_straight_rods:] = (
                self.periodic_boundary_nodes_idx[0, 0::3] + 1
            )
            self.end_idx_in_rod_nodes[n_straight_rods:] = (
                self.periodic_boundary_nodes_idx[0, 1::3]
            )

            self.start_idx_in_rod_elems[n_straight_rods:] = (
                self.periodic_boundary_elems_idx[0, 0::2] + 1
            )
            self.end_idx_in_rod_elems[n_straight_rods:] = (
                self.periodic_boundary_elems_idx[0, 1::2]
            )

            self.start_idx_in_rod_voronoi[n_straight_rods:] = (
                self.periodic_boundary_voronoi_idx[0, :] + 1
            )

        # Allocate block structure using system collection.
        self._allocate_block_variables_in_nodes(systems)
        self._allocate_block_variables_in_elements(systems)
        self._allocate_blocks_variables_in_voronoi(systems)
        self._allocate_blocks_variables_for_symplectic_stepper(systems)

        # Reset ghosts of mass, rest length and rest voronoi length to 1. Otherwise
        # since ghosts are not modified, this causes a division by zero error.
        _reset_scalar_ghost(self.mass, self.ghost_nodes_idx, 1.0)
        _reset_scalar_ghost(self.rest_lengths, self.ghost_elems_idx, 1.0)
        _reset_scalar_ghost(self.rest_voronoi_lengths, self.ghost_voronoi_idx, 1.0)

        # Compute strains for the block
        _compute_sigma_kappa_for_blockstructure(self)

        # If n_elems_with_boundary defined and passed with kwargs, then this rod is ring and we need to know
        # how many boundary elements is rod containing.
        if n_ring_rods != 0:
            for sys_idx, system_to_be_added in enumerate(system_ring_rod):
                if np.count_nonzero(system_to_be_added.rest_sigma) == 0:
                    # Ring rod has to have non-zero rest sigma. If user did not set something, then Elastica will
                    # calculate it.
                    system_to_be_added.rest_sigma[:] = system_to_be_added.sigma[:]
                if np.count_nonzero(system_to_be_added.rest_kappa) == 0:
                    # Ring rod has to have non-zero rest kappa. If user did not set something, then Elastica will
                    # calculate it.
                    system_to_be_added.rest_kappa[:] = system_to_be_added.kappa[:]

            # We update periodic elements and voronoi because they are used in difference and trapezoidal kernels.
            _synchronize_periodic_boundary_of_vector_collection(
                self.rest_sigma, self.periodic_boundary_elems_idx
            )
            _synchronize_periodic_boundary_of_vector_collection(
                self.rest_kappa, self.periodic_boundary_voronoi_idx
            )

        # Initialize the mixin class for symplectic time-stepper.
        _RodSymplecticStepperMixin.__init__(self)

    def _allocate_block_variables_in_nodes(self, systems: list[RodType]) -> None:
        """
        This function takes system collection and allocates the variables on
        node for block-structure and references allocated variables back to the
        systems.

        Parameters
        ----------
        systems

        Returns
        -------

        """

        # Things in nodes that are scalars
        #             0 ("mass", float64[:]),
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

        # Things in nodes that are vectors
        #             0 ("position_collection", float64[:, :]),
        #             1 ("internal_forces", float64[:, :]),
        #             2 ("external_forces", float64[:, :]),
        # 6 in total
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
        """
        This function takes system collection and allocates the variables on
        elements for block-structure and references allocated variables back to the
        systems.

        Parameters
        ----------
        systems

        Returns
        -------

        """

        # Things in elements that are scalars
        #             0 ("radius", float64[:]),
        #             1 ("volume", float64[:]),
        #             2 ("density", float64[:]),
        #             3 ("lengths", float64[:]),
        #             4 ("rest_lengths", float64[:]),
        #             5 ("dilatation", float64[:]),
        #             6 ("dilatation_rate", float64[:]),
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

        # Things in elements that are vectors
        #             0 ("tangents", float64[:, :]),
        #             1 ("sigma", float64[:, :]),
        #             2 ("rest_sigma", float64[:, :]),
        #             3 ("internal_torques", float64[:, :]),
        #             4 ("external_torques", float64[:, :]),
        #             6 ("internal_stress", float64[:, :]),
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

        # Things in elements that are matrices
        #             0 ("director_collection", float64[:, :, :]),
        #             1 ("mass_second_moment_of_inertia", float64[:, :, :]),
        #             2 ("inv_mass_second_moment_of_inertia", float64[:, :, :]),
        #             3 ("shear_matrix", float64[:, :, :]),
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
        """
        This function takes system collection and allocates the variables on
        voronoi for block-structure and references allocated variables back to the
        systems.

        Parameters
        ----------
        systems

        Returns
        -------

        """

        # Things in voronoi that are scalars
        #             0 ("voronoi_dilatation", float64[:]),
        #             1 ("rest_voronoi_lengths", float64[:]),
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

        # Things in voronoi that are vectors
        #             0 ("kappa", float64[:, :]),
        #             1 ("rest_kappa", float64[:, :]),
        #             2 ("internal_couple", float64[:, :]),
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

        # Things in voronoi that are matrices
        #             0 ("bend_matrix", float64[:, :, :]),
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
        """
        This function takes system collection and allocates the variables used by symplectic
        stepper for block-structure and references allocated variables back to the systems.
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

        This method take domain types (node, element, voronoi) and value
        types (scalar, vector, tensor) as inputs and compute internally how to
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

        # typedef
        start_idx_list: np.ndarray
        end_idx_list: np.ndarray
        periodic_boundary_idx: np.ndarray
        synchronize_periodic_boundary: Callable
        domain_num: int
        view_shape: tuple[int, ...]

        if domain_type == "node":
            start_idx_list = self.start_idx_in_rod_nodes.view()
            end_idx_list = self.end_idx_in_rod_nodes.view()
            domain_num = self.n_nodes
            periodic_boundary_idx = self.periodic_boundary_nodes_idx.view()

        elif domain_type == "element":
            start_idx_list = self.start_idx_in_rod_elems.view()
            end_idx_list = self.end_idx_in_rod_elems.view()
            domain_num = self.n_elems
            periodic_boundary_idx = self.periodic_boundary_elems_idx.view()

        elif domain_type == "voronoi":
            start_idx_list = self.start_idx_in_rod_voronoi.view()
            end_idx_list = self.end_idx_in_rod_voronoi.view()
            domain_num = self.n_voronoi
            periodic_boundary_idx = self.periodic_boundary_voronoi_idx.view()

        else:
            raise ValueError(
                "Incorrect domain type. Must be one of node, element, and voronoi"
            )

        if value_type == "scalar":
            view_shape = (domain_num,)
            synchronize_periodic_boundary = (
                _synchronize_periodic_boundary_of_scalar_collection
            )

        elif value_type == "vector":
            view_shape = (3, domain_num)
            synchronize_periodic_boundary = (
                _synchronize_periodic_boundary_of_vector_collection
            )

        elif value_type == "tensor":
            view_shape = (3, 3, domain_num)
            synchronize_periodic_boundary = (
                _synchronize_periodic_boundary_of_matrix_collection
            )

        else:
            raise ValueError(
                "Incorrect value type. Must be one of scalar, vector, and tensor."
            )

        for k, v in mapping_dict.items():
            # Map class attributes to block memory
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

            # Synchronize periodic boundaries
            synchronize_periodic_boundary(self.__dict__[k], periodic_boundary_idx)
