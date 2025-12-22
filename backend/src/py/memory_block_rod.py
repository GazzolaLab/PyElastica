"""Create block-structure class for collection of Cosserat rod systems."""
import numpy as np

from elastica.rod.cosserat_rod import CosseratRod
from elastica.rod.data_structures import _RodSymplecticStepperMixin
from elastica.typing import RodType, SystemIdxType

from elasticapp import BlockRodSystem

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

        # Get ghost indices from C++ block
        self.ghost_nodes_idx = np.array(self._block.ghost_nodes_idx, dtype=np.int32)
        self.ghost_elems_idx = np.array(self._block.ghost_elems_idx, dtype=np.int32)
        self.ghost_voronoi_idx = np.array(self._block.ghost_voronoi_idx, dtype=np.int32)

        # Compute metadata from block and element counts
        # n_elems is total elements including ghost elements
        # For n rods, there are (n-1) ghost nodes, and 2*(n-1) ghost elements
        self.n_elems = int(np.sum(n_elems_straight_rods) + 2 * (len(systems) - 1))
        self.n_nodes = self.n_elems + 1
        self.n_voronoi = self.n_elems - 1
        self.n_rods = len(systems)

        for idx, system in enumerate(systems):
            self.relink_system_properties_to_block_memory(system, idx)
        self.define_symplectic_stepper_variables()

        # Note: The C++ block constructor calls reset_ghost() which sets all ghosts to default values
        # If additional ghosting mechanism are needed, they can be added here. (like ring rods)

        # Compute strains for the block
        self._block.compute_internal_forces_and_torques()

    def relink_system_properties_to_block_memory(
        self, system: RodType, system_idx: int,
    ) -> None:
        """Relink system properties to block memory."""
        for py_key, cpp_key in PY2CPP_VARNAMES.items():
            assert hasattr(system, py_key), f"System {system} does not have attribute {py_key}."
            value = getattr(system, py_key).copy()
            block_memory = self._block.at(system_idx).get(cpp_key)
            block_memory[...] = value
            setattr(system, py_key, block_memory)

            full_block_memory = self._block.get(cpp_key)
            setattr(self, py_key, full_block_memory)
