__doc__ = """Create block-structure class for collection of Cosserat rod systems."""
import numpy as np
import numpy.typing as npt
from numpy.typing import NDArray


def make_block_memory_metadata(
    n_elems_in_rods: NDArray[np.int32],
) -> tuple[
    int,
    NDArray[np.int32],
    NDArray[np.int32],
    NDArray[np.int32],
]:
    """
    This function, takes number of elements of each rod as a numpy array and computes,
    ghost nodes, elements and voronoi element indexes and numbers and returns it.

    Parameters
    ----------
    n_elems_in_rods: NDArray
        An integer array containing the number of elements in each of the n rod.

    Returns
    -------
    n_elems_with_ghosts: np.int32
        Total number of elements with ghost elements included. There are two ghost elements
        between each pair of two rods adjacent in memory block.
    ghost_nodes_idx: NDArray[np.int32]
        An integer array of length n - 1 containing the indices of ghost nodes in memory block.
    ghost_elements_idx: NDArray[np.int32]
        An integer array of length 2 * (n - 1) containing the indices of ghost elements in memory block.
    ghost_voronoi_idx: NDArray[np.int32]
        An integer array of length 2 * (n - 1) containing the indices of ghost Voronoi nodes in memory block.
    """
    n_nodes_in_rods = n_elems_in_rods + 1
    n_rods = n_elems_in_rods.shape[0]

    # Gap between two rods have one ghost node
    # n_nodes_with_ghosts = np.sum(n_nodes_in_rods) + (n_rods - 1)
    # Gap between two rods have two ghost elements : comes out to n_nodes_with_ghosts - 1
    n_elems_with_ghosts = np.sum(n_elems_in_rods) + 2 * (n_rods - 1)
    # Gap between two rods have three ghost voronois : comes out to n_nodes_with_ghosts - 2
    # n_voronoi_with_ghosts = np.sum(n_voronois_in_rods) + 3 * (n_rods - 1)

    ghost_nodes_idx = np.cumsum(n_nodes_in_rods[:-1], dtype=np.int32)
    # Add [0, 1, 2, ... n_rods-2] to the ghost_nodes idx to accommodate miscounting
    ghost_nodes_idx += np.arange(n_rods - 1)

    ghost_elems_idx = np.zeros((2 * (n_rods - 1),), dtype=np.int32)
    ghost_elems_idx[::2] = ghost_nodes_idx - 1
    ghost_elems_idx[1::2] = ghost_nodes_idx.copy()

    ghost_voronoi_idx = np.zeros((3 * (n_rods - 1),), dtype=np.int32)
    ghost_voronoi_idx[::3] = ghost_nodes_idx - 2
    ghost_voronoi_idx[1::3] = ghost_nodes_idx - 1
    ghost_voronoi_idx[2::3] = ghost_nodes_idx.copy()

    return int(n_elems_with_ghosts), ghost_nodes_idx, ghost_elems_idx, ghost_voronoi_idx


def make_block_memory_periodic_boundary_metadata(
    n_elems_in_rods: NDArray[np.int32],
) -> tuple[
    NDArray[np.int32],
    NDArray[np.int32],
    NDArray[np.int32],
    NDArray[np.int32],
]:
    """
    This function, takes the number of elements of ring rods and computes the periodic boundary node,
    element and voronoi index.

    Parameters
    ----------
    n_elems_in_rods : NDArray
        1D (n_ring_rods,) array containing data with 'int' type. Elements of this array contains total number of
         elements of one rod, including periodic boundary elements.

    Returns
    -------
    n_elems

    periodic_boundary_node : NDArray
        2D (2, n_periodic_boundary_nodes) array containing data with 'int' type. Vector containing periodic boundary
        elements index. First dimension is the periodic boundary index, second dimension is the referenced cell index.

    periodic_boundary_elems_idx : NDArray
        2D (2, n_periodic_boundary_elems) array containing data with 'int' type. Vector containing periodic boundary
        nodes index. First dimension is the periodic boundary index, second dimension is the referenced cell index.

    periodic_boundary_voronoi_idx : NDArray
        2D (2, n_periodic_boundary_voronoi) array containing data with 'int' type. Vector containing periodic boundary
        voronoi index. First dimension is the periodic boundary index, second dimension is the referenced cell index.

    """

    n_elem: NDArray[np.int32] = n_elems_in_rods.copy()
    n_rods = n_elems_in_rods.shape[0]

    periodic_boundary_node_idx = np.zeros((2, 3 * n_rods), dtype=np.int32)
    # count ghost nodes, first rod does not have a ghost node at the start, so exclude first rod.
    periodic_boundary_node_idx[0, 0::3][1:] = 1
    # This is for the first periodic node at the end
    periodic_boundary_node_idx[0, 1::3] = 1 + n_elem
    # This is for the second periodic node at the end
    periodic_boundary_node_idx[0, 2::3] = 1
    periodic_boundary_node_idx[0, :] = np.cumsum(periodic_boundary_node_idx[0, :])
    # Add [0, 1, 2, ..., n_rods] to the periodic boundary nodes to accommodate miscounting
    periodic_boundary_node_idx[0, :] += np.repeat(np.arange(n_rods), 3)
    # Now fill the reference node idx, to copy and correct periodic boundary nodes
    # First fill with the reference node idx of the first periodic node. This is the last node of the actual rod
    # (without ghost and periodic nodes).
    periodic_boundary_node_idx[1, 0::3] = periodic_boundary_node_idx[0, 1::3] - 1
    # Second fill with the reference node idx of the second periodic node. This is the first node of the actual rod
    # (without ghost and periodic nodes).
    periodic_boundary_node_idx[1, 1::3] = periodic_boundary_node_idx[0, 0::3] + 1
    # Third fill with the reference node idx of the third periodic node. This is the second node of the actual rod
    # (without ghost and periodic nodes).
    periodic_boundary_node_idx[1, 2::3] = periodic_boundary_node_idx[0, 0::3] + 2

    periodic_boundary_elems_idx = np.zeros((2, 2 * n_rods), dtype=np.int32)
    # count ghost elems, first rod does not have a ghost elem at the start, so exclude first rod.
    periodic_boundary_elems_idx[0, 0::2][1:] = 2
    # This is for the first periodic elem at the end
    periodic_boundary_elems_idx[0, 1::2] = 1 + n_elem
    periodic_boundary_elems_idx[0, :] = np.cumsum(periodic_boundary_elems_idx[0, :])
    # Add [0, 1, 2, ..., n_rods] to the periodic boundary elems to accommodate miscounting
    periodic_boundary_elems_idx[0, :] += np.repeat(np.arange(n_rods), 2)
    # Now fill the reference element idx, to copy and correct periodic boundary elements
    # First fill with the reference element idx of the first periodic element. This is the last element of the actual
    # rod
    # (without ghost and periodic elements).
    periodic_boundary_elems_idx[1, 0::2] = periodic_boundary_elems_idx[0, 1::2] - 1
    # Second fill with the reference element idx of the second periodic element. This is the first element of the actual
    # rod
    # (without ghost and periodic elements).
    periodic_boundary_elems_idx[1, 1::2] = periodic_boundary_elems_idx[0, 0::2] + 1

    periodic_boundary_voronoi_idx = np.zeros((2, n_rods), dtype=np.int32)
    # count ghost voronoi, first rod does not have a ghost voronoi at the start, so exclude first rod.
    periodic_boundary_voronoi_idx[0, 0::1][1:] = 3
    # This is for the first periodic voronoi at the end
    periodic_boundary_voronoi_idx[0, 1:] += n_elem[:-1]
    periodic_boundary_voronoi_idx[0, :] = np.cumsum(periodic_boundary_voronoi_idx[0, :])
    # Add [0, 1, 2, ..., n_rods] to the periodic boundary voronoi to accommodate miscounting
    periodic_boundary_voronoi_idx[0, :] += np.repeat(np.arange(0, n_rods), 1)
    # Now fill the reference voronoi idx, to copy and correct periodic boundary voronoi
    # Fill with the reference voronoi idx of the  periodic voronoi. This is the last voronoi of the actual rod
    # (without ghost and periodic voronoi).
    periodic_boundary_voronoi_idx[1, :] = (
        periodic_boundary_voronoi_idx[0, :] + n_elem[:]
    )

    # Increase the n_elem in rods by 2 because we are adding two periodic boundary elements
    n_elem = n_elem + 2

    return (
        n_elem,
        periodic_boundary_node_idx,
        periodic_boundary_elems_idx,
        periodic_boundary_voronoi_idx,
    )
