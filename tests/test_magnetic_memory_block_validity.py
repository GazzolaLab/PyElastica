import numpy as np
from numpy.testing import assert_allclose
from elastica._linalg import _batch_matvec
from elastica.memory_block.memory_block_magnetic_rod import MemoryBlockMagneticRod
import pytest
from elastica.utils import Tolerance


class MagneticMockRod:
    def __init__(self, n_elems):
        self.n_elems = n_elems  # np.random.randint(10, 30 + 1)
        self.n_nodes = self.n_elems + 1
        self.n_voronoi = self.n_elems - 1

        # Things that are scalar mapped on nodes
        self.mass = np.random.randn(self.n_nodes)

        # Things that are vectors mapped on nodes
        self.position_collection = np.random.randn(3, self.n_nodes)
        self.velocity_collection = np.random.randn(3, self.n_nodes)
        self.acceleration_collection = np.random.randn(3, self.n_nodes)
        self.internal_forces = np.random.randn(3, self.n_nodes)
        self.external_forces = np.random.randn(3, self.n_nodes)
        self.damping_forces = np.random.randn(3, self.n_nodes)

        # Things that are scalar mapped on elements
        self.radius = np.random.rand(self.n_elems)
        self.volume = np.random.rand(self.n_elems)
        self.density = np.random.rand(self.n_elems)
        self.lengths = np.random.rand(self.n_elems)
        self.rest_lengths = self.lengths.copy()
        self.dilatation = np.random.rand(self.n_elems)
        self.dilatation_rate = np.random.rand(self.n_elems)
        self.dissipation_constant_for_forces = np.random.rand(self.n_elems)
        self.dissipation_constant_for_torques = np.random.rand(self.n_elems)

        # Things that are vector mapped on elements
        self.omega_collection = np.random.randn(3, self.n_elems)
        self.alpha_collection = np.random.randn(3, self.n_elems)
        self.tangents = np.random.randn(3, self.n_elems)
        self.sigma = np.random.randn(3, self.n_elems)
        self.rest_sigma = np.random.randn(3, self.n_elems)
        self.internal_torques = np.random.randn(3, self.n_elems)
        self.external_torques = np.random.randn(3, self.n_elems)
        self.damping_torques = np.random.randn(3, self.n_elems)
        self.internal_stress = np.random.randn(3, self.n_elems)

        # Things that are matrix mapped on elements
        self.director_collection = np.zeros((3, 3, self.n_elems))
        for i in range(3):
            for j in range(3):
                self.director_collection[i, j, ...] = 3 * i + j
        # self.director_collection *= np.random.randn()
        self.mass_second_moment_of_inertia = np.random.randn() * np.ones(
            (3, 3, self.n_elems)
        )
        self.inv_mass_second_moment_of_inertia = np.random.randn() * np.ones(
            (3, 3, self.n_elems)
        )
        self.shear_matrix = np.random.randn() * np.ones((3, 3, self.n_elems))

        # Things that are scalar mapped on voronoi
        self.voronoi_dilatation = np.random.rand(self.n_voronoi)
        self.rest_voronoi_lengths = np.random.rand(self.n_voronoi)

        # Things that are vectors mapped on voronoi
        self.kappa = np.random.randn(3, self.n_voronoi)
        self.rest_kappa = np.random.randn(3, self.n_voronoi)
        self.internal_couple = np.random.randn(3, self.n_voronoi)

        # Things that are matrix mapped on voronoi
        self.bend_matrix = np.random.randn() * np.ones((3, 3, self.n_voronoi))

        # self.density = fake_idx
        magnetization_direction = np.random.randn(3,self.n_elems)
        magnetization_direction_in_material_frame = _batch_matvec(
            self.director_collection, magnetization_direction
        )
        magnetization_density = np.random.randn(self.n_elems)
        self.magnetization_collection = (
            magnetization_density * self.volume * magnetization_direction_in_material_frame
        )




"""
Most of the magnetic memory block derived from the Magnetic rod.
Only test the vectors on elements part.
"""
@pytest.mark.parametrize("n_rods", [1, 2, 5, 6])
def test_magnetic_block_structure_vectors_on_elements_validity(n_rods):
    """
    This function is testing validity of vectors on elements. It is been
    tested that for vector element variables, if the block structure memory
    and values are set correctly and rods that belong to rod structure
    share the memory with block structure.

    Parameters
    ----------
    n_rods

    Returns
    -------

    """
    world_rods = [MagneticMockRod(np.random.randint(10, 30 + 1)) for _ in range(n_rods)]
    block_structure = MemoryBlockMagneticRod(world_rods)

    for i in range(n_rods):
        start_idx = block_structure.start_idx_in_rod_elems[i]
        end_idx = block_structure.end_idx_in_rod_elems[i]

        # tangents
        assert np.shares_memory(block_structure.tangents, world_rods[i].tangents)
        assert np.shares_memory(
            block_structure.vector_dofs_in_rod_elems, world_rods[i].tangents
        )
        assert_allclose(
            block_structure.tangents[..., start_idx:end_idx], world_rods[i].tangents
        )

        # sigma
        assert np.shares_memory(block_structure.sigma, world_rods[i].sigma)
        assert np.shares_memory(
            block_structure.vector_dofs_in_rod_elems, world_rods[i].sigma
        )
        assert_allclose(
            block_structure.sigma[..., start_idx:end_idx], world_rods[i].sigma
        )

        # rest sigma
        assert np.shares_memory(block_structure.rest_sigma, world_rods[i].rest_sigma)
        assert np.shares_memory(
            block_structure.vector_dofs_in_rod_elems, world_rods[i].rest_sigma
        )
        assert_allclose(
            block_structure.rest_sigma[..., start_idx:end_idx], world_rods[i].rest_sigma
        )

        # internal torques
        assert np.shares_memory(
            block_structure.internal_torques, world_rods[i].internal_torques
        )
        assert np.shares_memory(
            block_structure.vector_dofs_in_rod_elems, world_rods[i].internal_torques
        )
        assert_allclose(
            block_structure.internal_torques[..., start_idx:end_idx],
            world_rods[i].internal_torques,
        )

        # external torques
        assert np.shares_memory(
            block_structure.external_torques, world_rods[i].external_torques
        )
        assert np.shares_memory(
            block_structure.vector_dofs_in_rod_elems, world_rods[i].external_torques
        )
        assert_allclose(
            block_structure.external_torques[..., start_idx:end_idx],
            world_rods[i].external_torques,
        )

        # damping torques
        assert np.shares_memory(
            block_structure.damping_torques, world_rods[i].damping_torques
        )
        assert np.shares_memory(
            block_structure.vector_dofs_in_rod_elems, world_rods[i].damping_torques
        )
        assert_allclose(
            block_structure.damping_torques[..., start_idx:end_idx],
            world_rods[i].damping_torques,
        )

        # internal stress
        assert np.shares_memory(
            block_structure.internal_stress, world_rods[i].internal_stress
        )
        assert np.shares_memory(
            block_structure.vector_dofs_in_rod_elems, world_rods[i].internal_stress
        )
        assert_allclose(
            block_structure.internal_stress[..., start_idx:end_idx],
            world_rods[i].internal_stress,
        )

        # magnetization_collection
        assert np.shares_memory(
            block_structure.magnetization_collection, world_rods[i].magnetization_collection
        )
        assert np.shares_memory(
            block_structure.vector_dofs_in_rod_elems, world_rods[i].magnetization_collection
        )
        assert_allclose(
            block_structure.magnetization_collection[..., start_idx:end_idx],
            world_rods[i].magnetization_collection,
        )

if __name__ == "__main__":
    from pytest import main

    main([__file__])
