import numpy as np
from numpy.testing import assert_allclose

from elastica.memory_block.memory_block_rod import MemoryBlockCosseratRod
import pytest
from elastica.utils import Tolerance


class MockRod:
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


@pytest.mark.parametrize("n_rods", [1, 2, 5, 6])
def test_block_structure_scalar_on_nodes_validity(n_rods):
    """
    This function is testing validity of scalars on nodes. It is been
    tested that for scalar node variables, if the block structure memory
    and values are set correctly and rods that belong to rod structure
    share the memory with block structure.

    Parameters
    ----------
    n_rods

    Returns
    -------

    """

    world_rods = [MockRod(np.random.randint(10, 30 + 1)) for _ in range(n_rods)]
    block_structure = MemoryBlockCosseratRod(world_rods)

    for i in range(n_rods):
        start_idx = block_structure.start_idx_in_rod_nodes[i]
        end_idx = block_structure.end_idx_in_rod_nodes[i]

        assert np.shares_memory(block_structure.mass, world_rods[i].mass)
        assert np.shares_memory(
            block_structure.scalar_dofs_in_rod_nodes, world_rods[i].mass
        )
        assert_allclose(
            block_structure.mass[start_idx:end_idx],
            world_rods[i].mass,
            atol=Tolerance.atol(),
        )


@pytest.mark.parametrize("n_rods", [1, 2, 5, 6])
def test_block_structure_vectors_on_nodes_validity(n_rods):
    """
    This function is testing validity of vectors on nodes. It is been
    tested that for vectors node variables, if the block structure memory
    and values are set correctly and rods that belong to rod structure
    share the memory with block structure.

    Parameters
    ----------
    n_rods

    Returns
    -------

    """
    world_rods = [MockRod(np.random.randint(10, 30 + 1)) for _ in range(n_rods)]
    block_structure = MemoryBlockCosseratRod(world_rods)

    for i in range(n_rods):
        start_idx = block_structure.start_idx_in_rod_nodes[i]
        end_idx = block_structure.end_idx_in_rod_nodes[i]

        # position collection
        assert np.shares_memory(
            block_structure.position_collection, world_rods[i].position_collection
        )
        assert np.shares_memory(
            block_structure.vector_dofs_in_rod_nodes, world_rods[i].position_collection
        )
        assert_allclose(
            block_structure.position_collection[..., start_idx:end_idx],
            world_rods[i].position_collection,
        )

        # internal forces
        assert np.shares_memory(
            block_structure.internal_forces, world_rods[i].internal_forces
        )
        assert np.shares_memory(
            block_structure.vector_dofs_in_rod_nodes, world_rods[i].internal_forces
        )
        assert_allclose(
            block_structure.internal_forces[..., start_idx:end_idx],
            world_rods[i].internal_forces,
        )

        # external forces
        assert np.shares_memory(
            block_structure.external_forces, world_rods[i].external_forces
        )
        assert np.shares_memory(
            block_structure.vector_dofs_in_rod_nodes, world_rods[i].external_forces
        )
        assert_allclose(
            block_structure.external_forces[..., start_idx:end_idx],
            world_rods[i].external_forces,
        )

        # damping forces
        assert np.shares_memory(
            block_structure.damping_forces, world_rods[i].damping_forces
        )
        assert np.shares_memory(
            block_structure.vector_dofs_in_rod_nodes, world_rods[i].damping_forces
        )
        assert_allclose(
            block_structure.damping_forces[..., start_idx:end_idx],
            world_rods[i].damping_forces,
        )


@pytest.mark.parametrize("n_rods", [1, 2, 5, 6])
def test_block_structure_scalar_on_elements_validity(n_rods):
    """
    This function is testing validity of scalars on elements. It is been
    tested that for scalar element variables, if the block structure memory
    and values are set correctly and rods that belong to rod structure
    share the memory with block structure.

    Parameters
    ----------
    n_rods

    Returns
    -------

    """

    world_rods = [MockRod(np.random.randint(10, 30 + 1)) for _ in range(n_rods)]
    block_structure = MemoryBlockCosseratRod(world_rods)

    for i in range(n_rods):
        start_idx = block_structure.start_idx_in_rod_elems[i]
        end_idx = block_structure.end_idx_in_rod_elems[i]

        # radius
        assert np.shares_memory(block_structure.radius, world_rods[i].radius)
        assert np.shares_memory(
            block_structure.scalar_dofs_in_rod_elems, world_rods[i].radius
        )
        assert_allclose(block_structure.radius[start_idx:end_idx], world_rods[i].radius)

        # volume
        assert np.shares_memory(block_structure.volume, world_rods[i].volume)
        assert np.shares_memory(
            block_structure.scalar_dofs_in_rod_elems, world_rods[i].volume
        )
        assert_allclose(block_structure.volume[start_idx:end_idx], world_rods[i].volume)

        # density
        assert np.shares_memory(block_structure.density, world_rods[i].density)
        assert np.shares_memory(
            block_structure.scalar_dofs_in_rod_elems, world_rods[i].density
        )
        assert_allclose(
            block_structure.density[start_idx:end_idx], world_rods[i].density
        )

        # lengths
        assert np.shares_memory(block_structure.lengths, world_rods[i].lengths)
        assert np.shares_memory(
            block_structure.scalar_dofs_in_rod_elems, world_rods[i].lengths
        )
        assert_allclose(
            block_structure.lengths[start_idx:end_idx], world_rods[i].lengths
        )

        # rest lengths
        assert np.shares_memory(
            block_structure.rest_lengths, world_rods[i].rest_lengths
        )
        assert np.shares_memory(
            block_structure.scalar_dofs_in_rod_elems, world_rods[i].rest_lengths
        )
        assert_allclose(
            block_structure.rest_lengths[start_idx:end_idx], world_rods[i].rest_lengths
        )

        # dilatation
        assert np.shares_memory(block_structure.dilatation, world_rods[i].dilatation)
        assert np.shares_memory(
            block_structure.scalar_dofs_in_rod_elems, world_rods[i].dilatation
        )
        assert_allclose(
            block_structure.dilatation[start_idx:end_idx], world_rods[i].dilatation
        )

        # dilatation rate
        assert np.shares_memory(
            block_structure.dilatation_rate, world_rods[i].dilatation_rate
        )
        assert np.shares_memory(
            block_structure.scalar_dofs_in_rod_elems, world_rods[i].dilatation_rate
        )
        assert_allclose(
            block_structure.dilatation_rate[start_idx:end_idx],
            world_rods[i].dilatation_rate,
        )

        # dissipation constant for forces
        assert np.shares_memory(
            block_structure.dissipation_constant_for_forces,
            world_rods[i].dissipation_constant_for_forces,
        )
        assert np.shares_memory(
            block_structure.scalar_dofs_in_rod_elems,
            world_rods[i].dissipation_constant_for_forces,
        )
        assert_allclose(
            block_structure.dissipation_constant_for_forces[start_idx:end_idx],
            world_rods[i].dissipation_constant_for_forces,
        )

        # dissipation constant for torques
        assert np.shares_memory(
            block_structure.dissipation_constant_for_torques,
            world_rods[i].dissipation_constant_for_torques,
        )
        assert np.shares_memory(
            block_structure.scalar_dofs_in_rod_elems,
            world_rods[i].dissipation_constant_for_torques,
        )
        assert_allclose(
            block_structure.dissipation_constant_for_torques[start_idx:end_idx],
            world_rods[i].dissipation_constant_for_torques,
        )


@pytest.mark.parametrize("n_rods", [1, 2, 5, 6])
def test_block_structure_vectors_on_elements_validity(n_rods):
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
    world_rods = [MockRod(np.random.randint(10, 30 + 1)) for _ in range(n_rods)]
    block_structure = MemoryBlockCosseratRod(world_rods)

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


@pytest.mark.parametrize("n_rods", [1, 2, 5, 6])
def test_block_structure_matrices_on_elements_validity(n_rods):
    """
    This function is testing validity of matrices on elements. It is been
    tested that for matrices element variables, if the block structure memory
    and values are set correctly and rods that belong to rod structure
    share the memory with block structure.

    Parameters
    ----------
    n_rods

    Returns
    -------

    """
    world_rods = [MockRod(np.random.randint(10, 30 + 1)) for _ in range(n_rods)]
    block_structure = MemoryBlockCosseratRod(world_rods)

    for i in range(n_rods):
        start_idx = block_structure.start_idx_in_rod_elems[i]
        end_idx = block_structure.end_idx_in_rod_elems[i]

        # director collection
        assert np.shares_memory(
            block_structure.director_collection, world_rods[i].director_collection
        )
        assert np.shares_memory(
            block_structure.matrix_dofs_in_rod_elems, world_rods[i].director_collection
        )
        assert_allclose(
            block_structure.director_collection[..., start_idx:end_idx],
            world_rods[i].director_collection,
        )

        # mass second moment of inertia
        assert np.shares_memory(
            block_structure.mass_second_moment_of_inertia,
            world_rods[i].mass_second_moment_of_inertia,
        )
        assert np.shares_memory(
            block_structure.matrix_dofs_in_rod_elems,
            world_rods[i].mass_second_moment_of_inertia,
        )
        assert_allclose(
            block_structure.mass_second_moment_of_inertia[..., start_idx:end_idx],
            world_rods[i].mass_second_moment_of_inertia,
        )

        # inv mass second moment of inertia
        assert np.shares_memory(
            block_structure.inv_mass_second_moment_of_inertia,
            world_rods[i].inv_mass_second_moment_of_inertia,
        )
        assert np.shares_memory(
            block_structure.matrix_dofs_in_rod_elems,
            world_rods[i].inv_mass_second_moment_of_inertia,
        )
        assert_allclose(
            block_structure.inv_mass_second_moment_of_inertia[..., start_idx:end_idx],
            world_rods[i].inv_mass_second_moment_of_inertia,
        )

        # shear matrix
        assert np.shares_memory(
            block_structure.shear_matrix, world_rods[i].shear_matrix
        )
        assert np.shares_memory(
            block_structure.matrix_dofs_in_rod_elems, world_rods[i].shear_matrix
        )
        assert_allclose(
            block_structure.shear_matrix[..., start_idx:end_idx],
            world_rods[i].shear_matrix,
        )


@pytest.mark.parametrize("n_rods", [1, 2, 5, 6])
def test_block_structure_scalar_on_voronoi_validity(n_rods):
    """
    This function is testing validity of scalars on voronoi. It is been
    tested that for scalar voronoi variables, if the block structure memory
    and values are set correctly and rods that belong to rod structure
    share the memory with block structure.

    Parameters
    ----------
    n_rods

    Returns
    -------

    """
    world_rods = [MockRod(np.random.randint(10, 30 + 1)) for _ in range(n_rods)]
    block_structure = MemoryBlockCosseratRod(world_rods)

    for i in range(n_rods):
        start_idx = block_structure.start_idx_in_rod_voronoi[i]
        end_idx = block_structure.end_idx_in_rod_voronoi[i]

        # voronoi dilatation
        assert np.shares_memory(
            block_structure.voronoi_dilatation, world_rods[i].voronoi_dilatation
        )
        assert np.shares_memory(
            block_structure.scalar_dofs_in_rod_voronois,
            world_rods[i].voronoi_dilatation,
        )
        assert_allclose(
            block_structure.voronoi_dilatation[start_idx:end_idx],
            world_rods[i].voronoi_dilatation,
        )

        # rest voronoi lengths
        assert np.shares_memory(
            block_structure.rest_voronoi_lengths, world_rods[i].rest_voronoi_lengths
        )
        assert np.shares_memory(
            block_structure.scalar_dofs_in_rod_voronois,
            world_rods[i].rest_voronoi_lengths,
        )
        assert_allclose(
            block_structure.rest_voronoi_lengths[start_idx:end_idx],
            world_rods[i].rest_voronoi_lengths,
        )


@pytest.mark.parametrize("n_rods", [1, 2, 5, 6])
def test_block_structure_vectors_on_voronoi_validity(n_rods):
    """
    This function is testing validity of vectors on voronoi. It is been
    tested that for vectors voronoi variables, if the block structure memory
    and values are set correctly and rods that belong to rod structure
    share the memory with block structure.

    Parameters
    ----------
    n_rods

    Returns
    -------

    """
    world_rods = [MockRod(np.random.randint(10, 30 + 1)) for _ in range(n_rods)]
    block_structure = MemoryBlockCosseratRod(world_rods)

    for i in range(n_rods):
        start_idx = block_structure.start_idx_in_rod_voronoi[i]
        end_idx = block_structure.end_idx_in_rod_voronoi[i]

        # kappa
        assert np.shares_memory(block_structure.kappa, world_rods[i].kappa)
        assert np.shares_memory(
            block_structure.vector_dofs_in_rod_voronois, world_rods[i].kappa
        )
        assert_allclose(
            block_structure.kappa[..., start_idx:end_idx], world_rods[i].kappa
        )

        # rest kappa
        assert np.shares_memory(block_structure.rest_kappa, world_rods[i].rest_kappa)
        assert np.shares_memory(
            block_structure.vector_dofs_in_rod_voronois, world_rods[i].rest_kappa
        )
        assert_allclose(
            block_structure.rest_kappa[..., start_idx:end_idx], world_rods[i].rest_kappa
        )

        # internal couple
        assert np.shares_memory(
            block_structure.internal_couple, world_rods[i].internal_couple
        )
        assert np.shares_memory(
            block_structure.vector_dofs_in_rod_voronois, world_rods[i].internal_couple
        )
        assert_allclose(
            block_structure.internal_couple[..., start_idx:end_idx],
            world_rods[i].internal_couple,
        )


@pytest.mark.parametrize("n_rods", [1, 2, 5, 6])
def test_block_structure_matrices_on_voronoi_validity(n_rods):
    """
    This function is testing validity of matrices on voronoi. It is been
    tested that for matrices voronoi variables, if the block structure memory
    and values are set correctly and rods that belong to rod structure
    share the memory with block structure.

    Parameters
    ----------
    n_rods

    Returns
    -------

    """
    world_rods = [MockRod(np.random.randint(10, 30 + 1)) for _ in range(n_rods)]
    block_structure = MemoryBlockCosseratRod(world_rods)

    for i in range(n_rods):
        start_idx = block_structure.start_idx_in_rod_voronoi[i]
        end_idx = block_structure.end_idx_in_rod_voronoi[i]

        # bend matrix
        assert np.shares_memory(block_structure.bend_matrix, world_rods[i].bend_matrix)
        assert np.shares_memory(
            block_structure.matrix_dofs_in_rod_voronois, world_rods[i].bend_matrix
        )
        assert_allclose(
            block_structure.bend_matrix[..., start_idx:end_idx],
            world_rods[i].bend_matrix,
        )


@pytest.mark.parametrize("n_rods", [1, 2, 5, 6])
def test_block_structure_rate_collection_validity(n_rods):
    """
    This function is testing validity of rate collection vectors.
    Rate collection contains, velocity_collection, omega_collection,
    acceleration_collection and alpha_collection. Here we pack them
    together because it is more efficient for time-stepper. It is been
    tested that for vectors node variables, if the block structure memory
    and values are set correctly and rods that belong to rod structure
    share the memory with block structure.

    Parameters
    ----------
    n_rods

    Returns
    -------

    """
    world_rods = [MockRod(np.random.randint(10, 30 + 1)) for _ in range(n_rods)]
    block_structure = MemoryBlockCosseratRod(world_rods)

    # Test vectors on nodes
    for i in range(n_rods):
        start_idx = block_structure.start_idx_in_rod_nodes[i]
        end_idx = block_structure.end_idx_in_rod_nodes[i]

        # velocity collection
        assert np.shares_memory(
            block_structure.velocity_collection, world_rods[i].velocity_collection
        )
        assert np.shares_memory(
            block_structure.rate_collection, world_rods[i].velocity_collection
        )
        assert_allclose(
            block_structure.velocity_collection[..., start_idx:end_idx],
            world_rods[i].velocity_collection,
        )

        # acceleration collection
        assert np.shares_memory(
            block_structure.acceleration_collection,
            world_rods[i].acceleration_collection,
        )
        assert np.shares_memory(
            block_structure.rate_collection,
            world_rods[i].acceleration_collection,
        )
        assert_allclose(
            block_structure.acceleration_collection[..., start_idx:end_idx],
            world_rods[i].acceleration_collection,
        )

    # Test vectors on elements
    for i in range(n_rods):
        start_idx = block_structure.start_idx_in_rod_elems[i]
        end_idx = block_structure.end_idx_in_rod_elems[i]

        # omega collection
        assert np.shares_memory(
            block_structure.omega_collection, world_rods[i].omega_collection
        )
        assert np.shares_memory(
            block_structure.rate_collection, world_rods[i].omega_collection
        )
        assert_allclose(
            block_structure.omega_collection[..., start_idx:end_idx],
            world_rods[i].omega_collection,
        )

        # alpha collection
        assert np.shares_memory(
            block_structure.alpha_collection, world_rods[i].alpha_collection
        )
        assert np.shares_memory(
            block_structure.rate_collection, world_rods[i].alpha_collection
        )
        assert_allclose(
            block_structure.alpha_collection[..., start_idx:end_idx],
            world_rods[i].alpha_collection,
        )

    # Validity of the rate collection array
    assert np.shares_memory(
        block_structure.rate_collection, block_structure.v_w_collection
    )
    assert np.shares_memory(
        block_structure.rate_collection, block_structure.dvdt_dwdt_collection
    )
    assert block_structure.v_w_collection.shape == (2, 3 * block_structure.n_nodes)
    assert block_structure.dvdt_dwdt_collection.shape == (
        2,
        3 * block_structure.n_nodes,
    )


if __name__ == "__main__":
    from pytest import main

    main([__file__])
