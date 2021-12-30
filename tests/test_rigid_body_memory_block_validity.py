import numpy as np
from numpy.testing import assert_allclose

from elastica.memory_block.memory_block_rigid_body import MemoryBlockRigidBody
import pytest
from elastica.utils import Tolerance


class MockRigidBody:
    def __init__(self):

        self.radius = np.random.randn()
        self.length = np.random.randn()
        self.density = np.random.randn()
        self.volume = np.random.randn()
        self.mass = np.random.randn()

        self.position_collection = np.random.randn(3, 1)
        self.velocity_collection = np.random.randn(3, 1)
        self.acceleration_collection = np.random.randn(3, 1)
        self.omega_collection = np.random.randn(3, 1)
        self.alpha_collection = np.random.randn(3, 1)
        self.director_collection = np.random.randn(3, 3, 1)

        self.external_forces = np.random.randn(3, 1)
        self.external_torques = np.random.randn(3, 1)

        self.mass_second_moment_of_inertia = np.random.randn(3, 3, 1)
        self.inv_mass_second_moment_of_inertia = np.random.randn(3, 3, 1)


@pytest.mark.parametrize("n_rods", [1, 2, 5, 6])
def test_block_structure_scalar_validity(n_rods):
    """
    This function is testing validity of scalars. It is been tested that
    for scalar variables, if the block structure memory and values are
    set correctly and rods that belong to rod structure share the memory
    with block structure.

    Parameters
    ----------
    n_rods

    Returns
    -------

    """

    world_bodies = [MockRigidBody() for _ in range(n_rods)]
    block_structure = MemoryBlockRigidBody(world_bodies)

    for i in range(n_rods):

        # radius
        assert np.shares_memory(block_structure.radius, world_bodies[i].radius)
        assert np.shares_memory(
            block_structure.scalar_dofs_in_rigid_bodies, world_bodies[i].radius
        )
        assert_allclose(
            block_structure.radius[i : i + 1],
            world_bodies[i].radius,
            atol=Tolerance.atol(),
        )

        # length
        assert np.shares_memory(block_structure.length, world_bodies[i].length)
        assert np.shares_memory(
            block_structure.scalar_dofs_in_rigid_bodies, world_bodies[i].length
        )
        assert_allclose(
            block_structure.length[i : i + 1],
            world_bodies[i].length,
            atol=Tolerance.atol(),
        )

        # density
        assert np.shares_memory(block_structure.density, world_bodies[i].density)
        assert np.shares_memory(
            block_structure.scalar_dofs_in_rigid_bodies, world_bodies[i].density
        )
        assert_allclose(
            block_structure.density[i : i + 1],
            world_bodies[i].density,
            atol=Tolerance.atol(),
        )

        # volume
        assert np.shares_memory(block_structure.volume, world_bodies[i].volume)
        assert np.shares_memory(
            block_structure.scalar_dofs_in_rigid_bodies, world_bodies[i].volume
        )
        assert_allclose(
            block_structure.volume[i : i + 1],
            world_bodies[i].volume,
            atol=Tolerance.atol(),
        )

        # mass
        assert np.shares_memory(block_structure.mass, world_bodies[i].mass)
        assert np.shares_memory(
            block_structure.scalar_dofs_in_rigid_bodies, world_bodies[i].mass
        )
        assert_allclose(
            block_structure.mass[i : i + 1],
            world_bodies[i].mass,
            atol=Tolerance.atol(),
        )


@pytest.mark.parametrize("n_rods", [1, 2, 5, 6])
def test_block_structure_vectors_validity(n_rods):
    """
    This function is testing validity of vectors. It is been
    tested that for vector variables, if the block structure memory
    and values are set correctly and rods that belong to rod structure
    share the memory with block structure.

    Parameters
    ----------
    n_rods

    Returns
    -------

    """
    world_bodies = [MockRigidBody() for _ in range(n_rods)]
    block_structure = MemoryBlockRigidBody(world_bodies)

    for i in range(n_rods):

        # position collection
        assert np.shares_memory(
            block_structure.position_collection, world_bodies[i].position_collection
        )
        assert np.shares_memory(
            block_structure.vector_dofs_in_rigid_bodies,
            world_bodies[i].position_collection,
        )
        assert_allclose(
            block_structure.position_collection[..., i : i + 1],
            world_bodies[i].position_collection,
        )

        # external forces
        assert np.shares_memory(
            block_structure.external_forces, world_bodies[i].external_forces
        )
        assert np.shares_memory(
            block_structure.vector_dofs_in_rigid_bodies, world_bodies[i].external_forces
        )
        assert_allclose(
            block_structure.external_forces[..., i : i + 1],
            world_bodies[i].external_forces,
        )

        # external torques
        assert np.shares_memory(
            block_structure.external_torques, world_bodies[i].external_torques
        )
        assert np.shares_memory(
            block_structure.vector_dofs_in_rigid_bodies,
            world_bodies[i].external_torques,
        )
        assert_allclose(
            block_structure.external_torques[..., i : i + 1],
            world_bodies[i].external_torques,
        )


@pytest.mark.parametrize("n_rods", [1, 2, 5, 6])
def test_block_structure_matrix_validity(n_rods):
    """
    This function is testing validity of matrices. It is been
    tested that for matrix variables, if the block structure memory
    and values are set correctly and rods that belong to rod structure
    share the memory with block structure.

    Parameters
    ----------
    n_rods

    Returns
    -------

    """
    world_bodies = [MockRigidBody() for _ in range(n_rods)]
    block_structure = MemoryBlockRigidBody(world_bodies)

    for i in range(n_rods):
        # director collection
        assert np.shares_memory(
            block_structure.director_collection, world_bodies[i].director_collection
        )
        assert np.shares_memory(
            block_structure.matrix_dofs_in_rigid_bodies,
            world_bodies[i].director_collection,
        )
        assert_allclose(
            block_structure.director_collection[..., i : i + 1],
            world_bodies[i].director_collection,
        )

        # mass second moment of inertia
        assert np.shares_memory(
            block_structure.mass_second_moment_of_inertia,
            world_bodies[i].mass_second_moment_of_inertia,
        )
        assert np.shares_memory(
            block_structure.matrix_dofs_in_rigid_bodies,
            world_bodies[i].mass_second_moment_of_inertia,
        )
        assert_allclose(
            block_structure.mass_second_moment_of_inertia[..., i : i + 1],
            world_bodies[i].mass_second_moment_of_inertia,
        )

        # inv mass second moment of inertia
        assert np.shares_memory(
            block_structure.inv_mass_second_moment_of_inertia,
            world_bodies[i].inv_mass_second_moment_of_inertia,
        )
        assert np.shares_memory(
            block_structure.matrix_dofs_in_rigid_bodies,
            world_bodies[i].inv_mass_second_moment_of_inertia,
        )
        assert_allclose(
            block_structure.inv_mass_second_moment_of_inertia[..., i : i + 1],
            world_bodies[i].inv_mass_second_moment_of_inertia,
        )


@pytest.mark.parametrize("n_rods", [1, 2, 5, 6])
def test_block_structure_symplectic_stepper_variables_validity(n_rods):
    """
    This function is testing validity of vectors. It is been
    tested that for vector variables, if the block structure memory
    and values are set correctly and rods that belong to rod structure
    share the memory with block structure.

    Parameters
    ----------
    n_rods

    Returns
    -------

    """
    world_bodies = [MockRigidBody() for _ in range(n_rods)]
    block_structure = MemoryBlockRigidBody(world_bodies)

    for i in range(n_rods):

        # velocity collection
        assert np.shares_memory(
            block_structure.velocity_collection, world_bodies[i].velocity_collection
        )
        assert np.shares_memory(
            block_structure.rate_collection, world_bodies[i].velocity_collection
        )
        assert_allclose(
            block_structure.velocity_collection[..., i : i + 1],
            world_bodies[i].velocity_collection,
        )

        # omega collection
        assert np.shares_memory(
            block_structure.omega_collection, world_bodies[i].omega_collection
        )
        assert np.shares_memory(
            block_structure.rate_collection, world_bodies[i].omega_collection
        )
        assert_allclose(
            block_structure.omega_collection[..., i : i + 1],
            world_bodies[i].omega_collection,
        )

        # acceleration collection
        assert np.shares_memory(
            block_structure.acceleration_collection,
            world_bodies[i].acceleration_collection,
        )
        assert np.shares_memory(
            block_structure.rate_collection, world_bodies[i].acceleration_collection
        )
        assert_allclose(
            block_structure.acceleration_collection[..., i : i + 1],
            world_bodies[i].acceleration_collection,
        )

        # alpha collection
        assert np.shares_memory(
            block_structure.alpha_collection, world_bodies[i].alpha_collection
        )
        assert np.shares_memory(
            block_structure.rate_collection, world_bodies[i].alpha_collection
        )
        assert_allclose(
            block_structure.alpha_collection[..., i : i + 1],
            world_bodies[i].alpha_collection,
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
