__doc__ = """" Test wrapper to construct memory block """

import pytest
import numpy as np

from elastica.rod import RodBase
from elastica.wrappers.memory_block import construct_memory_block_structures
from elastica.memory_block.memory_block_rod import MemoryBlockCosseratRod
from elastica.memory_block.memory_block_rigid_body import MemoryBlockRigidBody
from elastica.rigidbody.rigid_body import RigidBodyBase
from elastica.memory_block.memory_block_magnetic_rod import MemoryBlockMagneticRod
from elastica.rod.magnetic_rod import MagneticRod


class BaseRodForTesting(RodBase):
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

class MagneticRodForTesting(MagneticRod):
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

        self.magnetization_collection = np.random.rand(3, n_elems)


class RigidBodyForTesting(RigidBodyBase):
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
def test_construct_memory_block_structures_for_Cosserat_rod(n_rods):
    """
    This test is only testing the validity of created block-structure class, using the
    construct_memory_block_structures function.

    Parameters
    ----------
    n_rods

    Returns
    -------

    """

    systems = [BaseRodForTesting(np.random.randint(10, 30 + 1)) for _ in range(n_rods)]

    memory_block_list = construct_memory_block_structures(systems)

    assert issubclass(memory_block_list[0].__class__, MemoryBlockCosseratRod)

@pytest.mark.parametrize("n_rods", [1, 2, 5, 6])
def test_construct_memory_block_structures_for_Magnetic_rod(n_rods):
    """
    This test is only testing the validity of created block-structure class, using the
    construct_memory_block_structures function for magnetic rods

    Parameters
    ----------
    n_rods

    Returns
    -------

    """

    systems = [MagneticRodForTesting(np.random.randint(10, 30 + 1)) for _ in range(n_rods)]

    memory_block_list = construct_memory_block_structures(systems)

    assert issubclass(memory_block_list[0].__class__, MemoryBlockMagneticRod)

@pytest.mark.parametrize("n_bodies", [1, 2, 5, 6])
def test_construct_memory_block_structures_for_Rigid_Body(n_bodies):
    """
    This test is only testing the validity of created block-structure class, using the
    construct_memory_block_structures function for rigid bodies.

    Parameters
    ----------
    n_bodies

    Returns
    -------

    """

    systems = [RigidBodyForTesting() for _ in range(n_bodies)]

    memory_block_list = construct_memory_block_structures(systems)

    assert issubclass(memory_block_list[0].__class__, MemoryBlockRigidBody)

@pytest.mark.parametrize("n_rods", [1, 2, 5, 6])
def test_construct_memory_block_structures_for_mixed_systems(n_rods):
    """
    This test is only testing the validity of created block-structure class, using the
    construct_memory_block_structures function for cosserat, magnetic rods and rigid bodies

    Parameters
    ----------
    n_rods

    Returns
    -------

    """

    cosserat_rod_systems = [BaseRodForTesting(np.random.randint(10, 30 + 1)) for _ in range(n_rods)]
    magnetic_rod_systems = [MagneticRodForTesting(np.random.randint(10, 30 + 1)) for _ in range(n_rods)]
    rigid_body_systems = [RigidBodyForTesting() for _ in range(n_rods)]
    systems = cosserat_rod_systems + magnetic_rod_systems + rigid_body_systems

    memory_block_list = construct_memory_block_structures(systems)

    assert issubclass(memory_block_list[0].__class__, MemoryBlockCosseratRod)
    assert issubclass(memory_block_list[1].__class__, MemoryBlockRigidBody)
    assert issubclass(memory_block_list[2].__class__, MemoryBlockMagneticRod)

@pytest.mark.xfail(raises=TypeError)
@pytest.mark.parametrize("n_rods", [1, 2, 5, 6])
def test_construct_memory_block_structures_for_unknown_systems(n_rods):
    """
    This test is only testing the validity of created block-structure class, using the
    construct_memory_block_structures function for the unknown systems. These systems are not
    derived from RodBase or RigidBodyBase

    Parameters
    ----------
    n_rods

    Returns
    -------

    """
    class UnknownSystem:
        pass


    systems = [UnknownSystem() for _ in range(n_rods)]

    _ = construct_memory_block_structures(systems)


if __name__ == "__main__":
    from pytest import main

    main([__file__])
