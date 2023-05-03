import numpy as np
from numpy.testing import assert_allclose
import pytest
from elastica.utils import Tolerance


from elastica.rod.data_structures import _RodSymplecticStepperMixin
from elastica._rotations import _get_rotation_matrix
from elastica.memory_block.memory_block_rod import MemoryBlockCosseratRod
from elastica.rod.data_structures import (
    overload_operator_dynamic_numba,
    overload_operator_kinematic_numba,
)


class MockRod:
    def __init__(self, n_elems):
        self.n_elems = n_elems  # np.random.randint(10, 30 + 1)
        self.n_nodes = self.n_elems + 1
        self.n_voronoi = self.n_elems - 1
        self.ring_rod_flag = False

        # Things that are scalar mapped on nodes
        self.mass = np.random.randn(self.n_nodes)

        # Things that are vectors mapped on nodes
        self.position_collection = np.random.randn(3, self.n_nodes)
        self.velocity_collection = np.random.randn(3, self.n_nodes)
        self.acceleration_collection = np.random.randn(3, self.n_nodes)
        self.internal_forces = np.random.randn(3, self.n_nodes)
        self.external_forces = np.random.randn(3, self.n_nodes)

        # Things that are scalar mapped on elements
        self.radius = np.random.rand(self.n_elems)
        self.volume = np.random.rand(self.n_elems)
        self.density = np.random.rand(self.n_elems)
        self.lengths = np.random.rand(self.n_elems)
        self.rest_lengths = self.lengths.copy()
        self.dilatation = np.random.rand(self.n_elems)
        self.dilatation_rate = np.random.rand(self.n_elems)

        # Things that are vector mapped on elements
        self.omega_collection = np.random.randn(3, self.n_elems)
        self.alpha_collection = np.random.randn(3, self.n_elems)
        self.tangents = np.random.randn(3, self.n_elems)
        self.sigma = np.random.randn(3, self.n_elems)
        self.rest_sigma = np.random.randn(3, self.n_elems)
        self.internal_torques = np.random.randn(3, self.n_elems)
        self.external_torques = np.random.randn(3, self.n_elems)
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


class BlockStructureWithSymplecticStepper(
    MemoryBlockCosseratRod, _RodSymplecticStepperMixin
):
    def __init__(self, systems):
        MemoryBlockCosseratRod.__init__(self, systems, [i for i in range(len(systems))])
        _RodSymplecticStepperMixin.__init__(self)

    def update_accelerations(self, time):
        pass


@pytest.mark.parametrize("n_rods", [1, 2, 5, 6])
def test_block_structure_kinematic_state_references(n_rods):
    """
    This function is testing validity of kinematic state views and compare them
    with the block structure vectors.

    Parameters
    ----------
    n_rods

    Returns
    -------

    """
    world_rods = [MockRod(np.random.randint(10, 30 + 1)) for _ in range(n_rods)]
    block_structure = BlockStructureWithSymplecticStepper(world_rods)

    assert_allclose(
        block_structure.position_collection,
        block_structure.kinematic_states.position_collection,
        atol=Tolerance.atol(),
    )
    assert np.shares_memory(
        block_structure.position_collection,
        block_structure.kinematic_states.position_collection,
    )

    assert_allclose(
        block_structure.director_collection,
        block_structure.kinematic_states.director_collection,
        atol=Tolerance.atol(),
    )
    assert np.shares_memory(
        block_structure.director_collection,
        block_structure.kinematic_states.director_collection,
    )


@pytest.mark.parametrize("n_rods", [1, 2, 5, 6])
def test_block_structure_kinematic_update(n_rods):
    """
    This function is testing validity __iadd__ operation of kinematic_states.

    Parameters
    ----------
    n_rods

    Returns
    -------

    """

    world_rods = [MockRod(np.random.randint(10, 30 + 1)) for _ in range(n_rods)]
    block_structure = BlockStructureWithSymplecticStepper(world_rods)

    position = block_structure.position_collection.copy()
    velocity = block_structure.velocity_collection.copy()

    directors = block_structure.director_collection.copy()
    omega = block_structure.omega_collection.copy()

    prefac = np.random.randn()

    correct_position = position + prefac * velocity
    correct_director = np.zeros(directors.shape)
    np.einsum(
        "ijk,jlk->ilk",
        _get_rotation_matrix(1.0, prefac * omega),
        directors.copy(),
        out=correct_director,
    )

    # block_structure.kinematic_states += block_structure.kinematic_rates(0, prefac)

    overload_operator_kinematic_numba(
        block_structure.n_nodes,
        prefac,
        block_structure.position_collection,
        block_structure.director_collection,
        block_structure.velocity_collection,
        block_structure.omega_collection,
    )

    assert_allclose(
        correct_position, block_structure.position_collection, atol=Tolerance.atol()
    )
    assert_allclose(
        correct_director, block_structure.director_collection, atol=Tolerance.atol()
    )


@pytest.mark.parametrize("n_rods", [1, 2, 5, 6])
def test_block_structure_dynamic_state_references(n_rods):
    """
    This function is testing validity of dynamic state views and compare them
    with the block structure vectors.

    Parameters
    ----------
    n_rods

    Returns
    -------

    """
    world_rods = [MockRod(np.random.randint(10, 30 + 1)) for _ in range(n_rods)]
    block_structure = BlockStructureWithSymplecticStepper(world_rods)

    assert_allclose(
        block_structure.velocity_collection,
        block_structure.dynamic_states.velocity_collection,
        atol=Tolerance.atol(),
    )
    assert np.shares_memory(
        block_structure.velocity_collection,
        block_structure.dynamic_states.velocity_collection,
    )

    assert_allclose(
        block_structure.omega_collection,
        block_structure.dynamic_states.omega_collection,
        atol=Tolerance.atol(),
    )
    assert np.shares_memory(
        block_structure.omega_collection,
        block_structure.dynamic_states.omega_collection,
    )

    assert_allclose(
        block_structure.v_w_collection,
        block_structure.dynamic_states.rate_collection,
        atol=Tolerance.atol(),
    )
    assert np.shares_memory(
        block_structure.v_w_collection, block_structure.dynamic_states.rate_collection
    )

    assert_allclose(
        block_structure.dvdt_dwdt_collection,
        block_structure.dynamic_states.dvdt_dwdt_collection,
        atol=Tolerance.atol(),
    )
    assert np.shares_memory(
        block_structure.dvdt_dwdt_collection,
        block_structure.dynamic_states.dvdt_dwdt_collection,
    )


@pytest.mark.parametrize("n_rods", [1, 2, 5, 6])
def test_block_structure_dynamic_state_kinematic_rates(n_rods):
    """
    This function is testing validity of dynamic state function and compare them
    with the block structure vectors.

    Parameters
    ----------
    n_rods

    Returns
    -------

    """
    world_rods = [MockRod(np.random.randint(10, 30 + 1)) for _ in range(n_rods)]
    block_structure = BlockStructureWithSymplecticStepper(world_rods)

    prefac = 1.0

    correct_velocity = prefac * block_structure.velocity_collection.copy()
    velocity_test = block_structure.kinematic_rates(0, prefac)[0].copy()

    assert_allclose(
        correct_velocity,
        velocity_test,
        atol=Tolerance.atol(),
    )

    correct_omega = prefac * block_structure.omega_collection.copy()
    omega_test = block_structure.kinematic_rates(0, prefac)[1].copy()

    assert_allclose(
        correct_omega,
        omega_test,
        atol=Tolerance.atol(),
    )


@pytest.mark.parametrize("n_rods", [1, 2, 5, 6])
def test_block_structure_dynamic_state_dynamic_rates(n_rods):
    """
    This function is testing validity of dynamic rates function and compare them
    with the block structure vector.

    Parameters
    ----------
    n_rods

    Returns
    -------

    """
    world_rods = [MockRod(np.random.randint(10, 30 + 1)) for _ in range(n_rods)]
    block_structure = BlockStructureWithSymplecticStepper(world_rods)

    assert_allclose(
        block_structure.dvdt_dwdt_collection,
        block_structure.dynamic_rates(0, prefac=1),
        atol=Tolerance.atol(),
    )


@pytest.mark.parametrize("n_rods", [1, 2, 5, 6])
def test_block_structure_dynamic_update(n_rods):
    """
    This function is testing validity __iadd__ operation of dynamic_states.

    Parameters
    ----------
    n_rods

    Returns
    -------

    """

    world_rods = [MockRod(np.random.randint(10, 30 + 1)) for _ in range(n_rods)]
    block_structure = BlockStructureWithSymplecticStepper(world_rods)

    v_w = block_structure.v_w_collection.copy()
    dvdt_dwdt = block_structure.dvdt_dwdt_collection.copy()

    prefac = np.random.randn()

    correct_v_w = v_w + prefac * dvdt_dwdt

    overload_operator_dynamic_numba(
        block_structure.v_w_collection, block_structure.dynamic_rates(0, prefac)
    )

    assert_allclose(correct_v_w, block_structure.v_w_collection, atol=Tolerance.atol())
    assert_allclose(
        correct_v_w[0], block_structure.rate_collection[0], atol=Tolerance.atol()
    )  # velocity collection
    assert_allclose(
        correct_v_w[1], block_structure.rate_collection[1], atol=Tolerance.atol()
    )  # omega collection


if __name__ == "__main__":
    from pytest import main

    main([__file__])
