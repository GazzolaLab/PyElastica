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
        self.n_elems = n_elems
        self.n_nodes = self.n_elems + 1
        self.n_voronoi = self.n_elems - 1
        self.ring_rod_flag = False

        # Fixed seed RNG for reproducible test data
        rng = np.random.default_rng(42)

        # Things that are scalar mapped on nodes
        self.mass = rng.standard_normal(self.n_nodes)

        # Things that are vectors mapped on nodes
        self.position_collection = rng.standard_normal((3, self.n_nodes))
        self.velocity_collection = rng.standard_normal((3, self.n_nodes))
        self.acceleration_collection = rng.standard_normal((3, self.n_nodes))
        self.internal_forces = rng.standard_normal((3, self.n_nodes))
        self.external_forces = rng.standard_normal((3, self.n_nodes))

        # Things that are scalar mapped on elements
        self.radius = rng.random(self.n_elems)
        self.volume = rng.random(self.n_elems)
        self.density = rng.random(self.n_elems)
        self.lengths = rng.random(self.n_elems)
        self.rest_lengths = self.lengths.copy()
        self.dilatation = rng.random(self.n_elems)
        self.dilatation_rate = rng.random(self.n_elems)

        # Things that are vector mapped on elements
        self.omega_collection = rng.standard_normal((3, self.n_elems))
        self.alpha_collection = rng.standard_normal((3, self.n_elems))
        self.tangents = rng.standard_normal((3, self.n_elems))
        self.sigma = rng.standard_normal((3, self.n_elems))
        self.rest_sigma = rng.standard_normal((3, self.n_elems))
        self.internal_torques = rng.standard_normal((3, self.n_elems))
        self.external_torques = rng.standard_normal((3, self.n_elems))
        self.internal_stress = rng.standard_normal((3, self.n_elems))

        # Things that are matrix mapped on elements
        self.director_collection = np.zeros((3, 3, self.n_elems))
        for i in range(3):
            for j in range(3):
                self.director_collection[i, j, ...] = 3 * i + j
        # self.director_collection *= rng.standard_normal()  # Commented out as before
        self.mass_second_moment_of_inertia = rng.standard_normal() * np.ones(
            (3, 3, self.n_elems)
        )
        self.inv_mass_second_moment_of_inertia = rng.standard_normal() * np.ones(
            (3, 3, self.n_elems)
        )
        self.shear_matrix = rng.standard_normal() * np.ones((3, 3, self.n_elems))

        # Things that are scalar mapped on voronoi
        self.voronoi_dilatation = rng.random(self.n_voronoi)
        self.rest_voronoi_lengths = rng.random(self.n_voronoi)

        # Things that are vectors mapped on voronoi
        self.kappa = rng.standard_normal((3, self.n_voronoi))
        self.rest_kappa = rng.standard_normal((3, self.n_voronoi))
        self.internal_couple = rng.standard_normal((3, self.n_voronoi))

        # Things that are matrix mapped on voronoi
        self.bend_matrix = rng.standard_normal() * np.ones((3, 3, self.n_voronoi))


class BlockStructureWithSymplecticStepper(
    MemoryBlockCosseratRod, _RodSymplecticStepperMixin
):
    def __init__(self, systems):
        MemoryBlockCosseratRod.__init__(self, systems, [i for i in range(len(systems))])
        _RodSymplecticStepperMixin.__init__(self)

    def update_accelerations(self, time):
        pass


@pytest.mark.parametrize("n_rods", [1, 2, 5, 6])
def test_block_structure_kinematic_update(n_rods, rng):
    """
    This function is testing validity __iadd__ operation of kinematic_states.

    Parameters
    ----------
    n_rods

    Returns
    -------

    """

    world_rods = [MockRod(rng.randint(10, 30 + 1)) for _ in range(n_rods)]
    block_structure = BlockStructureWithSymplecticStepper(world_rods)

    position = block_structure.position_collection.copy()
    velocity = block_structure.velocity_collection.copy()

    directors = block_structure.director_collection.copy()
    omega = block_structure.omega_collection.copy()

    prefac = rng.standard_normal()

    correct_position = position + prefac * velocity
    correct_director = np.zeros(directors.shape)
    np.einsum(
        "ijk,jlk->ilk",
        _get_rotation_matrix(1.0, prefac * omega),
        directors.copy(),
        out=correct_director,
    )

    overload_operator_kinematic_numba(
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
def test_block_structure_dynamic_update(n_rods, rng):
    """
    This function is testing validity __iadd__ operation of dynamic_states.

    Parameters
    ----------
    n_rods

    Returns
    -------

    """

    world_rods = [MockRod(rng.randint(10, 30 + 1)) for _ in range(n_rods)]
    block_structure = BlockStructureWithSymplecticStepper(world_rods)

    v_w = block_structure.v_w_collection.copy()
    dvdt_dwdt = block_structure.dvdt_dwdt_collection.copy()

    prefac = rng.standard_normal()

    correct_v_w = v_w + prefac * dvdt_dwdt

    overload_operator_dynamic_numba(
        prefac,
        block_structure.v_w_collection,
        block_structure.dvdt_dwdt_collection,
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
