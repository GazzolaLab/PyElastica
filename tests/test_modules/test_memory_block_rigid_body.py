import pytest
import numpy as np
import scipy as sp

from numpy.testing import assert_array_equal

from elastica.rigidbody import RigidBodyBase
from elastica.modules.memory_block import construct_memory_block_structures
from elastica.memory_block.memory_block_rigid_body import MemoryBlockRigidBody


class MockRigidBodyForTesting(RigidBodyBase):
    def __init__(self):
        super().__init__()
        self.radius = np.random.uniform(low=1, high=5)
        self.length = np.random.uniform(low=1, high=10)
        self.density = np.random.uniform(low=1, high=10)
        self.volume = self.length * self.radius * self.radius
        self.mass = np.array(self.volume * self.density)

        self.position_collection = np.random.rand(3, 1)
        self.external_forces = np.random.rand(3, 1)
        self.external_torques = np.random.rand(3, 1)

        self.director_collection = sp.linalg.qr(np.random.rand(3, 3))[0].reshape(
            3, 3, 1
        )

        mass_second_moi_diag = (
            np.random.uniform(low=1, high=2, size=(3,))
            * self.radius
            * self.length
            * self.density
        )

        self.mass_second_moment_of_inertia = np.diag(mass_second_moi_diag).reshape(
            (3, 3, 1)
        )
        self.inv_mass_second_moment_of_inertia = np.diag(
            1.0 / mass_second_moi_diag
        ).reshape((3, 3, 1))

        self.velocity_collection = np.random.rand(3, 1)
        self.acceleration_collection = np.random.rand(3, 1)
        self.omega_collection = np.random.rand(3, 1)
        self.alpha_collection = np.random.rand(3, 1)


@pytest.mark.parametrize("n_bodies", [1, 2, 5, 6])
def test_construct_memory_block_structures_for_rigid_bodies(n_bodies):
    """
    This test is only testing the validity of created rigid-body block-structure class,
    using the construct_memory_block_structures function.

    Parameters
    ----------
    n_bodies: int
        Number of rigid bodies to pass into memory block constructor.
    """

    systems = [MockRigidBodyForTesting() for _ in range(n_bodies)]

    memory_block_list = construct_memory_block_structures(systems)

    assert issubclass(memory_block_list[0].__class__, MemoryBlockRigidBody)


@pytest.mark.parametrize("n_bodies", [1, 2, 5, 6])
def test_memory_block_rigid_body(n_bodies):
    """
    Test memory block logic for rigid bodies.

    Parameters
    ----------
    n_bodies: int
        Number of rigid bodies to be passed into memory block.
    """

    systems = [MockRigidBodyForTesting() for _ in range(n_bodies)]
    system_idx_list = np.arange(n_bodies)

    memory_block = MemoryBlockRigidBody(systems, system_idx_list)

    assert memory_block.n_systems == n_bodies
    assert memory_block.n_elems == n_bodies
    assert memory_block.n_nodes == n_bodies

    attr_list = dir(memory_block)

    expected_attr_list = [
        "mass",
        "volume",
        "density",
        "position_collection",
        "external_forces",
        "external_torques",
        "director_collection",
        "mass_second_moment_of_inertia",
        "inv_mass_second_moment_of_inertia",
        "velocity_collection",
        "omega_collection",
        "acceleration_collection",
        "alpha_collection",
    ]

    # Cross check: make sure memory block and rod attributes are views of each other
    for attr in expected_attr_list:
        # Check if the memory block has the attribute
        assert attr in attr_list

        block_view = memory_block.__dict__[attr].view()

        for k in memory_block.system_idx_list:
            # Assert that the rod's and memory block's attributes share memory
            assert np.shares_memory(
                block_view[..., k : k + 1], systems[k].__dict__[attr]
            )

            # Assert that the rod's and memory block's attributes are equal in values
            assert_array_equal(block_view[..., k : k + 1], systems[k].__dict__[attr])

    # Self check: make sure memory block attributes do not share memory with each other
    for attr_x in expected_attr_list:
        for attr_y in expected_attr_list:
            if attr_x == attr_y:
                continue

            assert not np.may_share_memory(
                memory_block.__dict__[attr_x],
                memory_block.__dict__[attr_y],
            )
