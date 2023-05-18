__doc__ = (
    """ Test specific functions used in contact in Elastica.joint implementation"""
)

import numpy as np
from elastica.joint import _prune_using_aabbs_rod_rigid_body, _prune_using_aabbs_rod_rod


class TestPruneUsingAABBSRodRigidBody:
    "class to test the prune using aabbs rod rigid body function"

    def test_prune_using_aabbs_rod_rigid_body(self):
        "Testing function with analytically verified values"

        "Intersecting rod and cylinder"
        "dummy inputs were generated using chatgpt"
        rod_one_position_collection = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        rod_one_radius_collection = np.array([1.0, 0.5])
        rod_one_length_collection = np.array([2.0, 1.0])
        cylinder_position = np.array([[4, 5, 6]])
        cylinder_director = np.array(
            [[[1, 0, 0], [0, 0.707, -0.707], [0, 0.707, 0.707]]]
        )
        cylinder_radius = 1.5
        cylinder_length = 5.0
        assert (
            _prune_using_aabbs_rod_rigid_body(
                rod_one_position_collection,
                rod_one_radius_collection,
                rod_one_length_collection,
                cylinder_position,
                cylinder_director,
                cylinder_radius,
                cylinder_length,
            )
            == 0
        )

        "Non - Intersecting rod and cylinder"
        rod_one_position_collection = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        rod_one_radius_collection = np.array([1.0, 0.5])
        rod_one_length_collection = np.array([2.0, 1.0])
        cylinder_position = np.array([[20, 3, 4]])
        cylinder_director = np.array(
            [[[1, 0, 0], [0, 0.707, -0.707], [0, 0.707, 0.707]]]
        )
        cylinder_radius = 1.5
        cylinder_length = 5.0
        assert (
            _prune_using_aabbs_rod_rigid_body(
                rod_one_position_collection,
                rod_one_radius_collection,
                rod_one_length_collection,
                cylinder_position,
                cylinder_director,
                cylinder_radius,
                cylinder_length,
            )
            == 1
        )


class TestPruneUsingAABBSRodRod:
    "class to test the prune using aabbs rod rod function"

    def test_prune_using_aabbs_rod_rod(self):
        "Testing function with analytically verified values"

        "Intersecting rod and rod"
        "dummy inputs were generated using chatgpt"
        rod_one_position_collection = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        rod_one_radius_collection = np.array([1.0, 0.5])
        rod_one_length_collection = np.array([2.0, 1.0])
        rod_two_position_collection = np.array([[2, 3, 4], [5, 6, 7], [8, 9, 10]])
        rod_two_radius_collection = np.array([0.8, 1.2])
        rod_two_length_collection = np.array([1.5, 2.5])
        assert (
            _prune_using_aabbs_rod_rod(
                rod_one_position_collection,
                rod_one_radius_collection,
                rod_one_length_collection,
                rod_two_position_collection,
                rod_two_radius_collection,
                rod_two_length_collection,
            )
            == 0
        )

        "Non - Intersecting rod and rod"
        rod_one_position_collection = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        rod_one_radius_collection = np.array([0.5, 0.3])
        rod_one_length_collection = np.array([1, 2])
        rod_two_position_collection = np.array(
            [[15, 16, 17], [18, 19, 20], [21, 22, 23]]
        )
        rod_two_radius_collection = np.array([0.4, 0.2])
        rod_two_length_collection = np.array([2, 3])
        assert (
            _prune_using_aabbs_rod_rod(
                rod_one_position_collection,
                rod_one_radius_collection,
                rod_one_length_collection,
                rod_two_position_collection,
                rod_two_radius_collection,
                rod_two_length_collection,
            )
            == 1
        )
