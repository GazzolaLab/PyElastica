__doc__ = (
    """ Test specific functions used in contact in Elastica.joint implementation"""
)

import numpy as np
from numpy.testing import assert_allclose
from elastica.joint import (
    _prune_using_aabbs_rod_rigid_body,
    _prune_using_aabbs_rod_rod,
    _calculate_contact_forces_rod_rigid_body,
)


class TestPruneUsingAABBSRodRigidBody:
    "class to test the prune using aabbs rod rigid body function"

    def test_prune_using_aabbs_rod_rigid_body(self):
        "Testing function with analytically verified values"

        "Intersecting rod and cylinder"
        "dummy inputs were generated using chatgpt"
        rod_one_position_collection = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        rod_one_radius_collection = np.array([1.0, 0.5])
        rod_one_length_collection = np.array([2.0, 1.0])
        cylinder_position = np.array([[4], [5], [6]])
        cylinder_director = np.array(
            [[[1], [0], [0]], [[0], [0.707], [-0.707]], [[0], [0.707], [0.707]]]
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
        cylinder_position = np.array([[20], [3], [4]])
        cylinder_director = np.array(
            [[[1], [0], [0]], [[0], [0.707], [-0.707]], [[0], [0.707], [0.707]]]
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


class TestCalculateContactForcesRodRigidBody:
    "class to test the calculate contact forces rod rigid body function"

    def test_claculate_contact_forces_rod_rigid_body(self):
        "Testing function with analytically verified values"

        tol = 1e-5

        "initializing rod parameters"
        rod_position_collection = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        rod_element_position = 0.5 * (
            rod_position_collection[..., 1:] + rod_position_collection[..., :-1]
        )
        rod_radius_collection = np.array([1.3, 0.5])
        rod_length_collection = np.array([2.1, 0.9])
        rod_tangent_collection = np.array([[0, -0.29], [-0.029, 1.12], [-1.04, -1.165]])

        rod_internal_forces = np.array(
            [
                [0.59155762, -0.0059393, 0.39725841],
                [-2.45916096, 0.81593963, 1.19209269],
                [1.46996099, -0.52774132, -0.08287433],
            ]
        )
        rod_external_forces = np.array(
            [
                [-1.55308218, 1.00714652, 2.04033497],
                [-0.35890683, -0.22667242, -0.31385005],
                [-0.66969949, -0.67150666, 0.17822428],
            ]
        )

        rod_velocity_collection = np.array(
            [
                [-0.91917535, -1.04929632, -0.32208191],
                [0.13459384, -1.77261144, -0.28656239],
                [1.63574433, 0.31741648, 1.43917516],
            ]
        )

        "initializing cylinder parameters"
        cylinder_position = np.array([[4], [5], [6]])
        cylinder_director = np.array(
            [[[1], [0], [0]], [[0], [0.707], [-0.707]], [[0], [0.707], [0.707]]]
        )
        cylinder_radius = 1.5
        cylinder_length = 5.0
        x_cyl = (
            cylinder_position[..., 0]
            - 0.5 * cylinder_length * cylinder_director[2, :, 0]
        )

        cylinder_external_forces = np.array(
            [[-0.27817918], [-0.04400299], [1.36401515]]
        )

        cylinder_external_torques = np.array(
            [[-0.2338623], [-1.39748107], [0.31085926]]
        )

        cylinder_velocity_collection = np.array(
            [
                [0.63276313, -0.32444142, 0.61402734],
                [-0.01528792, -0.28025795, 0.32799382],
                [-2.22331567, -0.80881859, -0.82109278],
            ]
        )

        "initializing constants"
        k = 1.0
        nu = 0.5
        velocity_damping_coefficient = 0.1
        friction_coefficient = 0.2

        "Function call"
        _calculate_contact_forces_rod_rigid_body(
            rod_element_position,
            rod_length_collection * rod_tangent_collection,
            cylinder_position[..., 0],
            x_cyl,
            cylinder_length * cylinder_director[2, :, 0],
            rod_radius_collection + cylinder_radius,
            rod_length_collection + cylinder_length,
            rod_internal_forces,
            rod_external_forces,
            cylinder_external_forces,
            cylinder_external_torques,
            cylinder_director[:, :, 0],
            rod_velocity_collection,
            cylinder_velocity_collection,
            k,
            nu,
            velocity_damping_coefficient,
            friction_coefficient,
        )

        "Test values"
        assert_allclose(
            cylinder_external_forces,
            np.array([[-1.389433], [-0.180767], [1.662106]]),
            rtol=tol,
            atol=tol,
        )
        assert_allclose(
            cylinder_external_torques,
            np.array([[0.159707], [-2.261999], [0.310859]]),
            rtol=tol,
            atol=tol,
        )
        assert_allclose(
            rod_external_forces,
            np.array(
                [
                    [-1.383582, 1.647523, 2.341712],
                    [-0.350649, -0.154162, -0.257856],
                    [-0.702578, -0.836991, 0.078497],
                ]
            ),
            rtol=tol,
            atol=tol,
        )
