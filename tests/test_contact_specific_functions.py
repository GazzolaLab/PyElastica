__doc__ = (
    """ Test specific functions used in contact in Elastica.joint implementation"""
)

import numpy as np
from numpy.testing import assert_allclose
from elastica.joint import (
    _prune_using_aabbs_rod_rigid_body,
    _prune_using_aabbs_rod_rod,
    _calculate_contact_forces_rod_rigid_body,
    _calculate_contact_forces_rod_rod,
    _calculate_contact_forces_self_rod,
)


def test_prune_using_aabbs_rod_rigid_body():
    "Function to test the prune using aabbs rod rigid body function"

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


def test_prune_using_aabbs_rod_rod():
    "Function to test the prune using aabbs rod rod function"

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
    rod_two_position_collection = np.array([[15, 16, 17], [18, 19, 20], [21, 22, 23]])
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


def test_claculate_contact_forces_rod_rigid_body():
    "Function to test the calculate contact forces rod rigid body function"

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
        cylinder_position[..., 0] - 0.5 * cylinder_length * cylinder_director[2, :, 0]
    )

    cylinder_external_forces = np.array([[-0.27817918], [-0.04400299], [1.36401515]])

    cylinder_external_torques = np.array([[-0.2338623], [-1.39748107], [0.31085926]])

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


def test_calculate_contact_forces_rod_rod():
    "Function to test the calculate contact forces rod rod function"

    "Testing function with analytically verified values"

    tol = 1e-5

    "initializing rod 1 parameters"
    rod_one_position_collection = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    rod_one_radius_collection = np.array([1.3, 0.5])
    rod_one_length_collection = np.array([2.1, 0.9])
    rod_one_tangent_collection = np.array([[0, -0.29], [-0.029, 1.12], [-1.04, -1.165]])

    rod_one_internal_forces = np.array(
        [
            [0.59155762, -0.0059393, 0.39725841],
            [-2.45916096, 0.81593963, 1.19209269],
            [1.46996099, -0.52774132, -0.08287433],
        ]
    )
    rod_one_external_forces = np.array(
        [
            [-1.55308218, 1.00714652, 2.04033497],
            [-0.35890683, -0.22667242, -0.31385005],
            [-0.66969949, -0.67150666, 0.17822428],
        ]
    )

    rod_one_velocity_collection = np.array(
        [
            [-0.91917535, -1.04929632, -0.32208191],
            [0.13459384, -1.77261144, -0.28656239],
            [1.63574433, 0.31741648, 1.43917516],
        ]
    )

    "initializing rod 2 parameters"
    rod_two_position_collection = np.array(
        [[1.0000001, 2.0000034, 3], [4, 5, 6], [7, 8, 9]]
    )
    rod_two_radius_collection = np.array([2.3, 0.4])
    rod_two_length_collection = np.array([1.5, 1.0])
    rod_two_tangent_collection = np.array([[0, -0.39], [-0.29, 1.00], [-1.0, -0.165]])

    rod_two_internal_forces = np.array(
        [
            [-0.46552762, 1.16896583, 1.06695832],
            [0.35104219, 0.5868144, -0.79542854],
            [1.90896989, -1.7709093, 1.21209849],
        ]
    )
    rod_two_external_forces = np.array(
        [
            [0.12615472, -0.12539237, -1.01333332],
            [0.7193244, 1.12085914, 1.57535336],
            [1.79006353, 0.20498294, 1.11384582],
        ]
    )

    rod_two_velocity_collection = np.array(
        [
            [-1.09200111, -0.87381467, 0.44073417],
            [0.93961598, 1.25513012, 0.09103194],
            [1.02214026, -0.78790631, -0.74019659],
        ]
    )

    "initializing constants"
    k = 1.0
    nu = 0.5

    "Function call"
    _calculate_contact_forces_rod_rod(
        rod_one_position_collection[..., :-1],
        rod_one_radius_collection,
        rod_one_length_collection,
        rod_one_tangent_collection,
        rod_one_velocity_collection,
        rod_one_internal_forces,
        rod_one_external_forces,
        rod_two_position_collection[..., :-1],
        rod_two_radius_collection,
        rod_two_length_collection,
        rod_two_tangent_collection,
        rod_two_velocity_collection,
        rod_two_internal_forces,
        rod_two_external_forces,
        k,
        nu,
    )

    "Test values"
    assert_allclose(
        rod_one_external_forces,
        np.array(
            [
                [-2.752861, -2.349062, 1.083684],
                [-0.358907, 0.627338, 0.54016],
                [-0.669699, 0.387652, 1.237383],
            ]
        ),
        rtol=tol,
        atol=tol,
    )

    assert_allclose(
        rod_two_external_forces,
        np.array(
            [
                [0.23889, 2.687294, 1.573883],
                [-0.565314, -0.37547, 2.648301],
                [0.825641, -1.336387, 1.501321],
            ]
        ),
        rtol=tol,
        atol=tol,
    )


def test_calculate_contact_forces_self_rod():
    "Function to test the calculate contact forces self rod function"

    "Testing function with analytically verified values"

    tol = 1e-5

    "initializing rod parameters"
    rod_position_collection = np.array(
        [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]
    )
    rod_radius_collection = np.array(
        [5.12952263, 0.19493561, 0.72904683, 0.21676892, 0.18705351]
    )
    rod_length_collection = np.array(
        [0.89050271, 0.7076813, 0.21078658, 0.95372826, 0.86037329]
    )

    rod_tangent_collection = np.array(
        [
            [-0.92511524, 0.19778438, 0.42897242, 0.28430662, -0.57637184],
            [1.73622348, 1.55757074, -0.4606567, -1.30228854, -0.34647765],
            [0.61561226, -0.86529598, -0.9180072, 0.99279484, -2.09424394],
        ]
    )

    rod_external_forces = np.array(
        [
            [-0.77306421, -0.25648047, -0.93419262, -0.77665042, -0.33345937],
            [-1.04834225, -1.92250527, -1.46505411, -0.71192403, -0.99256648],
            [0.33465609, 1.22871475, 0.06250578, -0.49531749, 0.58044695],
        ]
    )

    rod_velocity_collection = np.array(
        [
            [0.70544082, 0.05240655, -1.93283144, -0.35381074, -0.1305802],
            [-0.15193337, -0.16199143, 0.94085659, 0.53076711, 2.15766298],
            [-0.60888955, 0.36362709, 1.31370542, -0.7457939, -0.78005834],
        ]
    )

    "initializing constants"
    k = 1.0
    nu = 0.5

    "Function call"
    _calculate_contact_forces_self_rod(
        rod_position_collection[..., :-1],
        rod_radius_collection,
        rod_length_collection,
        rod_tangent_collection,
        rod_velocity_collection,
        rod_external_forces,
        k,
        nu,
    )

    "Test values"
    assert_allclose(
        rod_external_forces,
        np.array(
            [
                [-0.976629, -0.663611, -0.934193, -0.471303, -0.028112],
                [-1.125742, -2.077304, -1.465054, -0.595825, -0.876467],
                [0.204132, 0.967667, 0.062506, -0.299531, 0.776233],
            ]
        ),
        rtol=tol,
        atol=tol,
    )