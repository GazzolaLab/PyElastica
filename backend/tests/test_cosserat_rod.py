#!/usr/bin/env python3

# This file is based on pyelastica tests

# System imports
import numpy as np
import pytest
from numpy.testing import assert_allclose

from elasticapp._PyArrays import Vector, Matrix, Tensor
from elasticapp._PyExamples import _CosseratRodBlock


def vector_copier(self, other):
    for i, val in enumerate(other):
        self[i] = val


def matrix_copier(self, other):
    for i, val1 in enumerate(other):
        for j, val2 in enumerate(val1):
            self[i, j] = val2


def tensor_copier(self, other):
    for i, val1 in enumerate(other):
        for j, val2 in enumerate(val1):
            for k, val3 in enumerate(val2):
                self[i, j, k] = val3


def make_random_rod(n_elems: int, ring_rod_flag: bool = False):
    ret = _CosseratRodBlock(n_elems)
    n_nodes = n_elems if ring_rod_flag else n_elems + 1
    n_voronoi = n_elems if ring_rod_flag else n_elems - 1
    ring_rod_flag = ring_rod_flag

    # Things that are scalar mapped on nodes
    vector_copier(ret.mass, np.random.randn(n_nodes))

    # Things that are vectors mapped on nodes
    matrix_copier(ret.position_collection, np.random.randn(3, n_nodes))
    matrix_copier(ret.velocity_collection, np.random.randn(3, n_nodes))
    matrix_copier(ret.acceleration_collection, np.random.randn(3, n_nodes))
    matrix_copier(ret.internal_forces, np.random.randn(3, n_nodes))
    matrix_copier(ret.external_forces, np.random.randn(3, n_nodes))

    # Things that are scalar mapped on elements
    vector_copier(ret.radius, np.random.rand(n_elems))
    vector_copier(ret.volume, np.random.rand(n_elems))
    # vector_copier(ret.density, np.random.rand(n_elems))
    vector_copier(ret.lengths, np.random.rand(n_elems))
    vector_copier(ret.rest_lengths, np.asarray(ret.lengths))
    vector_copier(ret.dilatation, np.random.rand(n_elems))
    # vector_copier(ret.dilatation_rate, np.random.rand(n_elems))

    # Things that are vector mapped on elements
    matrix_copier(ret.omega_collection, np.random.randn(3, n_elems))
    matrix_copier(ret.alpha_collection, np.random.randn(3, n_elems))
    matrix_copier(ret.tangents, np.random.randn(3, n_elems))
    # matrix_copier(ret.sigma, np.random.randn(3, n_elems))
    # matrix_copier(ret.rest_sigma, np.random.randn(3, n_elems))
    matrix_copier(ret.internal_torques, np.random.randn(3, n_elems))
    matrix_copier(ret.external_torques, np.random.randn(3, n_elems))
    matrix_copier(ret.internal_stress, np.random.randn(3, n_elems))

    # Things that are matrix mapped on elements
    tensor_copier(
        ret.director_collection, np.tile(np.eye(3).reshape(3, 3, 1), (1, 1, n_elems))
    )
    tensor_copier(
        ret.mass_second_moment_of_inertia, np.random.randn() * np.ones((3, 3, n_elems))
    )
    # tensor_copier(ret.inv_mass_second_moment_of_inertia, np.random.randn() * np.ones((3, 3, n_elems)))
    # tensor_copier(ret.shear_matrix, np.random.randn() * np.ones((3, 3, n_elems)))

    # Things that are scalar mapped on voronoi
    vector_copier(ret.voronoi_dilatation, np.random.rand(n_voronoi))
    vector_copier(ret.rest_voronoi_lengths, np.random.rand(n_voronoi))

    # Things that are vectors mapped on voronoi
    # matrix_copier(ret.kappa, np.random.randn(3, n_voronoi))
    # matrix_copier(ret.rest_kappa, np.random.randn(3, n_voronoi))
    matrix_copier(ret.internal_couple, np.random.randn(3, n_voronoi))

    # Things that are matrix mapped on voronoi
    # matrix_copier(ret.bend_matrix, np.random.randn() * np.ones((3, 3, n_voronoi)))
    return ret


@pytest.mark.parametrize("n_elems", [5])
def test_make_rod_and_test_attrs(n_elems):
    rod = make_random_rod(n_elems)

    # TODO: these are int arrays and are therefore not correctly exported
    # [
    #     "density",
    #     "n_elems",
    #     "shear_modulus",
    #     "youngs_modulus",
    # ]

    # test all methods are defined and callable in python
    for rod_method in [
        "get_acceleration",
        "get_angular_acceleration",
        "get_angular_velocity",
        "get_bending_twist_rigidity_matrix",
        "get_curvature",
        "get_density",
        "get_director",
        "get_element_dilatation",
        "get_element_dimension",
        "get_element_length",
        "get_element_volume",
        "get_external_loads",
        "get_external_torques",
        "get_internal_couple",
        "get_internal_loads",
        "get_internal_stress",
        "get_internal_torques",
        "get_inv_mass",
        "get_inv_mass_second_moment_of_inertia",
        "get_mass",
        "get_mass_second_moment_of_inertia",
        "get_n_element",
        "get_position",
        "get_reference_curvature",
        "get_reference_element_length",
        "get_reference_shear_stretch_strain",
        "get_reference_voronoi_length",
        "get_shear_modulus",
        "get_shear_stretch_rigidity_matrix",
        "get_shear_stretch_strain",
        "get_tangent",
        "get_velocity",
        "get_voronoi_dilatation",
        "get_voronoi_length",
        "get_youngs",
    ]:
        assert callable(getattr(rod, rod_method))

    # test all vector attributes are accessible as vectors in python
    for rod_attr_vector in [
        "dilatation",
        "inv_mass",
        "lengths",
        "mass",
        "radius",
        "rest_lengths",
        "rest_voronoi_lengths",
        "volume",
        "voronoi_dilatation",
        "voronoi_length",
    ]:
        assert isinstance(getattr(rod, rod_attr_vector), Vector)

    # test all matrix attributes are accessible as matrices in python
    for rod_attr_matrix in [
        "acceleration_collection",
        "alpha_collection",
        "bending_twist_rigidity_matrix",
        "curvature",
        "external_forces",
        "external_torques",
        "internal_couple",
        "internal_forces",
        "internal_stress",
        "internal_torques",
        "inv_mass_second_moment_of_inertia",
        "omega_collection",
        "position_collection",
        "reference_curvature",
        "reference_shear_stretch_strain",
        "shear_stretch_rigidity_matrix",
        "shear_stretch_strain",
        "tangents",
        "velocity_collection",
    ]:
        assert isinstance(getattr(rod, rod_attr_matrix), Matrix)

    # test all tensor attributes are accessible as tensors in python
    for rod_attr_tensor in [
        "director_collection",
        "mass_second_moment_of_inertia",
    ]:
        assert isinstance(getattr(rod, rod_attr_tensor), Tensor)
