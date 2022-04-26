"""
Implement magnetization implementation check here.
"""
import pytest
import numpy as np
from numpy.testing import assert_allclose
from elastica.utils import MaxDimension, Tolerance

from elastica.rod.magnetic_rod import MagneticRod
from elastica._linalg import _batch_norm, _batch_matvec

@pytest.mark.xfail(raises=AttributeError)
@pytest.mark.parametrize("n_elems", [ 2, 5, 50, 100])
def test_magnetic_magnetization_density_not_defined(n_elems):
    """
    Test attribute error in magnetic rod, if magnetization density is not defined.

    Parameters
    ----------
    n_elems

    Returns
    -------

    """
    start = np.random.rand(3)
    direction = 5 * np.random.rand(3)
    direction_norm = np.linalg.norm(direction)
    direction /= direction_norm
    normal = np.array((direction[1], -direction[0], 0))
    base_length = 10
    base_radius = np.random.uniform(1, 10)
    density = np.random.uniform(1, 10)
    nu = 0.1
    # Youngs Modulus [Pa]
    E = 1e6
    # poisson ratio
    poisson_ratio = 0.5
    # Shear Modulus [Pa]
    G = E / (1.0 + poisson_ratio) / 2

    magnetization_direction = np.ones((n_elems)) * direction.reshape(3,1)

    magnetic_rod = MagneticRod.straight_rod(
        n_elems,
        start,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        nu,
        E,
        shear_modulus=G,
        magnetization_direction = magnetization_direction,
    )


@pytest.mark.xfail(raises=AttributeError)
@pytest.mark.parametrize("n_elems", [ 2, 5, 50, 100])
def test_magnetic_magnetization_direction_not_defined(n_elems):
    """
    Test attribute error in magnetic rod, if magnetization direction is not defined.

    Parameters
    ----------
    n_elems

    Returns
    -------

    """
    start = np.random.rand(3)
    direction = 5 * np.random.rand(3)
    direction_norm = np.linalg.norm(direction)
    direction /= direction_norm
    normal = np.array((direction[1], -direction[0], 0))
    base_length = 10
    base_radius = np.random.uniform(1, 10)
    density = np.random.uniform(1, 10)
    nu = 0.1
    # Youngs Modulus [Pa]
    E = 1e6
    # poisson ratio
    poisson_ratio = 0.5
    # Shear Modulus [Pa]
    G = E / (1.0 + poisson_ratio) / 2

    magnetization_density = np.random.randn(n_elems)

    magnetic_rod = MagneticRod.straight_rod(
        n_elems,
        start,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        nu,
        E,
        shear_modulus=G,
        magnetization_density = magnetization_density,
    )


@pytest.mark.parametrize("n_elems", [ 2, 5, 50, 100])
def test_magnetic_magnetization_collection(n_elems):
    """
    Test if magnetization collection is set correctly.

    Parameters
    ----------
    n_elems

    Returns
    -------

    """
    start = np.random.rand(3)
    direction = 5 * np.random.rand(3)
    direction_norm = np.linalg.norm(direction)
    direction /= direction_norm
    normal = np.array((direction[1], -direction[0], 0))
    normal /= np.linalg.norm(normal)
    binormal = np.cross(direction,normal)
    binormal /= np.linalg.norm(binormal)
    base_length = 10
    base_radius = np.random.uniform(1, 10)
    density = np.random.uniform(1, 10)
    nu = 0.1
    # Youngs Modulus [Pa]
    E = 1e6
    # poisson ratio
    poisson_ratio = 0.5
    # Shear Modulus [Pa]
    G = E / (1.0 + poisson_ratio) / 2

    A0 = np.pi * base_radius * base_radius
    volume = A0 * base_length / n_elems

    director_collection = np.zeros((3,3,n_elems))
    for i in range(n_elems):
        director_collection[0,:,i] = normal
        director_collection[1,:,i] = binormal
        director_collection[2,:,i] = direction


    magnetization_density = np.random.randn(n_elems)
    magnetization_direction = np.random.randn(3,n_elems)
    magnetization_direction /= _batch_norm(magnetization_direction)

    magnetic_rod = MagneticRod.straight_rod(
        n_elems,
        start,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        nu,
        E,
        shear_modulus=G,
        magnetization_density = magnetization_density,
        magnetization_direction = magnetization_direction,
    )


    assert_allclose(magnetic_rod.director_collection, director_collection, atol=Tolerance.atol())

    magnetization_direction_in_material_frame = _batch_matvec(director_collection, magnetization_direction)
    correct_magnetization_collection = magnetization_density * volume * magnetization_direction_in_material_frame

    assert_allclose(
        magnetic_rod.magnetization_collection, correct_magnetization_collection, atol=Tolerance.atol()
    )



if __name__ == "__main__":
    from pytest import main

    main([__file__])