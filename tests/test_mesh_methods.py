__doc__ = """ Test mesh class's methods in Elastica """

from elastica.mesh.mesh_initializer import Mesh
import numpy as np
from numpy.testing import assert_allclose
from sys import platform


"""
A dummy cube mesh stl file is used for testing at tests/cube.stl
This dummy file was created using the open source code for creating meshes using 'numpy-stl'
in numpy-stl documentation (https://numpy-stl.readthedocs.io/en/latest/usage.html#initial-usage)
"""


def cube_mesh_stl_init():
    """
    This function initializes a new cube mesh in stl file format.
    """
    if platform == "win32":
        path = r"tests\cube.stl"
    else:
        path = r"tests/cube.stl"

    mockmesh = Mesh(path)
    return mockmesh


"""
A dummy cube mesh obj file is used for testing at tests/cube.obj
This dummy file was created manually by guidelines at https://all3dp.com/1/obj-file-format-3d-printing-cad/
"""


def cube_mesh_obj_init():
    """
    This function initializes a new cube mesh in obj file format.
    """
    if platform == "win32":
        path = r"tests\cube.obj"
    else:
        path = r"tests/cube.obj"

    mockmesh = Mesh(path)
    return mockmesh


"""def test_visualize():
    This function tests the visualization of the mesh.
    Running this test will open a new window with the mesh visualization.
    mockmesh = cube_mesh_init()
    try:
        mockmesh.visualize()
    except:
        raise RuntimeError("Visualization failed")"""


def test_mesh_translate():
    """
    This function tests the translation of the mesh.
    """
    mockmeshstl = cube_mesh_stl_init()
    mockmeshobj = cube_mesh_obj_init()

    """
    By default both the cubes's(stl & obj) centers are to be situated at the origin,
    lets move its center to [1,1,1]
    """
    target_center = np.array([1, 1, 1])

    """
    This is pyvista's numpy array that stores the bounds of the mesh.
    format: [xmin, xmax, ymin, ymax, zmin, zmax]
    since we translated the uniform cube with edge 2 to center [1,1,1],
        the new bounds are: [0,2,0,2,0,2]
    """
    target_bounds = np.array([0, 2, 0, 2, 0, 2])

    """
    Similarly the face centers will also be translated from initial position,
    or moved by [1, 1, 1];
    checkout initial position in test_mesh_init.py.
    Since mesh face centers are calculated using mesh vertices,
    if the face centers are correct, the mesh vertices are also correct.
    """
    target_face_centers_stl = np.array(
        [
            [
                0.666666,
                1.333333,
                0,
                0,
                1.333333,
                0.666666,
                2,
                2,
                1.333333,
                0.666666,
                1.333333,
                0.666666,
            ],
            [
                0.666666,
                1.333333,
                0.666666,
                1.333333,
                0.666666,
                1.333333,
                0.666666,
                1.333333,
                2,
                2,
                0,
                0,
            ],
            [
                0,
                0,
                1.333333,
                0.666666,
                2,
                2,
                0.666666,
                1.333333,
                0.666666,
                1.333333,
                0.666666,
                1.333333,
            ],
        ]
    )
    target_face_centers_obj = np.array(
        [[1, 1, 2, 1, 0, 1], [0, 2, 1, 1, 1, 1], [1, 1, 1, 0, 1, 2]]
    )
    "Translating the mesh"
    mockmeshstl.translate(target_center)
    mockmeshobj.translate(target_center)

    "Testing the translation for both the stl and obj meshes"
    assert_allclose(mockmeshstl.mesh_center, target_center)
    assert_allclose(mockmeshobj.mesh_center, target_center)
    assert_allclose(mockmeshstl.mesh.bounds, target_bounds)
    assert_allclose(mockmeshobj.mesh.bounds, target_bounds)
    assert_allclose(mockmeshstl.face_centers, target_face_centers_stl, atol=1e-6)
    assert_allclose(mockmeshobj.face_centers, target_face_centers_obj, atol=1e-6)


def test_mesh_scale():
    """
    This function tests the scaling of the mesh.
    """
    mockmeshstl = cube_mesh_stl_init()
    mockmeshobj = cube_mesh_obj_init()
    scaling_factor = np.array([2, 2, 2])

    """
    This is pyvista's numpy array that stores the bounds of the mesh.
    format: [xmin, xmax, ymin, ymax, zmin, zmax]
    since we scaled the uniform cube with edge 2 by 2 situated with center at [0, 0, 0],
        the new bounds are: [-2,2,-2,2,-2,2]
    """
    target_bounds = np.array([-2, 2, -2, 2, -2, 2])

    """
    Similarly the face centers will also be translated from initial position,
    or multiplied by 2;
    checkout initial position in test_mesh_init.py.
    Since mesh face centers are calculated using mesh vertices,
    if the face centers are correct, the mesh vertices are also correct.
    """
    target_face_centers_stl = np.array(
        [
            [
                -0.666666,
                0.666666,
                -2,
                -2,
                0.666666,
                -0.666666,
                2,
                2,
                0.666666,
                -0.666666,
                0.666666,
                -0.666666,
            ],
            [
                -0.666666,
                0.666666,
                -0.666666,
                0.666666,
                -0.666666,
                0.666666,
                -0.666666,
                0.666666,
                2,
                2,
                -2,
                -2,
            ],
            [
                -2,
                -2,
                0.666666,
                -0.666666,
                2,
                2,
                -0.666666,
                0.666666,
                -0.666666,
                0.666666,
                -0.666666,
                0.666666,
            ],
        ]
    )
    target_face_centers_obj = np.array(
        [[0, 0, 2, 0, -2, 0], [-2, 2, 0, 0, 0, 0], [0, 0, 0, -2, 0, 2]]
    )

    "Scaling the mesh"
    mockmeshstl.scale(scaling_factor)
    mockmeshobj.scale(scaling_factor)
    "Testing the scaling for both the stl and obj meshes"
    assert_allclose(mockmeshstl.mesh.bounds, target_bounds)
    assert_allclose(mockmeshobj.mesh.bounds, target_bounds)
    assert_allclose(mockmeshstl.face_centers, target_face_centers_stl, atol=1e-6)
    assert_allclose(mockmeshobj.face_centers, target_face_centers_obj, atol=1e-6)


def test_mesh_rotate():
    """
    This function tests the rotation of the mesh.
    """
    mockmeshstl = cube_mesh_stl_init()
    mockmeshobj = cube_mesh_obj_init()
    rotation_angle = 90.0
    rotation_axis = np.array([1, 0, 0])

    """
    Checkout the formatting of bounds in above tests.
    """
    target_bounds = np.array([-1, 1, -1, 1, -2, 2])

    """
    First we scale the uniform cube mesh in y direction
    by a factor of 2, so that it becomes a cuboid.
    Then we rotate the cuboid by 90 degrees about x axis,
    so the longer edge of the cuboid is now in z direction;
    then we test the bounds as formatted above.
    """
    mockmeshstl.scale(np.array([1, 2, 1]))
    mockmeshobj.scale(np.array([1, 2, 1]))
    "Rotating the mesh"
    mockmeshstl.rotate(rotation_axis, rotation_angle)
    mockmeshobj.rotate(rotation_axis, rotation_angle)
    "Testing the bounds for both the stl and obj meshes"
    assert_allclose(mockmeshstl.mesh.bounds, target_bounds)
    assert_allclose(mockmeshobj.mesh.bounds, target_bounds)

    """
    Testing the final orientation of the meshes
    """
    correct_orientation_after_rotation = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    assert_allclose(
        mockmeshstl.mesh_orientation, correct_orientation_after_rotation, atol=1e-6
    )
    assert_allclose(
        mockmeshobj.mesh_orientation, correct_orientation_after_rotation, atol=1e-6
    )
