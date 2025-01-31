__doc__ = """ Test mesh initialization with attributes in Elastica """

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


def test_mesh_faces():
    """
    This functions tests the geometry of faces generated.
    """

    """
    Testing cube mesh in stl file format.
    """
    mockmesh = cube_mesh_stl_init()
    calculated_faces = np.array(
        [
            [
                [-1, 1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1],
                [-1, -1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1],
                [1, 1, -1, -1, 1, -1, 1, 1, 1, 1, 1, -1],
            ],
            [
                [-1, -1, -1, -1, -1, -1, -1, -1, 1, 1, -1, -1],
                [1, 1, -1, 1, -1, 1, -1, 1, 1, 1, -1, -1],
                [-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1],
            ],
            [
                [-1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1],
                [-1, -1, 1, 1, 1, 1, -1, -1, -1, 1, -1, 1],
                [-1, -1, 1, -1, 1, 1, -1, 1, 1, 1, 1, 1],
            ],
        ]
    )

    assert_allclose(mockmesh.faces, calculated_faces)

    """
    Testing cube mesh in obj file format.
    """
    mockmesh = cube_mesh_obj_init()
    calculated_faces = np.array(
        [
            [
                [1, 1, 1, 1, -1, 1],
                [1, 1, 1, 1, -1, 1],
                [-1, -1, 1, -1, -1, -1],
                [-1, -1, 1, -1, -1, -1],
            ],
            [
                [-1, 1, -1, -1, -1, -1],
                [-1, 1, -1, 1, -1, 1],
                [-1, 1, 1, 1, 1, 1],
                [-1, 1, 1, -1, 1, -1],
            ],
            [
                [-1, -1, -1, -1, -1, 1],
                [1, 1, 1, -1, 1, 1],
                [1, 1, 1, -1, 1, 1],
                [-1, -1, -1, -1, -1, 1],
            ],
        ]
    )
    assert_allclose(mockmesh.faces, calculated_faces)


def test_face_normals():
    """
    This functions tests the face normals of the cube mesh.
    """

    """
    Testing cube mesh in stl file format.
    """
    mockmesh = cube_mesh_stl_init()
    calculated_face_normals = np.array(
        [
            [0, 0, -1, -1, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, -1, -1],
            [-1, -1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        ]
    )
    assert_allclose(mockmesh.face_normals, calculated_face_normals)

    """
    Testing cube mesh in obj file format.
    """
    mockmesh = cube_mesh_obj_init()
    calculated_face_normals = np.array(
        [[0, 0, 1, 0, -1, 0], [-1, 1, 0, 0, 0, 0], [0, 0, 0, -1, 0, 1]]
    )
    assert_allclose(mockmesh.face_normals, calculated_face_normals)


def test_face_centers():
    """
    This functions tests the face centers of the cube mesh.
    """

    """
    Testing cube mesh in stl file format.
    """
    mockmesh = cube_mesh_stl_init()
    calculated_face_centers = np.array(
        [
            [
                -0.333333,
                0.333333,
                -1,
                -1,
                0.333333,
                -0.333333,
                1,
                1,
                0.333333,
                -0.333333,
                0.333333,
                -0.333333,
            ],
            [
                -0.333333,
                0.333333,
                -0.333333,
                0.333333,
                -0.333333,
                0.333333,
                -0.333333,
                0.333333,
                1,
                1,
                -1,
                -1,
            ],
            [
                -1,
                -1,
                0.333333,
                -0.333333,
                1,
                1,
                -0.333333,
                0.333333,
                -0.333333,
                0.333333,
                -0.333333,
                0.333333,
            ],
        ]
    )
    assert_allclose(mockmesh.face_centers, calculated_face_centers, atol=1e-6)

    """
    Testing cube mesh in obj file format.
    """
    mockmesh = cube_mesh_obj_init()
    calculated_face_centers = np.array(
        [[0, 0, 1, 0, -1, 0], [-1, 1, 0, 0, 0, 0], [0, 0, 0, -1, 0, 1]]
    )
    assert_allclose(mockmesh.face_centers, calculated_face_centers, atol=1e-6)


def test_mesh_scale():
    """
    This functions tests the scaling of the cube mesh.
    """
    mockmeshstl = cube_mesh_stl_init()
    mockmeshobj = cube_mesh_obj_init()
    """
    The scale of both the cubes mesh(stl & obj) is 2 in all directions because
    its a uniform cube with side 2 situated at origin.
    """
    calculated_mesh_scale = np.array([2, 2, 2])
    assert_allclose(mockmeshstl.mesh_scale, calculated_mesh_scale)
    assert_allclose(mockmeshobj.mesh_scale, calculated_mesh_scale)


def test_mesh_center():
    """
    This functions tests the center of the cube mesh.
    """
    mockmeshstl = cube_mesh_stl_init()
    mockmeshobj = cube_mesh_obj_init()
    """
    Both the cubes(stl & obj) are situated at origin.
    """
    calculated_mesh_center = np.array([0, 0, 0])
    assert_allclose(mockmeshstl.mesh_center, calculated_mesh_center)
    assert_allclose(mockmeshobj.mesh_center, calculated_mesh_center)


def test_mesh_orientation():
    """
    This functions tests the orientation of the cube mesh.
    """
    mockmeshstl = cube_mesh_stl_init()
    mockmeshobj = cube_mesh_obj_init()
    """
    Both the cubes(stl & obj) are situated at origin and the initial orientation is upright
    in the general 3-D cartesian plane.
    """
    calculated_mesh_orientation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert_allclose(mockmeshstl.mesh_orientation, calculated_mesh_orientation)
    assert_allclose(mockmeshobj.mesh_orientation, calculated_mesh_orientation)
