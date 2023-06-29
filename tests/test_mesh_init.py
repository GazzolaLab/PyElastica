__doc__ = """ Test mesh initialization in Elastica """

from mesh.mesh_initializer import MeshInitialize
import numpy as np
from numpy.testing import assert_allclose
from sys import platform


"""
A dummy cube mesh stl file is used for testing.
This dummy file was created using the open source code for creating meshes using 'numpy-stl'
in numpy-stl documentation (https://numpy-stl.readthedocs.io/en/latest/usage.html#initial-usage)
"""


def test_mesh_faces():

    """
    This function initializes a new cube mesh checks for the geometry
    of faces generated.
    """

    if platform == "win32":
        path = r"assets\cube.stl"
    else:
        path = r"assets/cube.stl"
    mockmesh = MeshInitialize(path)
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
