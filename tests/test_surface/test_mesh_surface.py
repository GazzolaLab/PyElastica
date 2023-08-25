__doc__ = """Tests for mesh surface class"""

from elastica.surface.mesh_surface import MeshSurface
from sys import platform

"""
A dummy cube mesh stl file is used for testing at tests/cube.stl
This dummy file was created using the open source code for creating meshes using 'numpy-stl'
in numpy-stl documentation (https://numpy-stl.readthedocs.io/en/latest/usage.html#initial-usage)
"""


def test_mesh_surface_init():
    """
    Testing mesh_surface initialization by providing a valid path to a stl mesh file.
    """
    if platform == "win32":
        path = r"tests\cube.stl"
    else:
        path = r"tests/cube.stl"

    test_mesh_surface = MeshSurface(path)

    assert isinstance(test_mesh_surface, MeshSurface), True
    assert test_mesh_surface.model_path == path
