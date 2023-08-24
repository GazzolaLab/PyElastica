__doc__ = """Tests for mesh surface class"""

import pytest
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


def test_mesh_surface_init_error_message():
    """
    Testing mesh_surface error message by providing an invalid path to a stl mesh file.
    """
    if platform == "win32":
        path = r"cube.stl"
    else:
        path = r"cube.stl"

    with pytest.raises(FileNotFoundError) as excinfo:
        test_mesh_surface = MeshSurface(path)

    assert "Please check the filepath.\n"
    " Please be sure to add .stl / .obj at the end of the filepath, if already present, ignore" in str(
        excinfo.value
    )
