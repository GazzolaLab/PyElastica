__doc__ = """ Mesh Initializer using Pyvista """

import pyvista as pv


class MeshInitialize():
    """
    This Mesh Initializer class uses pyvista to import mesh files in the
    STL or OBJ file formats and initializes the necessary mesh information.

    How to initialize a mesh?
    -------------------------

    mesh = MeshInitialize(r('<filepath>'))

    PS: Please be sure to add .stl / .obj at the end of the filepath, if already present, ignore.
    """
    def __init__(self, filepath):
        self.mesh = pv.read(filepath)
