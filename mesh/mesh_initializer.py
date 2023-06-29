__doc__ = """ Mesh Initializer using Pyvista """

import pyvista as pv
import numpy as np


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
        self.pyvista_faces = self.mesh.faces
        self.number_of_faces = self.mesh.n_faces
        self.pyvista_points = self.mesh.points
        self.faces = self.face_calculation(self.pyvista_faces, self.pyvista_points, self.number_of_faces)

    def face_calculation(self, pvfaces, meshpoints, n_faces):
        """
        This function converts the faces from pyvista to pyelastica geometry
        """

        faces = np.zeros((3, 3, n_faces))
        vertice_no = 0

        for i in range(n_faces):
            vertice_no += 1
            for j in range(3):
                faces[..., j, i] = meshpoints[pvfaces[vertice_no]]
                vertice_no += 1

        return faces
