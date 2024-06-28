__doc__ = """ Mesh Initializer using Pyvista """

import pyvista as pv  # type: ignore
import numpy as np
from numpy.typing import NDArray


class Mesh:
    """
    This Mesh Initializer class uses pyvista to import mesh files in the
    STL or OBJ file formats and initializes the necessary mesh information.

    How to initialize a mesh?
    -------------------------

    mesh = Mesh(r"<filepath>")

    Notes:
    ------

    - Please be sure to add .stl / .obj at the end of the filepath, if already present, ignore.

    Attributes:
    -----------

    mesh.faces:
        - Stores the coordinates of the 3 vertices of each of the n faces of the imported mesh.
        - Dimension: (3 spatial coordinates, 3 vertices, n faces)

    mesh.face_normals:
        - Stores the coordinates of the unit normal vector of each of the n faces.
        - Dimension: (3 spatial coordinates, n faces)

    mesh.face_centers:
        - Stores the coordinates of the position vector of each of the n face centers.
        - Dimension: (3 spatial coordinates, n faces)

    mesh.mesh_scale:
        - Stores the 3 dimensions of the smallest box that could fit the mesh.
        - Dimension: (3 spatial lengths)

    mesh.mesh_center:
        - Stores the coordinates of the position vector of the center of the smallest box that could fit the mesh.
        - Dimension: (3 spatial coordinates)

    mesh.mesh_orientation:
        - store the 3 orthonormal basis vectors that define the mesh orientation.
        - Dimension: (3 spatial coordinates, 3 orthonormal basis vectors)
        - Initial value: [[1,0,0],[0,1,0],[0,0,1]]

    Methods:
    --------

    mesh.mesh_update():
    Parameters: None
        - This method updates/refreshes the mesh attributes in pyelastica geometry.
        - By default this method is called at initialization and after every method that might change the mesh attributes.

    mesh.visualize():
    Parameters: None
        - This method visualizes the mesh using pyvista.

    mesh.translate():
    Parameters: {numpy.ndarray-(3 spatial coordinates)}
    ex : mesh.translate(np.array([1,1,1]))
        - This method translates the mesh by a given vector.
        - By default, the mesh's center is at the origin;
          by calling this method, the mesh's center is translated to the given vector.

    mesh.scale():
    Parameters: {numpy.ndarray-(3 spatial constants)}
    ex : mesh.scale(np.array([1,1,1]))
        - This method scales the mesh by a given factor in respective axes.

    mesh.rotate():
    Parameters: {rotation_axis: unit vector[numpy.ndarray-(3 spatial coordinates)], angle: in degrees[float]}
    ex : mesh.rotate(np.array([1,0,0]), 90)
        - This method rotates the mesh by a given angle about a given axis.
    """

    def __init__(self, filepath: str) -> None:
        self.mesh = pv.read(filepath)
        self.orientation_cube = pv.Cube()
        self.mesh_update()

    def mesh_update(self) -> None:
        """
        This method updates/refreshes the mesh attributes in pyelastica geometry.
        This needs to be performed at the first initialization as well as
        after every method that might change the mesh attributes.
        """
        self.mesh_center = self.mesh.center
        self.face_normals = self.face_normal_calculation(self.mesh.face_normals)
        self.faces = self.face_calculation(
            self.mesh.faces, self.mesh.points, self.mesh.n_faces
        )
        self.face_centers = self.face_center_calculation(self.faces, self.mesh.n_faces)
        self.mesh_scale = self.mesh_scale_calculation(self.mesh.bounds)
        self.mesh_orientation = self.orientation_calculation(
            self.orientation_cube.face_normals
        )

    def face_calculation(
        self, pvfaces: NDArray, meshpoints: NDArray, n_faces: int
    ) -> NDArray:
        """
        This function converts the faces from pyvista to pyelastica geometry

        What the function does?:
        ------------------------

        # The pyvista's 'faces' attribute returns the connectivity array of the faces of the mesh.
            ex: [3, 0, 1, 2, 4, 0, 1, 3, 4]
            The faces array is organized as:
                [n0, p0_0, p0_1, ..., p0_n, n1, p1_0, p1_1, ..., p1_n, ...]
                    ,where n0 is the number of points in face 0, and pX_Y is the Y'th point in face X.
            For more info, refer to the api reference here - https://docs.pyvista.org/version/stable/api/core/_autosummary/pyvista.PolyData.faces.html

        # The pyvista's 'points' attribute returns the individual vertices of the mesh with no connection information.
            ex: [-1.  1. -1.]
                [ 1. -1. -1.]
                [ 1.  1. -1.]

        # This function takes the 'mesh.points' and numbers them as 0, 1, 2 ..., n_faces - 1;
          then establishes connection between verticies of same cell/face through the 'mesh.faces' array
          and returns an array with dimension (3 spatial coordinates, 3 vertices, n faces), where n_faces is the number of faces in the mesh.

        Notes:
        ------

        - This function works only if each face of the mesh has equal no. of vertices i.e
          all the faces of the mesh has similar geometry.
        """
        n_of_vertices_in_each_face = pvfaces[0]
        faces = np.zeros((3, n_of_vertices_in_each_face, n_faces))
        vertice_no = 0

        for i in range(n_faces):
            vertice_no += 1
            for j in range(n_of_vertices_in_each_face):
                faces[..., j, i] = meshpoints[pvfaces[vertice_no]]
                vertice_no += 1

        return faces

    def face_normal_calculation(self, pyvista_face_normals: NDArray) -> NDArray:
        """
        This function converts the face normals from pyvista to pyelastica geometry,
        in pyelastica the face are stored in the format of (n_faces, 3 spatial coordinates),
        this is converted into (3 spatial coordinates, n_faces).
        """
        face_normals = np.transpose(pyvista_face_normals)

        return face_normals

    def face_center_calculation(self, faces: NDArray, n_faces: int) -> NDArray:
        """
        This function calculates the position vector of each face of the mesh
        simply by averaging all the vertices of every face/cell.
        """
        face_centers = np.zeros((3, n_faces))

        for i in range(n_faces):
            for j in range(3):
                vertices = len(faces[j][..., i])
                temp_sum = faces[j][..., i].sum()
                face_centers[j][i] = temp_sum / vertices

        return face_centers

    def mesh_scale_calculation(self, bounds: NDArray) -> NDArray:
        """
        This function calculates scale of the mesh,
        for that it calculates the maximum distance between mesh's farthest verticies in each axis.
        """
        scale = np.zeros(3)
        axis = 0
        for i in range(0, 5, 2):
            scale[axis] = bounds[i + 1] - bounds[i]
            axis += 1

        return scale

    def orientation_calculation(self, cube_face_normals: NDArray) -> NDArray:
        """
        This function calculates the orientation of the mesh
        by using a dummy cube utility from pyvista.
        This dummy cube has minimal values for lengths to minimize the calculation time.

        How the orientation is calculated?:
        ------------------------------------
        The dummy cube's orthogonal face normals are extracted to keep record of the orientation of cube;
        the cube is rotated with the mesh everytime the mesh.rotate() is called, so the orientation
        of the cube is same as the mesh.
        Lastly we find the orientation of the mesh by extracting unit face normal vector for each axis.
        """
        orientation = np.zeros((3, 3))
        axis = 0
        for i in range(1, 6, 2):
            orientation[axis] = cube_face_normals[i]
            axis += 1

        return orientation

    def visualize(self) -> None:
        """
        This function visualizes the mesh using pyvista.
        """
        pyvista_plotter = pv.Plotter()
        pyvista_plotter.add_mesh(self.mesh)
        pyvista_plotter.show()

    def translate(self, target_center: NDArray) -> None:
        """
        Parameters: {numpy.ndarray-(3 spatial coordinates)}
        ex : mesh.translate(np.array([1,1,1]))

        This method moves the mesh by center to the
        the target point given by the user.
        """
        self.mesh = self.mesh.translate(target_center)
        self.mesh_update()

    def scale(self, factor: NDArray) -> None:
        """
        Parameters: {numpy.ndarray-(3 spatial constants)}
        ex : mesh.scale(np.array([1,1,1]))

        This method scales the mesh by the given factor.
        """
        self.mesh = self.mesh.scale(factor)
        self.mesh_update()

    def rotate(self, axis: NDArray, angle: float) -> None:
        """
        Parameters: {rotation_axis: unit vector[numpy.ndarray-(3 spatial coordinates)], angle: in degrees[float]}
        ex : mesh.rotate(np.array([1,0,0]), 90)

        This method rotates the mesh by the given angle
        on the give rotation axis.
        """
        self.mesh = self.mesh.rotate_vector(axis, angle)
        self.orientation_cube = self.orientation_cube.rotate_vector(axis, angle)
        self.mesh_update()
