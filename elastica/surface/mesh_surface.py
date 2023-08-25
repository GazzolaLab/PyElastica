__doc__ = """mesh surface class"""

from elastica.surface.surface_base import SurfaceBase
from elastica.mesh.mesh_initializer import Mesh


class MeshSurface(SurfaceBase, Mesh):
    def __init__(self, filepath: str) -> None:
        """
        Mesh surface initializer.

        Parameters
        ----------
        mesh file path (stl): str
            - path to the mesh file

        Attributes:
        -----------

        mesh_surface.faces:
            - Stores the coordinates of the 3 vertices of each of the n faces of the imported mesh.
            - Dimension: (3 spatial coordinates, 3 vertices, n faces)

        mesh_surface.face_normals:
            - Stores the coordinates of the unit normal vector of each of the n faces.
            - Dimension: (3 spatial coordinates, n faces)

        mesh_surface.face_centers:
            - Stores the coordinates of the position vector of each of the n face centers.
            - Dimension: (3 spatial coordinates, n faces)

        mesh_surface.mesh_scale:
            - Stores the 3 dimensions of the smallest box that could fit the mesh.
            - Dimension: (3 spatial lengths)

        mesh_surface.mesh_center:
            - Stores the coordinates of the position vector of the center of the smallest box that could fit the mesh.
            - Dimension: (3 spatial coordinates)

        mesh_surface.mesh_orientation:
            - store the 3 orthonormal basis vectors that define the mesh orientation.
            - Dimension: (3 spatial coordinates, 3 orthonormal basis vectors)
            - Initial value: [[1,0,0],[0,1,0],[0,0,1]]

        mesh_surface.model_path:
            - Stores the path to the mesh file.

        Methods:
        --------

        mesh_surface.mesh_update():
        Parameters: None
            - This method updates/refreshes the mesh attributes in pyelastica geometry.
            - By default this method is called at initialization and after every method that might change the mesh attributes.

        mesh_surface.visualize():
        Parameters: None
            - This method visualizes the mesh using pyvista.

        mesh_surface.translate():
        Parameters: {numpy.ndarray-(3 spatial coordinates)}
        ex : mesh.translate(np.array([1,1,1]))
            - This method translates the mesh by a given vector.
            - By default, the mesh's center is at the origin;
            by calling this method, the mesh's center is translated to the given vector.

        mesh_surface.scale():
        Parameters: {numpy.ndarray-(3 spatial constants)}
        ex : mesh.scale(np.array([1,1,1]))
            - This method scales the mesh by a given factor in respective axes.

        mesh_surface.rotate():
        Parameters: {rotation_axis: unit vector[numpy.ndarray-(3 spatial coordinates)], angle: in degrees[float]}
        ex : mesh.rotate(np.array([1,0,0]), 90)
            - This method rotates the mesh by a given angle about a given axis.
        """
        SurfaceBase.__init__(self)  # inhereting the parent class attributes, methods
        Mesh.__init__(self, filepath)

        mesh_surface = Mesh(filepath)
        mesh_surface.mesh_update()  # using a mesh method so flake doesn't throw the error of unused variable
