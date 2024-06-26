__doc__ = """rigid body class based on mesh"""

from numpy.typing import NDArray
from elastica.typing import MeshType

import numpy as np
import numba
from elastica._linalg import _batch_cross, _batch_norm
from elastica.utils import MaxDimension
from elastica.rigidbody.rigid_body import RigidBodyBase


class MeshRigidBody(RigidBodyBase):
    def __init__(
        self,
        mesh: MeshType,
        center_of_mass: NDArray[np.float64],
        mass_second_moment_of_inertia: NDArray[np.float64],
        density: np.float64,
        volume: np.float64,
    ) -> None:
        """
        Mesh rigid body initializer.

        Parameters
        ----------
        mesh: mesh object
            mesh object which the mesh rigid body is based on
        center_of_mass: numpy.ndarray
            1D (3,) array containing data with 'float' type
            center of mass of the mesh rigid body
        mass_second_moment_of_inertia: numpy.ndarray
            2D (3,3) array containing data with 'float' type
            mass second moment of inertia of the mesh rigid body
        density: float
            density of the mesh rigid body
        volume: float
            volume of the mesh rigid body

        """
        # rigid body does not have elements it only have one node. We are setting n_elems to
        # zero for only make code to work. _bootstrap_from_data requires n_elems to be defined
        self.n_elems: int = 1  # center_mass

        self.density = density
        self.volume = volume
        self.mass = np.float64(self.volume * self.density)
        self.mass_second_moment_of_inertia = mass_second_moment_of_inertia.reshape(
            MaxDimension.value(), MaxDimension.value(), 1
        )

        self.inv_mass_second_moment_of_inertia = np.linalg.inv(
            mass_second_moment_of_inertia
        ).reshape(MaxDimension.value(), MaxDimension.value(), 1)
        normal = np.array([1.0, 0.0, 0.0]).reshape(3, 1)
        tangents = np.array([0.0, 0.0, 1.0]).reshape(3, 1)
        binormal = _batch_cross(tangents, normal)

        # initialize material frame to be the same as lab frame
        self.director_collection = np.zeros(
            (MaxDimension.value(), MaxDimension.value(), 1)
        )
        self.director_collection[0, ...] = normal
        self.director_collection[1, ...] = binormal
        self.director_collection[2, ...] = tangents
        self.faces = mesh.faces.copy()
        self.n_faces = mesh.faces.shape[-1]
        self.face_centers = np.array(mesh.face_centers.copy())
        self.face_normals = np.array(mesh.face_normals.copy())

        # since material frame is the same as lab frame initially, no need to convert this from lab to material
        self.distance_to_face_centers_from_center_of_mass = _batch_norm(
            self.face_centers - center_of_mass.reshape(3, 1)
        )
        self.direction_to_face_centers_from_center_of_mass_in_material_frame = (
            self.face_centers - center_of_mass.reshape(3, 1)
        ) / self.distance_to_face_centers_from_center_of_mass
        self.distance_to_faces_from_center_of_mass = np.zeros(
            (MaxDimension.value(), self.n_faces)
        )
        self.direction_to_faces_from_center_of_mass_in_material_frame = np.zeros(
            (MaxDimension.value(), MaxDimension.value(), self.n_faces)
        )
        for i in range(MaxDimension.value()):
            for k in range(self.n_faces):
                self.distance_to_faces_from_center_of_mass[i, k] = np.linalg.norm(
                    self.faces[:, i, k] - center_of_mass
                )
                self.direction_to_faces_from_center_of_mass_in_material_frame[
                    :, i, k
                ] = (
                    self.faces[:, i, k] - center_of_mass
                ) / self.distance_to_faces_from_center_of_mass[
                    i, k
                ]

        self.face_normals_in_material_frame = np.zeros(
            (MaxDimension.value(), self.n_faces)
        )
        self.face_normals_in_material_frame = self.face_normals.copy()

        # position is at the center of mass
        self.position_collection = np.zeros((MaxDimension.value(), 1))
        self.position_collection[:, 0] = center_of_mass

        self.velocity_collection = np.zeros((MaxDimension.value(), 1))
        self.omega_collection = np.zeros((MaxDimension.value(), 1))
        self.acceleration_collection = np.zeros((MaxDimension.value(), 1))
        self.alpha_collection = np.zeros((MaxDimension.value(), 1))

        self.external_forces = np.zeros((MaxDimension.value())).reshape(
            MaxDimension.value(), 1
        )
        self.external_torques = np.zeros((MaxDimension.value())).reshape(
            MaxDimension.value(), 1
        )

    def update_faces(self) -> None:
        _update_faces(
            self.director_collection,
            self.face_centers,
            self.position_collection,
            self.distance_to_face_centers_from_center_of_mass,
            self.direction_to_face_centers_from_center_of_mass_in_material_frame,
            self.face_normals,
            self.face_normals_in_material_frame,
            self.faces,
            self.distance_to_faces_from_center_of_mass,
            self.direction_to_faces_from_center_of_mass_in_material_frame,
            self.n_faces,
        )


@numba.njit(cache=True)  # type: ignore
def _update_faces(
    director_collection,
    face_centers,
    center_of_mass,
    distance_to_face_centers_from_center_of_mass,
    direction_to_face_centers_from_center_of_mass_in_material_frame,
    face_normals,
    face_normals_in_material_frame,
    faces,
    distance_to_faces_from_center_of_mass,
    direction_to_faces_from_center_of_mass_in_material_frame,
    n_faces,
):
    # this function updates the face_centers, face_normals, and faces
    face_centers[:] = np.zeros((3, n_faces))  # dim,faces
    face_normals[:] = np.zeros((3, n_faces))  # dim,faces
    faces[:] = np.zeros((3, 3, n_faces))  # dim,vertices,faces

    for k in range(n_faces):  # loop through the faces
        for i in range(3):  # loop through the three dimensions
            face_centers[i, k] += center_of_mass[i, 0]
            faces[i, :, k] += center_of_mass[i, 0]
            for j in range(3):  # dummy variable for matrix multiplication
                # update face centers [face_centers_in_lab_frame = CoM + distance_to_face_centers_from_CoM@material_frame_to_lab_frame_director@direction_to_face_centers_from_CoM_in_material_frame]
                face_centers[i, k] += (
                    distance_to_face_centers_from_center_of_mass[i]
                    * director_collection[i, j, 0]
                    * direction_to_face_centers_from_center_of_mass_in_material_frame[
                        j, k
                    ]
                )
                # update face normals [face_normals_in_lab_frame = material_frame_to_lab_frame_director@face_normals_in_material_frame]
                face_normals[i, k] += (
                    director_collection[i, j, 0] * face_normals_in_material_frame[j, k]
                )
                for m in range(3):  # loop through the vertices
                    # update faces [face_vertices_in_lab_frame = CoM + distance_to_face_vertices_from_CoM@material_frame_to_lab_frame_director@direction_to_face_vertices_from_CoM_in_material_frame]
                    faces[i, m, k] += (
                        distance_to_faces_from_center_of_mass[m, k]
                        * director_collection[i, j, 0]
                        * direction_to_faces_from_center_of_mass_in_material_frame[
                            j, m, k
                        ]
                    )
