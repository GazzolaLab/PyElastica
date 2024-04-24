__doc__ = """rigid body class based on mesh"""

import numpy as np
import numba
from elastica._linalg import _batch_cross, _batch_norm
from elastica.utils import MaxDimension
from elastica.rigidbody.rigid_body import RigidBodyBase


class MeshRigidBody(RigidBodyBase):
    def __init__(
        self,
        mesh,
        center_of_mass,
        mass_second_moment_of_inertia,
        density,
        volume,
    ):
        """
        Mesh rigid body initializer.

        Parameters
        ----------
        mesh
        center_of_mass
        mass_second_moment_of_inertia
        density
        volume
        """
        # rigid body does not have elements it only have one node. We are setting n_elems to
        # zero for only make code to work. _bootstrap_from_data requires n_elems to be defined
        self.n_elems = 1  # center_mass

        self.density = density
        self.volume = volume
        self.mass = np.array([self.volume * self.density])
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
        self.distance_to_face_centers = _batch_norm(
            self.face_centers - center_of_mass.reshape(3, 1)
        )
        self.direction_to_face_centers = (
            self.face_centers - center_of_mass.reshape(3, 1)
        ) / self.distance_to_face_centers
        self.distance_to_faces = np.zeros((MaxDimension.value(), self.n_faces))
        self.direction_to_faces = np.zeros(
            (MaxDimension.value(), MaxDimension.value(), self.n_faces)
        )
        for i in range(MaxDimension.value()):
            for k in range(self.n_faces):
                self.distance_to_faces[i, k] = np.linalg.norm(
                    self.faces[:, i, k] - center_of_mass
                )
                self.direction_to_faces[:, i, k] = (
                    self.faces[:, i, k] - center_of_mass
                ) / self.distance_to_faces[i, k]

        self.mass_second_moment_of_inertia = mass_second_moment_of_inertia.reshape(
            MaxDimension.value(), MaxDimension.value(), 1
        )

        self.inv_mass_second_moment_of_inertia = np.linalg.inv(
            mass_second_moment_of_inertia
        ).reshape(MaxDimension.value(), MaxDimension.value(), 1)

        # position is at the center
        self.position_collection = np.zeros((MaxDimension.value(), 1))
        self.position_collection[:, 0] = center_of_mass

        self.velocity_collection = np.zeros((MaxDimension.value(), 1))
        self.omega_collection = np.zeros((MaxDimension.value(), 1))
        self.acceleration_collection = np.zeros((MaxDimension.value(), 1))
        self.alpha_collection = np.zeros((MaxDimension.value(), 1))
        self.face_normals_lagrangian = np.zeros((MaxDimension.value(), self.n_faces))

        # self.face_normals_lagrangian = self.director_collection[...,0].T@self.face_normals
        # for i in range(3):
        #     for j in range(3):
        #         for k in range(self.n_faces):
        #             self.face_normals_lagrangian[i, k] += (
        #                 self.director_collection[j, i, 0] * self.face_normals[j, k]
        #             )
        self.face_normals_lagrangian = self.face_normals.copy()

        self.external_forces = np.zeros((MaxDimension.value())).reshape(
            MaxDimension.value(), 1
        )
        self.external_torques = np.zeros((MaxDimension.value())).reshape(
            MaxDimension.value(), 1
        )

    def update_faces(self):
        _update_faces(
            self.director_collection,
            self.faces,
            self.position_collection,
            self.distance_to_faces,
            self.direction_to_faces,
            self.n_faces,
        )
        _update_face_centers(
            self.director_collection,
            self.face_centers,
            self.position_collection,
            self.distance_to_face_centers,
            self.direction_to_face_centers,
            self.n_faces,
        )
        _update_face_normals(
            self.director_collection,
            self.face_normals,
            self.face_normals_lagrangian,
            self.n_faces,
        )


@numba.njit(cache=True)
def _update_face_centers(
    director_collection,
    face_centers,
    center_of_mass,
    distance_to_face_centers,
    direction_to_face_centers,
    n_faces,
):
    face_centers[:] = np.zeros((3, n_faces))
    for k in range(n_faces):
        for i in range(3):
            face_centers[i, k] += center_of_mass[i, 0]
            for j in range(3):
                face_centers[i, k] += (
                    distance_to_face_centers[i]
                    * director_collection[i, j, 0]
                    * direction_to_face_centers[j, k]
                )


@numba.njit(cache=True)
def _update_faces(
    director_collection,
    faces,
    center_of_mass,
    distance_to_faces,
    direction_to_faces,
    n_faces,
):
    faces[:] = np.zeros((3, 3, n_faces))  # dim,vertices,faces
    for k in range(n_faces):
        for m in range(3):
            for i in range(3):
                faces[i, m, k] += center_of_mass[i, 0]
                for j in range(3):
                    faces[i, m, k] += (
                        distance_to_faces[m, k]
                        * director_collection[i, j, 0]
                        * direction_to_faces[j, m, k]
                    )


@numba.njit(cache=True)
def _update_face_normals(
    director_collection,
    face_normals,
    face_normals_lagrangian,
    n_faces,
):

    face_normals[:] = np.zeros((3, n_faces))
    for i in range(3):
        for j in range(3):
            for k in range(n_faces):
                face_normals[i, k] += (
                    director_collection[i, j, 0] * face_normals_lagrangian[j, k]
                )
