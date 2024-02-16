import numpy as np
from elastica import *
from elastica._rotations import _get_rotation_matrix, _rotate


class IsometricBC(ConstraintBase):
    """ """

    def __init__(
        self,
        constrain_start_time,
        constrain_start_positions,
        constrain_start_directors,
        constrained_nodes,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.constrain_start_time = constrain_start_time
        self.constrain_start_positions = constrain_start_positions
        self.constrain_start_directors = constrain_start_directors
        self.constrained_nodes = constrained_nodes

    def constrain_values(self, rod, time):
        if time > self.constrain_start_time:
            for i in self.constrained_nodes:
                rod.position_collection[..., i] = self.constrain_start_positions[..., i]
                rod.director_collection[..., i] = self.constrain_start_directors[..., i]
        else:
            self.constrain_start_positions[:] = rod.position_collection[:]
            self.constrain_start_directors[:] = rod.director_collection[:]

    def constrain_rates(self, rod, time):
        if time > self.constrain_start_time:
            for i in self.constrained_nodes:
                rod.velocity_collection[..., i] = 0.0
                rod.omega_collection[..., i] = 0.0


class IsometricStrainBC(ConstraintBase):
    """ """

    def __init__(
        self,
        desired_length,
        direction,
        constrain_start_positions,
        constrain_start_directors,
        length_node_idx,
        constraint_node_idx,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.desired_length = desired_length
        self.constrain_start_positions = constrain_start_positions
        self.constrain_start_directors = constrain_start_directors
        self.length_node_idx = length_node_idx
        self.constraint_node_idx = constraint_node_idx
        self.condition = False
        self.direction = direction
        self.tol = 1e-3

    def constrain_values(self, rod, time):
        length = np.linalg.norm(
            (
                rod.position_collection[..., self.length_node_idx[1]]
                - rod.position_collection[..., self.length_node_idx[0]]
            )
            * self.direction
        )
        if (
            abs(length - self.desired_length) < self.tol * self.desired_length
            or self.condition
        ):
            self.condition = True
            for i in self.constraint_node_idx:
                rod.position_collection[..., i] = self.constrain_start_positions[..., i]
                rod.director_collection[..., i] = self.constrain_start_directors[..., i]
        else:
            self.constrain_start_positions[:] = rod.position_collection[:]
            self.constrain_start_directors[:] = rod.director_collection[:]

    def constrain_rates(self, rod, time):
        length = np.linalg.norm(
            (
                rod.position_collection[..., self.length_node_idx[1]]
                - rod.position_collection[..., self.length_node_idx[0]]
            )
            * self.direction
        )
        if (
            abs(length - self.desired_length) < self.tol * self.desired_length
            or self.condition
        ):
            for i in self.constraint_node_idx:
                rod.velocity_collection[..., i] = 0.0
                rod.acceleration_collection[..., i] = 0.0
                rod.omega_collection[..., i] = 0.0
                rod.alpha_collection[..., i] = 0.0


class CoilTwistBC(ConstraintBase):
    """ """

    def __init__(
        self,
        constrain_start_time,
        coil_radius,
        coil_direction,
        twisting_angular_speed,
        twisted_nodes,
        dt,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.constrain_start_time = constrain_start_time
        self.coil_radius = coil_radius
        self.coil_direction = coil_direction
        self.twisting_angular_speed = twisting_angular_speed
        self.twisted_nodes = twisted_nodes
        self.dt = dt
        self.rotation_matrix = _get_rotation_matrix(
            self.twisting_angular_speed * self.dt, self.coil_direction.reshape(3, 1)
        )[
            ..., 0
        ]  # rotation matrix at each time step

    def constrain_values(self, rod, time):
        if time > self.constrain_start_time:
            for i in self.twisted_nodes:
                current_node_height = np.dot(
                    rod.position_collection[..., i], self.coil_direction
                )
                angle_vector = (
                    rod.position_collection[..., i]
                    - current_node_height * self.coil_direction
                )  # this gives you the angle with respect to the normal and binormal
                rod.position_collection[..., i] = (
                    current_node_height * self.coil_direction
                    + self.rotation_matrix @ angle_vector
                )
                rod.director_collection[..., i] = (
                    self.rotation_matrix @ rod.director_collection[..., i]
                )

    def constrain_rates(self, rod, time):
        if time > self.constrain_start_time:
            for i in self.twisted_nodes:
                current_node_height = np.dot(
                    rod.position_collection[..., i], self.coil_direction
                )
                angle_vector = (
                    rod.position_collection[..., i]
                    - current_node_height * self.coil_direction
                )  # this gives you the angle with respect to the normal and binormal
                rotation_normal_vector = np.cross(self.coil_direction, angle_vector)
                rod.velocity_collection[..., i] = (
                    self.twisting_angular_speed
                    * self.coil_radius
                    * rotation_normal_vector
                )
                rod.omega_collection[..., i] = self.twisting_angular_speed
