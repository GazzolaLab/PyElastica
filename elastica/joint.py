__doc__ = """ Joint between rods module """

import numpy as np

from ._linalg import _batch_matmul, _batch_matvec, _batch_cross


class FreeJoint:
    # pass the k and nu for the forces
    # also the rods for the joint
    # indices should be 0 or -1, we will provide wrappers for users later
    def __init__(self, k, nu, rod_one, rod_two, index_one, index_two):
        self.k = k
        self.nu = nu
        self.rod_one = rod_one
        self.rod_two = rod_two
        self.index_one = index_one
        self.index_two = index_two

    def apply_force(self):
        end_distance_vector = (self.rod_two.position[..., self.index_two]
                               - self.rod_one.position[..., self.index_one])
        end_distance = np.sqrt(np.sum(end_distance_vector * end_distance_vector))
        penetration = self.rod_one.radius + self.rod_two.radius - end_distance
        is_contact = (penetration > 0)
        elastic_force = self.k * penetration * end_distance_vector
        relative_velocity = (self.rod_two.velocity[..., self.index_two]
                             - self.rod_one.velocity[..., self.index_one])
        normal_relative_velocity = np.sum(relative_velocity * end_distance_vector) / end_distance
        damping_force = -self.nu * normal_relative_velocity * end_distance_vector
        contact_force = is_contact * (elastic_force + damping_force)
        self.rod_two.external_forces[..., self.index_two] += contact_force
        self.rod_one.external_forces[..., self.index_one] -= contact_force


class HingeJoint(FreeJoint):
    def __init__(k, nu, rod_one, rod_two, index_one, index_two):
        super().__init__(k, nu, rod_one, rod_two, index_one, index_two)

    def apply_torque(self):
        pass
