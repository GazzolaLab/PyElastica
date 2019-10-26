__doc__ = """ Joint between rods module """

import numpy as np

from ._linalg import _batch_matmul, _batch_matvec, _batch_cross


class FreeJoint:
    # pass the k and nu for the forces
    # also the necessary rods for the joint
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
        end_distance = np.sqrt(np.dot(end_distance_vector * end_distance_vector))
        elastic_force = self.k * end_distance_vector
        relative_velocity = (self.rod_two.velocity[..., self.index_two]
                             - self.rod_one.velocity[..., self.index_one])
        normal_relative_velocity = np.dot(relative_velocity,
                                          end_distance_vector) / end_distance
        damping_force = (-self.nu * normal_relative_velocity
                         * end_distance_vector) / end_distance
        contact_force = elastic_force + damping_force

        self.rod_two.external_forces[..., self.index_two] -= contact_force
        self.rod_one.external_forces[..., self.index_one] += contact_force
        return

    def apply_torque(self):
        pass


# this joint currently keeps rod one fixed and moves rod two
# how couples act needs to be reconfirmed
class HngeJoint(FreeJoint):
    # TODO: IN WRAPPER COMPUTE THE NORMAL DIRECTION OR ASK USER TO GIVE INPUT, IF NOT THROW ERROR
    def __init__(self, k, nu, rod_one, rod_two, index_one, index_two, kt, normal_direction):
        super().__init__(k, nu, rod_one, rod_two, index_one, index_two)
        # normal direction of the constraing plane
        # for example for yz plane (1,0,0)
        self.normal_direction = normal_direction
        # additional in-plane constraint through restoring torque
        # stiffness of the restoring constraint -- tuned emprically
        self.kt = kt

    def apply_torque(self):
        # current direction of the first element of link two
        # also NOTE: - rod two is hinged at first element
        link_direction = (self.rod_two.position[..., self.index_two + 1] -
                          self.rod_two.position[..., self.index_two])

        # projection of the linkdirection onto the plane normal
        force_direction = - np.dot(link_direction, self.normal_direction) * self.normal_direction

        # compute the restoring torque
        torque = self.kt * link_direction * force_direction

        # The opposite torque will be applied on link one (no effect in this case since we assume
        # link one is completely fixed.
        self.rod_one.torques[..., self.index_one] -= self.rod_one.Q[self.index_one] * torque
        self.rod_two.torques[..., self.index_two] += self.rod_two.Q[self.index_two] * torque


# class FixedJoint(FreeJoint)
#    def __init__(self, k, nu, rod_one, rod_two, index_one, index_two):
#        super().__init__(k, nu, rod_one, rod_two, index_one, index_two)
#
#    def
#
#
# class Run():
#
# hgjt = HingeJoint(1e8,1e-2,rod1,rod2,-1,0)
# hgjt.apply_force
# hgjt.apply_torque()
#
# spjt = SphericalJoint(1e8, 1e-2, rod1, rod2, -1, 0)
# spjt.apply_force()
