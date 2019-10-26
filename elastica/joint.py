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
        self.end_distance_vector = (self.rod_two.position[..., self.index_two]
                               - self.rod_one.position[..., self.index_one])
        self.end_distance = np.sqrt(np.sum(self.end_distance_vector * self.end_distance_vector))
        self.elastic_force = self.k * self.end_distance_vector
        relative_velocity = (self.rod_two.velocity[..., self.index_two]
                             - self.rod_one.velocity[..., self.index_one])
        normal_relative_velocity = np.sum(relative_velocity * self.end_distance_vector) / self.end_distance
        damping_force = -self.nu * normal_relative_velocity * self.end_distance_vector
        contact_force = self.elastic_force + damping_force

        self.rod_two.external_forces[..., self.index_two] -= contact_force
        self.rod_one.external_forces[..., self.index_one] += contact_force
        return
    def apply_torque(self):
        pass



class HingeJoint(FreeJoint):
    # TODO: IN WRAPPER COMPUTE THE NORMAL DIRECTION OR ASK USER TO GIVE INPUT, IF NOT THROW ERROR
    def __init__(self, k, nu,rod_one, rod_two, index_one, index_two, kt, normaldirection):
        super().__init__(k, nu, rod_one, rod_two, index_one, index_two)

        # normal direction of the constraing plane
        # for example for yz plane (1,0,0)
        self.normaldirection = normaldirection

        # additional in-plane constraint through restoring torque
        # stiffness of the restoring constraint -- tuned emprically
        self.kt = kt

    def apply_force(self):
        return super().apply_force

    def apply_torque(self):

        # current direction of the first element of link two
        linkdirection = rod_two.position[...,self.index_two + 1] - rod_two.position[...,self.index_two]

        # projection of the linkdirection onto the plane normal
        forcedirection = - np.dot(linkdirection,self.normaldirection) * self.normaldirection

        # compute the restoring torque
        torque = self.kt * linkdirection * forcedirection

        # The opposite torque will be applied on link one (no effect in this case since we assume
        # link one is completely fixed.
        self.rod_one.torques[...,self.index_one]  -= self.rod_one.Q[index_one] * torque
        self.rod_two.torques[..., self.index_two] += self.rod_two.Q[index_two] * torque





#class FixedJoint(FreeJoint)
#    def __init__(self, k, nu, rod_one, rod_two, index_one, index_two):
#        super().__init__(k, nu, rod_one, rod_two, index_one, index_two)
#
#    def
#
#
#class Run():
#
#hgjt = HingeJoint(1e8,1e-2,rod1,rod2,-1,0)
#hgjt.apply_force
#hgjt.apply_torque()
#
#spjt = SphericalJoint(1e8, 1e-2, rod1, rod2, -1, 0)
#spjt.apply_force()