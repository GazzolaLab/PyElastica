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
        end_distance = np.sqrt(np.dot(end_distance_vector, end_distance_vector))
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
class HingeJoint(FreeJoint):
    # TODO: IN WRAPPER COMPUTE THE NORMAL DIRECTION OR ASK USER TO GIVE INPUT, IF NOT THROW ERROR
    def __init__(self, k, nu, rod_one, rod_two, index_one, index_two, kt, normal_direction):
        super().__init__(k, nu, rod_one, rod_two, index_one, index_two)
        # normal direction of the constraing plane
        # for example for yz plane (1,0,0)
        # unitize the normal vector
        self.normal_direction = normal_direction/np.linalg.norm(normal_direction)
        # additional in-plane constraint through restoring torque
        # stiffness of the restoring constraint -- tuned emprically
        self.kt = kt

    # Apply force is same as free joint
    def apply_force(self):
        return super().apply_force()

    def apply_torque(self):
        # current direction of the first element of link two
        # also NOTE: - rod two is hinged at first element
        self.link_direction = (self.rod_two.position[..., self.index_two + 1] -
                          self.rod_two.position[..., self.index_two])

        # projection of the linkdirection onto the plane normal
        self.force_direction = - np.dot(self.link_direction, self.normal_direction) * self.normal_direction

        # compute the restoring torque
        self.torque = self.kt * self.link_direction * self.force_direction

        # The opposite torque will be applied on link one
        self.rod_one.external_torques[..., self.index_one] -= np.dot(self.rod_one.directors[..., self.index_one] * self.torque)
        self.rod_two.external_torques[..., self.index_two] += np.dot(self.rod_two.directors[..., self.index_two] * self.torque)


class FixedJoint(FreeJoint):
    def __init__(self, k, nu, rod_one, rod_two, index_one, index_two,kt):
        super().__init__(k, nu, rod_one, rod_two, index_one, index_two)
        # additional in-plane constraint through restoring torque
        # stiffness of the restoring constraint -- tuned emprically
        self.kt = kt

    # Apply force is same as free joint
    def apply_force(self):
        return super().apply_force()

    def apply_torque(self):
        # current direction of the first element of link two
        # also NOTE: - rod two is fixed at first element
        link_direction = (self.rod_two.position[..., self.index_two + 1] -
                          self.rod_two.position[..., self.index_two])

        # To constrain the orientation of link two, the second node of link tow should align with
        # the direction of link one. Thus, we compute the desired position of the second node of link two
        # as check1, and the current position of the second node of link two as check2. Check1 and check2
        # should overlap.

        check1 = self.rod_one.position[..., self.index_one] + self.rod_two.reset_lengths[self.index_two] \
                                    * self.rod_two.tangents[self.index_two]

        check2 = self.rod_two.position[...,self.index_two]

        # Compute the restoring torque
        forcedirection = -self.kt * (check2 - check1)
        torque = link_direction * forcedirection

        # The opposite torque will be applied on link one
        self.rod_one.external_torques[..., self.index_one] -= self.rod_one.directors[..., self.index_one] * torque
        self.rod_two.external_torques[..., self.index_two] += self.rod_two.directors[..., self.index_two] * torque



