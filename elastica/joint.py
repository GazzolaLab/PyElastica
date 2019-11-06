__doc__ = """ Joint between rods module """

import numpy as np
from elastica.utils import Tolerance


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
        end_distance_vector = (
            self.rod_two.position[..., self.index_two]
            - self.rod_one.position[..., self.index_one]
        )
        # Calculate norm of end_distance_vector
        # this implementation timed: 2.48 µs ± 126 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
        end_distance = np.sqrt(np.dot(end_distance_vector, end_distance_vector))

        # Below if check is not efficient find something else
        # We are checking if end of rod1 and start of rod2 are at the same point in space
        # If they are at the same point in space, it is a zero vector.
        if end_distance <= Tolerance.atol():
            normalized_end_distance_vector = np.array([0.0, 0.0, 0.0])
        else:
            normalized_end_distance_vector = end_distance_vector / end_distance

        elastic_force = self.k * end_distance_vector

        relative_velocity = (
            self.rod_two.velocity[..., self.index_two]
            - self.rod_one.velocity[..., self.index_one]
        )
        normal_relative_velocity = (
            np.dot(relative_velocity, normalized_end_distance_vector)
            * normalized_end_distance_vector
        )
        damping_force = -self.nu * normal_relative_velocity

        contact_force = elastic_force + damping_force

        self.rod_one.external_forces[..., self.index_one] += contact_force
        self.rod_two.external_forces[..., self.index_two] -= contact_force

        return

    def apply_torque(self):
        pass


class HingeJoint(FreeJoint):
    """ this joint currently keeps rod one fixed and moves rod two
        how couples act needs to be reconfirmed
    """

    # TODO: IN WRAPPER COMPUTE THE NORMAL DIRECTION OR ASK USER TO GIVE INPUT, IF NOT THROW ERROR
    def __init__(
        self, k, nu, rod_one, rod_two, index_one, index_two, kt, normal_direction
    ):
        super().__init__(k, nu, rod_one, rod_two, index_one, index_two)
        # normal direction of the constraing plane
        # for example for yz plane (1,0,0)
        # unitize the normal vector
        self.normal_direction = normal_direction / np.linalg.norm(normal_direction)
        # additional in-plane constraint through restoring torque
        # stiffness of the restoring constraint -- tuned emprically
        self.kt = kt

    # Apply force is same as free joint
    def apply_force(self):
        return super().apply_force()

    def apply_torque(self):
        # current direction of the first element of link two
        # also NOTE: - rod two is hinged at first element
        link_direction = (
            self.rod_two.position[..., self.index_two + 1]
            - self.rod_two.position[..., self.index_two]
        )

        # projection of the linkdirection onto the plane normal
        force_direction = (
            -np.dot(link_direction, self.normal_direction) * self.normal_direction
        )

        # compute the restoring torque
        torque = self.kt * np.cross(link_direction, force_direction)

        # The opposite torque will be applied on link one
        self.rod_one.external_torques[..., self.index_one] -= (
            self.rod_one.directors[..., self.index_one] @ torque
        )
        self.rod_two.external_torques[..., self.index_two] += (
            self.rod_two.directors[..., self.index_two] @ torque
        )


class FixedJoint(FreeJoint):
    def __init__(self, k, nu, rod_one, rod_two, index_one, index_two, kt):
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
        link_direction = (
            self.rod_two.position[..., self.index_two + 1]
            - self.rod_two.position[..., self.index_two]
        )

        # To constrain the orientation of link two, the second node of link two should align with
        # the direction of link one. Thus, we compute the desired position of the second node of link two
        # as check1, and the current position of the second node of link two as check2. Check1 and check2
        # should overlap.

        position_diff = (
            self.rod_one.position[..., self.index_one]
            - self.rod_one.position[..., self.index_one - 1]
        )
        length = np.sqrt(np.dot(position_diff, position_diff))
        tangent = position_diff / length

        tgt_destination = (
            self.rod_one.position[..., self.index_one]
            + self.rod_two.rest_lengths[self.index_two] * tangent
        )  # dl of rod 2 can be different than rod 1 so use restlengt of rod 2

        curr_destination = self.rod_two.position[
            ..., self.index_two + 1
        ]  # second element of rod2

        # Compute the restoring torque
        forcedirection = -self.kt * (
            curr_destination - tgt_destination
        )  # force direction is between rod2 2nd element and rod1
        torque = np.cross(link_direction, forcedirection)

        # The opposite torque will be applied on link one
        self.rod_one.external_torques[..., self.index_one] -= (
            self.rod_one.directors[..., self.index_one] @ torque
        )
        self.rod_two.external_torques[..., self.index_two] += (
            self.rod_two.directors[..., self.index_two] @ torque
        )
