__doc__ = """ Module containing joint classes to connect multiple rods together. """
__all__ = ["FreeJoint", "HingeJoint", "FixedJoint", "ExternalContact"]
import numpy as np
from elastica.utils import Tolerance, MaxDimension
from elastica import IMPORT_NUMBA


class FreeJoint:
    """
    This free joint class is the base class for all joints. Free or spherical
    joints constrains the relative movement between two nodes (chosen by the user)
    by applying restoring forces. For implementation details, refer to Zhang et al. Nature Communications (2019).

        Attributes
        ----------
        k: float
            Stiffness coefficient of the joint.
        nu: float
            Damping coefficient of the joint.

    Note
    ----
    Every new joint class must be derived from the FreeJoint class.


    """

    # pass the k and nu for the forces
    # also the necessary rods for the joint
    # indices should be 0 or -1, we will provide wrappers for users later
    def __init__(self, k, nu):
        """

        Parameters
        ----------
        k: float
           Stiffness coefficient of the joint.
        nu: float
           Damping coefficient of the joint.

        """
        self.k = k
        self.nu = nu

    def apply_forces(self, rod_one, index_one, rod_two, index_two):
        """
        Apply joint force to the connected rod objects.

        Parameters
        ----------
        rod_one : object
            Rod-like object
        index_one : int
            Index of first rod for joint.
        rod_two : object
            Rod-like object
        index_two : int
            Index of second rod for joint.

        Returns
        -------

        """
        end_distance_vector = (
            rod_two.position_collection[..., index_two]
            - rod_one.position_collection[..., index_one]
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
            rod_two.velocity_collection[..., index_two]
            - rod_one.velocity_collection[..., index_one]
        )
        normal_relative_velocity = (
            np.dot(relative_velocity, normalized_end_distance_vector)
            * normalized_end_distance_vector
        )
        damping_force = -self.nu * normal_relative_velocity

        contact_force = elastic_force + damping_force

        rod_one.external_forces[..., index_one] += contact_force
        rod_two.external_forces[..., index_two] -= contact_force

        return

    def apply_torques(self, rod_one, index_one, rod_two, index_two):
        """
        Apply restoring joint torques to the connected rod objects.

        In FreeJoint class, this routine simply passes.

        Parameters
        ----------
        rod_one : object
            Rod-like object
        index_one : int
            Index of first rod for joint.
        rod_two : object
            Rod-like object
        index_two : int
            Index of second rod for joint.

        Returns
        -------

        """
        pass


class HingeJoint(FreeJoint):
    """
    This hinge joint class constrains the relative movement and rotation
    (only one axis defined by the user) between two nodes and elements
    (chosen by the user) by applying restoring forces and torques. For
    implementation details, refer to Zhang et. al. Nature
    Communications (2019).

        Attributes
        ----------
        k: float
            Stiffness coefficient of the joint.
        nu: float
            Damping coefficient of the joint.
        kt: float
            Rotational stiffness coefficient of the joint.
        normal_direction: numpy.ndarray
            2D (dim, 1) array containing data with 'float' type. Constraint rotation direction.
    """

    # TODO: IN WRAPPER COMPUTE THE NORMAL DIRECTION OR ASK USER TO GIVE INPUT, IF NOT THROW ERROR
    def __init__(self, k, nu, kt, normal_direction):
        """

        Parameters
        ----------
        k: float
            Stiffness coefficient of the joint.
        nu: float
            Damping coefficient of the joint.
        kt: float
            Rotational stiffness coefficient of the joint.
        normal_direction: numpy.ndarray
            2D (dim, 1) array containing data with 'float' type. Constraint rotation direction.
        """
        super().__init__(k, nu)
        # normal direction of the constrain plane
        # for example for yz plane (1,0,0)
        # unitize the normal vector
        self.normal_direction = normal_direction / np.linalg.norm(normal_direction)
        # additional in-plane constraint through restoring torque
        # stiffness of the restoring constraint -- tuned empirically
        self.kt = kt

    # Apply force is same as free joint
    def apply_forces(self, rod_one, index_one, rod_two, index_two):
        return super().apply_forces(rod_one, index_one, rod_two, index_two)

    def apply_torques(self, rod_one, index_one, rod_two, index_two):
        # current direction of the first element of link two
        # also NOTE: - rod two is hinged at first element
        link_direction = (
            rod_two.position_collection[..., index_two + 1]
            - rod_two.position_collection[..., index_two]
        )

        # projection of the link direction onto the plane normal
        force_direction = (
            -np.dot(link_direction, self.normal_direction) * self.normal_direction
        )

        # compute the restoring torque
        torque = self.kt * np.cross(link_direction, force_direction)

        # The opposite torque will be applied on link one
        rod_one.external_torques[..., index_one] -= (
            rod_one.director_collection[..., index_one] @ torque
        )
        rod_two.external_torques[..., index_two] += (
            rod_two.director_collection[..., index_two] @ torque
        )


class FixedJoint(FreeJoint):
    """
    The fixed joint class restricts the relative movement and rotation
    between two nodes and elements by applying restoring forces and torques.
    For implementation details, refer to Zhang et al. Nature
    Communications (2019).

        Attributes
        ----------
        k: float
            Stiffness coefficient of the joint.
        nu: float
            Damping coefficient of the joint.
        kt: float
            Rotational stiffness coefficient of the joint.
    """

    def __init__(self, k, nu, kt):
        """

        Parameters
        ----------
        k: float
            Stiffness coefficient of the joint.
        nu: float
            Damping coefficient of the joint.
        kt: float
            Rotational stiffness coefficient of the joint.
        """
        super().__init__(k, nu)
        # additional in-plane constraint through restoring torque
        # stiffness of the restoring constraint -- tuned empirically
        self.kt = kt

    # Apply force is same as free joint
    def apply_forces(self, rod_one, index_one, rod_two, index_two):
        return super().apply_forces(rod_one, index_one, rod_two, index_two)

    def apply_torques(self, rod_one, index_one, rod_two, index_two):
        # current direction of the first element of link two
        # also NOTE: - rod two is fixed at first element
        link_direction = (
            rod_two.position_collection[..., index_two + 1]
            - rod_two.position_collection[..., index_two]
        )

        # To constrain the orientation of link two, the second node of link two should align with
        # the direction of link one. Thus, we compute the desired position of the second node of link two
        # as check1, and the current position of the second node of link two as check2. Check1 and check2
        # should overlap.

        tgt_destination = (
            rod_one.position_collection[..., index_one]
            + rod_two.rest_lengths[index_two] * rod_one.tangents[..., index_one]
        )  # dl of rod 2 can be different than rod 1 so use rest length of rod 2

        curr_destination = rod_two.position_collection[
            ..., index_two + 1
        ]  # second element of rod2

        # Compute the restoring torque
        forcedirection = -self.kt * (
            curr_destination - tgt_destination
        )  # force direction is between rod2 2nd element and rod1
        torque = np.cross(link_direction, forcedirection)

        # The opposite torque will be applied on link one
        rod_one.external_torques[..., index_one] -= (
            rod_one.director_collection[..., index_one] @ torque
        )
        rod_two.external_torques[..., index_two] += (
            rod_two.director_collection[..., index_two] @ torque
        )


if IMPORT_NUMBA:
    from elastica._elastica_numba._joint import ExternalContact
else:
    from elastica._elastica_numpy._joint import ExternalContact
