__doc__ = """ Module containing joint classes to connect multiple rods together. """
__all__ = ["FreeJoint", "HingeJoint", "FixedJoint", "get_relative_rotation_two_systems"]

from elastica._rotations import _inv_rotate
from elastica.typing import SystemType, RodType, ConnectionIndex, RigidBodyType

import numpy as np
from numpy.typing import NDArray


class FreeJoint:
    """
    This free joint class is the base class for all joints. Free or spherical
    joints constrains the relative movement between two nodes (chosen by the user)
    by applying restoring forces. For implementation details, refer to Zhang et al. Nature Communications (2019).

    Notes
    -----
    Every new joint class must be derived from the FreeJoint class.

        Attributes
        ----------
        k: float
            Stiffness coefficient of the joint.
        nu: float
            Damping coefficient of the joint.

    """

    # pass the k and nu for the forces
    # also the necessary rods for the joint
    # indices should be 0 or -1, we will provide wrappers for users later
    def __init__(self, k: float, nu: float) -> None:
        """

        Parameters
        ----------
        k: float
           Stiffness coefficient of the joint.
        nu: float
           Damping coefficient of the joint.

        """
        self.k = np.float64(k)
        self.nu = np.float64(nu)

    def apply_forces(
        self,
        system_one: "RodType | RigidBodyType",
        index_one: ConnectionIndex,
        system_two: "RodType | RigidBodyType",
        index_two: ConnectionIndex,
        time: np.float64 = np.float64(0.0),
    ) -> None:
        """
        Apply joint force to the connected rod objects.

        Parameters
        ----------
        system_one : RodType | RigidBodyType
            Rod or rigid-body object
        index_one : ConnectionIndex
            Index of first rod for joint.
        system_two : RodType | RigidBodyType
            Rod or rigid-body object
        index_two : ConnectionIndex
            Index of second rod for joint.

        Returns
        -------

        """
        end_distance_vector = (
            system_two.position_collection[..., index_two]
            - system_one.position_collection[..., index_one]
        )
        elastic_force = self.k * end_distance_vector

        relative_velocity = (
            system_two.velocity_collection[..., index_two]
            - system_one.velocity_collection[..., index_one]
        )
        damping_force = self.nu * relative_velocity

        contact_force = elastic_force + damping_force
        system_one.external_forces[..., index_one] += contact_force
        system_two.external_forces[..., index_two] -= contact_force

        return

    def apply_torques(
        self,
        system_one: "RodType | RigidBodyType",
        index_one: ConnectionIndex,
        system_two: "RodType | RigidBodyType",
        index_two: ConnectionIndex,
        time: np.float64 = np.float64(0.0),
    ) -> None:
        """
        Apply restoring joint torques to the connected rod objects.

        In FreeJoint class, this routine simply passes.

        Parameters
        ----------
        system_one : RodType | RigidBodyType
            Rod or rigid-body object
        index_one : ConnectionIndex
            Index of first rod for joint.
        system_two : RodType | RigidBodyType
            Rod or rigid-body object
        index_two : ConnectionIndex
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
    def __init__(
        self,
        k: float,
        nu: float,
        kt: float,
        normal_direction: NDArray[np.float64],
    ) -> None:
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
        self.kt = np.float64(kt)

    # Apply force is same as free joint
    def apply_forces(
        self,
        system_one: "RodType | RigidBodyType",
        index_one: ConnectionIndex,
        system_two: "RodType | RigidBodyType",
        index_two: ConnectionIndex,
        time: np.float64 = np.float64(0.0),
    ) -> None:
        return super().apply_forces(system_one, index_one, system_two, index_two)

    def apply_torques(
        self,
        system_one: "RodType | RigidBodyType",
        index_one: ConnectionIndex,
        system_two: "RodType | RigidBodyType",
        index_two: ConnectionIndex,
        time: np.float64 = np.float64(0.0),
    ) -> None:
        # current tangent direction of the `index_two` element of system two
        system_two_tangent = system_two.director_collection[2, :, index_two]

        # projection of the tangent of system two onto the plane normal
        force_direction = (
            -np.dot(system_two_tangent, self.normal_direction) * self.normal_direction
        )

        # compute the restoring torque
        torque = self.kt * np.cross(system_two_tangent, force_direction)

        # The opposite torque will be applied on link one
        system_one.external_torques[..., index_one] -= (
            system_one.director_collection[..., index_one] @ torque
        )
        system_two.external_torques[..., index_two] += (
            system_two.director_collection[..., index_two] @ torque
        )


class FixedJoint(FreeJoint):
    """
    The fixed joint class restricts the relative movement and rotation
    between two nodes and elements by applying restoring forces and torques.
    For implementation details, refer to Zhang et al. Nature
    Communications (2019).

        Notes
        -----
        Issue #131 : Add constraint in twisting, add rest_rotation_matrix (v0.3.0)

        Attributes
        ----------
        k: float
            Stiffness coefficient of the joint.
        nu: float
            Damping coefficient of the joint.
        kt: float
            Rotational stiffness coefficient of the joint.
        nut: float
            Rotational damping coefficient of the joint.
        rest_rotation_matrix: np.array
            2D (3,3) array containing data with 'float' type.
            Rest 3x3 rotation matrix from system one to system two at the connected elements.
            Instead of aligning the directors of both systems directly, a desired rest rotational matrix labeled C_12*
            is enforced.
    """

    def __init__(
        self,
        k: float,
        nu: float,
        kt: float,
        nut: float = 0.0,
        rest_rotation_matrix: NDArray[np.float64] | None = None,
    ) -> None:
        """

        Parameters
        ----------
        k: float
            Stiffness coefficient of the joint.
        nu: float
            Damping coefficient of the joint.
        kt: float
            Rotational stiffness coefficient of the joint.
        nut: float = 0.
            Rotational damping coefficient of the joint.
        rest_rotation_matrix: np.array | None
            2D (3,3) array containing data with 'float' type.
            Rest 3x3 rotation matrix from system one to system two at the connected elements.
            If provided, the rest rotation matrix is enforced between the two systems throughout the simulation.
            If not provided, `rest_rotation_matrix` is initialized to the identity matrix,
            which means that a restoring torque will be applied to align the directors of both systems directly.
            (default=None)
        """
        super().__init__(k, nu)
        # additional in-plane constraint through restoring torque
        # stiffness of the restoring constraint -- tuned empirically
        self.kt = np.float64(kt)
        self.nut = np.float64(nut)

        # TODO: compute the rest rotation matrix directly during initialization
        #  as soon as systems (e.g. `rod_one` and `rod_two`) and indices (e.g. `index_one` and `index_two`)
        #  are available in the __init__
        if rest_rotation_matrix is None:
            rest_rotation_matrix = np.eye(3)
        assert rest_rotation_matrix.shape == (3, 3), "Rest rotation matrix must be 3x3"
        self.rest_rotation_matrix = rest_rotation_matrix

    # Apply force is same as free joint
    def apply_forces(
        self,
        system_one: "RodType | RigidBodyType",
        index_one: ConnectionIndex,
        system_two: "RodType | RigidBodyType",
        index_two: ConnectionIndex,
        time: np.float64 = np.float64(0.0),
    ) -> None:
        return super().apply_forces(system_one, index_one, system_two, index_two)

    def apply_torques(
        self,
        system_one: "RodType | RigidBodyType",
        index_one: ConnectionIndex,
        system_two: "RodType | RigidBodyType",
        index_two: ConnectionIndex,
        time: np.float64 = np.float64(0.0),
    ) -> None:
        # collect directors of systems one and two
        # note that systems can be either rods or rigid bodies
        system_one_director = system_one.director_collection[..., index_one]
        system_two_director = system_two.director_collection[..., index_two]

        # rel_rot: C_12 = C_1I @ C_I2
        # C_12 is relative rotation matrix from system 1 to system 2
        # C_1I is the rotation from system 1 to the inertial frame (i.e. the world frame)
        # C_I2 is the rotation from the inertial frame to system 2 frame (inverse of system_two_director)
        rel_rot = system_one_director @ system_two_director.T
        # error_rot: C_22* = C_21 @ C_12*
        # C_22* is rotation matrix from current orientation of system 2 to desired orientation of system 2
        # C_21 is the inverse of C_12, which describes the relative (current) rotation from system 1 to system 2
        # C_12* is the desired rotation between systems one and two, which is saved in the static_rotation attribute
        dev_rot = rel_rot.T @ self.rest_rotation_matrix

        # compute rotation vectors based on C_22*
        # scipy implementation
        # rot_vec = Rotation.from_matrix(dev_rot).as_rotvec()
        #
        # implementation using custom _inv_rotate compiled with numba
        # rotation vector between identity matrix and C_22*
        rot_vec = _inv_rotate(np.dstack([np.eye(3), dev_rot.T])).squeeze()

        # rotate rotation vector into inertial frame
        rot_vec_inertial_frame = system_two_director.T @ rot_vec

        # deviation in rotation velocity between system 1 and system 2
        # first convert to inertial frame, then take differences
        dev_omega = (
            system_two_director.T @ system_two.omega_collection[..., index_two]
            - system_one_director.T @ system_one.omega_collection[..., index_one]
        )

        # we compute the constraining torque using a rotational spring - damper system in the inertial frame
        torque = self.kt * rot_vec_inertial_frame - self.nut * dev_omega

        # The opposite torques will be applied to system one and two after rotating the torques into the local frame
        system_one.external_torques[..., index_one] -= system_one_director @ torque
        system_two.external_torques[..., index_two] += system_two_director @ torque


def get_relative_rotation_two_systems(
    system_one: "RodType | RigidBodyType",
    index_one: ConnectionIndex,
    system_two: "RodType | RigidBodyType",
    index_two: ConnectionIndex,
) -> NDArray[np.float64]:
    """
    Compute the relative rotation matrix C_12 between system one and system two at the specified elements.

    Examples
    ----------
    How to get the relative rotation between two systems (e.g. the rotation from end of rod one to base of rod two):

        >>> rel_rot_mat = get_relative_rotation_two_systems(system1, -1, system2, 0)

    How to initialize a FixedJoint with a rest rotation between the two systems,
    which is enforced throughout the simulation:

        >>> simulator.connect(
        ...    first_rod=system1, second_rod=system2, first_connect_idx=-1, second_connect_idx=0
        ... ).using(
        ...    FixedJoint,
        ...    ku=1e6, nu=0.0, kt=1e3, nut=0.0,
        ...    rest_rotation_matrix=get_relative_rotation_two_systems(system1, -1, system2, 0)
        ... )

    See Also
    ---------
    FixedJoint

    Parameters
    ----------
    system_one : RodType | RigidBodyType
        Rod or rigid-body object
    index_one : ConnectionIndex
        Index of first rod for joint.
    system_two : RodType | RigidBodyType
        Rod or rigid-body object
    index_two : ConnectionIndex
        Index of second rod for joint.

    Returns
    -------
    relative_rotation_matrix : np.array
        Relative rotation matrix C_12 between the two systems for their current state.
    """
    return (
        system_one.director_collection[..., index_one]
        @ system_two.director_collection[..., index_two].T
    )
