__doc__ = (
    """ Module containing joint classes to connect rods and rigid bodies together. """
)
__all__ = ["GenericSystemTypeFreeJoint", "GenericSystemTypeFixedJoint"]
from elastica.joint import FreeJoint, FixedJoint
from elastica.typing import SystemType
from elastica.utils import Tolerance, MaxDimension
import numpy as np
from typing import Optional

# Experimental implementation for `joint` modules: #122 #149
#   - Enable the joint between `rod` and `rigid-body`
#   - Allow joint to form offset from the node/COM
#   - Generalized version for joint system
# Blocked by:
#   - #113
# TODO:
#    - [x] Tests
#    - [ ] Physical validation / theory / literature for reference
#    - [ ] Optimization / Numba
#    - [ ] Benchmark
#    - [x] Examples


class GenericSystemTypeFreeJoint(FreeJoint):
    """
    Constrains the relative movement between two nodes by applying restoring forces.

    Attributes
    ----------
    k : float
        Stiffness coefficient of the joint.
    nu : float
        Damping coefficient of the joint.
    point_system_one : numpy.ndarray
        Describes for system one in the local coordinate system the translation from the node `index_one` (for rods)
        or the center of mass (for rigid bodies) to the joint.
    point_system_two : numpy.ndarray
        Describes for system two in the local coordinate system the translation from the node `index_two` (for rods)
        or the center of mass (for rigid bodies) to the joint.


    Examples
    --------
    How to connect two Cosserat rods together using a spherical joint with a gap of 0.01 m in between.

    >>> simulator.connect(rod_one, rod_two, first_connect_idx=-1, second_connect_idx=0).using(
    ...    FreeJoint,
    ...    k=1e4,
    ...    nu=1,
    ...    point_system_one=np.array([0.0, 0.0, 0.005]),
    ...    point_system_two=np.array([0.0, 0.0, -0.005]),
    ... )

    How to connect the distal end of a CosseratRod with the base of a cylinder using a spherical joint.

    >>> simulator.connect(rod, cylinder, first_connect_idx=-1, second_connect_idx=0).using(
    ...    FreeJoint,
    ...    k=1e4,
    ...    nu=1,
    ...    point_system_two=np.array([0.0, 0.0, -cylinder.length / 2.]),
    ... )

    """

    # pass the k and nu for the forces
    # also the necessary systems for the joint
    def __init__(
        self,
        k: float,
        nu: float,
        point_system_one: Optional[np.ndarray] = None,
        point_system_two: Optional[np.ndarray] = None,
        **kwargs,
    ):
        """

        Parameters
        ----------
        k : float
            Stiffness coefficient of the joint.
        nu : float
            Damping coefficient of the joint.
        point_system_one : Optional[numpy.ndarray]
            Describes for system one in the local coordinate system the translation from the node `index_one` (for rods)
            or the center of mass (for rigid bodies) to the joint.
            (default = np.array([0.0, 0.0, 0.0]))
        point_system_two : Optional[numpy.ndarray]
            Describes for system two in the local coordinate system the translation from the node `index_two` (for rods)
            or the center of mass (for rigid bodies) to the joint.
            (default = np.array([0.0, 0.0, 0.0]))
        """
        super().__init__(k=k, nu=nu, **kwargs)

        self.point_system_one = (
            point_system_one
            if point_system_one is not None
            else np.array([0.0, 0.0, 0.0])
        )
        self.point_system_two = (
            point_system_two
            if point_system_two is not None
            else np.array([0.0, 0.0, 0.0])
        )

    def apply_forces(
        self,
        system_one: SystemType,
        index_one: int,
        system_two: SystemType,
        index_two: int,
    ):
        """
        Apply joint force to the connected systems.

        Parameters
        ----------
        system_one : SystemType
            System two of the joint connection.
        index_one : int
            Index of first system for joint.
        system_two : SystemType
            System two of the joint connection.
        index_two : int
            Index of second system for joint.

        """
        # Compute the position in the inertial frame of the specified point.
        # The point is defined in the local coordinate system of system one and used to attach to the joint.
        position_system_one = compute_position_of_point(
            system=system_one, point=self.point_system_one, index=index_one
        )
        # Compute the position in the inertial frame of the specified point.
        # The point is defined in the local coordinate system of system two and used to attach to the joint.
        position_system_two = compute_position_of_point(
            system=system_two, point=self.point_system_two, index=index_two
        )

        # Analogue to the positions, compute the velocities of the points in the inertial frames
        velocity_system_one = compute_velocity_of_point(
            system=system_one, point=self.point_system_one, index=index_one
        )
        velocity_system_two = compute_velocity_of_point(
            system=system_two, point=self.point_system_two, index=index_two
        )

        # Compute the translational deviation of the point belonging to system one
        # from the point belonging to system two
        distance_vector = position_system_two - position_system_one

        # Compute elastic force using a spring formulation as a linear function of the (undesired) distance between
        # the two systems.
        elastic_force = self.k * distance_vector

        # Compute the velocity deviation of the point belonging to system one from the point belonging to system two
        relative_velocity = velocity_system_two - velocity_system_one

        # Compute damping force considering the specified damping coefficient `nu`
        damping_force = self.nu * relative_velocity

        # compute contact force as addition of elastic force and damping force
        contact_force = elastic_force + damping_force

        # loop over the two systems
        for i, (system, index, point, system_position) in enumerate(
            zip(
                [system_one, system_two],
                [index_one, index_two],
                [self.point_system_one, self.point_system_two],
                [position_system_one, position_system_two],
            )
        ):
            # The external force has opposite signs for the two systems:
            # For system one: external_force = +contact_force
            # For system two: external_force = -contact_force
            external_force = (1 - 2 * i) * contact_force

            # the contact force needs to be applied at a distance from the Center of Mass (CoM) of the rigid body
            # or the selected node of the Cosserat rod.
            # This generates a torque, which we also need to apply to both systems.
            # We first compute the vector r from the node / CoM to the joint connection point.
            distance_system_point = (
                system_position - system.position_collection[..., index]
            )
            # The torque is the cross product of the distance vector and the contact force: tau = r x F
            external_torque = np.cross(distance_system_point, external_force)

            # Apply external forces and torques to both systems.
            system.external_forces[..., index] += external_force
            # the torque still needs to be rotated into the local coordinate system of the system
            system.external_torques[..., index] += (
                system.director_collection[..., index] @ external_torque
            )

    def apply_torques(
        self,
        system_one: SystemType,
        index_one: int,
        system_two: SystemType,
        index_two: int,
    ):
        """
        Apply restoring joint torques to the connected systems.

        In FreeJoint class, this routine simply passes.

        Parameters
        ----------
        system_one : SystemType
             System two of the joint connection.
        index_one : int
            Index of first system for joint.
        system_two : SystemType
            System two of the joint connection.
        index_two : int
            Index of second system for joint.

        """
        pass


class GenericSystemTypeFixedJoint(GenericSystemTypeFreeJoint):
    """
    The fixed joint class restricts the relative movement and rotation
    between two nodes and elements by applying restoring forces and torques.

        Attributes
        ----------
        k : float
            Stiffness coefficient of the joint.
        nu : float
            Damping coefficient of the joint.
        kt : float
            Rotational stiffness coefficient of the joint.
        nut : float
            Rotational damping coefficient of the joint.
        point_system_one : numpy.ndarray
            Describes for system one in the local coordinate system the translation from the node `index_one` (for rods)
            or the center of mass (for rigid bodies) to the joint.
        point_system_two : numpy.ndarray
            Describes for system two in the local coordinate system the translation from the node `index_two` (for rods)
            or the center of mass (for rigid bodies) to the joint.
        rest_rotation_matrix : np.ndarray
            2D (3,3) array containing data with 'float' type.
            Rest 3x3 rotation matrix from system one to system two at the connected elements.
            Instead of aligning the directors of both systems directly, a desired rest rotational matrix labeled C_12*
            is enforced.

        Examples
        --------
        How to connect two Cosserat rods together using a fixed joint while aligning the tangents (e.g. local z-axis).

        >>> simulator.connect(rod_one, rod_two).using(
        ...    FixedJoint,
        ...    k=1e4,
        ...    nu=1,
        ... )

        How to connect a cosserat rod with the base of a cylinder using a fixed joint, where the cylinder is rotated
        by 45 degrees around the y-axis.

        >>> from scipy.spatial.transform import Rotation
        ... simulator.connect(rod, cylinder).using(
        ...    FixedJoint,
        ...    k=1e5,
        ...    nu=1e0,
        ...    kt=1e3,
        ...    nut=1e-3,
        ...    point_system_two=np.array([0, 0, -cylinder.length / 2]),
        ...    rest_rotation_matrix=Rotation.from_euler('y', np.pi / 4, degrees=False).as_matrix(),
        ... )
    """

    def __init__(
        self,
        k: float,
        nu: float,
        kt: float,
        nut: float = 0.0,
        point_system_one: Optional[np.ndarray] = None,
        point_system_two: Optional[np.ndarray] = None,
        rest_rotation_matrix: Optional[np.ndarray] = None,
        **kwargs,
    ):
        """

        Parameters
        ----------
        k : float
            Stiffness coefficient of the joint.
        nu : float
            Damping coefficient of the joint.
        kt : float
            Rotational stiffness coefficient of the joint.
        nut : float
            Rotational damping coefficient of the joint. (default=0.0)
        point_system_one : Optional[numpy.ndarray]
            Describes for system one in the local coordinate system the translation from the node `index_one` (for rods)
            or the center of mass (for rigid bodies) to the joint.
            (default = np.array([0.0, 0.0, 0.0]))
        point_system_two : Optional[numpy.ndarray]
            Describes for system two in the local coordinate system the translation from the node `index_two` (for rods)
            or the center of mass (for rigid bodies) to the joint.
            (default = np.array([0.0, 0.0, 0.0]))
        rest_rotation_matrix : Optional[np.ndarray]
            2D (3,3) array containing data with 'float' type.
            Rest 3x3 rotation matrix from system one to system two at the connected elements.
            If provided, the rest rotation matrix is enforced between the two systems throughout the simulation.
            If not provided, `rest_rotation_matrix` is initialized to the identity matrix,
            which means that a restoring torque will be applied to align the directors of both systems directly.
            (default=None)
        """
        super().__init__(
            k=k,
            nu=nu,
            point_system_one=point_system_one,
            point_system_two=point_system_two,
            **kwargs,
        )

        # set rotational spring and damping coefficients
        self.kt = kt
        self.nut = nut

        # TODO: compute the rest rotation matrix directly during initialization
        #  as soon as systems (e.g. `system_one` and `system_two`) and indices (e.g. `index_one` and `index_two`)
        #  are available in the __init__
        if rest_rotation_matrix is None:
            rest_rotation_matrix = np.eye(3)
        assert rest_rotation_matrix.shape == (3, 3), "Rest rotation matrix must be 3x3"
        self.rest_rotation_matrix = rest_rotation_matrix

    # Use the `apply_torques` method of the `FixedJoint` class to apply restoring torques to the connected systems.
    apply_torques = FixedJoint.apply_torques


def compute_position_of_point(system: SystemType, point: np.ndarray, index: int):
    """
    Computes the position in the inertial frame of a point specified in the local frame of
    the specified node of the rod.

    Parameters
    ----------
    system: SystemType
        System to which the point belongs.
    point : np.ndarray
        1D (3,) numpy array containing 'float' data.
        The point describes a position in the local frame relative to the inertial position of node
        with index `index` and orientation of element with `index`.
    index : int
        Index of the node / element in the system.

    Returns
    -------
    position : np.ndarray
        1D (3,) numpy array containing 'float' data.
        Position of the point in the inertial frame.

    Examples
    --------
    Compute position in inertial frame for a point (0, 0, 1) relative to the last node of the rod.

    >>> system.compute_position_of_point(np.array([0, 0, 1]), -1)
    """
    position = (
        system.position_collection[..., index]
        + system.director_collection[..., index].T @ point
    )
    return position


def compute_velocity_of_point(system: SystemType, point: np.ndarray, index: int):
    """
    Computes the velocity in the inertial frame of point specified in the local frame of a node / element.

    Parameters
    ----------
    system: SystemType
        System to which the point belongs.
    point : np.ndarray
        1D (3,) numpy array containing 'float' data.
        The point describes a position in the local frame relative to the inertial position of node
        with index `index` and orientation of element with `index`.
    index : int
        Index of the node / element in the system.

    Returns
    -------
    velocity : np.ndarray
        1D (3,) numpy array containing 'float' data.
        Velocity of the point in the inertial frame.

    Examples
    --------
    Compute velocity in inertial frame for a point (0, 0, 1) relative to the last node of the rod.

    >>> system.compute_velocity_of_point(np.array([0, 0, 1]), -1)
    """
    # point rotated into the inertial frame
    point_inertial_frame = system.director_collection[..., index].T @ point

    # rotate angular velocity to inertial frame
    omega_inertial_frame = np.dot(
        system.director_collection[..., index].T, system.omega_collection[..., index]
    )

    # apply the euler differentiation rule
    velocity = system.velocity_collection[..., index] + np.cross(
        omega_inertial_frame, point_inertial_frame
    )

    return velocity
