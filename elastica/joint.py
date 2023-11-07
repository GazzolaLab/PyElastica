__doc__ = """ Module containing joint classes to connect multiple rods together. """
from elastica._rotations import _inv_rotate
from elastica.typing import SystemType, RodType
import numpy as np
import logging


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

    def apply_forces(
        self, system_one: SystemType, index_one, system_two: SystemType, index_two
    ):
        """
        Apply joint force to the connected rod objects.

        Parameters
        ----------
        system_one : object
            Rod or rigid-body object
        index_one : int
            Index of first rod for joint.
        system_two : object
            Rod or rigid-body object
        index_two : int
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
        self, system_one: SystemType, index_one, system_two: SystemType, index_two
    ):
        """
        Apply restoring joint torques to the connected rod objects.

        In FreeJoint class, this routine simply passes.

        Parameters
        ----------
        system_one : object
            Rod or rigid-body object
        index_one : int
            Index of first rod for joint.
        system_two : object
            Rod or rigid-body object
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
    def apply_forces(
        self,
        system_one: SystemType,
        index_one,
        system_two: SystemType,
        index_two,
    ):
        return super().apply_forces(system_one, index_one, system_two, index_two)

    def apply_torques(
        self,
        system_one: SystemType,
        index_one,
        system_two: SystemType,
        index_two,
    ):
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

    def __init__(self, k, nu, kt, nut=0.0, rest_rotation_matrix=None):
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
        rest_rotation_matrix: np.array
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
        self.kt = kt
        self.nut = nut

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
        system_one: SystemType,
        index_one,
        system_two: SystemType,
        index_two,
    ):
        return super().apply_forces(system_one, index_one, system_two, index_two)

    def apply_torques(
        self,
        system_one: SystemType,
        index_one,
        system_two: SystemType,
        index_two,
    ):
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
    system_one: SystemType,
    index_one,
    system_two: SystemType,
    index_two,
):
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
    system_one : SystemType
        Rod or rigid-body object
    index_one : int
        Index of first rod for joint.
    system_two : SystemType
        Rod or rigid-body object
    index_two : int
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


# everything below this comment should be removed beyond v0.4.0
def _dot_product(a, b):
    raise NotImplementedError(
        "This function is removed in v0.3.2. Please use\n"
        "elastica.contact_utils._dot_product()\n"
        "instead for find the dot product between a and b."
    )


def _norm(a):
    raise NotImplementedError(
        "This function is removed in v0.3.2. Please use\n"
        "elastica.contact_utils._norm()\n"
        "instead for finding the norm of a."
    )


def _clip(x, low, high):
    raise NotImplementedError(
        "This function is removed in v0.3.2. Please use\n"
        "elastica.contact_utils._clip()\n"
        "instead for clipping x."
    )


def _out_of_bounds(x, low, high):
    raise NotImplementedError(
        "This function is removed in v0.3.2. Please use\n"
        "elastica.contact_utils._out_of_bounds()\n"
        "instead for checking if x is out of bounds."
    )


def _find_min_dist(x1, e1, x2, e2):
    raise NotImplementedError(
        "This function is removed in v0.3.2. Please use\n"
        "elastica.contact_utils._find_min_dist()\n"
        "instead for finding minimum distance between contact points."
    )


def _calculate_contact_forces_rod_rigid_body(
    x_collection_rod,
    edge_collection_rod,
    x_cylinder_center,
    x_cylinder_tip,
    edge_cylinder,
    radii_sum,
    length_sum,
    internal_forces_rod,
    external_forces_rod,
    external_forces_cylinder,
    external_torques_cylinder,
    cylinder_director_collection,
    velocity_rod,
    velocity_cylinder,
    contact_k,
    contact_nu,
    velocity_damping_coefficient,
    friction_coefficient,
):
    raise NotImplementedError(
        "This function is removed in v0.3.2. Please use\n"
        "elastica._contact_functions._calculate_contact_forces_rod_cylinder()\n"
        "instead for calculating rod cylinder contact forces."
    )


def _calculate_contact_forces_rod_rod(
    x_collection_rod_one,
    radius_rod_one,
    length_rod_one,
    tangent_rod_one,
    velocity_rod_one,
    internal_forces_rod_one,
    external_forces_rod_one,
    x_collection_rod_two,
    radius_rod_two,
    length_rod_two,
    tangent_rod_two,
    velocity_rod_two,
    internal_forces_rod_two,
    external_forces_rod_two,
    contact_k,
    contact_nu,
):
    raise NotImplementedError(
        "This function is removed in v0.3.2. Please use\n"
        "elastica._contact_functions._calculate_contact_forces_rod_rod()\n"
        "instead for calculating rod rod contact forces."
    )


def _calculate_contact_forces_self_rod(
    x_collection_rod,
    radius_rod,
    length_rod,
    tangent_rod,
    velocity_rod,
    external_forces_rod,
    contact_k,
    contact_nu,
):
    raise NotImplementedError(
        "This function is removed in v0.3.2. Please use\n"
        "elastica._contact_functions._calculate_contact_forces_self_rod()\n"
        "instead for calculating rod self-contact forces."
    )


def _aabbs_not_intersecting(aabb_one, aabb_two):
    raise NotImplementedError(
        "This function is removed in v0.3.2. Please use\n"
        "elastica.contact_utils._aabbs_not_intersecting()\n"
        "instead for checking aabbs intersection."
    )


def _prune_using_aabbs_rod_rigid_body(
    rod_one_position_collection,
    rod_one_radius_collection,
    rod_one_length_collection,
    cylinder_position,
    cylinder_director,
    cylinder_radius,
    cylinder_length,
):
    raise NotImplementedError(
        "This function is removed in v0.3.2. Please use\n"
        "elastica.contact_utils._prune_using_aabbs_rod_cylinder()\n"
        "instead for checking rod cylinder intersection."
    )


def _prune_using_aabbs_rod_rod(
    rod_one_position_collection,
    rod_one_radius_collection,
    rod_one_length_collection,
    rod_two_position_collection,
    rod_two_radius_collection,
    rod_two_length_collection,
):
    raise NotImplementedError(
        "This function is removed in v0.3.2. Please use\n"
        "elastica.contact_utils._prune_using_aabbs_rod_rod()\n"
        "instead for checking rod rod intersection."
    )


class ExternalContact(FreeJoint):
    """
    This class is for applying contact forces between rod-cylinder and rod-rod.
    If you are want to apply contact forces between rod and cylinder, first system is always rod and second system
    is always cylinder.
    In addition to the contact forces, user can define apply friction forces between rod and cylinder that
    are in contact. For details on friction model refer to this [1]_.
    TODO: Currently friction force is between rod-cylinder, in future implement friction forces between rod-rod.

    Notes
    -----
    The `velocity_damping_coefficient` is set to a high value (e.g. 1e4) to minimize slip and simulate stiction
    (static friction), while friction_coefficient corresponds to the Coulombic friction coefficient.

    Examples
    --------
    How to define contact between rod and cylinder.

    >>> simulator.connect(rod, cylinder).using(
    ...    ExternalContact,
    ...    k=1e4,
    ...    nu=10,
    ...    velocity_damping_coefficient=10,
    ...    kinetic_friction_coefficient=10,
    ... )

    How to define contact between rod and rod.

    >>> simulator.connect(rod, rod).using(
    ...    ExternalContact,
    ...    k=1e4,
    ...    nu=10,
    ... )

    .. [1] Preclik T., Popa Constantin., Rude U., Regularizing a Time-Stepping Method for Rigid Multibody Dynamics, Multibody Dynamics 2011, ECCOMAS. URL: https://www10.cs.fau.de/publications/papers/2011/Preclik_Multibody_Ext_Abstr.pdf
    """

    # Dev note:
    # Most of the cylinder-cylinder contact SHOULD be implemented
    # as given in this `paper <http://larochelle.sdsmt.edu/publications/2005-2009/Collision%20Detection%20of%20Cylindrical%20Rigid%20Bodies%20Using%20Line%20Geometry.pdf>`,
    # but the elastica-cpp kernels are implemented.
    # This is maybe to speed-up the kernel, but it's
    # potentially dangerous as it does not deal with "end" conditions
    # correctly.

    def __init__(self, k, nu, velocity_damping_coefficient=0, friction_coefficient=0):
        """

        Parameters
        ----------
        k : float
            Contact spring constant.
        nu : float
            Contact damping constant.
        velocity_damping_coefficient : float
            Velocity damping coefficient between rigid-body and rod contact is used to apply friction force in the
            slip direction.
        friction_coefficient : float
            For Coulombic friction coefficient for rigid-body and rod contact.
        """
        super().__init__(k, nu)
        self.velocity_damping_coefficient = velocity_damping_coefficient
        self.friction_coefficient = friction_coefficient
        log = logging.getLogger(self.__class__.__name__)
        log.warning(
            # Remove warning and add error if ExternalContact is used in v0.3.3
            # Remove the option to use ExternalContact, beyond v0.3.3
            "The option to use the ExternalContact joint for the rod-rod and rod-cylinder contact is now deprecated.\n"
            "Instead, for rod-rod contact or rod-cylinder contact,use RodRodContact or RodCylinderContact from the add-on Contact mixin class.\n"
            "For reference see the classes elastica.contact_forces.RodRodContact() and elastica.contact_forces.RodCylinderContact().\n"
            "For usage check examples/RigidbodyCases/RodRigidBodyContact/rod_cylinder_contact.py and examples/RodContactCase/RodRodContact/rod_rod_contact_parallel_validation.py.\n"
            " The option to use the ExternalContact joint for the rod-rod and rod-cylinder will be removed in the future (v0.3.3).\n"
        )

    def apply_forces(
        self,
        rod_one: RodType,
        index_one,
        rod_two: SystemType,
        index_two,
    ):
        # del index_one, index_two
        from elastica.contact_utils import (
            _prune_using_aabbs_rod_cylinder,
            _prune_using_aabbs_rod_rod,
        )
        from elastica._contact_functions import (
            _calculate_contact_forces_rod_cylinder,
            _calculate_contact_forces_rod_rod,
        )

        # TODO: raise error during the initialization if rod one is rigid body.

        # If rod two has one element, then it is rigid body.
        if rod_two.n_elems == 1:
            cylinder_two = rod_two
            # First, check for a global AABB bounding box, and see whether that
            # intersects
            if _prune_using_aabbs_rod_cylinder(
                rod_one.position_collection,
                rod_one.radius,
                rod_one.lengths,
                cylinder_two.position_collection,
                cylinder_two.director_collection,
                cylinder_two.radius[0],
                cylinder_two.length[0],
            ):
                return

            x_cyl = (
                cylinder_two.position_collection[..., 0]
                - 0.5 * cylinder_two.length * cylinder_two.director_collection[2, :, 0]
            )

            rod_element_position = 0.5 * (
                rod_one.position_collection[..., 1:]
                + rod_one.position_collection[..., :-1]
            )
            _calculate_contact_forces_rod_cylinder(
                rod_element_position,
                rod_one.lengths * rod_one.tangents,
                cylinder_two.position_collection[..., 0],
                x_cyl,
                cylinder_two.length * cylinder_two.director_collection[2, :, 0],
                rod_one.radius + cylinder_two.radius,
                rod_one.lengths + cylinder_two.length,
                rod_one.internal_forces,
                rod_one.external_forces,
                cylinder_two.external_forces,
                cylinder_two.external_torques,
                cylinder_two.director_collection[:, :, 0],
                rod_one.velocity_collection,
                cylinder_two.velocity_collection,
                self.k,
                self.nu,
                self.velocity_damping_coefficient,
                self.friction_coefficient,
            )

        else:
            # First, check for a global AABB bounding box, and see whether that
            # intersects

            if _prune_using_aabbs_rod_rod(
                rod_one.position_collection,
                rod_one.radius,
                rod_one.lengths,
                rod_two.position_collection,
                rod_two.radius,
                rod_two.lengths,
            ):
                return

            _calculate_contact_forces_rod_rod(
                rod_one.position_collection[
                    ..., :-1
                ],  # Discount last node, we want element start position
                rod_one.radius,
                rod_one.lengths,
                rod_one.tangents,
                rod_one.velocity_collection,
                rod_one.internal_forces,
                rod_one.external_forces,
                rod_two.position_collection[
                    ..., :-1
                ],  # Discount last node, we want element start position
                rod_two.radius,
                rod_two.lengths,
                rod_two.tangents,
                rod_two.velocity_collection,
                rod_two.internal_forces,
                rod_two.external_forces,
                self.k,
                self.nu,
            )


class SelfContact(FreeJoint):
    """
    This class is modeling self contact of rod.

    """

    def __init__(self, k, nu):
        super().__init__(k, nu)
        log = logging.getLogger(self.__class__.__name__)
        log.warning(
            # Remove warning and add error if SelfContact is used in v0.3.3
            # Remove the option to use SelfContact, beyond v0.3.3
            "The option to use the SelfContact joint for the rod self contact is now deprecated.\n"
            "Instead, for rod self contact use RodSelfContact from the add-on Contact mixin class.\n"
            "For reference see the class elastica.contact_forces.RodSelfContact(), and for usage check examples/RodContactCase/RodSelfContact/solenoids.py.\n"
            "The option to use the SelfContact joint for the rod self contact will be removed in the future (v0.3.3).\n"
        )

    def apply_forces(self, rod_one: RodType, index_one, rod_two: SystemType, index_two):
        # del index_one, index_two
        from elastica._contact_functions import (
            _calculate_contact_forces_self_rod,
        )

        _calculate_contact_forces_self_rod(
            rod_one.position_collection[
                ..., :-1
            ],  # Discount last node, we want element start position
            rod_one.radius,
            rod_one.lengths,
            rod_one.tangents,
            rod_one.velocity_collection,
            rod_one.external_forces,
            self.k,
            self.nu,
        )
