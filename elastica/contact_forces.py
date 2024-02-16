__doc__ = """ Numba implementation module containing contact between rods and rigid bodies and other rods rigid bodies or surfaces."""

from elastica.typing import RodType, SystemType, AllowedContactType
from elastica.rod import RodBase
from elastica.rigidbody import Cylinder, Sphere
from elastica.surface import Plane
from elastica.contact_utils import (
    _prune_using_aabbs_rod_cylinder,
    _prune_using_aabbs_rod_rod,
    _prune_using_aabbs_rod_sphere,
)
from elastica._contact_functions import (
    _calculate_contact_forces_rod_cylinder,
    _calculate_contact_forces_rod_rod,
    _calculate_contact_forces_self_rod,
    _calculate_contact_forces_rod_sphere,
    _calculate_contact_forces_rod_plane,
    _calculate_contact_forces_rod_plane_with_anisotropic_friction,
    _calculate_contact_forces_cylinder_plane,
)
import numpy as np


class NoContact:
    """
    This is the base class for contact applied between rod-like objects and allowed contact objects.

    Notes
    -----
    Every new contact class must be derived
    from NoContact class.

    """

    def __init__(self):
        """
        NoContact class does not need any input parameters.
        """

    def _check_systems_validity(
        self,
        system_one: SystemType,
        system_two: AllowedContactType,
    ) -> None:
        """
        This checks the contact order between a SystemType object and an AllowedContactType object, the order should follow: Rod, Rigid body, Surface.
        In NoContact class, this just checks if system_two is a rod then system_one must be a rod.


        Parameters
        ----------
        system_one
            SystemType
        system_two
            AllowedContactType
        """
        if issubclass(system_two.__class__, RodBase):
            if not issubclass(system_one.__class__, RodBase):
                raise TypeError(
                    "Systems provided to the contact class have incorrect order. \n"
                    " First system is {0} and second system is {1}. \n"
                    " If the first system is a rod, the second system can be a rod, rigid body or surface. \n"
                    " If the first system is a rigid body, the second system can be a rigid body or surface.".format(
                        system_one.__class__, system_two.__class__
                    )
                )

    def apply_contact(
        self,
        system_one: SystemType,
        system_two: AllowedContactType,
    ) -> None:
        """
        Apply contact forces and torques between SystemType object and AllowedContactType object.

        In NoContact class, this routine simply passes.

        Parameters
        ----------
        system_one : SystemType
            Rod or rigid-body object
        system_two : AllowedContactType
            Rod, rigid-body, or surface object
        """
        pass


class RodRodContact(NoContact):
    """
    This class is for applying contact forces between rod-rod.

    Examples
    --------
    How to define contact between rod and rod.

    >>> simulator.detect_contact_between(first_rod, second_rod).using(
    ...    RodRodContact,
    ...    k=1e4,
    ...    nu=10,
    ... )

    """

    def __init__(self, k: float, nu: float):
        """
        Parameters
        ----------
        k : float
            Contact spring constant.
        nu : float
            Contact damping constant.
        """
        super(RodRodContact, self).__init__()
        self.k = k
        self.nu = nu

    def _check_systems_validity(
        self,
        system_one: SystemType,
        system_two: AllowedContactType,
    ) -> None:
        """
        This checks the contact order and type of a SystemType object and an AllowedContactType object.
        For the RodRodContact class both systems must be distinct rods.

        Parameters
        ----------
        system_one
            SystemType
        system_two
            AllowedContactType
        """
        if not issubclass(system_one.__class__, RodBase) or not issubclass(
            system_two.__class__, RodBase
        ):
            raise TypeError(
                "Systems provided to the contact class have incorrect order. \n"
                " First system is {0} and second system is {1}. \n"
                " Both systems must be distinct rods".format(
                    system_one.__class__, system_two.__class__
                )
            )
        if system_one == system_two:
            raise TypeError(
                "First rod is identical to second rod. \n"
                "Rods must be distinct for RodRodConact. \n"
                "If you want self contact, use RodSelfContact instead"
            )

    def apply_contact(self, system_one: RodType, system_two: RodType) -> None:
        """
        Apply contact forces and torques between RodType object and RodType object.

        Parameters
        ----------
        system_one: object
            Rod object.
        system_two: object
            Rod object.

        """
        # First, check for a global AABB bounding box, and see whether that
        # intersects

        if _prune_using_aabbs_rod_rod(
            system_one.position_collection,
            system_one.radius,
            system_one.lengths,
            system_two.position_collection,
            system_two.radius,
            system_two.lengths,
        ):
            return

        _calculate_contact_forces_rod_rod(
            system_one.position_collection[
                ..., :-1
            ],  # Discount last node, we want element start position
            system_one.radius,
            system_one.lengths,
            system_one.tangents,
            system_one.velocity_collection,
            system_one.internal_forces,
            system_one.external_forces,
            system_two.position_collection[
                ..., :-1
            ],  # Discount last node, we want element start position
            system_two.radius,
            system_two.lengths,
            system_two.tangents,
            system_two.velocity_collection,
            system_two.internal_forces,
            system_two.external_forces,
            self.k,
            self.nu,
        )


class RodCylinderContact(NoContact):
    """
    This class is for applying contact forces between rod-cylinder.
    If you are want to apply contact forces between rod and cylinder, first system is always rod and second system
    is always cylinder.
    In addition to the contact forces, user can define apply friction forces between rod and cylinder that
    are in contact. For details on friction model refer to this [1]_.

    Notes
    -----
    The `velocity_damping_coefficient` is set to a high value (e.g. 1e4) to minimize slip and simulate stiction
    (static friction), while friction_coefficient corresponds to the Coulombic friction coefficient.

    Examples
    --------
    How to define contact between rod and cylinder.

    >>> simulator.detect_contact_between(rod, cylinder).using(
    ...    RodCylinderContact,
    ...    k=1e4,
    ...    nu=10,
    ... )


    .. [1] Preclik T., Popa Constantin., Rude U., Regularizing a Time-Stepping Method for Rigid Multibody Dynamics, Multibody Dynamics 2011, ECCOMAS. URL: https://www10.cs.fau.de/publications/papers/2011/Preclik_Multibody_Ext_Abstr.pdf
    """

    def __init__(
        self,
        k: float,
        nu: float,
        velocity_damping_coefficient=0.0,
        friction_coefficient=0.0,
    ):
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
        super(RodCylinderContact, self).__init__()
        self.k = k
        self.nu = nu
        self.velocity_damping_coefficient = velocity_damping_coefficient
        self.friction_coefficient = friction_coefficient

    def _check_systems_validity(
        self,
        system_one: SystemType,
        system_two: AllowedContactType,
    ) -> None:
        """
        This checks the contact order and type of a SystemType object and an AllowedContactType object.
        For the RodCylinderContact class first_system should be a rod and second_system should be a cylinder.

        Parameters
        ----------
        system_one
            SystemType
        system_two
            AllowedContactType
        """
        if not issubclass(system_one.__class__, RodBase) or not issubclass(
            system_two.__class__, Cylinder
        ):
            raise TypeError(
                "Systems provided to the contact class have incorrect order/type. \n"
                " First system is {0} and second system is {1}. \n"
                " First system should be a rod, second should be a cylinder".format(
                    system_one.__class__, system_two.__class__
                )
            )

    def apply_contact(self, system_one: RodType, system_two: SystemType) -> None:
        # First, check for a global AABB bounding box, and see whether that
        # intersects
        if _prune_using_aabbs_rod_cylinder(
            system_one.position_collection,
            system_one.radius,
            system_one.lengths,
            system_two.position_collection,
            system_two.director_collection,
            system_two.radius[0],
            system_two.length[0],
        ):
            return

        x_cyl = (
            system_two.position_collection[..., 0]
            - 0.5 * system_two.length * system_two.director_collection[2, :, 0]
        )

        rod_element_position = 0.5 * (
            system_one.position_collection[..., 1:]
            + system_one.position_collection[..., :-1]
        )
        _calculate_contact_forces_rod_cylinder(
            rod_element_position,
            system_one.lengths * system_one.tangents,
            system_two.position_collection[..., 0],
            x_cyl,
            system_two.length * system_two.director_collection[2, :, 0],
            system_one.radius + system_two.radius,
            system_one.lengths + system_two.length,
            system_one.internal_forces,
            system_one.external_forces,
            system_two.external_forces,
            system_two.external_torques,
            system_two.director_collection[:, :, 0],
            system_one.velocity_collection,
            system_two.velocity_collection,
            self.k,
            self.nu,
            self.velocity_damping_coefficient,
            self.friction_coefficient,
        )


class RodSelfContact(NoContact):
    """
    This class is modeling self contact of rod.

    Examples
    --------
    How to define contact rod self contact.

    >>> simulator.detect_contact_between(rod, rod).using(
    ...    RodSelfContact,
    ...    k=1e4,
    ...    nu=10,
    ... )

    """

    def __init__(self, k: float, nu: float):
        """

        Parameters
        ----------
        k : float
            Contact spring constant.
        nu : float
            Contact damping constant.
        """
        super(RodSelfContact, self).__init__()
        self.k = k
        self.nu = nu

    def _check_systems_validity(
        self,
        system_one: SystemType,
        system_two: AllowedContactType,
    ) -> None:
        """
        This checks the contact order and type of a SystemType object and an AllowedContactType object.
        For the RodSelfContact class first_system and second_system should be the same rod.

        Parameters
        ----------
        system_one
            SystemType
        system_two
            AllowedContactType
        """
        if (
            not issubclass(system_one.__class__, RodBase)
            or not issubclass(system_two.__class__, RodBase)
            or system_one != system_two
        ):
            raise TypeError(
                "Systems provided to the contact class have incorrect order/type. \n"
                " First system is {0} and second system is {1}. \n"
                " First system and second system should be the same rod \n"
                " If you want rod rod contact, use RodRodContact instead".format(
                    system_one.__class__, system_two.__class__
                )
            )

    def apply_contact(self, system_one: RodType, system_two: RodType) -> None:
        """
        Apply contact forces and torques between RodType object and itself.

        Parameters
        ----------
        system_one: object
            Rod object.
        system_two: object
            Rod object.

        """
        _calculate_contact_forces_self_rod(
            system_one.position_collection[
                ..., :-1
            ],  # Discount last node, we want element start position
            system_one.radius,
            system_one.lengths,
            system_one.tangents,
            system_one.velocity_collection,
            system_one.external_forces,
            self.k,
            self.nu,
        )


class RodSphereContact(NoContact):
    """
    This class is for applying contact forces between rod-sphere.
    First system is always rod and second system is always sphere.
    In addition to the contact forces, user can define apply friction forces between rod and sphere that
    are in contact. For details on friction model refer to this [1]_.

    Notes
    -----
    The `velocity_damping_coefficient` is set to a high value (e.g. 1e4) to minimize slip and simulate stiction
    (static friction), while friction_coefficient corresponds to the Coulombic friction coefficient.

    Examples
    --------
    How to define contact between rod and sphere.

    >>> simulator.detect_contact_between(rod, sphere).using(
    ...    RodSphereContact,
    ...    k=1e4,
    ...    nu=10,
    ... )

    .. [1] Preclik T., Popa Constantin., Rude U., Regularizing a Time-Stepping Method for Rigid Multibody Dynamics, Multibody Dynamics 2011, ECCOMAS. URL: https://www10.cs.fau.de/publications/papers/2011/Preclik_Multibody_Ext_Abstr.pdf
    """

    def __init__(
        self,
        k: float,
        nu: float,
        velocity_damping_coefficient=0.0,
        friction_coefficient=0.0,
    ):
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
        super(RodSphereContact, self).__init__()
        self.k = k
        self.nu = nu
        self.velocity_damping_coefficient = velocity_damping_coefficient
        self.friction_coefficient = friction_coefficient

    def _check_systems_validity(
        self,
        system_one: SystemType,
        system_two: AllowedContactType,
    ) -> None:
        """
        This checks the contact order and type of a SystemType object and an AllowedContactType object.
        For the RodSphereContact class first_system should be a rod and second_system should be a sphere.
        Parameters
        ----------
        system_one
            SystemType
        system_two
            AllowedContactType
        """
        if not issubclass(system_one.__class__, RodBase) or not issubclass(
            system_two.__class__, Sphere
        ):
            raise TypeError(
                "Systems provided to the contact class have incorrect order/type. \n"
                " First system is {0} and second system is {1}. \n"
                " First system should be a rod, second should be a sphere".format(
                    system_one.__class__, system_two.__class__
                )
            )

    def apply_contact(self, system_one: RodType, system_two: SystemType) -> None:
        """
        Apply contact forces and torques between RodType object and Sphere object.

        Parameters
        ----------
        system_one: object
            Rod object.
        system_two: object
            Sphere object.

        """
        # First, check for a global AABB bounding box, and see whether that
        # intersects
        if _prune_using_aabbs_rod_sphere(
            system_one.position_collection,
            system_one.radius,
            system_one.lengths,
            system_two.position_collection,
            system_two.director_collection,
            system_two.radius[0],
        ):
            return

        x_sph = (
            system_two.position_collection[..., 0]
            - system_two.radius * system_two.director_collection[2, :, 0]
        )

        rod_element_position = 0.5 * (
            system_one.position_collection[..., 1:]
            + system_one.position_collection[..., :-1]
        )
        _calculate_contact_forces_rod_sphere(
            rod_element_position,
            system_one.lengths * system_one.tangents,
            system_two.position_collection[..., 0],
            x_sph,
            system_two.radius * system_two.director_collection[2, :, 0],
            system_one.radius + system_two.radius,
            system_one.lengths + 2 * system_two.radius,
            system_one.internal_forces,
            system_one.external_forces,
            system_two.external_forces,
            system_two.external_torques,
            system_two.director_collection[:, :, 0],
            system_one.velocity_collection,
            system_two.velocity_collection,
            self.k,
            self.nu,
            self.velocity_damping_coefficient,
            self.friction_coefficient,
        )


class RodPlaneContact(NoContact):
    """
    This class is for applying contact forces between rod-plane.
    First system is always rod and second system is always plane.
    For more details regarding the contact module refer to
    Eqn 4.8 of Gazzola et al. RSoS (2018).

    Examples
    --------
    How to define contact between rod and plane.

    >>> simulator.detect_contact_between(rod, plane).using(
    ...    RodPlaneContact,
    ...    k=1e4,
    ...    nu=10,
    ... )
    """

    def __init__(
        self,
        k: float,
        nu: float,
    ):
        """
        Parameters
        ----------
        k : float
            Contact spring constant.
        nu : float
            Contact damping constant.
        """
        super(RodPlaneContact, self).__init__()
        self.k = k
        self.nu = nu
        self.surface_tol = 1e-4

    def _check_systems_validity(
        self,
        system_one: SystemType,
        system_two: AllowedContactType,
    ) -> None:
        """
        This checks the contact order and type of a SystemType object and an AllowedContactType object.
        For the RodPlaneContact class first_system should be a rod and second_system should be a plane.
        Parameters
        ----------
        system_one
            SystemType
        system_two
            AllowedContactType
        """
        if not issubclass(system_one.__class__, RodBase) or not issubclass(
            system_two.__class__, Plane
        ):
            raise TypeError(
                "Systems provided to the contact class have incorrect order/type. \n"
                " First system is {0} and second system is {1}. \n"
                " First system should be a rod, second should be a plane".format(
                    system_one.__class__, system_two.__class__
                )
            )

    def apply_contact(self, system_one: RodType, system_two: SystemType) -> None:
        """
        Apply contact forces and torques between RodType object and Plane object.

        Parameters
        ----------
        system_one: object
            Rod object.
        system_two: object
            Plane object.

        """
        _calculate_contact_forces_rod_plane(
            system_two.origin,
            system_two.normal,
            self.surface_tol,
            self.k,
            self.nu,
            system_one.radius,
            system_one.mass,
            system_one.position_collection,
            system_one.velocity_collection,
            system_one.internal_forces,
            system_one.external_forces,
        )


class RodPlaneContactWithAnisotropicFriction(NoContact):
    """
    This class is for applying contact forces between rod-plane with friction.
    First system is always rod and second system is always plane.
    For more details regarding the contact module refer to
    Eqn 4.8 of Gazzola et al. RSoS (2018).

    Examples
    --------
    How to define contact between rod and plane.

    >>> simulator.detect_contact_between(rod, plane).using(
    ...    RodPlaneContactWithAnisotropicFriction,
    ...    k=1e4,
    ...    nu=10,
    ...    slip_velocity_tol = 1e-4,
    ...    static_mu_array = np.array([0.0,0.0,0.0]),
    ...    kinetic_mu_array = np.array([1.0,2.0,3.0]),
    ... )
    """

    def __init__(
        self,
        k: float,
        nu: float,
        slip_velocity_tol: float,
        static_mu_array: np.ndarray,
        kinetic_mu_array: np.ndarray,
    ):
        """
        Parameters
        ----------
        k : float
            Contact spring constant.
        nu : float
            Contact damping constant.
        slip_velocity_tol: float
            Velocity tolerance to determine if the element is slipping or not.
        static_mu_array: numpy.ndarray
            1D (3,) array containing data with 'float' type.
            [forward, backward, sideways] static friction coefficients.
        kinetic_mu_array: numpy.ndarray
            1D (3,) array containing data with 'float' type.
            [forward, backward, sideways] kinetic friction coefficients.
        """
        super(RodPlaneContactWithAnisotropicFriction, self).__init__()
        self.k = k
        self.nu = nu
        self.surface_tol = 1e-4
        self.slip_velocity_tol = slip_velocity_tol
        (
            self.static_mu_forward,
            self.static_mu_backward,
            self.static_mu_sideways,
        ) = static_mu_array
        (
            self.kinetic_mu_forward,
            self.kinetic_mu_backward,
            self.kinetic_mu_sideways,
        ) = kinetic_mu_array

    def _check_systems_validity(
        self,
        system_one: SystemType,
        system_two: AllowedContactType,
    ) -> None:
        """
        This checks the contact order and type of a SystemType object and an AllowedContactType object.
        For the RodSphereContact class first_system should be a rod and second_system should be a plane.
        Parameters
        ----------
        system_one
            SystemType
        system_two
            AllowedContactType
        """
        if not issubclass(system_one.__class__, RodBase) or not issubclass(
            system_two.__class__, Plane
        ):
            raise TypeError(
                "Systems provided to the contact class have incorrect order/type. \n"
                " First system is {0} and second system is {1}. \n"
                " First system should be a rod, second should be a plane".format(
                    system_one.__class__, system_two.__class__
                )
            )

    def apply_contact(self, system_one: RodType, system_two: SystemType) -> None:
        """
        Apply contact forces and torques between RodType object and Plane object with anisotropic friction.

        Parameters
        ----------
        system_one: object
            Rod object.
        system_two: object
            Plane object.

        """

        _calculate_contact_forces_rod_plane_with_anisotropic_friction(
            system_two.origin,
            system_two.normal,
            self.surface_tol,
            self.slip_velocity_tol,
            self.k,
            self.nu,
            self.kinetic_mu_forward,
            self.kinetic_mu_backward,
            self.kinetic_mu_sideways,
            self.static_mu_forward,
            self.static_mu_backward,
            self.static_mu_sideways,
            system_one.radius,
            system_one.mass,
            system_one.tangents,
            system_one.position_collection,
            system_one.director_collection,
            system_one.velocity_collection,
            system_one.omega_collection,
            system_one.internal_forces,
            system_one.external_forces,
            system_one.internal_torques,
            system_one.external_torques,
        )


class CylinderPlaneContact(NoContact):
    """
    This class is for applying contact forces between cylinder-plane.
    First system is always cylinder and second system is always plane.
    For more details regarding the contact module refer to
    Eqn 4.8 of Gazzola et al. RSoS (2018).

    Examples
    --------
    How to define contact between cylinder and plane.

    >>> simulator.detect_contact_between(cylinder, plane).using(
    ...    CylinderPlaneContact,
    ...    k=1e4,
    ...    nu=10,
    ... )
    """

    def __init__(
        self,
        k: float,
        nu: float,
    ):
        """
        Parameters
        ----------
        k : float
            Contact spring constant.
        nu : float
            Contact damping constant.
        """
        super(CylinderPlaneContact, self).__init__()
        self.k = k
        self.nu = nu
        self.surface_tol = 1e-4

    def _check_systems_validity(
        self,
        system_one: SystemType,
        system_two: AllowedContactType,
    ) -> None:
        """
        This checks the contact order and type of a SystemType object and an AllowedContactType object.
        For the RodPlaneContact class first_system should be a cylinder and second_system should be a plane.
        Parameters
        ----------
        system_one
            SystemType
        system_two
            AllowedContactType
        """
        if not issubclass(system_one.__class__, Cylinder) or not issubclass(
            system_two.__class__, Plane
        ):
            raise TypeError(
                "Systems provided to the contact class have incorrect order/type. \n"
                " First system is {0} and second system is {1}. \n"
                " First system should be a cylinder, second should be a plane".format(
                    system_one.__class__, system_two.__class__
                )
            )

    def apply_contact(self, system_one: Cylinder, system_two: SystemType):
        """
        This function computes the plane force response on the cylinder, in the
        case of contact. Contact model given in Eqn 4.8 Gazzola et. al. RSoS 2018 paper
        is used.

        Parameters
        ----------
        system_one: object
            Cylinder object.
        system_two: object
            Plane object.

        """
        return _calculate_contact_forces_cylinder_plane(
            system_two.origin,
            system_two.normal,
            self.surface_tol,
            self.k,
            self.nu,
            system_one.length,
            system_one.position_collection,
            system_one.velocity_collection,
            system_one.external_forces,
        )
