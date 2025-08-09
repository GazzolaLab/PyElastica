__doc__ = """ Numba implementation module containing contact between rods and rigid bodies and other rods rigid bodies or surfaces."""

from typing import TypeVar, Generic, Type
from elastica.typing import RodType, SystemType, SurfaceType

from elastica.rod.rod_base import RodBase
from elastica.rigidbody.cylinder import Cylinder
from elastica.rigidbody.sphere import Sphere
from elastica.surface.plane import Plane
from elastica.surface.surface_base import SurfaceBase
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
from numpy.typing import NDArray


S1 = TypeVar("S1")  # TODO: Find bound
S2 = TypeVar("S2")


class NoContact(Generic[S1, S2]):
    """
    This is the base class for contact applied between rod-like objects and allowed contact objects.

    Notes
    -----
    Every new contact class must be derived
    from NoContact class.

    """

    def __init__(self) -> None:
        """
        NoContact class does not need any input parameters.
        """

    @property
    def _allowed_system_one(self) -> list[Type]:
        # Modify this list to include the allowed system types for contact
        return [RodBase]

    @property
    def _allowed_system_two(self) -> list[Type]:
        # Modify this list to include the allowed system types for contact
        return [RodBase]

    def _check_systems_validity(
        self,
        system_one: S1,
        system_two: S2,
    ) -> None:
        """
        Here, we check the allowed system types for contact.
        For derived classes, this method can be overridden to enforce specific system types
        for contact model.
        """

        common_check_systems_validity(system_one, self._allowed_system_one)
        common_check_systems_validity(system_two, self._allowed_system_two)

        common_check_systems_identity(system_one, system_two)

    def apply_contact(
        self,
        system_one: S1,
        system_two: S2,
        time: np.float64 = np.float64(0.0),
    ) -> None:
        """
        Apply contact forces and torques between two system object..

        In NoContact class, this routine simply passes.

        Parameters
        ----------
        system_one
        system_two
        """


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

    def __init__(self, k: np.float64, nu: np.float64) -> None:
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

    def apply_contact(
        self,
        system_one: RodType,
        system_two: RodType,
        time: np.float64 = np.float64(0.0),
    ) -> None:
        """
        Apply contact forces and torques between RodType object and RodType object.

        Parameters
        ----------
        system_one: RodType
        system_two: RodType

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
        velocity_damping_coefficient: float = 0.0,
        friction_coefficient: float = 0.0,
    ) -> None:
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
        self.k = np.float64(k)
        self.nu = np.float64(nu)
        self.velocity_damping_coefficient = np.float64(velocity_damping_coefficient)
        self.friction_coefficient = np.float64(friction_coefficient)

    @property
    def _allowed_system_two(self) -> list[Type]:
        # Modify this list to include the allowed system types for contact
        return [Cylinder]

    def apply_contact(
        self,
        system_one: RodType,
        system_two: Cylinder,
        time: np.float64 = np.float64(0.0),
    ) -> None:
        # First, check for a global AABB bounding box, and see whether that
        # intersects
        if _prune_using_aabbs_rod_cylinder(
            system_one.position_collection,
            system_one.radius,
            system_one.lengths,
            system_two.position_collection,
            system_two.director_collection,
            system_two.radius,
            system_two.length,
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

    def __init__(self, k: float, nu: float) -> None:
        """

        Parameters
        ----------
        k : float
            Contact spring constant.
        nu : float
            Contact damping constant.
        """
        super(RodSelfContact, self).__init__()
        self.k = np.float64(k)
        self.nu = np.float64(nu)

    def _check_systems_validity(
        self,
        system_one: RodType,
        system_two: RodType,
    ) -> None:
        """
        Overriding the base class method to check if the two systems are identical.
        """
        common_check_systems_validity(system_one, self._allowed_system_one)
        common_check_systems_validity(system_two, self._allowed_system_two)
        common_check_systems_different(system_one, system_two)

    def apply_contact(
        self,
        system_one: RodType,
        system_two: RodType,
        time: np.float64 = np.float64(0.0),
    ) -> None:
        """
        Apply contact forces and torques between RodType object and itself.

        Parameters
        ----------
        system_one: RodType
        system_two: RodType

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
        velocity_damping_coefficient: float = 0.0,
        friction_coefficient: float = 0.0,
    ) -> None:
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
        self.k = np.float64(k)
        self.nu = np.float64(nu)
        self.velocity_damping_coefficient = np.float64(velocity_damping_coefficient)
        self.friction_coefficient = np.float64(friction_coefficient)

    @property
    def _allowed_system_two(self) -> list[Type]:
        return [Sphere]

    def apply_contact(
        self,
        system_one: RodType,
        system_two: Sphere,
        time: np.float64 = np.float64(0.0),
    ) -> None:
        """
        Apply contact forces and torques between RodType object and Sphere object.

        Parameters
        ----------
        system_one: RodType
        system_two: Sphere

        """
        # First, check for a global AABB bounding box, and see whether that
        # intersects
        if _prune_using_aabbs_rod_sphere(
            system_one.position_collection,
            system_one.radius,
            system_one.lengths,
            system_two.position_collection,
            system_two.director_collection,
            system_two.radius,
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
    ) -> None:
        """
        Parameters
        ----------
        k : float
            Contact spring constant.
        nu : float
            Contact damping constant.
        """
        super(RodPlaneContact, self).__init__()
        self.k = np.float64(k)
        self.nu = np.float64(nu)
        self.surface_tol = np.float64(1.0e-4)

    @property
    def _allowed_system_two(self) -> list[Type]:
        return [SurfaceBase]

    def apply_contact(
        self,
        system_one: RodType,
        system_two: SurfaceType,
        time: np.float64 = np.float64(0.0),
    ) -> None:
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
        static_mu_array: NDArray[np.float64],
        kinetic_mu_array: NDArray[np.float64],
    ) -> None:
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
        self.k = np.float64(k)
        self.nu = np.float64(nu)
        self.surface_tol = np.float64(1.0e-4)
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

    @property
    def _allowed_system_two(self) -> list[Type]:
        return [SurfaceBase]

    def apply_contact(
        self,
        system_one: RodType,
        system_two: SurfaceType,
        time: np.float64 = np.float64(0.0),
    ) -> None:
        """
        Apply contact forces and torques between RodType object and Plane object with anisotropic friction.

        Parameters
        ----------
        system_one: RodType
        system_two: SurfaceType

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
    ) -> None:
        """
        Parameters
        ----------
        k : float
            Contact spring constant.
        nu : float
            Contact damping constant.
        """
        super(CylinderPlaneContact, self).__init__()
        self.k = np.float64(k)
        self.nu = np.float64(nu)
        self.surface_tol = np.float64(1.0e-4)

    @property
    def _allowed_system_one(self) -> list[Type]:
        return [Cylinder]

    @property
    def _allowed_system_two(self) -> list[Type]:
        return [SurfaceBase]

    def apply_contact(
        self,
        system_one: Cylinder,
        system_two: SurfaceType,
        time: np.float64 = np.float64(0.0),
    ) -> None:
        """
        This function computes the plane force response on the cylinder, in the
        case of contact. Contact model given in Eqn 4.8 Gazzola et. al. RSoS 2018 paper
        is used.

        Parameters
        ----------
        system_one: Cylinder
        system_two: SurfaceBase

        """
        _calculate_contact_forces_cylinder_plane(
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


def common_check_systems_identity(
    system_one: S1,
    system_two: S2,
) -> None:
    """
    This checks if two objects are identical.

    Raises
    ------
    TypeError
        If two objects are identical.
    """
    if system_one == system_two:
        raise TypeError(
            "First system is identical to second system. Systems must be distinct for contact."
        )


def common_check_systems_different(
    system_one: S1,
    system_two: S2,
) -> None:
    """
    This checks if two objects are identical.

    Raises
    ------
    TypeError
        If two objects are not identical.
    """
    if system_one != system_two:
        raise TypeError("First system must be identical to the second system.")


def common_check_systems_validity(
    system: S1 | S2, allowed_system: list[Type[S1] | Type[S2]]
) -> None:
    # Check validity
    if not isinstance(system, tuple(allowed_system)):
        system_name = system.__class__.__name__
        allowed_system_names = [candidate.__name__ for candidate in allowed_system]
        raise TypeError(
            f"System provided ({system_name}) must be derived from {allowed_system_names}."
        )
