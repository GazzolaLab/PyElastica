from elastica import *
from elastica.joint import (
    _prune_using_aabbs_rod_rod,
    _prune_using_aabbs_rod_rigid_body,
    _calculate_contact_forces_rod_rod,
    _calculate_contact_forces_rod_rigid_body,
)
import numpy as np


class ExternalContactMemoryBlock(FreeJoint):
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

    def __init__(
        self, k, nu, velocity_damping_coefficient=0, friction_coefficient=0, **kwargs
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
        super().__init__(k, nu)
        self.velocity_damping_coefficient = velocity_damping_coefficient
        self.friction_coefficient = friction_coefficient
        self.k = np.array(k)
        self.nu = np.array(nu)

    def apply_forces(
        self,
        rod_one: RodType,
        index_one,
        rod_two: SystemType,
        index_two,
    ):
        # del index_one, index_two

        # TODO: raise error during the initialization if rod one is rigid body.

        # If rod two has one element, then it is rigid body.
        if rod_two.n_elems == 1:
            cylinder_two = rod_two
            # First, check for a global AABB bounding box, and see whether that
            # intersects
            if _prune_using_aabbs_rod_rigid_body(
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
            _calculate_contact_forces_rod_rigid_body(
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
