from elastica._linalg import _batch_matvec


class RodBase:
    """
    Base class for all rods
    # TODO : What needs to be ported here?

    # The interface class, as seen from global scope
    # Can be made common to all entities in the code
    """

    def __init__(self):
        pass

    def get_velocity(self):
        return self.velocity_collection

    def get_angular_velocity(self):
        return self.omega_collection

    def get_acceleration(self):
        return (self._compute_internal_forces() + self.external_forces) / self.mass

    def get_angular_acceleration(self):
        return (
            _batch_matvec(
                self.inv_mass_second_moment_of_inertia,
                (self._compute_internal_torques() + self.external_torques),
            )
            * self.dilatation
        )
