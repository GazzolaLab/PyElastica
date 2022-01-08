import numpy as np
from elastica.external_forces import NoForces


# Force class. This force class is used only for joint test cases
class EndpointForcesSinusoidal(NoForces):
    """Applies sinusoidal forces on endpoints"""

    def __init__(
        self,
        start_force_mag,
        end_force_mag,
        ramp_up_time=0.0,
        tangent_direction=np.array([0, 0, 1]),
        normal_direction=np.array([0, 1, 0]),
    ):
        super(EndpointForcesSinusoidal, self).__init__()
        # Start force
        self.start_force_mag = start_force_mag
        self.end_force_mag = end_force_mag

        # Applied force directions
        self.normal_direction = normal_direction
        # self.roll_direction = np.cross(tangent_direction, normal_direction)
        self.roll_direction = np.cross(normal_direction, tangent_direction)

        assert ramp_up_time >= 0.0
        self.ramp_up_time = ramp_up_time

    def apply_forces(self, system, time=0.0):

        if time < self.ramp_up_time:
            # When time smaller than ramp up time apply the force in normal direction
            # First pull the rod upward or downward direction some time.
            start_force = -2.0 * self.start_force_mag * self.normal_direction
            end_force = -2.0 * self.end_force_mag * self.normal_direction

            system.external_forces[..., 0] += start_force
            system.external_forces[..., -1] += end_force

        else:
            # When time is greater than ramp up time, forces are applied in normal
            # and roll direction or forces are in a plane perpendicular to the
            # direction.

            # First force applied to start of the rod
            roll_forces_start = (
                self.start_force_mag
                * np.cos(0.5 * np.pi * (time - self.ramp_up_time))
                * self.roll_direction
            )
            normal_forces_start = (
                self.start_force_mag
                * np.sin(0.5 * np.pi * (time - self.ramp_up_time))
                * self.normal_direction
            )
            start_force = roll_forces_start + normal_forces_start
            # Now force applied to end of the rod
            roll_forces_end = (
                self.end_force_mag
                * np.cos(0.5 * np.pi * (time - self.ramp_up_time))
                * self.roll_direction
            )
            normal_forces_end = (
                self.end_force_mag
                * np.sin(0.5 * np.pi * (time - self.ramp_up_time))
                * self.normal_direction
            )
            end_force = roll_forces_end + normal_forces_end
            # Update external forces
            system.external_forces[..., 0] += start_force
            system.external_forces[..., -1] += end_force
