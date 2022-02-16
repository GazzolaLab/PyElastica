# Localized Force and Torque

> Originated from the inquiry in the issue #39 

## Discussion

## Comparison

<F24><F25>

## Modified Implementation

```py
class EndpointForcesWithTorques(NoForces):
    """
    This class applies constant forces on the endpoint nodes.
    """

    def __init__(self, end_force, ramp_up_time=0.0):
        """

        Parameters
        ----------
        start_force: numpy.ndarray
            2D (dim, 1) array containing data with 'float' type.
            Force applied to first node of the rod-like object.
        end_force: numpy.ndarray
            2D (dim, 1) array containing data with 'float' type.
            Force applied to last node of the rod-like object.
        ramp_up_time: float
            Applied forces are ramped up until ramp up time.

        """
        self.end_force = end_force
        assert ramp_up_time >= 0.0
        self.ramp_up_time = ramp_up_time

    def apply_forces(self, system, time=0.0):
        # factor = min(1.0, time / self.ramp_up_time)
        #
        # system.external_forces[..., 0] += self.start_force * factor
        # system.external_forces[..., -1] += self.end_force * factor

        factor = min(1.0, time / self.ramp_up_time)
        self.external_forces[..., -1] += self.end_force * factor

    def apply_torques(self, system, time: np.float64 = 0.0):
		factor = min(1.0, time / self.ramp_up_time)
		arm_length = system.lengths[...,-1]
		director = system.director_collection[..., -1]
        self.external_torques[..., -1] += np.cross(
            [0.0, 0.0, 0.5 * arm_length], director @ self.end_force
        )

```
