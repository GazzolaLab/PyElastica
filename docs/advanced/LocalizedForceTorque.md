# Localized Force and Torque

> Originated by the investigation in the [issue #39](https://github.com/GazzolaLab/PyElastica/issues/39)

## Discussion

In elastica, __a force is applied on a node__ while __a torque is applied on an element__.
For example, a localized force `EndpointForce` is applied only on a node. However, we found that adding additional torque on a neighboring elements, such that the torque represent a local moment induced by the point-force, could yield better convergence.
We haven't found any evidence (yet) that this actually changes the steady-state configuration and kinematics, since it is two different implementation of the same point-load.
We suspect the improvement by adding additional torque is due to explicitly giving the force-boundary condition that match the final internal-stress state.

## Comparison

Factoring the additional-torque on a neighboring element leads to slightly better error estimates for the Timoshenko beam example. The results are condensed here.
With new implementation, we achieved the same error with less number of discretization, but it also requires additional torque computation.

![image](https://github.com/GazzolaLab/PyElastica/blob/assets/docs/assets/plot/error_EndpointForcesWithTorques.png?raw=true)

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
