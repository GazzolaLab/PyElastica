from typing import TypeAlias, Callable

import numpy as np
from numba import njit
from elastica.typing import SystemType
from elastica.external_forces import NoForces

Position: TypeAlias = np.ndarray  # vector (3)
Orientation: TypeAlias = np.ndarray  # SO3 matrix (3, 3)
Pose: TypeAlias = tuple[Position, Orientation]


class TargetPoseProportionalControl(NoForces):
    """
    This class applies directional forces on the end node towards a sequence of targets.
    """

    def __init__(
        self,
        elem_index: int,
        p_linear_value: float,
        p_angular_value: float,
        target: Pose | Callable[[float, SystemType], Pose],
        target_history: list[Pose],
        ramp_up_time=1.0,
    ):
        """

        Parameters
        ----------
        elem_index: int
            index of the element to apply the force
        p_linear_value: float
            proportional linear gain
        p_angular_value: float
            proportional angular gain
        target: Pose | Callable[[float, SystemType], Pose]
            Target position and orientation.
            array (3,) containing data with 'float' type, or a function that returns the target Pose
            given time and rod.
        ramp_up_time: float
            Applied forces are ramped up until ramp up time.
        """
        super().__init__()
        assert ramp_up_time > 0.0
        self.elem_index = elem_index
        self.linear_gain = p_linear_value
        self.angular_gain = p_angular_value
        self.ramp_up_time = ramp_up_time
        self.target = target
        self.target_history = target_history
        self.save_counter = 0
        self.save_every = 200

        if isinstance(target, np.ndarray):
            self.target = lambda t: target

    def apply_forces(self, system: SystemType, time=0.0):
        target_position, target_orientation = self.target(time, system)
        if self.save_counter % self.save_every == 0:
            self.target_history.append((target_position, target_orientation))
            self.save_counter = 0
        self.save_counter += 1

        self.compute_node_force(
            system.external_forces,
            system.external_torques,
            system.position_collection,
            system.director_collection,
            self.linear_gain,
            self.angular_gain,
            time,
            self.ramp_up_time,
            target_position,
            target_orientation,
            self.elem_index,
        )

    @staticmethod
    @njit(cache=True)
    def compute_node_force(
        external_forces,
        external_torques,
        positions,
        orientations,
        linear_gain,
        angular_gain,
        time,
        ramp_up_time,
        target_position,
        target_orientation,
        index,
    ):
        factor = min(1.0, time / ramp_up_time)

        # Linear
        position = 0.5 * (positions[..., index] + positions[..., index + 1])
        force = target_position - position
        external_forces[..., index] += 0.5 * linear_gain * factor * force
        external_forces[..., index + 1] += 0.5 * linear_gain * factor * force

        # Angular
        orientation = orientations[..., index]
        rotation = orientation.T @ target_orientation
        angle = np.arccos((np.trace(rotation) - 1) / 2 - 1e-10)
        vector = (1.0 / (2 * np.sin(angle) + 1e-14)) * np.array(
            [
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            ]
        )
        torque = factor * angular_gain * angle * vector

        external_torques[..., index] -= orientation @ torque
