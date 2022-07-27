__doc__ = """Example where the tip of a pendulum is tracking a sphere."""

import numpy as np
from scipy.spatial.transform import Rotation
import sys

# FIXME without appending sys.path make it more generic
sys.path.append("../../")
from elastica import *
from elastica.utils import Tolerance
from examples.ControllerCases.pendulum_tracking_sphere_postprocessing import (
    plot_video,
    plot_video_xy,
)


class ControlSimulator(
    BaseSystemCollection, Constraints, Connections, Forcing, CallBacks, Control
):
    pass


class SphereCircularConstraint(ConstraintBase):
    def __init__(self, *args, circle_radius: float, frequency: float, **kwargs):
        """
        Parameters
        ----------
        circle_radius : float
            Radius of the circle which sphere is tracking.
        frequency : float
            Rotational frequency [Hz] of the sphere moving around the circle.
        """
        super(SphereCircularConstraint, self).__init__(*args, **kwargs)
        self.circle_radius = circle_radius
        self.frequency = frequency

    def constrain_values(self, system: SystemType, time: float):
        # reset position values on z-axis
        system.position_collection[2, :] = 0.0

        # desired rotation angle around y-axis at the current time step
        theta = self.frequency * 2 * np.pi * time

        # this assumes that for a zero theta angle, the pendulum is aligned with the x-axis.
        system.position_collection[0, :] = np.cos(theta) * self.circle_radius
        system.position_collection[1, :] = np.sin(theta) * self.circle_radius

    def constrain_rates(self, system: SystemType, time: float) -> None:
        """We directly correct the positional values instead of bothering with the velocities."""
        pass


class PendulumTrackingController(ControllerBase):
    """
    PID Controller to track the sphere as closely as possible with the end of the pendulum
    """

    def __init__(self, dt: float, P: float = 0.0, I: float = 0.0, D: float = 0.0):
        """
        Parameters
        ----------
        dt: float
            Time step of the simulation.
        P : float
            Proportional gain [Nm / rad]
        I : float
            Integral gain [Nm / rad / s].
        D : float
            Derivative gain. [Nm s / rad]
        """
        super().__init__()
        self.dt = dt

        self.P = P  # proportional gain
        self.I = I  # integral gain
        self.D = D  # derivative gain

        self.integrator = 0.0

    def apply_torques(self, systems: Dict[str, SystemType], time: np.float64 = 0.0):
        # collect positions of the sphere and the pendulum
        sphere_position = systems["sphere"].position_collection[:, 0]
        # current rotation angle of the sphere around the z-axis
        sphere_theta = np.arctan2(sphere_position[1], sphere_position[0])
        # construct rotation matrix for sphere
        sphere_rot_mat = Rotation.from_euler(
            "z", sphere_theta, degrees=False
        ).as_matrix()

        pendulum_tip_local_frame = np.array(
            [0.0, 0.0, systems["pendulum"].length.item() / 2]
        )
        # compute tip position of pendulum in inertial frame
        pendulum_tip = (
            systems["pendulum"].director_collection[..., 0].T @ pendulum_tip_local_frame
        )
        pendulum_theta = np.arctan2(pendulum_tip[1], pendulum_tip[0])
        # construct rotation matrix for pendulum
        pendulum_rot_mat = Rotation.from_euler(
            "z", pendulum_theta, degrees=False
        ).as_matrix()

        # rotation matrix from current rotation of pendulum to rotation of sphere
        dev_rot_mat = pendulum_rot_mat.T @ sphere_rot_mat
        # rotation vector
        dev_rot_vec = Rotation.from_matrix(dev_rot_mat).as_rotvec()

        # error in the rotation angle
        error_theta = dev_rot_vec[2]

        # get angular velocity around z-axis of inertial frame
        angular_velocity = (
            systems["pendulum"].director_collection[..., 0].T
            @ systems["pendulum"].omega_collection[..., 0]
        )

        # compute torque with PID controller
        torsional_torque = (
            self.P * error_theta
            + self.I * self.integrator
            - self.D * angular_velocity[2]
        )

        # compute torque in inertial frame
        torque_lab_frame = torsional_torque * np.array([0.0, 0.0, 1.0])

        # rotate torque to body frame
        systems["pendulum"].external_torques[:, 0] = (
            systems["pendulum"].director_collection[..., 0] @ torque_lab_frame
        )

        # update the integrator
        self.integrator += error_theta * self.dt


control_simulator = ControlSimulator()

# setting up test params
direction = np.array([1.0, 0.0, 0.0])
normal = np.array([0.0, 1.0, 0.0])
base_length = 0.2
base_radius = 0.01
density = 1000
poisson_ratio = 0.5

# setting up timestepper and video
final_time = 10
dt = 5e-5
total_steps = int(final_time / dt)
fps = 100  # frames per second of the video
diagnostic_step_skip = int(1 / (fps * dt))

# Create pendulum with cylinder
pendulum = Cylinder(
    start=np.array([-base_length / 2, 0.0, 0.0]),
    direction=direction,
    normal=normal,
    base_length=base_length,
    base_radius=base_radius,
    density=density,
)
control_simulator.append(pendulum)

# Create sphere system
sphere = Sphere(
    center=base_length / 2 * direction, base_radius=base_radius, density=density
)
control_simulator.append(sphere)

# Apply boundary conditions to pendulum at base: only allow yaw (e.g. rotation around z-axis)
control_simulator.constrain(pendulum).using(
    GeneralConstraint,
    constrained_position_idx=(0,),
    constrained_director_idx=(0,),
    translational_constraint_selector=np.array([True, True, True]),
    rotational_constraint_selector=np.array([True, True, False]),
)

# Move sphere along a circle in the XY plane (e.g. rotate around z-axis)
control_simulator.constrain(sphere).using(
    SphereCircularConstraint, circle_radius=base_length / 2, frequency=0.1
)

# add controller to pendulum
control_simulator.control(systems={"pendulum": pendulum, "sphere": sphere}).using(
    PendulumTrackingController,
    dt=dt,
    P=1e-3,
    I=2e-5,
    D=2e-5,
)

pp_list_pendulum = defaultdict(list)
pp_list_sphere = defaultdict(list)

control_simulator.collect_diagnostics(pendulum).using(
    MyCallBack, step_skip=diagnostic_step_skip, callback_params=pp_list_pendulum
)
control_simulator.collect_diagnostics(sphere).using(
    MyCallBack, step_skip=diagnostic_step_skip, callback_params=pp_list_sphere
)

control_simulator.finalize()
timestepper = PositionVerlet()

print("Total steps", total_steps)
integrate(timestepper, control_simulator, final_time, total_steps)

filename = "pendulum_tracking_sphere_example"
plot_video(
    plot_params_pendulum=pp_list_pendulum,
    pendulum=pendulum,
    plot_params_sphere=pp_list_sphere,
    video_name=filename + ".mp4",
    fps=fps,
)
plot_video_xy(
    plot_params_pendulum=pp_list_pendulum,
    pendulum=pendulum,
    plot_params_sphere=pp_list_sphere,
    video_name=filename + "_xy.mp4",
    fps=fps,
)
