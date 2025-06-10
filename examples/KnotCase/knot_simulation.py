__doc__ = """Simulating overhand-knot, a degenerated case of Trefoil knot.
A demonstration includes how to create an arbitrary controller for a node in a rod,
resembling a proportional-controller of SO3 Pose. The same class can be used further
to mimic the MPC control or trajectory-tracing."""

import numpy as np
import elastica as ea

from knot_forcing import TargetPoseProportionalControl
from knot_visualization import plot_video3D


class SoftRodSimulator(
    ea.BaseSystemCollection,
    ea.Constraints,
    ea.Forcing,
    ea.Damping,
    ea.CallBacks,
    ea.Contact,
):
    pass


class AxialStretchingCallBack(ea.CallBackBaseClass):
    """
    Records the position of the rod
    """

    def __init__(self, step_skip: int, callback_params: dict):
        ea.CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.callback_params = callback_params

    def make_callback(self, system, time, current_step: int):
        if current_step % self.every == 0:

            self.callback_params["time"].append(time)
            self.callback_params["step"].append(current_step)
            self.callback_params["position"].append(system.position_collection.copy())
            self.callback_params["orientation"].append(
                system.director_collection.copy()
            )
            return


if __name__ == "__main__":
    # Options
    GENERATE_2D_VIDEO = False
    GENERATE_3D_VIDEO = True

    simulator = SoftRodSimulator()
    recorded_history = ea.defaultdict(list)
    final_time = 5
    dt = 0.0002

    # setting up test params
    n_elem = 50
    start = np.zeros((3,))
    direction = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 1.0, 0.0])
    base_length = 1.2
    base_radius = 0.025
    density = 2000
    youngs_modulus = 1e6
    poisson_ratio = 0.5
    shear_modulus = youngs_modulus / (2 * (poisson_ratio + 1.0))

    stretchable_rod = ea.CosseratRod.straight_rod(
        n_elem,
        start,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        youngs_modulus=youngs_modulus,
        shear_modulus=shear_modulus,
    )
    simulator.append(stretchable_rod)

    run_time = 4

    def base_target(t, rod):
        target_position = direction * base_length - 5 * base_radius * normal
        if t <= run_time / 2:
            ratio = min(2 * t / run_time, 1.0)
            angular_ratio = ratio * np.pi * 2
            position = target_position * ratio
            orientation_twist = np.array(
                [
                    [0, np.cos(angular_ratio), np.sin(angular_ratio)],
                    [0, -np.sin(angular_ratio), np.cos(angular_ratio)],
                    [1, 0, 0],
                ],
                dtype=float,
            )
        else:
            ratio = min(2 * (t - run_time / 2) / run_time, 1.0)
            R = 8
            position = np.array(
                [
                    target_position[0] * (1 - ratio),
                    -R * base_radius * np.cos(2 * ratio * 12) * (1 - ratio),
                    -R * base_radius * np.sin(2 * ratio * 12) * (1 - ratio),
                ]
            )
            angular_ratio = (1 - ratio) * np.pi * 2
            orientation_twist = np.array(
                [
                    [0, np.cos(angular_ratio), -np.sin(angular_ratio)],
                    [0, np.sin(angular_ratio), np.cos(angular_ratio)],
                    [1, 0, 0],
                ],
                dtype=float,
            )
        return position, orientation_twist

    # Control point
    p = 3e3
    pt = 5e0
    simulator.add_forcing_to(stretchable_rod).using(
        TargetPoseProportionalControl,
        elem_index=0,
        p_linear_value=p,
        p_angular_value=pt,
        target=base_target,
        ramp_up_time=1e-6,
        target_history=recorded_history["base_pose"],
    )

    # Boundary conditions
    simulator.constrain(stretchable_rod).using(
        ea.FixedConstraint, constrained_position_idx=(-1, -20)
    )

    # Self contact
    simulator.detect_contact_between(stretchable_rod, stretchable_rod).using(
        ea.RodSelfContact, k=1e4, nu=3
    )

    # Gravity
    simulator.add_forcing_to(stretchable_rod).using(
        ea.GravityForces, acc_gravity=np.array([0.0, 0.0, -9.80665])
    )

    # Damping
    damping_constant = 5.0
    simulator.dampen(stretchable_rod).using(
        ea.AnalyticalLinearDamper,
        translational_damping_constant=damping_constant,
        rotational_damping_constant=damping_constant * 0.01,
        time_step=dt,
    )
    simulator.dampen(stretchable_rod).using(ea.LaplaceDissipationFilter, filter_order=5)

    simulator.collect_diagnostics(stretchable_rod).using(
        AxialStretchingCallBack, step_skip=1, callback_params=recorded_history
    )

    # Finalize and run the simulation
    simulator.finalize()
    timestepper = ea.PositionVerlet()
    total_steps = int(final_time / dt)
    print("Total steps", total_steps)
    ea.integrate(timestepper, simulator, final_time, total_steps)

    if GENERATE_3D_VIDEO:
        filename_video = "knot3D.mp4"
        plot_video3D(recorded_history, video_name=filename_video, margin=0.2, fps=10)
