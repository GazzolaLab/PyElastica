__doc__ = """Snake friction case from X. Zhang et. al. Nat. Comm. 2021"""

import os
import numpy as np
import pickle

from elastica import *

from continuum_snake_postprocessing import (
    plot_snake_velocity,
    plot_video,
    compute_projected_velocity,
    plot_curvature,
)
from snake_forcing import (
    MuscleTorquesLifting,
)
from snake_contact import SnakeRodPlaneContact


class SnakeSimulator(
    BaseSystemCollection, Constraints, Forcing, Damping, CallBacks, Contact
):
    pass


def run_snake(
    b_coeff_lat,
    PLOT_FIGURE=False,
    SAVE_FIGURE=False,
    SAVE_VIDEO=False,
    SAVE_RESULTS=False,
):
    # Initialize the simulation class
    snake_sim = SnakeSimulator()

    # Simulation parameters
    period = 2.0
    final_time = 20.0
    time_step = 5e-5
    total_steps = int(final_time / time_step)
    rendering_fps = 100
    step_skip = int(1.0 / (rendering_fps * time_step))

    # collection of snake characteristics
    n_elem = 25
    base_length = 0.35
    base_radius = 0.009
    snake_torque_ratio = 30.0
    snake_torque_liftratio = 10.0

    start = np.array([0.0, 0.0, 0.0 + base_radius])
    direction = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    density = 1000
    E = 1e6
    poisson_ratio = 0.5
    shear_modulus = E / (poisson_ratio + 1.0)

    shearable_rod = CosseratRod.straight_rod(
        n_elem,
        start,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        youngs_modulus=E,
        shear_modulus=shear_modulus,
    )

    snake_sim.append(shearable_rod)
    damping_constant = 1e-1

    # use linear damping with constant damping ratio
    snake_sim.dampen(shearable_rod).using(
        AnalyticalLinearDamper,
        damping_constant=damping_constant,
        time_step=time_step,
    )

    # Add gravitational forces
    gravitational_acc = -9.80665

    snake_sim.add_forcing_to(shearable_rod).using(
        GravityForces, acc_gravity=np.array([0.0, 0.0, gravitational_acc])
    )

    # 1. Add muscle torques -- lateral wave
    # Define lateral wave parameters
    lateral_wave_length = 1.0
    lateral_amp = b_coeff_lat[:-1]

    lateral_ratio = 1.0  # switch of lateral wave
    snake_sim.add_forcing_to(shearable_rod).using(
        MuscleTorques,
        base_length=base_length,
        b_coeff=snake_torque_ratio * lateral_ratio * lateral_amp,
        period=period,
        wave_number=2.0 * np.pi / (lateral_wave_length),
        phase_shift=0.0,
        rest_lengths=shearable_rod.rest_lengths,
        ramp_up_time=period,
        direction=normal,
        with_spline=True,
    )

    # 2. Add muscle torques -- lifting wave
    # Define lifting wave parameters
    lift_wave_length = lateral_wave_length
    lift_amp = np.array([1e-3, 2e-3, 2e-3, 2e-3, 2e-3, 1e-3])

    lift_ratio = 1.0  # switch of lifting wave
    phase = 0.5
    snake_sim.add_forcing_to(shearable_rod).using(
        MuscleTorquesLifting,
        b_coeff=snake_torque_liftratio * lift_ratio * lift_amp,
        period=period,
        wave_number=2.0 * np.pi / (lift_wave_length),
        phase_shift=phase * period,
        rest_lengths=shearable_rod.rest_lengths,
        ramp_up_time=0.01,
        direction=normal,
        with_spline=True,
        switch_on_time=2.0,
    )

    # Some common parameters first - define friction ratio etc.
    slip_velocity_tol = 1e-8
    froude = 0.1
    mu = base_length / (period * period * np.abs(gravitational_acc) * froude)
    kinetic_mu_array = np.array(
        [mu, 1.5 * mu, 2.0 * mu]
    )  # [forward, backward, sideways]
    normal_plane = normal
    origin_plane = np.array([0.0, 0.0, 0.0])
    ground_plane = Plane(plane_normal=normal_plane, plane_origin=origin_plane)
    snake_sim.append(ground_plane)

    snake_sim.detect_contact_between(shearable_rod, ground_plane).using(
        SnakeRodPlaneContact,
        k=1e2,
        nu=1e-1,
        slip_velocity_tol=slip_velocity_tol,
        kinetic_mu_array=kinetic_mu_array,
    )

    # Add call backs
    class ContinuumSnakeCallBack(CallBackBaseClass):
        """
        Call back function for continuum snake
        """

        def __init__(self, step_skip: int, callback_params: dict):
            CallBackBaseClass.__init__(self)
            self.every = step_skip
            self.callback_params = callback_params

        def make_callback(self, system, time, current_step: int):

            if current_step % self.every == 0:

                self.callback_params["time"].append(time)
                self.callback_params["step"].append(current_step)
                self.callback_params["position"].append(
                    system.position_collection.copy()
                )
                self.callback_params["radius"].append(system.radius.copy())
                self.callback_params["velocity"].append(
                    system.velocity_collection.copy()
                )
                self.callback_params["avg_velocity"].append(
                    system.compute_velocity_center_of_mass()
                )

                self.callback_params["center_of_mass"].append(
                    system.compute_position_center_of_mass()
                )
                self.callback_params["curvature"].append(system.kappa.copy())

                return

    pp_list = defaultdict(list)
    snake_sim.collect_diagnostics(shearable_rod).using(
        ContinuumSnakeCallBack, step_skip=step_skip, callback_params=pp_list
    )

    snake_sim.finalize()

    timestepper = PositionVerlet()
    integrate(timestepper, snake_sim, final_time, total_steps)

    if PLOT_FIGURE:
        filename_plot = "continuum_snake_velocity.png"
        plot_snake_velocity(pp_list, period, filename_plot, SAVE_FIGURE)
        plot_curvature(pp_list, shearable_rod.rest_lengths, period, SAVE_FIGURE)

        if SAVE_VIDEO:
            filename_video = "continuum_snake_with_lifting_wave.mp4"
            plot_video(
                pp_list,
                video_name=filename_video,
                fps=rendering_fps,
                xlim=(0, 3),
                ylim=(-1, 1),
            )

    if SAVE_RESULTS:

        filename = "continuum_snake.dat"
        file = open(filename, "wb")
        pickle.dump(pp_list, file)
        file.close()

    # Compute the average forward velocity. These will be used for optimization.
    [_, _, avg_forward, avg_lateral] = compute_projected_velocity(pp_list, period)

    return avg_forward, avg_lateral, pp_list


if __name__ == "__main__":
    # Options
    PLOT_FIGURE = True
    SAVE_FIGURE = False
    SAVE_VIDEO = True
    SAVE_RESULTS = True
    CMA_OPTION = False

    if CMA_OPTION:
        import cma

        SAVE_OPTIMIZED_COEFFICIENTS = False

        def optimize_snake(spline_coefficient):
            [avg_forward, _, _] = run_snake(
                spline_coefficient,
                PLOT_FIGURE=False,
                SAVE_FIGURE=False,
                SAVE_VIDEO=False,
                SAVE_RESULTS=False,
            )
            return -avg_forward

        # Optimize snake for forward velocity. In cma.fmin first input is function
        # to be optimized, second input is initial guess for coefficients you are optimizing
        # for and third input is standard deviation you initially set.
        optimized_spline_coefficients = cma.fmin(optimize_snake, 7 * [0], 0.5)

        # Save the optimized coefficients to a file
        filename_data = "optimized_coefficients.txt"
        if SAVE_OPTIMIZED_COEFFICIENTS:
            assert filename_data != "", "provide a file name for coefficients"
            np.savetxt(filename_data, optimized_spline_coefficients, delimiter=",")

    else:
        # Add muscle forces on the rod
        if os.path.exists("optimized_coefficients.txt"):
            t_coeff_optimized = np.genfromtxt(
                "optimized_coefficients.txt", delimiter=","
            )
        else:
            wave_length = 1.0
            t_coeff_optimized = np.array(
                [4e-3, 4e-3, 4e-3, 4e-3, 4e-3, 4e-3]
                # [3.4e-3, 3.3e-3, 5.7e-3, 2.8e-3, 3.0e-3, 3.0e-3]
            )
            t_coeff_optimized = np.hstack((t_coeff_optimized, wave_length))

        # run the simulation
        [avg_forward, avg_lateral, pp_list] = run_snake(
            t_coeff_optimized, PLOT_FIGURE, SAVE_FIGURE, SAVE_VIDEO, SAVE_RESULTS
        )

        print("average forward velocity:", avg_forward)
        print("average forward lateral:", avg_lateral)
