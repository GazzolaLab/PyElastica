__doc__ = """Continuum flagella example, for detailed explanation refer to Gazzola et. al. R. Soc. 2018
section 5.2.1 """

import numpy as np
import os
import elastica as ea
from examples.ContinuumFlagellaCase.continuum_flagella_postprocessing import (
    plot_velocity,
    plot_video,
    compute_projected_velocity,
)


class FlagellaSimulator(
    ea.BaseSystemCollection, ea.Constraints, ea.Forcing, ea.Damping, ea.CallBacks
):
    pass


def run_flagella(
    b_coeff, PLOT_FIGURE=False, SAVE_FIGURE=False, SAVE_VIDEO=False, SAVE_RESULTS=False
):

    flagella_sim = FlagellaSimulator()

    # setting up test params
    n_elem = 50
    start = np.zeros((3,))
    direction = np.array([0.0, 0.0, 1.0])
    normal = np.array([0.0, 1.0, 0.0])
    base_length = 1.0
    base_radius = 0.025
    density = 1000
    E = 1e7
    poisson_ratio = 0.5
    shear_modulus = E / (poisson_ratio + 1.0)

    shearable_rod = ea.CosseratRod.straight_rod(
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

    flagella_sim.append(shearable_rod)

    period = 1.0
    wave_length = b_coeff[-1]
    # Head and tail control points are zero.
    control_points = np.hstack((0, b_coeff[:-1], 0))
    flagella_sim.add_forcing_to(shearable_rod).using(
        ea.MuscleTorques,
        base_length=base_length,
        b_coeff=control_points,
        period=period,
        wave_number=2.0 * np.pi / (wave_length),
        phase_shift=0.0,
        rest_lengths=shearable_rod.rest_lengths,
        ramp_up_time=period,
        direction=normal,
        with_spline=True,
    )

    # Add slender body forces
    fluid_density = 1.0
    reynolds_number = 1e-4
    dynamic_viscosity = (
        fluid_density * base_length * base_length / (period * reynolds_number)
    )
    flagella_sim.add_forcing_to(shearable_rod).using(
        ea.SlenderBodyTheory, dynamic_viscosity=dynamic_viscosity
    )

    # add damping
    damping_constant = 0.625
    dt = 1e-4 * period
    flagella_sim.dampen(shearable_rod).using(
        ea.AnalyticalLinearDamper,
        damping_constant=damping_constant,
        time_step=dt,
    )

    # Add call backs
    class ContinuumFlagellaCallBack(ea.CallBackBaseClass):
        """
        Call back function for continuum snake
        """

        def __init__(self, step_skip: int, callback_params: dict):
            ea.CallBackBaseClass.__init__(self)
            self.every = step_skip
            self.callback_params = callback_params

        def make_callback(self, system, time, current_step: int):

            if current_step % self.every == 0:

                self.callback_params["time"].append(time)
                self.callback_params["step"].append(current_step)
                self.callback_params["position"].append(
                    system.position_collection.copy()
                )
                self.callback_params["velocity"].append(
                    system.velocity_collection.copy()
                )
                self.callback_params["avg_velocity"].append(
                    system.compute_velocity_center_of_mass()
                )
                self.callback_params["center_of_mass"].append(
                    system.compute_position_center_of_mass()
                )

                return

    pp_list = ea.defaultdict(list)
    flagella_sim.collect_diagnostics(shearable_rod).using(
        ContinuumFlagellaCallBack, step_skip=200, callback_params=pp_list
    )

    flagella_sim.finalize()
    timestepper = ea.PositionVerlet()
    # timestepper = PEFRL()

    final_time = (10.0 + 0.01) * period
    total_steps = int(final_time / dt)
    print("Total steps", total_steps)
    ea.integrate(timestepper, flagella_sim, final_time, total_steps)

    if PLOT_FIGURE:
        filename_plot = "continuum_flagella_velocity.png"
        plot_velocity(pp_list, period, filename_plot, SAVE_FIGURE)

        if SAVE_VIDEO:
            filename_video = "continuum_flagella.mp4"
            plot_video(pp_list, video_name=filename_video, margin=0.2, fps=200)

    if SAVE_RESULTS:
        import pickle

        filename = "continuum_flagella.dat"
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
    SAVE_VIDEO = False
    SAVE_RESULTS = False
    CMA_OPTION = False

    if CMA_OPTION:
        import cma

        SAVE_OPTIMIZED_COEFFICIENTS = False

        def optimize_snake(spline_coefficient):
            [avg_forward, _, _] = run_flagella(
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
        optimized_spline_coefficients = cma.fmin(optimize_snake, 5 * [0], 0.5)

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
            wave_length = (
                0.3866575573648976 * 1.0
            )  # 1.0 is base length, wave number is 16.25
            t_coeff_optimized = np.hstack((t_coeff_optimized, wave_length))
        else:
            t_coeff_optimized = np.array([17.4, 48.5, 5.4, 14.7, 0.38])

        # run the simulation
        [avg_forward, avg_lateral, pp_list] = run_flagella(
            t_coeff_optimized, PLOT_FIGURE, SAVE_FIGURE, SAVE_VIDEO, SAVE_RESULTS
        )

        print("average forward velocity:", avg_forward)
        print("average forward lateral:", avg_lateral)
