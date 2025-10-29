"""
Continuum Snake
===============

Snake friction case from X. Zhang et. al. Nat. Comm. 2021

This Elastica tutorial explains how to setup a Cosserat rod simulation to simulate a slithering snake. It covers many of the basics of setting up and running simulations with Elastica.

This slithering snake example includes gravitational forces, friction forces, and internal muscle torques. It also introduces the use of call back functions to allow logging of simulations data for post-processing after the simulation is over.

.. video:: ../../../assets/continuum_snake.mp4
   :width: 720
   :autoplay:
   :muted:
   :loop:

Getting Started
---------------
To set up the simulation, the first thing you need to do is import the necessary classes. As with the Timoshenko bean, we need to import modules which allow us to more easily construct different simulation systems. We also need to import a rod class, all the necessary forces to be applied, timestepping functions, and callback classes.
"""

import os
import numpy as np
import elastica as ea
from numpy.typing import NDArray
from elastica.typing import RodType

# %%
# Initialize System and Add Rod
# -----------------------------
# The first thing to do is initialize the simulator class by combining all the imported modules. After initializing, we will generate a rod and add it to the simulation.

from continuum_snake_postprocessing import (
    plot_snake_velocity,
    plot_video,
    compute_projected_velocity,
    plot_curvature,
)


class SnakeSimulator(
    ea.BaseSystemCollection,
    ea.Constraints,
    ea.Forcing,
    ea.Damping,
    ea.CallBacks,
    ea.Contact,
):
    pass


def run_snake(
    b_coeff: NDArray[np.float64],
    PLOT_FIGURE: bool = False,
    SAVE_FIGURE: bool = False,
    SAVE_VIDEO: bool = False,
    SAVE_RESULTS: bool = False,
) -> tuple[float, float, dict]:
    # Initialize the simulation class
    snake_sim = SnakeSimulator()

    # Simulation parameters
    period = 2
    final_time = (11.0 + 0.01) * period

    # setting up test params
    n_elem = 50
    start = np.zeros((3,))
    direction = np.array([0.0, 0.0, 1.0])
    normal = np.array([0.0, 1.0, 0.0])
    base_length = 0.35
    base_radius = base_length * 0.011
    density = 1000
    E = 1e6
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

    snake_sim.append(shearable_rod)

    # Add gravitational forces
    gravitational_acc = -9.80665
    snake_sim.add_forcing_to(shearable_rod).using(
        ea.GravityForces, acc_gravity=np.array([0.0, gravitational_acc, 0.0])
    )

    # %%
    # Muscle Torques
    # --------------
    # A snake generates torque throughout its body through muscle activations. While these muscle activations are generated internally by the snake, it is simpler to treat them as applied external forces, allowing us to apply them to the rod in the same manner as the other external forces.
    #
    # You may notice that the muscle torque parameters appear to have special values. These are optimized coefficients for a snake gait.

    # Add muscle torques
    wave_length = b_coeff[-1]
    snake_sim.add_forcing_to(shearable_rod).using(
        ea.MuscleTorques,
        base_length=base_length,
        b_coeff=b_coeff[:-1],
        period=period,
        wave_number=2.0 * np.pi / (wave_length),
        phase_shift=0.0,
        rest_lengths=shearable_rod.rest_lengths,
        ramp_up_time=period,
        direction=normal,
        with_spline=True,
    )

    # Anisotropic Friction Forces
    # ---------------------------
    # The last force that needs to be added is the friction force between the snake and the ground. Snakes exhibits anisotropic friction where the friction coefficient is different in different directions. You can also define both static and kinematic friction coefficients. This is accomplished by defining some small velocity threshold `slip_velocity_tol` that defines the transitions between static and kinematic friction.

    # Add friction forces
    ground_plane = ea.Plane(
        plane_origin=np.array([0.0, -base_radius, 0.0]), plane_normal=normal
    )
    snake_sim.append(ground_plane)
    slip_velocity_tol = 1e-8
    froude = 0.1
    mu = base_length / (period * period * np.abs(gravitational_acc) * froude)
    kinetic_mu_array = np.array(
        [mu, 1.5 * mu, 2.0 * mu]
    )  # [forward, backward, sideways]
    static_mu_array = np.zeros(kinetic_mu_array.shape)
    snake_sim.detect_contact_between(shearable_rod, ground_plane).using(
        ea.RodPlaneContactWithAnisotropicFriction,
        k=1.0,
        nu=1e-6,
        slip_velocity_tol=slip_velocity_tol,
        static_mu_array=static_mu_array,
        kinetic_mu_array=kinetic_mu_array,
    )

    # add damping
    damping_constant = 2e-3
    time_step = 1e-4
    snake_sim.dampen(shearable_rod).using(
        ea.AnalyticalLinearDamper,
        damping_constant=damping_constant,
        time_step=time_step,
    )

    # Add Callback Function
    # ---------------------
    # The simulation is now setup, but before it is run, we want to define a callback function. A callback function allows us to record time-series data throughout the simulation. If you do not define a callback function, you will only have access to the final configuration of the system. If you want to be able to analyze how the system evolves over time, it is critical that you record the appropriate quantities.
    #
    # To create a callback function, begin with the `CallBackBaseClass`. You can then define which state quantities you wish to record by having them appended to the `self.callback_params` dictionary as well as how often you wish to save the data by defining `skip_step`.

    # Add call backs
    class ContinuumSnakeCallBack(ea.CallBackBaseClass):
        """
        Call back function for continuum snake
        """

        def __init__(self, step_skip: int, callback_params: dict) -> None:
            ea.CallBackBaseClass.__init__(self)
            self.every = step_skip
            self.callback_params = callback_params

        def make_callback(
            self, system: ea.CosseratRod, time: float, current_step: int
        ) -> None:

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
                self.callback_params["curvature"].append(system.kappa.copy())

                self.callback_params["tangents"].append(system.tangents.copy())

                self.callback_params["friction"].append(
                    self.get_slip_velocity(system).copy()
                )

                return

        def get_slip_velocity(self, system: RodType) -> NDArray[np.float64]:
            from elastica.contact_utils import (
                _find_slipping_elements,
                _node_to_element_velocity,
            )
            from elastica._linalg import _batch_product_k_ik_to_ik, _batch_dot

            axial_direction = system.tangents
            element_velocity = _node_to_element_velocity(
                mass=system.mass, node_velocity_collection=system.velocity_collection
            )
            velocity_mag_along_axial_direction = _batch_dot(
                element_velocity, axial_direction
            )
            velocity_along_axial_direction = _batch_product_k_ik_to_ik(
                velocity_mag_along_axial_direction, axial_direction
            )
            slip_function_along_axial_direction = _find_slipping_elements(
                velocity_along_axial_direction, slip_velocity_tol
            )
            return slip_function_along_axial_direction

    total_steps = int(final_time / time_step)
    rendering_fps = 60
    step_skip = int(1.0 / (rendering_fps * time_step))

    pp_list: dict[str, list] = ea.defaultdict(list)
    snake_sim.collect_diagnostics(shearable_rod).using(
        ContinuumSnakeCallBack, step_skip=step_skip, callback_params=pp_list
    )

    # With the callback function added, we can now finalize the system and also define the time stepping parameters of the simulation such as the time step, final time, and time stepping algorithm to use.

    snake_sim.finalize()

    # Now all that is left is to run the simulation. Using the default parameters the simulation takes about 2-3 minutes to complete.

    timestepper = ea.PositionVerlet()
    dt = final_time / total_steps
    time = 0.0
    for i in range(total_steps):
        time = timestepper.step(snake_sim, time, dt)

    if PLOT_FIGURE:
        filename_plot = "continuum_snake_velocity.png"
        plot_snake_velocity(pp_list, period, filename_plot, SAVE_FIGURE)
        plot_curvature(pp_list, shearable_rod.rest_lengths, period, SAVE_FIGURE)

        if SAVE_VIDEO:
            filename_video = "continuum_snake.mp4"
            plot_video(
                pp_list,
                video_name=filename_video,
                fps=rendering_fps,
                xlim=(0, 4),
                ylim=(-1, 1),
            )

    if SAVE_RESULTS:
        import pickle

        filename = "continuum_snake.dat"
        file = open(filename, "wb")
        pickle.dump(pp_list, file)
        file.close()

    # Compute the average forward velocity. These will be used for optimization.
    [_, _, avg_forward, avg_lateral] = compute_projected_velocity(pp_list, period)

    return avg_forward, avg_lateral, pp_list


# %%
# Post-Process Data
# -----------------
# With the simulation complete, we want to analyze the simulation. Because we added a callback function, we can analyze how the snake evolves over time. All of the data from the callback function is located in the `pp_list` dictionary. Here we will use this information to compute and plot the velocity of the snake in the forward, lateral, and normal directions. We do this by using a pre-written analysis function `compute_projected_velocity`.
#
# In the plotted graph, you can see that it takes about one period for the snake to begin moving before rapidly reaching a steady gait over just 2-3 periods. We also see that the normal velocity is zero since we are only actuating the snake in a 2D plane.


# %%
# Gait Optimization with CMA
# --------------------------
# The following block of code in the main script demonstrates how to use the
# Covariance Matrix Adaptation Evolution Strategy (CMA-ES) to optimize the
# snake's gait. CMA-ES is a stochastic, derivative-free method for numerical
# optimization of non-linear or non-convex continuous optimization problems.
# Here, we use it to find the optimal set of muscle torque coefficients
# (`b_coeff`) that maximize the snake's average forward velocity. The
# `optimize_snake` function serves as the objective function for the
# optimization, which takes the spline coefficients as input and returns the
# negative of the average forward velocity, as CMA-ES is a minimization
# algorithm.


if __name__ == "__main__":

    # Options
    PLOT_FIGURE = True
    SAVE_FIGURE = True
    SAVE_VIDEO = True
    SAVE_RESULTS = False
    CMA_OPTION = False

    if CMA_OPTION:
        import cma

        SAVE_OPTIMIZED_COEFFICIENTS = False

        def optimize_snake(spline_coefficient: NDArray[np.float64]) -> float:
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
                [3.4e-3, 3.3e-3, 4.2e-3, 2.6e-3, 3.6e-3, 3.5e-3]
            )
            t_coeff_optimized = np.hstack((t_coeff_optimized, wave_length))

        # run the simulation
        [avg_forward, avg_lateral, pp_list] = run_snake(
            t_coeff_optimized, PLOT_FIGURE, SAVE_FIGURE, SAVE_VIDEO, SAVE_RESULTS
        )

        print("average forward velocity:", avg_forward)
        print("average forward lateral:", avg_lateral)
