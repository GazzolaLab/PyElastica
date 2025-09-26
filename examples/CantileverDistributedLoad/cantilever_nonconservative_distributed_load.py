from matplotlib import pyplot as plt
import numpy as np
import elastica as ea
import json

from cantilever_distrubuted_load_postprecessing import (
    plot_video_with_surface,
    find_tip_position,
    adjust_square_cross_section,
    NonconserativeForce,
)


class SquareRodSimulator(
    ea.BaseSystemCollection, ea.Constraints, ea.Forcing, ea.Damping, ea.CallBacks
):
    pass


def cantilever_subjected_to_a_nonconservative_load(
    n_elem,
    base_length,
    side_length,
    base_radius,
    youngs_modulus,
    dimentionless_varible,
    animation=False,
    plot_figure_equilibrium=False,
):
    square_rod_sim = SquareRodSimulator()

    square_rod = ea.CosseratRod.straight_rod(
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

    adjust_square_cross_section(square_rod, youngs_modulus, side_length)

    square_rod_sim.append(square_rod)

    square_rod_sim.constrain(square_rod).using(
        ea.OneEndFixedBC, constrained_position_idx=(0,), constrained_director_idx=(0,)
    )

    load = (youngs_modulus * I * dimentionless_varible) / (
        density * base_area * (base_length**3)
    )

    square_rod_sim.add_forcing_to(square_rod).using(NonconserativeForce, load)

    # add damping
    dl = base_length / n_elem
    dt = 0.1 * dl / 50
    damping_constant = 0.2

    square_rod_sim.dampen(square_rod).using(
        ea.AnalyticalLinearDamper,
        damping_constant=damping_constant,
        time_step=dt,
    )

    # Add call backs
    class NonConservativeDistributedLoadCallBack(ea.CallBackBaseClass):
        """
        Tracks the velocity norms of the rod
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
                self.callback_params["com"].append(
                    system.compute_position_center_of_mass()
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
                self.callback_params["velocity_magnitude"].append(
                    (
                        square_rod.velocity_collection[-1][0] ** 2
                        + square_rod.velocity_collection[-1][1] ** 2
                        + square_rod.velocity_collection[-1][2] ** 2
                    )
                    ** 0.5
                )

    recorded_history = ea.defaultdict(list)

    square_rod_sim.collect_diagnostics(square_rod).using(
        NonConservativeDistributedLoadCallBack,
        step_skip=200,
        callback_params=recorded_history,
    )

    square_rod_sim.finalize()
    timestepper = ea.PositionVerlet()

    total_steps = int(final_time / dt)
    print("Total steps", total_steps)
    dt = final_time / total_steps
    time = 0.0
    for i in range(total_steps):
        time = timestepper.step(square_rod_sim, time, dt)

    if plot_figure_equilibrium:

        plt.plot(
            recorded_history["time"],
            recorded_history["velocity_magnitude"],
            lw=1.0,
            label="velocity_magnitude",
        )
        plt.xlabel("Time (s)")
        plt.ylabel("Speed (m/s)")
        plt.title("Tip Position and Speed vs. Time")
        plt.legend()
        plt.grid()
        plt.show()

    rendering_fps = 30
    if animation:
        plot_video_with_surface(
            [recorded_history],
            video_name="cantilever_Non-conservative_distributed_load.mp4",
            fps=rendering_fps,
            step=1,
            # The following parameters are optional
            x_limits=(-0.0, 0.5),  # Set bounds on x-axis
            y_limits=(-0.5, 0.0),  # Set bounds on y-axis
            z_limits=(-0.0, 0.5),  # Set bounds on z-axis
            dpi=100,  # Set the quality of the image
            vis3D=True,  # Turn on 3D visualization
            vis2D=False,  # Turn on projected (2D) visualization
        )

    pos = square_rod.position_collection.view()

    tip_position = find_tip_position(square_rod, n_elem)
    relative_tip_position = np.zeros((2,))

    relative_tip_position[0] = tip_position[0] / base_length
    relative_tip_position[1] = -tip_position[1] / base_length

    print(relative_tip_position)
    return relative_tip_position


if __name__ == "__main__":
    final_time = 10
    # setting up test params
    n_elem = 100
    start = np.zeros((3,))
    direction = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 1.0, 0.0])
    base_length = 0.5
    side_length = 0.01
    base_radius = 0.01 / (np.pi ** (1 / 2))
    base_area = np.pi * base_radius**2
    density = 1000
    dimentionless_varible = 15
    youngs_modulus = 1.2e7
    # For shear modulus of 1e4, nu is 99!
    poisson_ratio = 0
    shear_modulus = youngs_modulus / (2 * (poisson_ratio + 1.0))
    I = (0.01**4) / 12

    cantilever_subjected_to_a_nonconservative_load(
        n_elem, base_length, side_length, base_radius, youngs_modulus, -15, True, False
    )

    with open("cantilever_distributed_load_data.json", "r") as file:
        tip_position_paper = json.load(file)
        tip_position_paper = tip_position_paper["non_conservative"]
    x_tip_experiment = []
    y_tip_experiment = []
    x_tip_paper = tip_position_paper["x_tip_position"]
    y_tip_paper = tip_position_paper["y_tip_position"]

    load_on_rod = np.arange(1, 26, 2)
    for i in load_on_rod:
        x_tip_experiment.append(
            cantilever_subjected_to_a_nonconservative_load(
                n_elem, base_length, base_radius, youngs_modulus, i, False, False
            )[0]
        )
        y_tip_experiment.append(
            -cantilever_subjected_to_a_nonconservative_load(
                n_elem, base_length, base_radius, youngs_modulus, i, False, False
            )[1]
        )

    plt.plot(
        load_on_rod,
        x_tip_paper,
        color="black",
        marker="*",
        linestyle="--",
        label="Theoretical_x",
    )
    plt.plot(
        load_on_rod,
        y_tip_paper,
        color="black",
        marker="*",
        linestyle=":",
        label="Theoretical_y",
    )
    plt.scatter(
        load_on_rod,
        x_tip_experiment,
        color="blue",
        marker="o",
        linestyle="None",
        label="x_tip/L",
    )
    plt.scatter(
        load_on_rod,
        y_tip_experiment,
        color="red",
        marker="o",
        linestyle="None",
        label="y_tip/L",
    )

    plt.title("Non-Conservative-Load Elastica Simulation Results")
    # Title
    plt.xlabel("Load")  # X-axis label
    plt.ylabel("x_tip/L and y_tip/L")  # Y-axis label
    plt.grid()
    plt.legend()  # Optional: Add a grid
    plt.show()  # Display the plot
