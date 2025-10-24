from matplotlib import pyplot as plt
import numpy as np
import elastica as ea
import json
from cantilever_distrubuted_load_postprecessing import (
    plot_video_with_surface,
    find_tip_position,
    adjust_square_cross_section,
)


def conservative_force_simulator(load, animation=False):
    class SquareRodSimulator(
        ea.BaseSystemCollection, ea.Constraints, ea.Forcing, ea.Damping, ea.CallBacks
    ):
        pass

    square_rod_sim = SquareRodSimulator()
    final_time = 10

    # setting up test params
    n_elem = 100
    start = np.zeros((3,))
    direction = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 1.0, 0.0])
    base_length = 0.5
    side_length = 0.01
    base_radius = 0.01 / (
        np.pi ** (1 / 2)
    )  # The Cross-sectional area is 1e-4(we assume its equivalent to a square cross-sectional surface with same area)
    base_area = np.pi * base_radius**2
    density = 1000  # nomilized with conservative case F=15
    youngs_modulus = 1.2e7
    dl = base_length / n_elem
    dt = 0.1 * dl / 50
    I = (0.01**4) / 12
    end_force_x = (youngs_modulus * I * load) / (density * base_area * (base_length**3))
    # For shear modulus of 1e4, nu is 99!
    poisson_ratio = 0.0
    shear_modulus = youngs_modulus / (2 * (poisson_ratio + 1.0))

    rendering_fps = 30

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

    conservative_load = np.array([0.0, -end_force_x, 0.0])

    square_rod_sim.add_forcing_to(square_rod).using(
        ea.GravityForces, acc_gravity=conservative_load
    )

    # add damping

    damping_constant = 0.1

    square_rod_sim.dampen(square_rod).using(
        ea.AnalyticalLinearDamper,
        damping_constant=damping_constant,
        time_step=dt,
    )

    # Add call backs
    class CantileverDistributedLoadCallBack(ea.CallBackBaseClass):
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
        CantileverDistributedLoadCallBack,
        step_skip=200,
        callback_params=recorded_history,
    )

    square_rod_sim.finalize()
    timestepper = ea.PositionVerlet()

    total_steps = int(final_time / dt)
    dt = final_time / total_steps
    time = 0.0
    for i in range(total_steps):
        time = timestepper.step(square_rod_sim, time, dt)

    relative_tip_position = np.zeros(
        2,
    )
    relative_tip_position[0] = find_tip_position(square_rod, n_elem)[0] / base_length
    relative_tip_position[1] = -find_tip_position(square_rod, n_elem)[1] / base_length

    if animation:
        plot_video_with_surface(
            [recorded_history],
            video_name="cantilever_conservative_distributed_load.mp4",
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

    relative_tip_position = np.zeros(
        2,
    )
    relative_tip_position[0] = find_tip_position(square_rod, n_elem)[0] / base_length
    relative_tip_position[1] = -find_tip_position(square_rod, n_elem)[1] / base_length

    print(relative_tip_position)
    return relative_tip_position


if __name__ == "__main__":

    with open("cantilever_distributed_load_data.json", "r") as file:
        tip_position_paper = json.load(file)
        tip_position_paper = tip_position_paper["conservative"]

    x_tip_experiment = []
    y_tip_experiment = []
    x_tip_paper = tip_position_paper["x_tip_position"]
    y_tip_paper = tip_position_paper["y_tip_position"]

    load_on_rod = np.arange(1, 26, 2)
    for load in load_on_rod:
        tip_displacement = conservative_force_simulator(load)
        x_tip_experiment.append(tip_displacement[0])
        y_tip_experiment.append(tip_displacement[1])

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
        marker="s",
        linestyle="None",
        label="x_tip/L",
    )
    plt.scatter(
        load_on_rod,
        y_tip_experiment,
        color="red",
        marker="s",
        linestyle="None",
        label="y_tip/L",
    )

    plt.title("Conservative-Load Elastica Simulation Results")
    # Title
    plt.xlabel("Load")  # X-axis label
    plt.ylabel("x_tip/L and y_tip/L")  # Y-axis label
    plt.grid()
    plt.legend()  # Optional: Add a grid
    plt.show()  # Display the plot
