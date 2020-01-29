import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb


def plot_velocity(
    list_input, period, filename="continuum_flagella_velocity.png", SAVE_FIGURE=False
):

    time_per_period = np.array(list_input["time"]) / period
    avg_velocity = np.array(list_input["avg_velocity"])

    [
        velocity_in_direction_of_rod,
        velocity_in_rod_roll_dir,
        _,
        _,
    ] = compute_projected_velocity(list_input, period)

    fig = plt.figure(figsize=(10, 8), frameon=True, dpi=150)
    ax = fig.add_subplot(111)
    ax.grid(b=True, which="minor", color="k", linestyle="--")
    ax.grid(b=True, which="major", color="k", linestyle="-")
    ax.plot(
        time_per_period[:], velocity_in_direction_of_rod[:, 2], "r-", label="forward"
    )
    ax.plot(
        time_per_period[:],
        velocity_in_rod_roll_dir[:, 0],
        c=to_rgb("xkcd:bluish"),
        label="lateral",
    )
    ax.plot(time_per_period[:], avg_velocity[:, 1], "k-", label="normal")
    fig.legend(prop={"size": 20})
    plt.show()

    if SAVE_FIGURE:
        fig.savefig(filename)


def plot_video(
    list_input, video_name="video.mp4", margin=0.2, fps=15
):  # (time step, x/y/z, node)
    import matplotlib.animation as manimation

    positions_over_time = np.array(list_input["position"])

    print("plot video")
    FFMpegWriter = manimation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    fig = plt.figure()
    plt.axis("equal")
    with writer.saving(fig, video_name, 100):
        for time in range(1, len(list_input["time"])):
            x = positions_over_time[time][2]
            y = positions_over_time[time][0]
            fig.clf()
            plt.plot(x, y, "o")
            plt.xlim([0 - margin, 2.5 + margin])
            plt.ylim([-1.25 - margin, 1.25 + margin])
            writer.grab_frame()


def compute_projected_velocity(list_input, period):

    time_per_period = np.array(list_input["time"]) / period
    avg_velocity = np.array(list_input["avg_velocity"])
    center_of_mass = np.array(list_input["center_of_mass"])

    # Compute rod velocity in rod direction. We need to compute that because,
    # after snake starts to move it chooses an arbitrary direction, which does not
    # have to be initial tangent direction of the rod. Thus we need to project the
    # snake velocity with respect to its new tangent and roll direction, after that
    # we will get the correct forward and lateral speed. After this projection
    # lateral velocity of the snake has to be oscillating between + and - values with
    # zero mean.

    # Number of steps in one period.
    period_step = int(period / (time_per_period[-1] - time_per_period[-2])) + 1
    number_of_period = int(time_per_period.shape[0] / period_step)
    # Center of mass position averaged in one period
    center_of_mass_averaged_over_one_period = np.zeros((number_of_period - 2, 3))
    for i in range(1, number_of_period - 1):
        # position of center of mass in rolling direction averaged over one period
        center_of_mass_averaged_over_one_period[i - 1, 0] = np.mean(
            center_of_mass[(i + 1) * period_step : (i + 2) * period_step, 0]
            - center_of_mass[(i + 0) * period_step : (i + 1) * period_step, 0]
        )
        # position of center of mass in normal direction averaged over one period
        center_of_mass_averaged_over_one_period[i - 1, 1] = np.mean(
            center_of_mass[(i + 1) * period_step : (i + 2) * period_step, 1]
            - center_of_mass[(i + 0) * period_step : (i + 1) * period_step, 1]
        )
        # position of center of mass in normal direction averaged over one period
        center_of_mass_averaged_over_one_period[i - 1, 2] = np.mean(
            center_of_mass[(i + 1) * period_step : (i + 2) * period_step, 2]
            - center_of_mass[(i + 0) * period_step : (i + 1) * period_step, 2]
        )

    # Average the rod directions over multiple periods and get the direction of the rod.
    direction_of_rod = np.array(
        [
            np.mean(center_of_mass_averaged_over_one_period[:, 0]),
            np.mean(center_of_mass_averaged_over_one_period[:, 1]),
            np.mean(center_of_mass_averaged_over_one_period[:, 2]),
        ]
    )
    direction_of_rod /= np.linalg.norm(direction_of_rod, ord=2)

    # Compute the projected rod velocity in the direction of the rod
    velocity_mag_in_direction_of_rod = np.einsum(
        "ji,i->j", avg_velocity, direction_of_rod
    )
    velocity_in_direction_of_rod = np.einsum(
        "j,i->ji", velocity_mag_in_direction_of_rod, direction_of_rod
    )

    # Get the lateral or roll velocity of the rod after subtracting its projected
    # velocity in the direction of rod
    velocity_in_rod_roll_dir = avg_velocity - velocity_in_direction_of_rod

    # Compute the average velocity over the simulation, this can be used for optimizing snake
    # for fastest forward velocity. We start after first period, because of the ramping up happens
    # in first period.
    average_forward_velocity_over_simulation = np.mean(
        velocity_in_direction_of_rod[period_step * 2 :, 2]
    )
    average_lateral_velocity_over_simulation = np.mean(
        velocity_in_rod_roll_dir[period_step * 2 :, 0]
    )

    return (
        velocity_in_direction_of_rod,
        velocity_in_rod_roll_dir,
        average_forward_velocity_over_simulation,
        average_lateral_velocity_over_simulation,
    )
