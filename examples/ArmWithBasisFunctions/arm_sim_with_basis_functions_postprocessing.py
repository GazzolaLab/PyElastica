import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb


def plot_video(
    plot_params: dict,
    video_name="video.mp4",
    margin=0.2,
    fps=15,
    step=100,
    *args,
    **kwargs
):  # (time step, x/y/z, node)
    import matplotlib.animation as manimation

    positions_over_time = np.array(plot_params["position"])

    print("plot video")
    FFMpegWriter = manimation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    fig = plt.figure()
    plt.gca().set_aspect("equal", adjustable="box")
    # plt.axis("square")
    time = np.array(plot_params["time"])
    with writer.saving(fig, video_name, 100):
        for time in range(1, time.shape[0], int(step)):
            x = positions_over_time[time][2]
            y = positions_over_time[time][1]
            fig.clf()
            if kwargs.__contains__("target"):
                plt.plot(kwargs["target"][2], kwargs["target"][1], "*", markersize=12)
            plt.plot(x, y, "o")
            plt.gca().set_aspect("equal", adjustable="box")
            plt.xlim([0 - margin, 1.0 + margin])
            plt.ylim([-1.00 - margin, 0.00 + margin])
            writer.grab_frame()


def plot_video_actiavation_muscle(
    activation_list: dict,
    torque_list: dict,
    video_name="video.mp4",
    margin=0.2,
    fps=15,
    step=100,
):
    import matplotlib.animation as manimation

    time = np.array(activation_list["time"])
    if "activation_signal" in activation_list:
        activation = np.array(activation_list["activation_signal"])
        first_activation = None
        second_activation = None
    else:
        activation = None
        first_activation = np.array(activation_list["first_activation_signal"])
        second_activation = np.array(activation_list["second_activation_signal"])

    torque = np.array(torque_list["torque"])
    if "torque_mag" in torque_list:
        torque_mag = np.array(torque_list["torque_mag"])
        first_torque_mag = None
        second_torque_mag = None
    else:
        torque_mag = None
        first_torque_mag = np.array(torque_list["first_torque_mag"])
        second_torque_mag = np.array(torque_list["second_torque_mag"])
    element_position = np.array(torque_list["element_position"])

    print("plot video")
    FFMpegWriter = manimation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    fig = plt.figure()
    plt.subplot(311)
    plt.axis("equal")
    with writer.saving(fig, video_name, 100):
        for time in range(1, time.shape[0], int(step)):
            # ax1 = plt.subplot(2, 2, 1)
            # ax2 = plt.subplot(222, frameon=False)
            # x = activation[time][2]
            torq = torque[time][0]
            pos = element_position[time]
            fig.clf()
            plt.subplot(3, 1, 1)
            if activation is not None:
                plt.plot(activation[time], "-")
            else:
                plt.plot(first_activation[time])
                plt.plot(second_activation[time])
            plt.ylim([-1 - margin, 1 + margin])
            plt.subplot(3, 1, 2)

            if torque_mag is not None:
                plt.plot(pos, torque_mag[time], "-")
            else:
                plt.plot(pos, first_torque_mag[time], pos, second_torque_mag[time])
            plt.ylim(
                [
                    min(np.amin(first_torque_mag), np.amin(second_torque_mag))
                    - 50 * margin,
                    max(np.amax(first_torque_mag), np.amax(second_torque_mag))
                    + 50 * margin,
                ]
            )
            plt.subplot(3, 1, 3)
            plt.plot(pos, torq, "-")
            # plt.xlim([0 - margin, 2.5 + margin])

            writer.grab_frame()

    # plt.subplot(221)
    #
    # # equivalent but more general
    # ax1 = plt.subplot(2, 2, 1)
    #
    # # add a subplot with no frame
    # ax2 = plt.subplot(222, frameon=False)
    #
    # # add a polar subplot
    # plt.subplot(223, projection='polar')
    #
    # # add a red subplot that shares the x-axis with ax1
    # plt.subplot(224, sharex=ax1, facecolor='red')
    #
    # # delete ax2 from the figure
    # plt.delaxes(ax2)
    #
    # # add ax2 to the figure again
    # plt.subplot(ax2)


def plot_arm_tip_sensor_values(sensor: dict, filename, SAVE_FIGURE=False):
    time = np.array(sensor["time"])
    sensor_value = np.array(sensor["sensor"])
    fig, axs = plt.subplots(3, 1, constrained_layout=False)
    axs[0].plot(time, sensor_value[:, 0], "-")
    axs[0].set_title("sensor value - x")
    axs[0].set_xlabel("time [s]")
    axs[0].set_ylabel("sensor value")

    axs[1].plot(time, sensor_value[:, 1], "-")
    axs[1].set_title("sensor value - y")
    axs[1].set_xlabel("time [s]")
    axs[1].set_ylabel("sensor value")

    axs[2].plot(time, sensor_value[:, 2], "-")
    axs[2].set_title("sensor value - z")
    axs[2].set_xlabel("time [s]")
    axs[2].set_ylabel("sensor value")

    plt.show()
    if SAVE_FIGURE:
        fig.savefig(filename)


def plot_video_zx(
    plot_params: dict,
    video_name="video.mp4",
    margin=0.2,
    fps=15,
    step=100,
    *args,
    **kwargs
):  # (time step, x/y/z, node)
    import matplotlib.animation as manimation

    positions_over_time = np.array(plot_params["position"])

    print("plot video")
    FFMpegWriter = manimation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    fig = plt.figure()
    plt.gca().set_aspect("equal", adjustable="box")
    # plt.axis("square")
    time = np.array(plot_params["time"])
    with writer.saving(fig, video_name, 100):
        for time in range(1, time.shape[0], int(step)):
            x = positions_over_time[time][2]
            y = positions_over_time[time][0]
            fig.clf()
            if kwargs.__contains__("target"):
                plt.plot(kwargs["target"][2], kwargs["target"][0], "*", markersize=12)
            plt.plot(x, y, "o")
            plt.gca().set_aspect("equal", adjustable="box")
            plt.xlim([0 - margin, 1.0 + margin])
            plt.ylim([-1.00 - margin, 0.00 + margin])
            writer.grab_frame()


def plot_video3d(
    plot_params: dict, video_name="video.mp4", margin=0.2, fps=15, step=1, **kwargs
):  # (time step, x/y/z, node)
    import matplotlib.animation as manimation
    from mpl_toolkits import mplot3d

    time = plot_params["time"]
    position = np.array(plot_params["position"])
    radius = np.array(plot_params["radius"])

    print("Plotting video")
    FFMpegWriter = manimation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=fps, metadata=metadata)

    fig = plt.figure(figsize=(10, 8), frameon=True, dpi=150)
    ax = plt.axes(projection="3d")  # fig.add_subplot(111)
    ax.grid(b=True, which="minor", color="k", linestyle="--")
    ax.grid(b=True, which="major", color="k", linestyle="-")

    time_idx = 0
    scatt = ax.scatter(
        position[time_idx, 0],
        position[time_idx, 1],
        position[time_idx, 2],
        s=np.pi * radius[time_idx] ** 2 * 10000,
    )

    ax.set_xlim(0 - margin, 1.0 + margin)
    ax.set_ylim(-1.00 - margin, 0.00 + margin)
    ax.set_zlim(0.0, 1.0 + margin)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    step = int(step)
    dpi = kwargs.get("dpi", 100)
    with writer.saving(fig, video_name, dpi):
        for time_idx in range(1, len(time), step):
            scatt._offsets3d = (
                position[time_idx, 0],
                position[time_idx, 1],
                position[time_idx, 2],
            )
            scatt.set_sizes(np.pi * radius[time_idx] ** 2 * 1e6)
            writer.grab_frame()


def plot_video_with_surface(
    plot_params: dict,
    cylinder_history: dict,
    cylinder_radius,
    cylinder_height,
    cylinder_direction,
    video_name="video.mp4",
    fps=60,
    step=1,
    **kwargs
):  # (time step, x/y/z, node)
    import matplotlib.animation as manimation
    from mpl_toolkits.mplot3d import proj3d, Axes3D

    plt.rcParams.update({"font.size": 22})

    # Should give a (n_time, 3, n_elem) array
    positions = np.array(plot_params["position"])
    radius = np.array(plot_params["radius"])
    # Should give a (n_time, 3) array
    com = np.array(plot_params["com"])

    cylinder_com = np.array(cylinder_history["com"])
    cylinder_origin = cylinder_com - 0.5 * cylinder_height * cylinder_direction.reshape(
        3
    )

    print("Plotting videos!")
    FFMpegWriter = manimation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    dpi = kwargs.get("dpi", 100)

    def make_data_for_cylinder_along_z(cstart, cradius, cheight):
        center_x, center_y = cstart[0], cstart[1]
        z = np.linspace(0, cheight, 3)
        theta = np.linspace(0, 2 * np.pi, 25)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_grid = cradius * np.cos(theta_grid) + center_x
        y_grid = cradius * np.sin(theta_grid) + center_y
        z_grid += cstart[2]
        return [x_grid, y_grid, z_grid]

    xlim = kwargs.get("x_limits", (-1.0, 1.0))
    ylim = kwargs.get("y_limits", (-1.0, 1.0))
    zlim = kwargs.get("z_limits", (-0.05, 1.0))
    difference = lambda x: x[1] - x[0]
    max_axis_length = max(difference(xlim), difference(ylim))
    # The scaling factor from physical space to matplotlib space
    scaling_factor = (2 * 0.1) / max_axis_length  # Octopus head dimension
    scaling_factor *= 2.6e3  # Along one-axis

    if kwargs.get("vis3D", True):
        fig = plt.figure(1, figsize=(10, 8), frameon=True, dpi=dpi)
        ax = plt.axes(projection="3d")
        # seaborn takes care
        # ax.grid(b=True, which="minor", color="k", linestyle="--")
        # ax.grid(b=True, which="major", color="k", linestyle="-")
        # Newer versions of matplotlib complain here, let's avoid!
        # plt.axis("square")

        # Surfaces first
        time_idx = 0
        XC, YC, ZC = make_data_for_cylinder_along_z(
            cylinder_origin[time_idx, ...], cylinder_radius, cylinder_height
        )
        surf = ax.plot_surface(XC, YC, ZC, color="g", alpha=0.5)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        # Rods next
        scatt = ax.scatter(
            positions[time_idx, 0],
            positions[time_idx, 1],
            positions[time_idx, 2],
            s=np.pi * radius[time_idx] ** 2 * 1e4,
        )

        # min_limits = global_rot_mat @ np.array([0.0, -0.5 * cylinder_height, 0.0])
        # min_limits = -np.abs(min_limits)
        # max_limits = min_limits + cylinder_height

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)

        with writer.saving(fig, video_name, dpi):
            with plt.style.context("seaborn-whitegrid"):
                for time_idx in range(0, positions.shape[0], int(step)):
                    scatt._offsets3d = (
                        positions[time_idx, 0],
                        positions[time_idx, 1],
                        positions[time_idx, 2],
                    )
                    scatt.set_sizes(np.pi * radius[time_idx] ** 2 * 1e4)

                    XC, YC, ZC = make_data_for_cylinder_along_z(
                        cylinder_origin[time_idx, ...], cylinder_radius, cylinder_height
                    )
                    surf.remove()
                    surf = ax.plot_surface(XC, YC, ZC, color="g", alpha=0.5)
                    writer.grab_frame()
    if kwargs.get("vis2D", True):
        from matplotlib.patches import Circle

        fig = plt.figure(2, figsize=(10, 8), frameon=True, dpi=dpi)
        ax = fig.add_subplot(111)
        time_idx = 0
        (rod_line,) = ax.plot(
            positions[time_idx, 0], positions[time_idx, 1], "r", lw=0.5
        )
        (tip_line,) = ax.plot(com[time_idx, 0], com[time_idx, 1], "k--", lw=2.0)
        scatt = ax.scatter(
            positions[time_idx, 0],
            positions[time_idx, 1],
            s=np.pi * (scaling_factor * radius[time_idx]) ** 2,
        )

        # min_limits = np.array([0.0, -0.5 * cylinder_height, 0.0])
        # max_limits = min_limits + cylinder_height

        # ax.set_xlim([min_limits[0], max_limits[0]])
        # ax.set_ylim([min_limits[1], max_limits[1]])

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        circle_artist = Circle(
            (cylinder_origin[time_idx, 0], cylinder_origin[time_idx, 1]),
            cylinder_radius,
            color="g",
        )
        ax.add_artist(circle_artist)
        ax.set_aspect("equal")
        video_name = "2D_" + video_name
        with writer.saving(fig, video_name, dpi):
            with plt.style.context("seaborn-whitegrid"):
                for time_idx in range(0, positions.shape[0], int(step)):
                    rod_line.set_xdata(positions[time_idx, 0])
                    rod_line.set_ydata(positions[time_idx, 1])
                    tip_line.set_xdata(com[:time_idx, 0])
                    tip_line.set_ydata(com[:time_idx, 1])
                    scatt.set_offsets(positions[time_idx, :2].T)
                    scatt.set_sizes(np.pi * (scaling_factor * radius[time_idx]) ** 2)
                    circle_artist.center = (
                        cylinder_origin[time_idx, 0],
                        cylinder_origin[time_idx, 1],
                    )
                    writer.grab_frame()
