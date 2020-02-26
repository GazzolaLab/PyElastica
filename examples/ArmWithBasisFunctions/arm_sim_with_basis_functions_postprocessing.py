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
            torq = torque[time][1]
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
    plot_params: dict,
    video_name="video.mp4",
    margin=0.2,
    fps=15,
    step=100,
    *args,
    **kwargs
):  # (time step, x/y/z, node)
    import matplotlib.animation as manimation
    from mpl_toolkits import mplot3d

    time = plot_params["time"]
    position = np.array(plot_params["position"])
    radius = np.array(plot_params["radius"])

    print("plot video")
    FFMpegWriter = manimation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    fig = plt.figure(figsize=(10, 8), frameon=True, dpi=150)
    with writer.saving(fig, video_name, 100):
        for time in range(1, len(time)):
            fig.clf()
            ax = plt.axes(projection="3d")  # fig.add_subplot(111)
            ax.grid(b=True, which="minor", color="k", linestyle="--")
            ax.grid(b=True, which="major", color="k", linestyle="-")
            ax.scatter(
                position[time, 2],
                position[time, 0],
                position[time, 1],
                s=np.pi * radius[time] ** 2 * 10000,
            )
            ax.set_xlim(0 - margin, 1.0 + margin)
            ax.set_ylim(-1.00 - margin, 0.00 + margin)
            ax.set_zlim(0.0, 1.0 + margin)
            ax.set_xlabel("z positon")
            ax.set_ylabel("x position")
            ax.set_zlabel("y position")
            writer.grab_frame()
