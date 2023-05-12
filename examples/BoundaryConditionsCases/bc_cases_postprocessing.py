import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation


def plot_position(
    plot_params_rod1: dict,
    filename="tip_position.png",
    SAVE_FIGURE=False,
):
    position_of_rod1 = np.array(plot_params_rod1["position"])

    fig = plt.figure(figsize=(10, 10), frameon=True, dpi=150)
    ax = fig.add_subplot(111)

    ax.grid(which="minor", color="k", linestyle="--")
    ax.grid(which="major", color="k", linestyle="-")
    ax.plot(position_of_rod1[:, 0, -1], position_of_rod1[:, 1, -1], "r-", label="rod1")

    fig.legend(prop={"size": 20})
    plt.xlabel("x")
    plt.ylabel("y")

    plt.show()

    if SAVE_FIGURE:
        fig.savefig(filename)


def plot_orientation(title, time, directors):
    quat = []
    for t in range(len(time)):
        quat_t = Rotation.from_matrix(directors[t].T).as_quat()
        quat.append(quat_t)
    quat = np.array(quat)

    plt.figure(num=title)
    plt.plot(time, quat[:, 0], label="x")
    plt.plot(time, quat[:, 1], label="y")
    plt.plot(time, quat[:, 2], label="z")
    plt.plot(time, quat[:, 3], label="w")
    plt.title(title)
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Quaternion")
    plt.show()


def plot_video(
    plot_params_rod1: dict,
    video_name="video.mp4",
    fps=100,
):  # (time step, x/y/z, node)
    import matplotlib.animation as manimation

    time = plot_params_rod1["time"]
    position_of_rod1 = np.array(plot_params_rod1["position"])

    print("plot video")
    FFMpegWriter = manimation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    fig = plt.figure(figsize=(10, 8), frameon=True, dpi=150)
    with writer.saving(fig, video_name, 100):
        for time in range(1, len(time)):
            fig.clf()
            ax = plt.axes(projection="3d")  # fig.add_subplot(111)
            ax.grid(which="minor", color="k", linestyle="--")
            ax.grid(which="major", color="k", linestyle="-")
            ax.plot(
                position_of_rod1[time, 0],
                position_of_rod1[time, 1],
                position_of_rod1[time, 2],
                "or",
                label="rod1",
            )

            ax.set_xlim(-0.25, 0.25)
            ax.set_ylim(-0.25, 0.25)
            ax.set_zlim(0, 0.61)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            writer.grab_frame()


def plot_video_xy(
    plot_params_rod1: dict,
    video_name="video_xy.mp4",
    fps=100,
):  # (time step, x/y/z, node)
    import matplotlib.animation as manimation

    time = plot_params_rod1["time"]
    position_of_rod1 = np.array(plot_params_rod1["position"])

    print("plot video xy")
    FFMpegWriter = manimation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    fig = plt.figure()
    plt.axis("equal")
    with writer.saving(fig, video_name, 100):
        for time in range(1, len(time)):
            fig.clf()
            plt.plot(
                position_of_rod1[time, 0], position_of_rod1[time, 1], "or", label="rod1"
            )

            plt.xlim([-0.25, 0.25])
            plt.ylim([-0.25, 0.25])
            plt.xlabel("x")
            plt.ylabel("y")
            writer.grab_frame()


def plot_video_xz(
    plot_params_rod1: dict,
    video_name="video_xz.mp4",
    fps=100,
):  # (time step, x/y/z, node)
    import matplotlib.animation as manimation

    time = plot_params_rod1["time"]
    position_of_rod1 = np.array(plot_params_rod1["position"])

    print("plot video xz")
    FFMpegWriter = manimation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    fig = plt.figure()
    plt.axis("equal")
    with writer.saving(fig, video_name, 100):
        for time in range(1, len(time)):
            fig.clf()
            plt.plot(
                position_of_rod1[time, 0], position_of_rod1[time, 2], "or", label="rod1"
            )

            plt.xlim([-0.25, 0.25])
            plt.ylim([0, 0.61])
            plt.xlabel("x")
            plt.ylabel("z")
            writer.grab_frame()
