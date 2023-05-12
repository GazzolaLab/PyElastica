import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb
from scipy.spatial.transform import Rotation


def plot_position(
    plot_params_rod1: dict,
    plot_params_rod2: dict,
    filename="spherical_joint_test.png",
    SAVE_FIGURE=False,
):

    position_of_rod1 = np.array(plot_params_rod1["position"])
    position_of_rod2 = np.array(plot_params_rod2["position"])

    fig = plt.figure(figsize=(10, 10), frameon=True, dpi=150)
    ax = fig.add_subplot(111)

    ax.grid(which="minor", color="k", linestyle="--")
    ax.grid(which="major", color="k", linestyle="-")
    ax.plot(position_of_rod1[:, 0, -1], position_of_rod1[:, 1, -1], "r-", label="rod1")
    ax.plot(
        position_of_rod2[:, 0, -1],
        position_of_rod2[:, 1, -1],
        c=to_rgb("xkcd:bluish"),
        label="rod2",
    )

    fig.legend(prop={"size": 20})

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
    plot_params_rod2: dict,
    video_name="video.mp4",
    margin=0.2,
    fps=15,
):  # (time step, x/y/z, node)
    import matplotlib.animation as manimation

    time = plot_params_rod1["time"]
    position_of_rod1 = np.array(plot_params_rod1["position"])
    position_of_rod2 = np.array(plot_params_rod2["position"])

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
            ax.plot(
                position_of_rod2[time, 0],
                position_of_rod2[time, 1],
                position_of_rod2[time, 2],
                "o",
                c=to_rgb("xkcd:bluish"),
                label="rod2",
            )

            ax.set_xlim(-0.25, 0.25)
            ax.set_ylim(-0.25, 0.25)
            ax.set_zlim(0, 0.4)
            writer.grab_frame()


def plot_video_xy(
    plot_params_rod1: dict,
    plot_params_rod2: dict,
    video_name="video.mp4",
    margin=0.2,
    fps=15,
):  # (time step, x/y/z, node)
    import matplotlib.animation as manimation

    time = plot_params_rod1["time"]
    position_of_rod1 = np.array(plot_params_rod1["position"])
    position_of_rod2 = np.array(plot_params_rod2["position"])

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
            plt.plot(
                position_of_rod2[time, 0],
                position_of_rod2[time, 1],
                "o",
                c=to_rgb("xkcd:bluish"),
                label="rod2",
            )

            plt.xlim([-0.25, 0.25])
            plt.ylim([-0.25, 0.25])
            writer.grab_frame()


def plot_video_xz(
    plot_params_rod1: dict,
    plot_params_rod2: dict,
    video_name="video.mp4",
    margin=0.2,
    fps=15,
):  # (time step, x/y/z, node)
    import matplotlib.animation as manimation

    time = plot_params_rod1["time"]
    position_of_rod1 = np.array(plot_params_rod1["position"])
    position_of_rod2 = np.array(plot_params_rod2["position"])

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
            plt.plot(
                position_of_rod2[time, 0],
                position_of_rod2[time, 2],
                "o",
                c=to_rgb("xkcd:bluish"),
                label="rod2",
            )

            plt.xlim([-0.25, 0.25])
            plt.ylim([0, 0.41])
            writer.grab_frame()
