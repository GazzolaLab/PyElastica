import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb
from scipy.spatial.transform import Rotation

from elastica.rigidbody import Cylinder
from elastica._linalg import _batch_matvec


def plot_position(
    plot_params_rod1: dict,
    plot_params_rod2: dict,
    plot_params_cylinder: dict = None,
    filename="joint_cases_last_node_pos_xy.png",
    SAVE_FIGURE=False,
):
    position_of_rod1 = np.array(plot_params_rod1["position"])
    position_of_rod2 = np.array(plot_params_rod2["position"])
    position_of_cylinder = (
        np.array(plot_params_cylinder["position"])
        if plot_params_cylinder is not None
        else None
    )

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
    if position_of_cylinder is not None:
        ax.plot(
            position_of_cylinder[:, 0, -1],
            position_of_cylinder[:, 1, -1],
            c=to_rgb("xkcd:greenish"),
            label="cylinder",
        )

    fig.legend(prop={"size": 20})
    plt.xlabel("x")
    plt.ylabel("y")

    plt.show()

    if SAVE_FIGURE:
        fig.savefig(filename)


def plot_orientation(title, time, directors):
    """
    Plot the orientation of one node
    """
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
    plot_params_cylinder: dict = None,
    video_name="joint_cases_video.mp4",
    fps=100,
    cylinder: Cylinder = None,
):  # (time step, x/y/z, node)
    import matplotlib.animation as manimation

    time = plot_params_rod1["time"]
    position_of_rod1 = np.array(plot_params_rod1["position"])
    position_of_rod2 = np.array(plot_params_rod2["position"])
    position_of_cylinder = (
        np.array(plot_params_cylinder["position"])
        if plot_params_cylinder is not None
        else None
    )
    director_of_cylinder = (
        np.array(plot_params_cylinder["directors"])
        if plot_params_cylinder is not None
        else None
    )

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
            if position_of_cylinder is not None:
                ax.plot(
                    position_of_cylinder[time, 0],
                    position_of_cylinder[time, 1],
                    position_of_cylinder[time, 2],
                    "o",
                    c=to_rgb("xkcd:greenish"),
                    label="Cylinder CoM",
                )
                if cylinder is not None:
                    cylinder_axis_points = np.zeros((3, 10))
                    cylinder_axis_points[2, :] = np.linspace(
                        start=-cylinder.length.squeeze() / 2,
                        stop=cylinder.length.squeeze() / 2,
                        num=cylinder_axis_points.shape[1],
                    )
                    cylinder_director = director_of_cylinder[time, ...]
                    cylinder_director_batched = np.repeat(
                        cylinder_director, repeats=cylinder_axis_points.shape[1], axis=2
                    )
                    # rotate points into inertial frame
                    cylinder_axis_points = _batch_matvec(
                        cylinder_director_batched.transpose((1, 0, 2)),
                        cylinder_axis_points,
                    )
                    # add offset position of CoM
                    cylinder_axis_points += position_of_cylinder[time, ...]
                    ax.plot(
                        cylinder_axis_points[0, :],
                        cylinder_axis_points[1, :],
                        cylinder_axis_points[2, :],
                        c=to_rgb("xkcd:greenish"),
                        label="Cylinder axis",
                    )

            ax.set_xlim(-0.25, 0.25)
            ax.set_ylim(-0.25, 0.25)
            ax.set_zlim(0, 0.61)
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.set_zlabel("z [m]")
            writer.grab_frame()


def plot_video_xy(
    plot_params_rod1: dict,
    plot_params_rod2: dict,
    plot_params_cylinder: dict = None,
    video_name="joint_cases_video_xy.mp4",
    fps=100,
    cylinder: Cylinder = None,
):  # (time step, x/y/z, node)
    import matplotlib.animation as manimation

    time = plot_params_rod1["time"]
    position_of_rod1 = np.array(plot_params_rod1["position"])
    position_of_rod2 = np.array(plot_params_rod2["position"])
    position_of_cylinder = (
        np.array(plot_params_cylinder["position"])
        if plot_params_cylinder is not None
        else None
    )
    director_of_cylinder = (
        np.array(plot_params_cylinder["directors"])
        if plot_params_cylinder is not None
        else None
    )

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
            if position_of_cylinder is not None:
                plt.plot(
                    position_of_cylinder[time, 0],
                    position_of_cylinder[time, 1],
                    "o",
                    c=to_rgb("xkcd:greenish"),
                    label="cylinder",
                )
                if cylinder is not None:
                    cylinder_axis_points = np.zeros((3, 10))
                    cylinder_axis_points[2, :] = np.linspace(
                        start=-cylinder.length.squeeze() / 2,
                        stop=cylinder.length.squeeze() / 2,
                        num=cylinder_axis_points.shape[1],
                    )
                    cylinder_director = director_of_cylinder[time, ...]
                    cylinder_director_batched = np.repeat(
                        cylinder_director, repeats=cylinder_axis_points.shape[1], axis=2
                    )
                    # rotate points into inertial frame
                    cylinder_axis_points = _batch_matvec(
                        cylinder_director_batched.transpose((1, 0, 2)),
                        cylinder_axis_points,
                    )
                    # add offset position of CoM
                    cylinder_axis_points += position_of_cylinder[time, ...]
                    plt.plot(
                        cylinder_axis_points[0, :],
                        cylinder_axis_points[1, :],
                        c=to_rgb("xkcd:greenish"),
                        label="Cylinder axis",
                    )

            plt.xlim([-0.25, 0.25])
            plt.ylim([-0.25, 0.25])
            plt.xlabel("x [m]")
            plt.ylabel("y [m]")
            writer.grab_frame()


def plot_video_xz(
    plot_params_rod1: dict,
    plot_params_rod2: dict,
    plot_params_cylinder: dict = None,
    video_name="joint_cases_video_xz.mp4",
    fps=100,
    cylinder: Cylinder = None,
):  # (time step, x/y/z, node)
    import matplotlib.animation as manimation

    time = plot_params_rod1["time"]
    position_of_rod1 = np.array(plot_params_rod1["position"])
    position_of_rod2 = np.array(plot_params_rod2["position"])
    position_of_cylinder = (
        np.array(plot_params_cylinder["position"])
        if plot_params_cylinder is not None
        else None
    )
    director_of_cylinder = (
        np.array(plot_params_cylinder["directors"])
        if plot_params_cylinder is not None
        else None
    )

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
            if position_of_cylinder is not None:
                plt.plot(
                    position_of_cylinder[time, 0],
                    position_of_cylinder[time, 2],
                    "o",
                    c=to_rgb("xkcd:greenish"),
                    label="cylinder",
                )
                if cylinder is not None:
                    cylinder_axis_points = np.zeros((3, 10))
                    cylinder_axis_points[2, :] = np.linspace(
                        start=-cylinder.length.squeeze() / 2,
                        stop=cylinder.length.squeeze() / 2,
                        num=cylinder_axis_points.shape[1],
                    )
                    cylinder_director = director_of_cylinder[time, ...]
                    cylinder_director_batched = np.repeat(
                        cylinder_director, repeats=cylinder_axis_points.shape[1], axis=2
                    )
                    # rotate points into inertial frame
                    cylinder_axis_points = _batch_matvec(
                        cylinder_director_batched.transpose((1, 0, 2)),
                        cylinder_axis_points,
                    )
                    # add offset position of CoM
                    cylinder_axis_points += position_of_cylinder[time, ...]
                    plt.plot(
                        cylinder_axis_points[0, :],
                        cylinder_axis_points[2, :],
                        c=to_rgb("xkcd:greenish"),
                        label="Cylinder axis",
                    )

            plt.xlim([-0.25, 0.25])
            plt.ylim([0, 0.61])
            plt.xlabel("x [m]")
            plt.ylabel("z [m]")
            writer.grab_frame()
