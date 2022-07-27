import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb
from mpl_toolkits import mplot3d
from scipy.spatial.transform import Rotation
from typing import Dict

from elastica.rigidbody import Cylinder
from elastica._linalg import _batch_matvec


def plot_video(
    plot_params_pendulum: Dict,
    pendulum: Cylinder,
    plot_params_sphere: Dict,
    video_name="pendulum_tracking_sphere_example.mp4",
    fps=100,
):  # (time step, x/y/z, node)
    import matplotlib.animation as manimation

    time = plot_params_pendulum["time"]
    position_of_pendulum = np.array(plot_params_pendulum["position"])
    director_of_pendulum = np.array(plot_params_pendulum["directors"])
    position_of_sphere = np.array(plot_params_sphere["position"])

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

            # plot pendulum center of mass
            ax.plot(
                position_of_pendulum[time, 0],
                position_of_pendulum[time, 1],
                position_of_pendulum[time, 2],
                "or",
                label="Pendulum CoM",
            )

            # plot pendulum axis
            pendulum_axis_points = np.zeros((3, 10))
            pendulum_axis_points[2, :] = np.linspace(
                start=-pendulum.length.squeeze() / 2,
                stop=pendulum.length.squeeze() / 2,
                num=pendulum_axis_points.shape[1],
            )
            pendulum_director = director_of_pendulum[time, ...]
            pendulum_director_batched = np.repeat(
                pendulum_director, repeats=pendulum_axis_points.shape[1], axis=2
            )
            # rotate points into inertial frame
            pendulum_axis_points = _batch_matvec(
                pendulum_director_batched.transpose((1, 0, 2)),
                pendulum_axis_points,
            )
            # add offset position of CoM
            pendulum_axis_points += position_of_pendulum[time, ...]
            ax.plot(
                pendulum_axis_points[0, :],
                pendulum_axis_points[1, :],
                pendulum_axis_points[2, :],
                c="r",
                label="Pendulum",
            )

            ax.plot(
                position_of_sphere[time, 0],
                position_of_sphere[time, 1],
                position_of_sphere[time, 2],
                "x",
                c=to_rgb("xkcd:bluish"),
                label="Sphere",
            )

            ax.set_xlim(-0.25, 0.25)
            ax.set_ylim(-0.25, 0.25)
            ax.set_zlim(0.0, 0.1)
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.set_zlabel("z [m]")
            plt.legend()

            writer.grab_frame()


def plot_video_xy(
    plot_params_pendulum: Dict,
    pendulum: Cylinder,
    plot_params_sphere: Dict,
    video_name="pendulum_tracking_sphere_example_xy.mp4",
    fps=100,
):  # (time step, x/y/z, node)
    import matplotlib.animation as manimation

    time = plot_params_pendulum["time"]
    position_of_pendulum = np.array(plot_params_pendulum["position"])
    director_of_pendulum = np.array(plot_params_pendulum["directors"])
    position_of_sphere = np.array(plot_params_sphere["position"])

    print("plot video xy")
    FFMpegWriter = manimation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    fig = plt.figure()
    plt.axis("equal")
    with writer.saving(fig, video_name, 100):
        for time in range(1, len(time)):
            fig.clf()

            # plot pendulum center of mass
            plt.plot(
                position_of_pendulum[time, 0],
                position_of_pendulum[time, 1],
                "or",
                label="Pendulum CoM",
            )

            # plot pendulum axis
            pendulum_axis_points = np.zeros((3, 10))
            pendulum_axis_points[2, :] = np.linspace(
                start=-pendulum.length.squeeze() / 2,
                stop=pendulum.length.squeeze() / 2,
                num=pendulum_axis_points.shape[1],
            )
            pendulum_director = director_of_pendulum[time, ...]
            pendulum_director_batched = np.repeat(
                pendulum_director, repeats=pendulum_axis_points.shape[1], axis=2
            )
            # rotate points into inertial frame
            pendulum_axis_points = _batch_matvec(
                pendulum_director_batched.transpose((1, 0, 2)),
                pendulum_axis_points,
            )
            # add offset position of CoM
            pendulum_axis_points += position_of_pendulum[time, ...]
            plt.plot(
                pendulum_axis_points[0, :],
                pendulum_axis_points[1, :],
                c="r",
                label="Pendulum",
            )

            plt.plot(
                position_of_sphere[time, 0],
                position_of_sphere[time, 1],
                "x",
                c=to_rgb("xkcd:bluish"),
                label="Sphere",
            )

            plt.xlim([-0.25, 0.25])
            plt.ylim([-0.25, 0.25])
            plt.xlabel("x [m]")
            plt.ylabel("y [m]")
            plt.legend()

            writer.grab_frame()
