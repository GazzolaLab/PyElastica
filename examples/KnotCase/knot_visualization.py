from matplotlib import pyplot as plt
import matplotlib.animation as manimation
import numpy as np


def plot_video3D(plot_params: dict, video_name="video.mp4", margin=0.2, fps=15):
    t = np.array(plot_params["time"])
    positions_over_time = np.array(plot_params["position"])
    directors_over_time = np.array(plot_params["orientation"])

    base_position, base_orientation = [], []
    for p, q in plot_params["base_pose"]:
        base_position.append(p)
        base_orientation.append(q)

    total_time = int(np.around(t[..., -1], 1))
    total_frames = fps * total_time
    step = round(len(t) / total_frames)
    print("Total time:", total_time, "Total frames:", total_frames)

    FFMpegWriter = manimation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=fps, metadata=metadata)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(-0.0 - margin, 1.0 + margin)
    ax.set_ylim(-0.3 - margin, 0.3 + margin)
    ax.set_zlim(-0.3 - margin, 0.3 + margin)
    ax.view_init(elev=10, azim=-45)
    ax.set_aspect("equal")
    rod_lines_3d = ax.plot(
        *positions_over_time[0],
        linewidth=3,
    )[0]
    quiver_length = 0.3
    targets_orientation_normal = ax.quiver(
        *base_position[0],
        *base_orientation[0][0],
        color="r",
        length=quiver_length,
    )
    targets_orientation_binormal = ax.quiver(
        *base_position[0],
        *base_orientation[0][1],
        color="g",
        length=quiver_length,
    )
    targets_orientation_tangent = ax.quiver(
        *base_position[0],
        *base_orientation[0][2],
        color="b",
        length=quiver_length,
    )
    elem_positions = 0.5 * (
        positions_over_time[..., 1:] + positions_over_time[..., :-1]
    )
    normal = ax.quiver(
        *elem_positions[0],
        *directors_over_time[0][0],
        color="k",
        alpha=0.5,
        length=quiver_length * 0.8,
    )
    with writer.saving(fig, video_name, dpi=100):
        with plt.style.context("seaborn-v0_8-whitegrid"):
            for time in range(1, len(t), step):
                rod_lines_3d.set_xdata(positions_over_time[time][0])
                rod_lines_3d.set_ydata(positions_over_time[time][1])
                rod_lines_3d.set_3d_properties(positions_over_time[time][2])

                targets_orientation_normal.remove()
                targets_orientation_normal = ax.quiver(
                    *base_position[time],
                    *base_orientation[time][0],
                    color="r",
                    length=quiver_length,
                )

                targets_orientation_binormal.remove()
                targets_orientation_binormal = ax.quiver(
                    *base_position[time],
                    *base_orientation[time][1],
                    color="g",
                    length=quiver_length,
                )

                targets_orientation_tangent.remove()
                targets_orientation_tangent = ax.quiver(
                    *base_position[time],
                    *base_orientation[time][2],
                    color="b",
                    length=quiver_length,
                )

                normal.remove()
                normal = ax.quiver(
                    *elem_positions[time],
                    *directors_over_time[time][1],
                    color="k",
                    alpha=0.5,
                    length=quiver_length * 0.8,
                )

                writer.grab_frame()
    plt.close(fig)
    print("Video saved as", video_name)
