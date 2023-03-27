import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from typing import Dict, Sequence


def plot_video(
    rods_history: Sequence[Dict],
    video_name="video.mp4",
    fps=60,
    step=1,
    **kwargs,
):
    plt.rcParams.update({"font.size": 22})

    # 2d case <always 2d case for now>
    import matplotlib.animation as animation

    # simulation time
    sim_time = np.array(rods_history[0]["time"])

    # Rod
    n_visualized_rods = len(rods_history)  # should be one for now
    # Rod info
    rod_history_unpacker = lambda rod_idx, t_idx: (
        rods_history[rod_idx]["position"][t_idx],
        rods_history[rod_idx]["radius"][t_idx],
    )
    # Rod center of mass
    com_history_unpacker = lambda rod_idx, t_idx: rods_history[rod_idx]["com"][time_idx]

    # video pre-processing
    print("plot scene visualization video")
    FFMpegWriter = animation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    dpi = kwargs.get("dpi", 100)

    xlim = kwargs.get("x_limits", (-1.0, 1.0))
    ylim = kwargs.get("y_limits", (-1.0, 1.0))
    zlim = kwargs.get("z_limits", (-0.0, 1.0))

    difference = lambda x: x[1] - x[0]
    max_axis_length = max(difference(xlim), difference(ylim))
    # The scaling factor from physical space to matplotlib space
    scaling_factor = (2 * 0.1) / max_axis_length  # Octopus head dimension
    scaling_factor *= 2.6e3  # Along one-axis

    fig = plt.figure(2, figsize=(10, 8), frameon=True, dpi=dpi)
    # ax = fig.add_subplot(111)
    ax = plt.axes(projection="3d")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)

    time_idx = 0
    rod_lines = [None for _ in range(n_visualized_rods)]
    rod_com_lines = [None for _ in range(n_visualized_rods)]
    rod_scatters = [None for _ in range(n_visualized_rods)]

    for rod_idx in range(n_visualized_rods):
        inst_position, inst_radius = rod_history_unpacker(rod_idx, time_idx)
        inst_position = 0.5 * (inst_position[..., 1:] + inst_position[..., :-1])
        # rod_lines[rod_idx] = ax.plot(inst_position[0], inst_position[1],inst_position[2], "r", lw=0.5)[0]
        inst_com = com_history_unpacker(rod_idx, time_idx)
        # rod_com_lines[rod_idx] = ax.plot(inst_com[0], inst_com[1],inst_com[2], "k--", lw=2.0)[0]

        rod_scatters[rod_idx] = ax.scatter(
            inst_position[2],
            inst_position[0],
            inst_position[1],
            s=np.pi * (scaling_factor * inst_radius) ** 2,
        )

    # ax.set_aspect("equal")

    with writer.saving(fig, video_name, dpi):
        with plt.style.context("seaborn-whitegrid"):
            for time_idx in tqdm(range(0, sim_time.shape[0], int(step))):

                for rod_idx in range(n_visualized_rods):
                    inst_position, inst_radius = rod_history_unpacker(rod_idx, time_idx)
                    inst_position = 0.5 * (
                        inst_position[..., 1:] + inst_position[..., :-1]
                    )

                    # rod_lines[rod_idx].set_xdata(inst_position[0])
                    # rod_lines[rod_idx].set_ydata(inst_position[1])
                    # rod_lines[rod_idx].set_zdata(inst_position[2])

                    com = com_history_unpacker(rod_idx, time_idx)
                    # rod_com_lines[rod_idx].set_xdata(com[0])
                    # rod_com_lines[rod_idx].set_ydata(com[1])
                    # rod_com_lines[rod_idx].set_zdata(com[2])

                    # rod_scatters[rod_idx].set_offsets(inst_position[:3].T)
                    rod_scatters[rod_idx]._offsets3d = (
                        inst_position[2],
                        inst_position[0],
                        inst_position[1],
                    )

                    rod_scatters[rod_idx].set_sizes(
                        np.pi * (scaling_factor * inst_radius) ** 2 * 0.1
                    )

                writer.grab_frame()

    # Be a good boy and close figures
    # https://stackoverflow.com/a/37451036
    # plt.close(fig) alone does not suffice
    # See https://github.com/matplotlib/matplotlib/issues/8560/
    plt.close(plt.gcf())


def plot_video_2D(
    rods_history: Sequence[Dict],
    video_name="video_2D.mp4",
    fps=60,
    step=1,
    vis2D=True,
    **kwargs,
):
    plt.rcParams.update({"font.size": 22})

    # 2d case <always 2d case for now>
    import matplotlib.animation as animation

    # simulation time
    sim_time = np.array(rods_history[0]["time"])

    # Rod
    n_visualized_rods = len(rods_history)  # should be one for now
    # Rod info
    rod_history_unpacker = lambda rod_idx, t_idx: (
        rods_history[rod_idx]["position"][t_idx],
        rods_history[rod_idx]["radius"][t_idx],
    )
    # Rod center of mass
    com_history_unpacker = lambda rod_idx, t_idx: rods_history[rod_idx]["com"][time_idx]

    # video pre-processing
    print("plot scene visualization video")
    FFMpegWriter = animation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    dpi = kwargs.get("dpi", 100)

    xlim = kwargs.get("x_limits", (-1.0, 1.0))
    ylim = kwargs.get("y_limits", (-1.0, 1.0))
    zlim = kwargs.get("z_limits", (-0.05, 1.0))

    difference = lambda x: x[1] - x[0]
    max_axis_length = max(difference(xlim), difference(ylim))
    # The scaling factor from physical space to matplotlib space
    scaling_factor = (2 * 0.1) / max_axis_length  # Octopus head dimension
    scaling_factor *= 2.6e3  # Along one-axis

    # xy plot
    video_name_xy = "xy_" + video_name
    fig = plt.figure(2, figsize=(10, 8), frameon=True, dpi=dpi)
    ax = fig.add_subplot(111)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    time_idx = 0
    rod_lines = [None for _ in range(n_visualized_rods)]
    rod_com_lines = [None for _ in range(n_visualized_rods)]
    rod_scatters = [None for _ in range(n_visualized_rods)]

    for rod_idx in range(n_visualized_rods):
        inst_position, inst_radius = rod_history_unpacker(rod_idx, time_idx)
        inst_position = 0.5 * (inst_position[..., 1:] + inst_position[..., :-1])
        rod_lines[rod_idx] = ax.plot(inst_position[0], inst_position[1], "r", lw=0.5)[0]
        inst_com = com_history_unpacker(rod_idx, time_idx)
        rod_com_lines[rod_idx] = ax.plot(inst_com[0], inst_com[1], "k--", lw=2.0)[0]

        rod_scatters[rod_idx] = ax.scatter(
            inst_position[0],
            inst_position[1],
            s=np.pi * (scaling_factor * inst_radius) ** 2,
        )

    ax.set_aspect("equal")

    with writer.saving(fig, video_name_xy, dpi):
        with plt.style.context("seaborn-whitegrid"):
            for time_idx in tqdm(range(0, sim_time.shape[0], int(step))):

                for rod_idx in range(n_visualized_rods):
                    inst_position, inst_radius = rod_history_unpacker(rod_idx, time_idx)
                    inst_position = 0.5 * (
                        inst_position[..., 1:] + inst_position[..., :-1]
                    )

                    rod_lines[rod_idx].set_xdata(inst_position[0])
                    rod_lines[rod_idx].set_ydata(inst_position[1])

                    com = com_history_unpacker(rod_idx, time_idx)
                    rod_com_lines[rod_idx].set_xdata(com[0])
                    rod_com_lines[rod_idx].set_ydata(com[1])

                    rod_scatters[rod_idx].set_offsets(inst_position[:2].T)
                    rod_scatters[rod_idx].set_sizes(
                        np.pi * (scaling_factor * inst_radius) ** 2
                    )

                writer.grab_frame()

    # Be a good boy and close figures
    # https://stackoverflow.com/a/37451036
    # plt.close(fig) alone does not suffice
    # See https://github.com/matplotlib/matplotlib/issues/8560/
    plt.close(plt.gcf())

    # xz plot
    video_name_xz = "xz_" + video_name
    fig = plt.figure(2, figsize=(10, 8), frameon=True, dpi=dpi)
    ax = fig.add_subplot(111)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    time_idx = 0
    rod_lines = [None for _ in range(n_visualized_rods)]
    rod_com_lines = [None for _ in range(n_visualized_rods)]
    rod_scatters = [None for _ in range(n_visualized_rods)]

    for rod_idx in range(n_visualized_rods):
        inst_position, inst_radius = rod_history_unpacker(rod_idx, time_idx)
        inst_position = 0.5 * (inst_position[..., 1:] + inst_position[..., :-1])
        rod_lines[rod_idx] = ax.plot(inst_position[0], inst_position[2], "r", lw=0.5)[0]
        inst_com = com_history_unpacker(rod_idx, time_idx)
        rod_com_lines[rod_idx] = ax.plot(inst_com[0], inst_com[2], "k--", lw=2.0)[0]

        rod_scatters[rod_idx] = ax.scatter(
            inst_position[0],
            inst_position[2],
            s=np.pi * (scaling_factor * inst_radius) ** 2,
        )

    ax.set_aspect("equal")

    with writer.saving(fig, video_name_xz, dpi):
        with plt.style.context("seaborn-whitegrid"):
            for time_idx in tqdm(range(0, sim_time.shape[0], int(step))):

                for rod_idx in range(n_visualized_rods):
                    inst_position, inst_radius = rod_history_unpacker(rod_idx, time_idx)
                    inst_position = 0.5 * (
                        inst_position[..., 1:] + inst_position[..., :-1]
                    )

                    rod_lines[rod_idx].set_xdata(inst_position[0])
                    rod_lines[rod_idx].set_ydata(inst_position[2])

                    com = com_history_unpacker(rod_idx, time_idx)
                    rod_com_lines[rod_idx].set_xdata(com[0])
                    rod_com_lines[rod_idx].set_ydata(com[2])

                    rod_scatters[rod_idx].set_offsets(
                        np.vstack((inst_position[0], inst_position[2])).T
                    )
                    rod_scatters[rod_idx].set_sizes(
                        np.pi * (scaling_factor * inst_radius) ** 2
                    )

                writer.grab_frame()

    # Be a good boy and close figures
    # https://stackoverflow.com/a/37451036
    # plt.close(fig) alone does not suffice
    # See https://github.com/matplotlib/matplotlib/issues/8560/
    plt.close(plt.gcf())


def plot_com_position_vs_time(
    rods_history, file_name="muscular_flagella_com_pos_vs_time.png"
):

    time = rods_history["time"]
    # rod_com_position = np.array(rods_history["com"]) * -1e3
    rod_com_position = np.array(rods_history["position"])[:, :, 9] * -1e3

    # We are interested in dx, subtract initial position.
    rod_com_position -= rod_com_position[0, :]

    plt.rcParams.update({"font.size": 22})
    fig = plt.figure(figsize=(10, 10), frameon=True, dpi=150)

    axs = []
    axs.append(plt.subplot2grid((3, 1), (0, 0)))
    axs.append(plt.subplot2grid((3, 1), (1, 0)))
    axs.append(plt.subplot2grid((3, 1), (2, 0)))

    axs[0].plot(time, rod_com_position[:, 0], "-", linewidth=3)
    axs[0].set_ylabel("com x [micro m]", fontsize=15)

    axs[1].plot(time, rod_com_position[:, 1], "-", linewidth=3)
    axs[1].set_ylabel("com y [micro m]", fontsize=15)

    axs[2].plot(time, rod_com_position[:, 2], "-", linewidth=3)
    axs[2].set_ylabel("com z [micro m]", fontsize=15)

    axs[2].set_xlabel("time [s]", fontsize=15)

    plt.tight_layout()
    fig.align_ylabels()
    fig.savefig(file_name)
    plt.show()
    plt.close(plt.gcf())


def plot_position_vs_time_comparison_cpp(
    rods_history, file_name="muscular_flagella_com_pos_vs_time.png"
):
    time = rods_history["time"]
    rod_com_position = np.array(rods_history["position"])[:, :, 9] * -1e3

    # We are interested in dx, subtract initial position.
    rod_com_position -= rod_com_position[0, :]

    # Load data from cpp simulation
    data = np.loadtxt("data/Flagella.txt")

    min_y = np.min(rod_com_position[:, 0])
    max_y = np.max(rod_com_position[:, 0])

    plt.rcParams.update({"font.size": 22})
    fig = plt.figure(figsize=(12, 8), frameon=True, dpi=150)

    axs = []
    axs.append(plt.subplot2grid((1, 1), (0, 0)))

    axs[0].plot(time, rod_com_position[:, 0], "-", linewidth=3, label="PyElastica")
    axs[0].plot(
        data[:, 0],
        (data[:, 1] - data[0, 1]) * -1e3,
        "-.",
        linewidth=3,
        label="Elastica Cpp",
    )
    axs[0].set_yticks(np.arange(min_y, max_y, 5.0))
    axs[0].set_ylim(min_y, max_y)
    axs[0].set_ylabel("com x [micro m]", fontsize=20)

    axs[0].set_xlabel("time [s]", fontsize=20)

    plt.tight_layout()
    fig.align_ylabels()
    fig.legend(prop={"size": 20})
    fig.savefig(file_name)
    plt.show()
    plt.close(plt.gcf())
