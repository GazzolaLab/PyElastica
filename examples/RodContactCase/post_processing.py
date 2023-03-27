import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from typing import Dict, Sequence


def plot_video_with_surface(
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
    zlim = kwargs.get("z_limits", (-0.05, 1.0))

    difference = lambda x: x[1] - x[0]
    max_axis_length = max(difference(xlim), difference(ylim))
    # The scaling factor from physical space to matplotlib space
    scaling_factor = (2 * 0.1) / max_axis_length  # Octopus head dimension
    scaling_factor *= 2.6e3  # Along one-axis

    if kwargs.get("vis3D", True):
        fig = plt.figure(1, figsize=(10, 8), frameon=True, dpi=dpi)
        ax = plt.axes(projection="3d")

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)

        time_idx = 0
        rod_lines = [None for _ in range(n_visualized_rods)]
        rod_com_lines = [None for _ in range(n_visualized_rods)]
        rod_scatters = [None for _ in range(n_visualized_rods)]

        for rod_idx in range(n_visualized_rods):
            inst_position, inst_radius = rod_history_unpacker(rod_idx, time_idx)
            if not inst_position.shape[1] == inst_radius.shape[0]:
                inst_position = 0.5 * (inst_position[..., 1:] + inst_position[..., :-1])

            rod_scatters[rod_idx] = ax.scatter(
                inst_position[0],
                inst_position[1],
                inst_position[2],
                s=np.pi * (scaling_factor * inst_radius) ** 2,
            )

        # ax.set_aspect("equal")
        video_name_3D = "3D_" + video_name

        with writer.saving(fig, video_name_3D, dpi):
            with plt.style.context("seaborn-whitegrid"):
                for time_idx in tqdm(range(0, sim_time.shape[0], int(step))):

                    for rod_idx in range(n_visualized_rods):
                        inst_position, inst_radius = rod_history_unpacker(
                            rod_idx, time_idx
                        )
                        if not inst_position.shape[1] == inst_radius.shape[0]:
                            inst_position = 0.5 * (
                                inst_position[..., 1:] + inst_position[..., :-1]
                            )

                        rod_scatters[rod_idx]._offsets3d = (
                            inst_position[0],
                            inst_position[1],
                            inst_position[2],
                        )

                        # rod_scatters[rod_idx].set_offsets(inst_position[:2].T)
                        rod_scatters[rod_idx].set_sizes(
                            np.pi * (scaling_factor * inst_radius) ** 2
                        )

                    writer.grab_frame()

        # Be a good boy and close figures
        # https://stackoverflow.com/a/37451036
        # plt.close(fig) alone does not suffice
        # See https://github.com/matplotlib/matplotlib/issues/8560/
        plt.close(plt.gcf())

    if kwargs.get("vis2D", True):
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
            if not inst_position.shape[1] == inst_radius.shape[0]:
                inst_position = 0.5 * (inst_position[..., 1:] + inst_position[..., :-1])
            rod_lines[rod_idx] = ax.plot(
                inst_position[0], inst_position[1], "r", lw=0.5
            )[0]
            inst_com = com_history_unpacker(rod_idx, time_idx)
            rod_com_lines[rod_idx] = ax.plot(inst_com[0], inst_com[1], "k--", lw=2.0)[0]

            rod_scatters[rod_idx] = ax.scatter(
                inst_position[0],
                inst_position[1],
                s=np.pi * (scaling_factor * inst_radius) ** 2,
            )

        ax.set_aspect("equal")
        video_name_2D = "2D_xy_" + video_name

        with writer.saving(fig, video_name_2D, dpi):
            with plt.style.context("seaborn-whitegrid"):
                for time_idx in tqdm(range(0, sim_time.shape[0], int(step))):

                    for rod_idx in range(n_visualized_rods):
                        inst_position, inst_radius = rod_history_unpacker(
                            rod_idx, time_idx
                        )
                        if not inst_position.shape[1] == inst_radius.shape[0]:
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

        # Plot zy

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
            if not inst_position.shape[1] == inst_radius.shape[0]:
                inst_position = 0.5 * (inst_position[..., 1:] + inst_position[..., :-1])
            rod_lines[rod_idx] = ax.plot(
                inst_position[2], inst_position[1], "r", lw=0.5
            )[0]
            inst_com = com_history_unpacker(rod_idx, time_idx)
            rod_com_lines[rod_idx] = ax.plot(inst_com[2], inst_com[1], "k--", lw=2.0)[0]

            rod_scatters[rod_idx] = ax.scatter(
                inst_position[2],
                inst_position[1],
                s=np.pi * (scaling_factor * inst_radius) ** 2,
            )

        ax.set_aspect("equal")
        video_name_2D = "2D_zy_" + video_name

        with writer.saving(fig, video_name_2D, dpi):
            with plt.style.context("seaborn-whitegrid"):
                for time_idx in tqdm(range(0, sim_time.shape[0], int(step))):

                    for rod_idx in range(n_visualized_rods):
                        inst_position, inst_radius = rod_history_unpacker(
                            rod_idx, time_idx
                        )
                        if not inst_position.shape[1] == inst_radius.shape[0]:
                            inst_position = 0.5 * (
                                inst_position[..., 1:] + inst_position[..., :-1]
                            )

                        rod_lines[rod_idx].set_xdata(inst_position[2])
                        rod_lines[rod_idx].set_ydata(inst_position[1])

                        com = com_history_unpacker(rod_idx, time_idx)
                        rod_com_lines[rod_idx].set_xdata(com[2])
                        rod_com_lines[rod_idx].set_ydata(com[1])

                        rod_scatters[rod_idx].set_offsets(
                            np.vstack((inst_position[2], inst_position[1])).T
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

        # Plot xz
        fig = plt.figure(2, figsize=(10, 8), frameon=True, dpi=dpi)
        ax = fig.add_subplot(111)
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)

        time_idx = 0
        rod_lines = [None for _ in range(n_visualized_rods)]
        rod_com_lines = [None for _ in range(n_visualized_rods)]
        rod_scatters = [None for _ in range(n_visualized_rods)]

        for rod_idx in range(n_visualized_rods):
            inst_position, inst_radius = rod_history_unpacker(rod_idx, time_idx)
            if not inst_position.shape[1] == inst_radius.shape[0]:
                inst_position = 0.5 * (inst_position[..., 1:] + inst_position[..., :-1])
            rod_lines[rod_idx] = ax.plot(
                inst_position[0], inst_position[2], "r", lw=0.5
            )[0]
            inst_com = com_history_unpacker(rod_idx, time_idx)
            rod_com_lines[rod_idx] = ax.plot(inst_com[0], inst_com[2], "k--", lw=2.0)[0]

            rod_scatters[rod_idx] = ax.scatter(
                inst_position[0],
                inst_position[2],
                s=np.pi * (scaling_factor * inst_radius) ** 2,
            )

        ax.set_aspect("equal")
        video_name_2D = "2D_xz_" + video_name

        with writer.saving(fig, video_name_2D, dpi):
            with plt.style.context("seaborn-whitegrid"):
                for time_idx in tqdm(range(0, sim_time.shape[0], int(step))):

                    for rod_idx in range(n_visualized_rods):
                        inst_position, inst_radius = rod_history_unpacker(
                            rod_idx, time_idx
                        )
                        if not inst_position.shape[1] == inst_radius.shape[0]:
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


def plot_velocity(
    plot_params_rod_one: dict,
    plot_params_rod_two: dict,
    filename="velocity.png",
    SAVE_FIGURE=False,
):
    time = np.array(plot_params_rod_one["time"])
    avg_velocity_rod_one = np.array(plot_params_rod_one["com_velocity"])
    avg_velocity_rod_two = np.array(plot_params_rod_two["com_velocity"])
    total_energy_rod_one = np.array(plot_params_rod_one["total_energy"])
    total_energy_rod_two = np.array(plot_params_rod_two["total_energy"])

    fig = plt.figure(figsize=(12, 10), frameon=True, dpi=150)
    axs = []
    axs.append(plt.subplot2grid((4, 1), (0, 0)))
    axs.append(plt.subplot2grid((4, 1), (1, 0)))
    axs.append(plt.subplot2grid((4, 1), (2, 0)))
    axs.append(plt.subplot2grid((4, 1), (3, 0)))

    axs[0].plot(time[:], avg_velocity_rod_one[:, 0], linewidth=3, label="rod_one")
    axs[0].plot(time[:], avg_velocity_rod_two[:, 0], linewidth=3, label="rod_two")
    axs[0].plot(
        time[:],
        avg_velocity_rod_one[:, 0] + avg_velocity_rod_two[:, 0],
        "--",
        linewidth=3,
        label="total",
    )
    axs[0].set_ylabel("x velocity", fontsize=20)

    axs[1].plot(
        time[:],
        avg_velocity_rod_one[:, 1],
        linewidth=3,
    )
    axs[1].plot(
        time[:],
        avg_velocity_rod_two[:, 1],
        linewidth=3,
    )
    axs[1].plot(
        time[:],
        avg_velocity_rod_one[:, 1] + avg_velocity_rod_two[:, 1],
        "--",
        linewidth=3,
    )
    axs[1].set_ylabel("y velocity", fontsize=20)

    axs[2].plot(
        time[:],
        avg_velocity_rod_one[:, 2],
        linewidth=3,
    )
    axs[2].plot(
        time[:],
        avg_velocity_rod_two[:, 2],
        linewidth=3,
    )
    axs[2].plot(
        time[:],
        avg_velocity_rod_one[:, 2] + avg_velocity_rod_two[:, 2],
        "--",
        linewidth=3,
    )
    axs[2].set_ylabel("z velocity", fontsize=20)

    axs[3].semilogy(
        time[:],
        total_energy_rod_one[:],
        linewidth=3,
    )
    axs[3].semilogy(
        time[:],
        total_energy_rod_two[:],
        linewidth=3,
    )
    axs[3].semilogy(
        time[:],
        np.abs(total_energy_rod_one[:] - total_energy_rod_two[:]),
        "--",
        linewidth=3,
    )
    axs[3].set_ylabel("total_energy", fontsize=20)
    axs[3].set_xlabel("time [s]", fontsize=20)

    plt.tight_layout()
    # fig.align_ylabels()
    fig.legend(prop={"size": 20})
    # fig.savefig(filename)
    plt.show()
    plt.close(plt.gcf())

    if SAVE_FIGURE:
        fig.savefig(filename)
        # plt.savefig(filename)


def plot_link_writhe_twist(twist_density, total_twist, total_writhe, total_link):

    plt.rcParams.update({"font.size": 22})
    fig = plt.figure(figsize=(10, 10), frameon=True, dpi=150)

    axs = []
    axs.append(plt.subplot2grid((1, 1), (0, 0)))
    axs[0].plot(
        twist_density,
        total_twist,
        label="twist",
    )
    axs[0].plot(
        twist_density,
        total_writhe,
        label="writhe",
    )
    axs[0].plot(
        twist_density,
        total_link,
        label="link",
    )
    axs[0].set_xlabel("twist density", fontsize=20)
    axs[0].set_ylabel("link-twist-writhe", fontsize=20)
    axs[0].set_xlim(0, 2.0)
    plt.tight_layout()
    fig.align_ylabels()
    fig.legend(prop={"size": 20})
    fig.savefig("link_twist_writhe.png")
    plt.close(plt.gcf())
