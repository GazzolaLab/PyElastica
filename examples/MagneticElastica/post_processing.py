import numpy as np
import matplotlib

matplotlib.use("Agg")  # Must be before importing matplotlib.pyplot or pylab!
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib import cm
from mpl_toolkits.mplot3d import proj3d, Axes3D
from tqdm import tqdm
from matplotlib.patches import Circle
from typing import Dict, Sequence
from collections import defaultdict


def plot_final_position(rod):
    plt.rcParams.update({"font.size": 22})
    fig = plt.figure(figsize=(10, 10), frameon=True, dpi=150)

    axs = []
    axs.append(plt.subplot2grid((1, 1), (0, 0)))

    axs[0].plot(
        rod.position_collection[0, ...],
        rod.position_collection[1, ...],
        c=to_rgb("xkcd:bluish"),
        label="n=" + str(rod.n_elems),
    )
    axs[0].set_xlabel("x", fontsize=20)
    axs[0].set_ylabel("y", fontsize=20)
    plt.tight_layout()
    fig.align_ylabels()
    fig.legend(prop={"size": 20})
    fig.savefig("magnetic_beam_test" + str(rod.n_elems) + ".png")
    plt.close(plt.gcf())


def plot_tip_position_history(post_processing_dict, filename="tip_position"):
    time = np.array(post_processing_dict["time"])
    rod_position_history = np.array(post_processing_dict["position"])[:, :, -1]

    plt.rcParams.update({"font.size": 22})
    fig = plt.figure(figsize=(10, 10), frameon=True, dpi=150)

    axs = []
    axs.append(plt.subplot2grid((1, 1), (0, 0)))

    axs[0].plot(
        rod_position_history[:, 0],
        rod_position_history[:, 1],
        c=to_rgb("xkcd:bluish"),
    )
    axs[0].set_xlabel("x", fontsize=20)
    axs[0].set_ylabel("y", fontsize=20)
    plt.tight_layout()
    fig.align_ylabels()
    fig.savefig(filename + "_xy_" + ".png")
    plt.close(plt.gcf())

    plt.rcParams.update({"font.size": 22})
    fig = plt.figure(figsize=(10, 10), frameon=True, dpi=150)

    axs = []
    axs.append(plt.subplot2grid((1, 1), (0, 0)))

    axs[0].plot(
        rod_position_history[:, 0],
        rod_position_history[:, 2],
        c=to_rgb("xkcd:bluish"),
    )
    axs[0].set_xlabel("x", fontsize=20)
    axs[0].set_ylabel("z", fontsize=20)
    plt.tight_layout()
    fig.align_ylabels()
    fig.savefig(filename + "_xz_" + ".png")
    plt.close(plt.gcf())

    plt.rcParams.update({"font.size": 22})
    fig = plt.figure(figsize=(10, 10), frameon=True, dpi=150)

    axs = []
    axs.append(plt.subplot2grid((1, 1), (0, 0)))

    axs[0].plot(
        rod_position_history[:, 1],
        rod_position_history[:, 2],
        c=to_rgb("xkcd:bluish"),
    )
    axs[0].set_xlabel("y", fontsize=20)
    axs[0].set_ylabel("z", fontsize=20)
    plt.tight_layout()
    fig.align_ylabels()
    fig.savefig(filename + "_yz_" + ".png")
    plt.close(plt.gcf())

    plt.rcParams.update({"font.size": 22})
    fig = plt.figure(figsize=(10, 10), frameon=True, dpi=150)

    axs = []
    axs.append(plt.subplot2grid((1, 1), (0, 0)))

    axs[0].plot(
        time,
        rod_position_history[:, 0],
        c=to_rgb("xkcd:bluish"),
    )
    axs[0].set_xlabel("time [s]", fontsize=20)
    axs[0].set_ylabel("x", fontsize=20)
    plt.tight_layout()
    fig.align_ylabels()
    fig.savefig(filename + "_time_vs_x_" + ".png")
    plt.close(plt.gcf())

    plt.rcParams.update({"font.size": 22})
    fig = plt.figure(figsize=(10, 10), frameon=True, dpi=150)

    axs = []
    axs.append(plt.subplot2grid((1, 1), (0, 0)))

    axs[0].plot(
        time,
        rod_position_history[:, 1],
        c=to_rgb("xkcd:bluish"),
    )
    axs[0].set_xlabel("time [s]", fontsize=20)
    axs[0].set_ylabel("y", fontsize=20)
    plt.tight_layout()
    fig.align_ylabels()
    fig.savefig(filename + "_time_vs_y_" + ".png")
    plt.close(plt.gcf())

    plt.rcParams.update({"font.size": 22})
    fig = plt.figure(figsize=(10, 10), frameon=True, dpi=150)

    axs = []
    axs.append(plt.subplot2grid((1, 1), (0, 0)))

    axs[0].plot(
        time,
        rod_position_history[:, 2],
        c=to_rgb("xkcd:bluish"),
    )
    axs[0].set_xlabel("time [s]", fontsize=20)
    axs[0].set_ylabel("z", fontsize=20)
    plt.tight_layout()
    fig.align_ylabels()
    fig.savefig(filename + "_time_vs_z_" + ".png")
    plt.close(plt.gcf())


def plot_video_with_surface(
    rods_history: Sequence[Dict],
    video_name="video.mp4",
    fps=60,
    step=1,
    vis2D=True,
    **kwargs,
):
    plt.rcParams.update({"font.size": 22})

    folder_name = kwargs.get("folder_name", "")

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

    # Generate target sphere data
    sphere_flag = False
    if kwargs.__contains__("sphere_history"):
        sphere_flag = True
        sphere_history = kwargs.get("sphere_history")
        n_visualized_spheres = len(sphere_history)  # should be one for now
        sphere_history_unpacker = lambda sph_idx, t_idx: (
            sphere_history[sph_idx]["position"][t_idx],
            sphere_history[sph_idx]["radius"][t_idx],
        )
        # color mapping
        sphere_cmap = cm.get_cmap("Spectral", n_visualized_spheres)

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

        if sphere_flag:
            sphere_artists = [None for _ in range(n_visualized_spheres)]
            for sphere_idx in range(n_visualized_spheres):
                sphere_position, sphere_radius = sphere_history_unpacker(
                    sphere_idx, time_idx
                )
                sphere_artists[sphere_idx] = ax.scatter(
                    sphere_position[0],
                    sphere_position[1],
                    sphere_position[2],
                    s=np.pi * (scaling_factor * sphere_radius) ** 2,
                )
                # sphere_radius,
                # color=sphere_cmap(sphere_idx),)
                ax.add_artist(sphere_artists[sphere_idx])

        # ax.set_aspect("equal")
        video_name_3D = folder_name + "3D_" + video_name

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

                    if sphere_flag:
                        for sphere_idx in range(n_visualized_spheres):
                            sphere_position, _ = sphere_history_unpacker(
                                sphere_idx, time_idx
                            )
                            sphere_artists[sphere_idx]._offsets3d = (
                                sphere_position[0],
                                sphere_position[1],
                                sphere_position[2],
                            )

                    writer.grab_frame()

        # Be a good boy and close figures
        # https://stackoverflow.com/a/37451036
        # plt.close(fig) alone does not suffice
        # See https://github.com/matplotlib/matplotlib/issues/8560/
        plt.close(plt.gcf())

    if kwargs.get("vis2D", True):
        max_axis_length = max(difference(xlim), difference(ylim))
        # The scaling factor from physical space to matplotlib space
        scaling_factor = (2 * 0.1) / max_axis_length  # Octopus head dimension
        scaling_factor *= 2.6e3  # Along one-axis

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

        if sphere_flag:
            sphere_artists = [None for _ in range(n_visualized_spheres)]
            for sphere_idx in range(n_visualized_spheres):
                sphere_position, sphere_radius = sphere_history_unpacker(
                    sphere_idx, time_idx
                )
                sphere_artists[sphere_idx] = Circle(
                    (sphere_position[0], sphere_position[1]),
                    sphere_radius,
                    color=sphere_cmap(sphere_idx),
                )
                ax.add_artist(sphere_artists[sphere_idx])

        ax.set_aspect("equal")
        video_name_2D = folder_name + "2D_xy_" + video_name

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

                    if sphere_flag:
                        for sphere_idx in range(n_visualized_spheres):
                            sphere_position, _ = sphere_history_unpacker(
                                sphere_idx, time_idx
                            )
                            sphere_artists[sphere_idx].center = (
                                sphere_position[0],
                                sphere_position[1],
                            )

                    writer.grab_frame()

        # Be a good boy and close figures
        # https://stackoverflow.com/a/37451036
        # plt.close(fig) alone does not suffice
        # See https://github.com/matplotlib/matplotlib/issues/8560/
        plt.close(plt.gcf())

        # Plot zy
        max_axis_length = max(difference(zlim), difference(ylim))
        # The scaling factor from physical space to matplotlib space
        scaling_factor = (2 * 0.1) / max_axis_length  # Octopus head dimension
        scaling_factor *= 2.6e3  # Along one-axis

        fig = plt.figure(2, figsize=(10, 8), frameon=True, dpi=dpi)
        ax = fig.add_subplot(111)
        ax.set_xlim(*zlim)
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

        if sphere_flag:
            sphere_artists = [None for _ in range(n_visualized_spheres)]
            for sphere_idx in range(n_visualized_spheres):
                sphere_position, sphere_radius = sphere_history_unpacker(
                    sphere_idx, time_idx
                )
                sphere_artists[sphere_idx] = Circle(
                    (sphere_position[2], sphere_position[1]),
                    sphere_radius,
                    color=sphere_cmap(sphere_idx),
                )
                ax.add_artist(sphere_artists[sphere_idx])

        ax.set_aspect("equal")
        video_name_2D = folder_name + "2D_zy_" + video_name

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

                    if sphere_flag:
                        for sphere_idx in range(n_visualized_spheres):
                            sphere_position, _ = sphere_history_unpacker(
                                sphere_idx, time_idx
                            )
                            sphere_artists[sphere_idx].center = (
                                sphere_position[2],
                                sphere_position[1],
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
        ax.set_xlim(*xlim)
        ax.set_ylim(*zlim)

        # The scaling factor from physical space to matplotlib space
        max_axis_length = max(difference(zlim), difference(xlim))
        scaling_factor = (2 * 0.1) / (max_axis_length)  # Octopus head dimension
        scaling_factor *= 2.6e3  # Along one-axis

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

        if sphere_flag:
            sphere_artists = [None for _ in range(n_visualized_spheres)]
            for sphere_idx in range(n_visualized_spheres):
                sphere_position, sphere_radius = sphere_history_unpacker(
                    sphere_idx, time_idx
                )
                sphere_artists[sphere_idx] = Circle(
                    (sphere_position[0], sphere_position[2]),
                    sphere_radius,
                    color=sphere_cmap(sphere_idx),
                )
                ax.add_artist(sphere_artists[sphere_idx])

        ax.set_aspect("equal")
        video_name_2D = folder_name + "2D_xz_" + video_name

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

                    if sphere_flag:
                        for sphere_idx in range(n_visualized_spheres):
                            sphere_position, _ = sphere_history_unpacker(
                                sphere_idx, time_idx
                            )
                            sphere_artists[sphere_idx].center = (
                                sphere_position[0],
                                sphere_position[2],
                            )

                    writer.grab_frame()

        # Be a good boy and close figures
        # https://stackoverflow.com/a/37451036
        # plt.close(fig) alone does not suffice
        # See https://github.com/matplotlib/matplotlib/issues/8560/
        plt.close(plt.gcf())
