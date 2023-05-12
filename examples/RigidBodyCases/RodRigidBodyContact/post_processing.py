import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from typing import Dict, Sequence
from tqdm import tqdm


def make_data_for_cylinder_along_y(cstart, cradius, cheight):
    center_x, center_z = cstart[0], cstart[1]
    y = np.linspace(0, cheight, 5)
    theta = np.linspace(0, 2 * np.pi, 20)
    theta_grid, y_grid = np.meshgrid(theta, y)
    x_grid = cradius * np.cos(theta_grid) + center_x
    z_grid = cradius * np.sin(theta_grid) + center_z
    y_grid += cstart[2]
    return [x_grid, y_grid, z_grid]


def plot_video(
    rod_history: dict,
    cylinder_history: dict,
    video_name="video.mp4",
    margin=0.2,
    fps=60,
    step=1,
    *args,
    **kwargs,
):  # (time step, x/y/z, node)

    cylinder_start = np.array(cylinder_history["position"])[0, ...]
    cylinder_radius = kwargs.get("cylinder_radius")
    cylinder_height = kwargs.get("cylinder_height")
    cylinder_direction = kwargs.get("cylinder_direction")

    XC, YC, ZC = make_data_for_cylinder_along_y(
        cylinder_start, cylinder_radius, cylinder_height
    )

    import matplotlib.animation as manimation

    plt.rcParams.update({"font.size": 22})

    # Should give a (n_time, 3, n_elem) array
    positions = np.array(rod_history["position"])
    # (n_time, 3) array
    com = np.array(rod_history["com"])

    cylinder_com = np.array(cylinder_history["com"])
    cylinder_origin = cylinder_com - 0.5 * cylinder_height * cylinder_direction

    print("plot video")
    FFMpegWriter = manimation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    dpi = 50

    # min_limits = np.roll(np.array([0.0, -0.5 * cylinder_height, 0.0]), _roll_key)

    fig = plt.figure(1, figsize=(10, 8), frameon=True, dpi=dpi)
    ax = plt.axes(projection="3d")  # fig.add_subplot(111)
    ax.grid(which="minor", color="k", linestyle="--")
    ax.grid(which="major", color="k", linestyle="-")
    # plt.axis("square")
    i = 0
    (rod_line,) = ax.plot(positions[i, 0], positions[i, 1], positions[i, 2], lw=3.0)
    XC, YC, ZC = make_data_for_cylinder_along_y(
        cylinder_origin[i, ...], cylinder_radius, cylinder_height
    )
    surf = ax.plot_surface(XC, YC, ZC, color="g", alpha=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    min_limits = np.array([0.0, 0.0, -0.5 * cylinder_height])
    min_limits = -np.abs(min_limits)
    max_limits = min_limits + cylinder_height

    ax.set_xlim([min_limits[0], max_limits[0]])
    ax.set_ylim([min_limits[1], max_limits[1]])
    ax.set_zlim([min_limits[2], max_limits[2]])
    with writer.saving(fig, video_name, dpi):
        with plt.style.context("seaborn-white"):
            for i in range(0, positions.shape[0], int(step)):
                rod_line.set_xdata(positions[i, 0])
                rod_line.set_ydata(positions[i, 1])
                rod_line.set_3d_properties(positions[i, 2])
                XC, YC, ZC = make_data_for_cylinder_along_y(
                    cylinder_origin[i, ...], cylinder_radius, cylinder_height
                )
                surf.remove()
                surf = ax.plot_surface(XC, YC, ZC, color="g", alpha=0.5)
                writer.grab_frame()

    from matplotlib.patches import Circle

    fig = plt.figure(2, figsize=(10, 8), frameon=True, dpi=dpi)
    ax = fig.add_subplot(111)
    i = 0
    cstart = cylinder_origin
    (rod_line,) = ax.plot(positions[i, 0], positions[i, 1], lw=3.0)
    (tip_line,) = ax.plot(com[i, 0], com[i, 1], "k--")

    min_limits = np.array([0.0, 0.0, -0.5 * cylinder_height])
    max_limits = min_limits + cylinder_height

    ax.set_xlim([min_limits[0], max_limits[0]])
    ax.set_ylim([min_limits[2], max_limits[2]])

    circle_artist = Circle((cstart[i, 0], cstart[i, 2]), cylinder_radius, color="g")
    ax.add_artist(circle_artist)
    ax.set_aspect("equal")
    video_name = "2D_" + video_name
    with writer.saving(fig, video_name, dpi):
        with plt.style.context("fivethirtyeight"):
            for i in range(0, positions.shape[0], int(step)):
                rod_line.set_xdata(positions[i, 0])
                rod_line.set_ydata(positions[i, 2])
                tip_line.set_xdata(com[:i, 0])
                tip_line.set_ydata(com[:i, 2])
                circle_artist.center = cstart[i, 0], cstart[i, 2]
                writer.grab_frame()


def plot_cylinder_rod_position(
    rod_history,
    cylinder_history,
    cylinder_radius,
    rod_base_radius,
    TIP_COLLISION,
    TIP_CHOICE,
    _roll_key=0,
):
    cylinder_start = np.array(cylinder_history["position"])[0, ...]
    positions = np.array(rod_history["position"])
    sim_time = np.array(rod_history["time"])

    n_elem = positions.shape[-1]

    fig = plt.figure(figsize=(10, 8), frameon=True, dpi=150)
    plt.rcParams.update({"font.size": 18})
    ax = fig.add_subplot(111)
    colliding_element_idx = n_elem // 2
    if TIP_COLLISION:
        colliding_element_idx = 0 if TIP_CHOICE == 1 else -1
    colliding_element_history = positions[:, :, colliding_element_idx]
    # fig = plt.figure(3, figsize=(8, 5))
    # ax = fig.add_subplot(111)
    ax.plot(sim_time, colliding_element_history[:, _roll_key], label="rod")
    ax.hlines(
        cylinder_start[_roll_key] - cylinder_radius - rod_base_radius,
        sim_time[0],
        sim_time[-1],
        "k",
        linestyle="dashed",
        label="cylinder",
    )
    plt.xlabel("Time [s]", fontsize=20)
    plt.ylabel("Position", fontsize=20)
    fig.legend(prop={"size": 20})
    plt.show()


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
    # plt.show()
    plt.close(plt.gcf())

    if SAVE_FIGURE:
        fig.savefig(filename)


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

            # for reference see
            # https://stackoverflow.com/questions/48172928/scale-matplotlib-pyplot
            # -axes-scatter-markersize-by-x-scale/48174228#48174228
            scaling_factor = (
                ax.get_window_extent().width / (max_axis_length) * 72.0 / fig.dpi
            )
            rod_scatters[rod_idx] = ax.scatter(
                inst_position[0],
                inst_position[1],
                inst_position[2],
                # for circle s = 4/pi*area = 4 * r^2
                s=4 * (scaling_factor * inst_radius) ** 2,
            )

        if sphere_flag:
            sphere_artists = [None for _ in range(n_visualized_spheres)]
            for sphere_idx in range(n_visualized_spheres):
                sphere_position, sphere_radius = sphere_history_unpacker(
                    sphere_idx, time_idx
                )
                scaling_factor = (
                    ax.get_window_extent().width / (max_axis_length) * 72.0 / fig.dpi
                )
                sphere_artists[sphere_idx] = ax.scatter(
                    sphere_position[0],
                    sphere_position[1],
                    sphere_position[2],
                    s=4 * (scaling_factor * sphere_radius) ** 2,
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

                        scaling_factor = (
                            ax.get_window_extent().width
                            / (max_axis_length)
                            * 72.0
                            / fig.dpi
                        )
                        # rod_scatters[rod_idx].set_offsets(inst_position[:2].T)
                        rod_scatters[rod_idx].set_sizes(
                            4 * (scaling_factor * inst_radius) ** 2
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

            scaling_factor = (
                ax.get_window_extent().width / (max_axis_length) * 72.0 / fig.dpi
            )
            rod_scatters[rod_idx] = ax.scatter(
                inst_position[0],
                inst_position[1],
                s=4 * (scaling_factor * inst_radius) ** 2,
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
                        scaling_factor = (
                            ax.get_window_extent().width
                            / (max_axis_length)
                            * 72.0
                            / fig.dpi
                        )
                        rod_scatters[rod_idx].set_sizes(
                            4 * (scaling_factor * inst_radius) ** 2
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

            scaling_factor = (
                ax.get_window_extent().width / (max_axis_length) * 72.0 / fig.dpi
            )
            rod_scatters[rod_idx] = ax.scatter(
                inst_position[2],
                inst_position[1],
                s=4 * (scaling_factor * inst_radius) ** 2,
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
                        scaling_factor = (
                            ax.get_window_extent().width
                            / (max_axis_length)
                            * 72.0
                            / fig.dpi
                        )
                        rod_scatters[rod_idx].set_sizes(
                            4 * (scaling_factor * inst_radius) ** 2
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

            scaling_factor = (
                ax.get_window_extent().width / (max_axis_length) * 72.0 / fig.dpi
            )
            rod_scatters[rod_idx] = ax.scatter(
                inst_position[0],
                inst_position[2],
                s=4 * (scaling_factor * inst_radius) ** 2,
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
                        scaling_factor = (
                            ax.get_window_extent().width
                            / (max_axis_length)
                            * 72.0
                            / fig.dpi
                        )
                        rod_scatters[rod_idx].set_sizes(
                            4 * (scaling_factor * inst_radius) ** 2
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


def plot_force_vs_energy(
    normalized_force,
    total_final_energy,
    friction_coefficient,
    filename="energy_vs_force.png",
    SAVE_FIGURE=False,
):

    fig = plt.figure(figsize=(12, 10), frameon=True, dpi=150)
    axs = []
    axs.append(plt.subplot2grid((1, 1), (0, 0)))

    axs[0].plot(
        normalized_force,
        total_final_energy,
        linewidth=3,
    )
    plt.axvline(x=friction_coefficient, linewidth=3, color="r", label="threshold")
    axs[0].set_ylabel("total energy", fontsize=20)
    axs[0].set_xlabel("normalized force", fontsize=20)

    plt.tight_layout()
    # fig.align_ylabels()
    fig.legend(prop={"size": 20})
    # fig.savefig(filename)
    # plt.show()
    plt.close(plt.gcf())

    if SAVE_FIGURE:
        fig.savefig(filename)
