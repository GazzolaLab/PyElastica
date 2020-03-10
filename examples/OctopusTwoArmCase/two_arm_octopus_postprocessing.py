import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from tqdm import tqdm

from typing import Dict, Sequence


def plot_video_with_surface(
        rods_history: Sequence[Dict],
        cylinders_history: Sequence[Dict],
        video_name="video.mp4",
        fps=60,
        step=1,
        **kwargs
):  # (time step, x/y/z, node)
    import matplotlib.animation as manimation
    from mpl_toolkits.mplot3d import proj3d, Axes3D

    plt.rcParams.update({"font.size": 22})

    # Cylinders first
    n_visualized_cylinders = len(cylinders_history)
    # n_cyl, n_time, 3,
    # cylinder_com = np.array([x["com"] for x in cylinders_history])
    # n_cyl floats
    cylinder_heights = [x["height"] for x in cylinders_history]
    cylinder_radii = [x["radius"] for x in cylinders_history]
    sim_time = np.array(cylinders_history[0]["time"])

    cylinder_cmap = cm.get_cmap('Spectral', n_visualized_cylinders)

    # Rods next
    n_visualized_rods = len(rods_history)

    # TODO : Should be a generator rather a function
    rod_history_unpacker = lambda rod_idx, t_idx: (
        rods_history[rod_idx]["position"][time_idx], rods_history[rod_idx]["radius"][t_idx])
    com_history_unpacker = lambda rod_idx, t_idx: rods_history[rod_idx]["com"][time_idx]
    cylinder_history_unpacker = lambda cyl_idx, t_idx: (
        cylinders_history[cyl_idx]["com"][t_idx] - 0.5 * cylinder_heights[cyl_idx] * cylinders_history[cyl_idx][
            "direction"].reshape(3), cylinder_radii[cyl_idx], cylinder_heights[cyl_idx])

    print("Plotting videos!")
    FFMpegWriter = manimation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    dpi = kwargs.get("dpi", 100)

    def make_data_for_cylinder_along_z(cstart, cradius, cheight):
        center_x, center_y = cstart[0], cstart[1]
        z = np.linspace(0, cheight, 3)
        theta = np.linspace(0, 2 * np.pi, 25)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_grid = cradius * np.cos(theta_grid) + center_x
        y_grid = cradius * np.sin(theta_grid) + center_y
        z_grid += cstart[2]
        return [x_grid, y_grid, z_grid]

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

        # Surfaces (cylinders, spheres) first
        time_idx = 0
        cylinder_surfs = [None for _ in range(n_visualized_cylinders)]

        for cylinder_idx in range(n_visualized_cylinders):
            XC, YC, ZC = make_data_for_cylinder_along_z(
                *cylinder_history_unpacker(cylinder_idx, time_idx)
            )
            cylinder_surfs[cylinder_idx] = ax.plot_surface(XC, YC, ZC, color=cylinder_cmap(cylinder_idx), alpha=1.0)

        # Rods next
        rod_scatters = [None for _ in range(n_visualized_cylinders)]

        for rod_idx in range(n_visualized_rods):
            inst_position, inst_radius = rod_history_unpacker(rod_idx, time_idx)
            inst_position = 0.5 * (inst_position[..., 1:] + inst_position[..., :-1])
            rod_scatters[rod_idx] = ax.scatter(
                inst_position[0],
                inst_position[1],
                inst_position[2],
                s=np.pi * inst_radius ** 2 * 1e4,
            )

        # min_limits = global_rot_mat @ np.array([0.0, -0.5 * cylinder_height, 0.0])
        # min_limits = -np.abs(min_limits)
        # max_limits = min_limits + cylinder_height

        with writer.saving(fig, video_name, dpi):
            with plt.style.context("seaborn-whitegrid"):
                for time_idx in tqdm(range(0, sim_time.shape[0], int(step))):
                    for rod_idx in range(n_visualized_rods):
                        inst_position, inst_radius = rod_history_unpacker(rod_idx, time_idx)
                        inst_position = 0.5 * (inst_position[..., 1:] + inst_position[..., :-1])
                        rod_scatters[rod_idx]._offsets3d = (
                            inst_position[0],
                            inst_position[1],
                            inst_position[2],
                        )
                        rod_scatters[rod_idx].set_sizes(np.pi * inst_radius ** 2 * 1e4)

                    for cylinder_idx in range(n_visualized_cylinders):
                        XC, YC, ZC = make_data_for_cylinder_along_z(*cylinder_history_unpacker(cylinder_idx, time_idx))
                        cylinder_surfs[cylinder_idx].remove()
                        cylinder_surfs[cylinder_idx] = ax.plot_surface(XC, YC, ZC, color=cylinder_cmap(cylinder_idx),
                                                                       alpha=1.0)

                    writer.grab_frame()

        # Delete all variables within scope
        # Painful
        del rod_scatters, cylinder_surfs,
        del time_idx, rod_idx, cylinder_idx
        del inst_position, inst_radius
        del XC, YC, ZC

        # Be a good boy and close figures
        # https://stackoverflow.com/a/37451036
        # plt.close(fig) alone does not suffice
        # See https://github.com/matplotlib/matplotlib/issues/8560/
        plt.close(plt.gcf())

    if kwargs.get("vis2D", True):
        from matplotlib.patches import Circle

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

        # min_limits = np.array([0.0, -0.5 * cylinder_height, 0.0])
        # max_limits = min_limits + cylinder_height

        # ax.set_xlim([min_limits[0], max_limits[0]])
        # ax.set_ylim([min_limits[1], max_limits[1]])

        cylinder_artists = [None for _ in range(n_visualized_cylinders)]
        for cylinder_idx in range(n_visualized_cylinders):
            cylinder_origin, cylinder_radius, _ = cylinder_history_unpacker(cylinder_idx, time_idx)
            cylinder_artists[cylinder_idx] = Circle(
                (cylinder_origin[0], cylinder_origin[1]),
                cylinder_radius,
                color=cylinder_cmap(cylinder_idx),
            )
            ax.add_artist(cylinder_artists[cylinder_idx])
        ax.set_aspect("equal")
        video_name = "2D_" + video_name

        with writer.saving(fig, video_name, dpi):
            with plt.style.context("seaborn-whitegrid"):
                for time_idx in tqdm(range(0, sim_time.shape[0], int(step))):

                    for rod_idx in range(n_visualized_rods):
                        inst_position, inst_radius = rod_history_unpacker(rod_idx, time_idx)
                        inst_position = 0.5 * (inst_position[..., 1:] + inst_position[..., :-1])

                        rod_lines[rod_idx].set_xdata(inst_position[0])
                        rod_lines[rod_idx].set_ydata(inst_position[1])

                        com = com_history_unpacker(rod_idx, time_idx)
                        rod_com_lines[rod_idx].set_xdata(com[0])
                        rod_com_lines[rod_idx].set_ydata(com[1])

                        rod_scatters[rod_idx].set_offsets(
                            inst_position[:2].T,
                        )
                        rod_scatters[rod_idx].set_sizes(np.pi * (scaling_factor * inst_radius) ** 2)

                    for cylinder_idx in range(n_visualized_cylinders):
                        cylinder_origin, _, _ = cylinder_history_unpacker(cylinder_idx, time_idx)
                        cylinder_artists[cylinder_idx].center = (cylinder_origin[0], cylinder_origin[1])

                    writer.grab_frame()

        # Be a good boy and close figures
        # https://stackoverflow.com/a/37451036
        # plt.close(fig) alone does not suffice
        # See https://github.com/matplotlib/matplotlib/issues/8560/
        plt.close(plt.gcf())
