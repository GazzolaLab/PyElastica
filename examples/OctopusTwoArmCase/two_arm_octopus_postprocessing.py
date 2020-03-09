import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb


def plot_video_with_surface(
    plot_params: dict,
    cylinder_history: dict,
    cylinder_radius,
    cylinder_height,
    cylinder_direction,
    video_name="video.mp4",
    fps=60,
    step=1,
    **kwargs
):  # (time step, x/y/z, node)
    import matplotlib.animation as manimation
    from mpl_toolkits.mplot3d import proj3d, Axes3D

    plt.rcParams.update({"font.size": 22})

    # Should give a (n_time, 3, n_elem) array
    positions = np.array(plot_params["position"])
    radius = np.array(plot_params["radius"])
    # Should give a (n_time, 3) array
    com = np.array(plot_params["com"])

    cylinder_com = np.array(cylinder_history["com"])
    cylinder_origin = cylinder_com - 0.5 * cylinder_height * cylinder_direction.reshape(
        3
    )

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
        # seaborn takes care
        # ax.grid(b=True, which="minor", color="k", linestyle="--")
        # ax.grid(b=True, which="major", color="k", linestyle="-")
        # Newer versions of matplotlib complain here, let's avoid!
        # plt.axis("square")

        # Surfaces first
        time_idx = 0
        XC, YC, ZC = make_data_for_cylinder_along_z(
            cylinder_origin[time_idx, ...], cylinder_radius, cylinder_height
        )
        surf = ax.plot_surface(XC, YC, ZC, color="g", alpha=0.5)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        # Rods next
        scatt = ax.scatter(
            positions[time_idx, 0],
            positions[time_idx, 1],
            positions[time_idx, 2],
            s=np.pi * radius[time_idx] ** 2 * 1e4,
        )

        # min_limits = global_rot_mat @ np.array([0.0, -0.5 * cylinder_height, 0.0])
        # min_limits = -np.abs(min_limits)
        # max_limits = min_limits + cylinder_height

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)

        with writer.saving(fig, video_name, dpi):
            with plt.style.context("seaborn-whitegrid"):
                for time_idx in range(0, positions.shape[0], int(step)):
                    scatt._offsets3d = (
                        positions[time_idx, 0],
                        positions[time_idx, 1],
                        positions[time_idx, 2],
                    )
                    scatt.set_sizes(np.pi * radius[time_idx] ** 2 * 1e4)

                    XC, YC, ZC = make_data_for_cylinder_along_z(
                        cylinder_origin[time_idx, ...], cylinder_radius, cylinder_height
                    )
                    surf.remove()
                    surf = ax.plot_surface(XC, YC, ZC, color="g", alpha=0.5)
                    writer.grab_frame()
    if kwargs.get("vis2D", True):
        from matplotlib.patches import Circle

        fig = plt.figure(2, figsize=(10, 8), frameon=True, dpi=dpi)
        ax = fig.add_subplot(111)
        time_idx = 0
        (rod_line,) = ax.plot(
            positions[time_idx, 0], positions[time_idx, 1], "r", lw=0.5
        )
        (tip_line,) = ax.plot(com[time_idx, 0], com[time_idx, 1], "k--", lw=2.0)
        scatt = ax.scatter(
            positions[time_idx, 0],
            positions[time_idx, 1],
            s=np.pi * (scaling_factor * radius[time_idx]) ** 2,
        )

        # min_limits = np.array([0.0, -0.5 * cylinder_height, 0.0])
        # max_limits = min_limits + cylinder_height

        # ax.set_xlim([min_limits[0], max_limits[0]])
        # ax.set_ylim([min_limits[1], max_limits[1]])

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        circle_artist = Circle(
            (cylinder_origin[time_idx, 0], cylinder_origin[time_idx, 1]),
            cylinder_radius,
            color="g",
        )
        ax.add_artist(circle_artist)
        ax.set_aspect("equal")
        video_name = "2D_" + video_name
        with writer.saving(fig, video_name, dpi):
            with plt.style.context("seaborn-whitegrid"):
                for time_idx in range(0, positions.shape[0], int(step)):
                    rod_line.set_xdata(positions[time_idx, 0])
                    rod_line.set_ydata(positions[time_idx, 1])
                    tip_line.set_xdata(com[:time_idx, 0])
                    tip_line.set_ydata(com[:time_idx, 1])
                    scatt.set_offsets(positions[time_idx, :2].T)
                    scatt.set_sizes(np.pi * (scaling_factor * radius[time_idx]) ** 2)
                    circle_artist.center = (
                        cylinder_origin[time_idx, 0],
                        cylinder_origin[time_idx, 1],
                    )
                    writer.grab_frame()
