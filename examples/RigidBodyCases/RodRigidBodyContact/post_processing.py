import numpy as np
from matplotlib import pyplot as plt


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
    **kwargs
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
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(1, figsize=(10, 8), frameon=True, dpi=dpi)
    ax = plt.axes(projection="3d")  # fig.add_subplot(111)
    ax.grid(b=True, which="minor", color="k", linestyle="--")
    ax.grid(b=True, which="major", color="k", linestyle="-")
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
