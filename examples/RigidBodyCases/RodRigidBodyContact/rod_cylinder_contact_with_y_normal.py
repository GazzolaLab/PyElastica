import numpy as np

# FIXME without appending sys.path make it more generic
import sys

sys.path.append("../../../")
from elastica import *
from matplotlib import pyplot as plt


class SingleRodSingleCylinderInteractionSimulator(
    BaseSystemCollection, Constraints, Connections, Forcing, CallBacks
):
    pass


single_rod_sim = SingleRodSingleCylinderInteractionSimulator()
# setting up test params
n_elem = 50

inclination = np.deg2rad(30)
direction = np.array([0.0, np.sin(inclination), -np.cos(inclination)])
normal = np.array([0.0, np.cos(inclination), np.sin(inclination)])

base_length = 0.5
base_radius = 0.01
base_area = np.pi * base_radius ** 2
density = 1750
nu = 0.001
E = 3e5
poisson_ratio = 0.5
shear_modulus = E / (1 + poisson_ratio)

cylinder_start = np.array([0.3, 0.0, 0.0])
cylinder_direction = np.array([0.0, 1.0, 0.0])
cylinder_normal = np.array([1.0, 0.0, 0.0])

cylinder_height = 0.4
cylinder_radius = 10.0 * base_radius

# can be set to True, checks collision at tips of rod
tip_collision = False

# Cylinder surface starts at 0.2
tip_offset = 0.0

if tip_collision:
    # The random choice decides which tip of the rod intersects with cylinder
    tip_choice = np.random.choice([1, -1])
    tip_offset = 0.5 * tip_choice * base_length * np.cos(inclination)

start_rod_1 = np.array(
    [
        0.15,
        0.5 * cylinder_height - 0.5 * base_length * np.sin(inclination),
        0.5 * base_length * np.cos(inclination) + tip_offset,
    ]
)
# start_rod_1[2] = cylinder_radius + base_length

rod1 = CosseratRod.straight_rod(
    n_elem,
    start_rod_1,
    direction,
    normal,
    base_length,
    base_radius,
    density,
    nu,
    E,
    shear_modulus=shear_modulus,
)
# Give it an initial push
rod1.velocity_collection[0, ...] = 0.05
single_rod_sim.append(rod1)


cylinder = Cylinder(
    cylinder_start,
    cylinder_direction,
    cylinder_normal,
    cylinder_height,
    cylinder_radius,
    density,
)
single_rod_sim.append(cylinder)

single_rod_sim.connect(rod1, cylinder).using(ExternalContact, 1e2, 0.1)


# Add call backs
class PositionCollector(CallBackBaseClass):
    """
    Call back function for continuum snake
    """

    def __init__(self, step_skip: int, callback_params: dict):
        CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.callback_params = callback_params

    def make_callback(self, system, time, current_step: int):
        if current_step % self.every == 0:
            self.callback_params["time"].append(time)
            # Collect only x
            self.callback_params["position"].append(system.position_collection.copy())
            self.callback_params["com"].append(system.compute_position_center_of_mass())
            return


recorded_rod_history = defaultdict(list)
single_rod_sim.collect_diagnostics(rod1).using(
    PositionCollector, step_skip=200, callback_params=recorded_rod_history
)
recorded_cyl_history = defaultdict(list)
single_rod_sim.collect_diagnostics(cylinder).using(
    PositionCollector, step_skip=200, callback_params=recorded_cyl_history
)

single_rod_sim.finalize()
timestepper = PositionVerlet()
final_time = 2.0
dl = base_length / n_elem
dt = 1e-4
total_steps = int(final_time / dt)
print("Total steps", total_steps)

integrate(timestepper, single_rod_sim, final_time, total_steps)

if True:

    def make_data_for_cylinder_along_y(cstart, cradius, cheight):
        center_x, center_z = cstart[0], cstart[1]
        y = np.linspace(0, cheight, 5)
        theta = np.linspace(0, 2 * np.pi, 20)
        theta_grid, y_grid = np.meshgrid(theta, y)
        x_grid = cradius * np.cos(theta_grid) + center_x
        z_grid = cradius * np.sin(theta_grid) + center_z
        y_grid += cstart[2]
        return [x_grid, y_grid, z_grid]

    XC, YC, ZC = make_data_for_cylinder_along_y(
        cylinder_start, cylinder_radius, cylinder_height
    )

    def plot_video(
        plot_params: dict,
        cylinder_history: dict,
        video_name="video.mp4",
        margin=0.2,
        fps=60,
        step=1,
        *args,
        **kwargs
    ):  # (time step, x/y/z, node)
        import matplotlib.animation as manimation

        plt.rcParams.update({"font.size": 22})

        # Should give a (n_time, 3, n_elem) array
        positions = np.array(plot_params["position"])
        # (n_time, 3) array
        com = np.array(plot_params["com"])

        cylinder_com = np.array(cylinder_history["com"])
        cylinder_origin = cylinder_com - 0.5 * cylinder_height * cylinder_direction

        print("plot video")
        FFMpegWriter = manimation.writers["ffmpeg"]
        metadata = dict(
            title="Movie Test", artist="Matplotlib", comment="Movie support!"
        )
        writer = FFMpegWriter(fps=fps, metadata=metadata)
        dpi = 50

        # min_limits = np.roll(np.array([0.0, -0.5 * cylinder_height, 0.0]), _roll_key)
        if True:
            from mpl_toolkits.mplot3d import Axes3D

            fig = plt.figure(1, figsize=(10, 8), frameon=True, dpi=dpi)
            ax = plt.axes(projection="3d")  # fig.add_subplot(111)
            ax.grid(b=True, which="minor", color="k", linestyle="--")
            ax.grid(b=True, which="major", color="k", linestyle="-")
            # plt.axis("square")
            i = 0
            (rod_line,) = ax.plot(
                positions[i, 0], positions[i, 1], positions[i, 2], lw=3.0
            )
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
        if True:
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

            circle_artist = Circle(
                (cstart[i, 0], cstart[i, 2]), cylinder_radius, color="g"
            )
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

    plot_video(recorded_rod_history, recorded_cyl_history, "cylinder_rod_collision.mp4")

    positions = np.array(recorded_rod_history["position"])
    sim_time = np.array(recorded_rod_history["time"])
    colliding_element_idx = n_elem // 2
    if tip_collision:
        colliding_element_idx = -1 if tip_choice == 1 else 0
    colliding_element_history = positions[:, :, colliding_element_idx]
    fig = plt.figure(3, figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.plot(sim_time, colliding_element_history[:, 0])
    ax.hlines(
        cylinder_start[0] - cylinder_radius - base_radius,
        sim_time[0],
        sim_time[-1],
        "k",
        linestyle="dashed",
    )
    plt.show()
