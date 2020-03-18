""" Flexible swinging pendulum test-case
    isort:skip_file
"""
# FIXME without appending sys.path make it more generic
import sys

sys.path.append("../../")  # isort:skip

# from collections import defaultdict

# import numpy as np
from matplotlib import pyplot as plt

# from elastica.boundary_conditions import FreeRod
# from elastica.callback_functions import CallBackBaseClass
# from elastica.external_forces import GravityForces
# from elastica.rod.cosserat_rod import CosseratRod
# from elastica.timestepper import integrate
# from elastica.timestepper.symplectic_steppers import PEFRL, PositionVerlet
# from elastica.wrappers import BaseSystemCollection, CallBacks, Constraints, Forcing
from elastica import *
from elastica._rotations import _get_rotation_matrix


class GammaOmegaQuasistaticSimulator(
    BaseSystemCollection, Constraints, Forcing, CallBacks
):
    pass


# Options
PLOT_FIGURE = True
PLOT_VIDEO = False
SAVE_FIGURE = False
SAVE_RESULTS = False

# For 10 elements, the prefac is  0.0007
gamma_omega_sim = GammaOmegaQuasistaticSimulator()
final_time = 0.5

# setting up test params
n_elem = 10 if SAVE_RESULTS else 10
start = np.zeros((3,))
direction = np.array([0.0, 0.0, 1.0])
normal = np.array([1.0, 0.0, 0.0])
base_length = 1.0
base_radius = 0.005
base_area = np.pi * base_radius ** 2
density = 1100.0
nu = 0.1
youngs_modulus = 5e6
# For shear modulus of 1e4, nu is 99!
poisson_ratio = 0.5

gamma_omega_rod = CosseratRod.straight_rod(
    n_elem,
    start,
    direction,
    normal,
    base_length,
    base_radius,
    density,
    nu,
    youngs_modulus,
    poisson_ratio,
)

gamma_omega_sim.append(gamma_omega_rod)


# Bad name : whats a FreeRod anyway?
class GammaOmegaBC(FreeRod):
    """
    the end of the rod fixed x[0]
    """

    def __init__(self, first_end, last_end, first_director, last_director):
        FreeRod.__init__(self)
        self.first_end = first_end.copy()
        self.last_end = last_end.copy()

        self.first_director = first_director.copy()
        self.last_director = last_director.copy()

    def constrain_values(self, rod, time):
        non_dimensional_time = time / final_time
        inst_first_position = 0.5 * base_length * non_dimensional_time
        self.first_end[2] = inst_first_position
        rod.position_collection[..., 0] = self.first_end
        if non_dimensional_time < 0.1:
            rod.position_collection[1, 0] = 1e-5 * np.sin(
                np.pi * non_dimensional_time / 0.1
            )

        inst_last_position = base_length * (1.0 - 0.5 * non_dimensional_time)
        self.last_end[2] = inst_last_position
        rod.position_collection[..., -1] = self.last_end
        if non_dimensional_time < 0.1:
            rod.position_collection[1, -1] = 1e-5 * np.sin(
                np.pi * non_dimensional_time / 0.1
            )

        # Slerp for rotation matrices
        # First angle is 0.0, last angle is np.pi on the first director
        inst_first_angle = (np.pi) * non_dimensional_time
        rotation_axes = np.array([0.0, 0.0, 1.0]).reshape(-1, 1)
        inst_first_dir = _get_rotation_matrix(inst_first_angle, rotation_axes)
        # print(inst_first_dir)
        rod.director_collection[..., 0] = inst_first_dir[..., 0] @ self.first_director

        inst_last_angle = (-np.pi) * non_dimensional_time
        inst_last_dir = _get_rotation_matrix(inst_last_angle, rotation_axes)
        # print(inst_last_dir)
        rod.director_collection[..., -1] = inst_last_dir[..., 0] @ self.last_director

    def constrain_rates(self, rod, time):
        non_dimensional_time = time / final_time

        rod.velocity_collection[..., 0] = 0.0
        rod.velocity_collection[2, 0] = 0.5 * base_length / final_time
        if non_dimensional_time < 0.1:
            rod.velocity_collection[1, 0] = (
                1e-5
                * np.pi
                / final_time
                / 0.1
                * np.cos(np.pi * non_dimensional_time / 0.1)
            )

        rod.velocity_collection[..., -1] = 0.0
        rod.velocity_collection[2, -1] = -0.5 * base_length / final_time
        if non_dimensional_time < 0.1:
            rod.velocity_collection[1, -1] = (
                1e-5
                * np.pi
                / final_time
                / 0.1
                * np.cos(np.pi * non_dimensional_time / 0.1)
            )

        # This is in d3! but d3 is equivalent to e3 in our case
        rod.omega_collection[..., 0] = 0.0
        rod.omega_collection[2, 0] = np.pi / final_time

        rod.omega_collection[..., -1] = 0.0
        rod.omega_collection[2, -1] = -np.pi / final_time


gamma_omega_sim.constrain(gamma_omega_rod).using(
    GammaOmegaBC, constrained_position_idx=(0, -1), constrained_director_idx=(0, -1)
)


# Add call backs
class GammaOmegaCallBack(CallBackBaseClass):
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
            self.callback_params["position"].append(system.position_collection.copy())
            self.callback_params["directors"].append(system.director_collection.copy())
            self.callback_params["dilatation"].append(system.dilatation.copy())
        return


dl = base_length / n_elem
dt = 0.00005 * dl
total_steps = int(final_time / dt)

print("Total steps", total_steps)
recorded_history = defaultdict(list)
step_skip = 60
gamma_omega_sim.collect_diagnostics(gamma_omega_rod).using(
    GammaOmegaCallBack, step_skip=step_skip, callback_params=recorded_history,
)

gamma_omega_sim.finalize()
timestepper = PositionVerlet()
# timestepper = PEFRL()

integrate(timestepper, gamma_omega_sim, final_time, total_steps)

if PLOT_VIDEO:

    def plot_video(
        plot_params: dict,
        video_name="video.mp4",
        margin=0.2,
        fps=60,
        step=1,
        *args,
        **kwargs
    ):
        # (time step, x/y/z, node)
        import matplotlib.animation as manimation
        from mpl_toolkits.mplot3d import Axes3D

        plt.rcParams.update({"font.size": 22})

        print("Plotting video")
        time = plot_params["time"]
        position = np.array(plot_params["position"])
        radius = np.array(plot_params["radius"])
        FFMpegWriter = manimation.writers["ffmpeg"]
        metadata = dict(
            title="Movie Test", artist="Matplotlib", comment="Movie support!"
        )
        writer = FFMpegWriter(fps=fps, metadata=metadata)
        dpi = 150
        fig = plt.figure(figsize=(10, 8), frameon=True, dpi=dpi)
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        i = 0
        (rod_line,) = ax.plot(positions[i, 2], positions[i, 0], positions[i, 1], lw=3.0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(0 - margin, 1.0 + margin)
        ax.set_ylim(-1.00 - margin, 0.00 + margin)
        ax.set_zlim(0.0, 1.0 + margin)
        ax.set_xlabel("z positon")
        ax.set_ylabel("x position")
        ax.set_zlabel("y position")
        # ax.grid(b=True, which="minor", color="k", linestyle="--")
        # ax.grid(b=True, which="major", color="k", linestyle="-")
        with writer.saving(fig, video_name, dpi):
            with plt.style.context("fivethirtyeight"):
                for temporal_idx in range(1, len(time), int(step)):
                    rod_line.set_xdata(position[temporal_idx, 2])
                    rod_line.set_ydata(position[temporal_idx, 0])
                    rod_line.set_zdata(position[temporal_idx, 1])
                    writer.grab_frame()

    plot_video(recorded_history, "quas_gamma_omega.mp4")

if PLOT_FIGURE:
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(10, 8), frameon=True, dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    # ax.set_aspect('equal', adjustable='box')
    # Should give a (n_time, 3, n_elem) array
    positions = np.array(recorded_history["position"])
    for i in range(positions.shape[0]):
        ax.plot(positions[i, 2], positions[i, 0], positions[i, 1], lw=2.0)
    fig.show()
    plt.show()

if SAVE_RESULTS:
    import pickle as pickle

    filename = "gamma_omega_quasistatic.dat"
    with open(filename, "wb") as file:
        pickle.dump(recorded_history, file)
