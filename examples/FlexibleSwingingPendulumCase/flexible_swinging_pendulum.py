""" Flexible swinging pendulum test-case
    isort:skip_file
"""

import numpy as np
from matplotlib import pyplot as plt

import elastica as ea


class SwingingFlexiblePendulumSimulator(
    ea.BaseSystemCollection, ea.Constraints, ea.Forcing, ea.CallBacks
):
    pass


# Options
PLOT_FIGURE = False
PLOT_VIDEO = False
SAVE_FIGURE = False
SAVE_RESULTS = True

# For 10 elements, the prefac is  0.0007
pendulum_sim = SwingingFlexiblePendulumSimulator()
final_time = 1.0 if SAVE_RESULTS else 5.0

# setting up test params
n_elem = 10 if SAVE_RESULTS else 50
start = np.zeros((3,))
direction = np.array([0.0, 0.0, 1.0])
normal = np.array([1.0, 0.0, 0.0])
base_length = 1.0
base_radius = 0.005
base_area = np.pi * base_radius ** 2
density = 1100.0
youngs_modulus = 5e6
# For shear modulus of 1e4, nu is 99!
poisson_ratio = 0.5

pendulum_rod = ea.CosseratRod.straight_rod(
    n_elem,
    start,
    direction,
    normal,
    base_length,
    base_radius,
    density,
    youngs_modulus=youngs_modulus,
    shear_modulus=youngs_modulus / (poisson_ratio + 1.0),
)

pendulum_sim.append(pendulum_rod)


# Bad name : whats a FreeRod anyway?
class HingeBC(ea.ConstraintBase):
    """
    the end of the rod fixed x[0]
    """

    def __init__(self, fixed_position, fixed_directors, **kwargs):
        super().__init__(**kwargs)
        self.fixed_position = np.array(fixed_position)
        self.fixed_directors = np.array(fixed_directors)

    def constrain_values(self, rod, time):
        rod.position_collection[..., 0] = self.fixed_position

    def constrain_rates(self, rod, time):
        rod.velocity_collection[..., 0] = 0.0


pendulum_sim.constrain(pendulum_rod).using(
    HingeBC, constrained_position_idx=(0,), constrained_director_idx=(0,)
)

# Add gravitational forces
gravitational_acc = -9.80665
pendulum_sim.add_forcing_to(pendulum_rod).using(
    ea.GravityForces, acc_gravity=np.array([gravitational_acc, 0.0, 0.0])
)


# Add call backs
class PendulumCallBack(ea.CallBackBaseClass):
    """
    Call back function for continuum snake
    """

    def __init__(self, step_skip: int, callback_params: dict):
        ea.CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.callback_params = callback_params

    def make_callback(self, system, time, current_step: int):
        if current_step % self.every == 0:
            self.callback_params["time"].append(time)
            self.callback_params["position"].append(system.position_collection.copy())
            self.callback_params["directors"].append(system.director_collection.copy())
            if time > 0.0:
                self.callback_params["internal_stress"].append(
                    system.internal_stress.copy()
                )
                self.callback_params["internal_couple"].append(
                    system.internal_couple.copy()
                )
        return


dl = base_length / n_elem
dt = (0.0007 if SAVE_RESULTS else 0.002) * dl
total_steps = int(final_time / dt)

print("Total steps", total_steps)
recorded_history = ea.defaultdict(list)
step_skip = (
    60
    if PLOT_VIDEO
    else (int(total_steps / 10) if PLOT_FIGURE else int(total_steps / 200))
)
pendulum_sim.collect_diagnostics(pendulum_rod).using(
    PendulumCallBack, step_skip=step_skip, callback_params=recorded_history
)

pendulum_sim.finalize()
timestepper = ea.PositionVerlet()
# timestepper = PEFRL()

ea.integrate(timestepper, pendulum_sim, final_time, total_steps)

if PLOT_VIDEO:

    def plot_video(
        plot_params: dict,
        video_name="video.mp4",
        margin=0.2,
        fps=60,
        step=1,
        *args,
        **kwargs,
    ):  # (time step, x/y/z, node)
        import matplotlib.animation as manimation

        plt.rcParams.update({"font.size": 22})

        # Should give a (n_time, 3, n_elem) array
        positions = np.array(plot_params["position"])

        print("plot video")
        FFMpegWriter = manimation.writers["ffmpeg"]
        metadata = dict(
            title="Movie Test", artist="Matplotlib", comment="Movie support!"
        )
        writer = FFMpegWriter(fps=fps, metadata=metadata)
        dpi = 300
        fig = plt.figure(figsize=(10, 8), frameon=True, dpi=dpi)
        ax = fig.add_subplot(111)
        ax.set_aspect("equal", adjustable="box")
        # plt.axis("square")
        i = 0
        (rod_line,) = ax.plot(positions[i, 2], positions[i, 0], lw=3.0)
        (tip_line,) = ax.plot(positions[:i, 2, -1], positions[:i, 0, -1], "k--")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim([-1.0 - margin, 1.0 + margin])
        ax.set_ylim([-1.0 - margin, 0.0 + margin])
        with writer.saving(fig, video_name, dpi):
            with plt.style.context("seaborn-white"):
                for i in range(0, positions.shape[0], int(step)):
                    rod_line.set_xdata(positions[i, 2])
                    rod_line.set_ydata(positions[i, 0])
                    tip_line.set_xdata(positions[:i, 2, -1])
                    tip_line.set_ydata(positions[:i, 0, -1])
                    writer.grab_frame()

    plot_video(recorded_history, "swinging_flexible_pendulum.mp4")

if PLOT_FIGURE:
    fig = plt.figure(figsize=(10, 8), frameon=True, dpi=150)
    ax = fig.add_subplot(111)
    ax.set_aspect("equal", adjustable="box")
    # Should give a (n_time, 3, n_elem) array
    positions = np.array(recorded_history["position"])
    for i in range(positions.shape[0]):
        ax.plot(positions[i, 2], positions[i, 0], lw=2.0)
    fig.show()
    plt.show()

if SAVE_RESULTS:
    import pickle as pickle

    filename = "flexible_swinging_pendulum.dat"
    with open(filename, "wb") as file:
        pickle.dump(recorded_history, file)
