import sys
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb

# FIXME without appending sys.path make it more generic
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")
from elastica import *


SAVE_FIGURE = False
PLOT_FIGURE = True


class MagneticBeamSimulator(BaseSystemCollection, Constraints, Forcing, CallBacks):
    pass


magnetic_beam_sim = MagneticBeamSimulator()

# setting up test params
n_elem = 100
start = np.zeros((3,))
direction = np.array([1.0, 0.0, 0.0])
normal = np.array([0.0, 1.0, 0.0])
base_length = 6.0
base_radius = 0.15
base_area = np.pi * base_radius**2
density = 5000
nu = 10.0
E = 1e6
I = np.pi / 4 * base_radius**4
poisson_ratio = 0.5
shear_modulus = E / (2 * poisson_ratio + 1.0)
base_radius = 0.15

# setting up magnetic properties
magnetization_density = 1e5
magnetic_field_angle = 2 * np.pi / 3
magnetic_field = 1e-2

magnetic_rod = MagneticRod.straight_rod(
    n_elem,
    start,
    direction,
    normal,
    base_length,
    base_radius,
    density,
    nu,
    E,
    shear_modulus=shear_modulus,
    magnetization_density=magnetization_density,
    magnetization_direction=np.ones((n_elem)) * direction.reshape(3, 1),
)
magnetic_beam_sim.append(magnetic_rod)

# Add boundary conditions, one end of rod is clamped
magnetic_beam_sim.constrain(magnetic_rod).using(
    OneEndFixedBC, constrained_position_idx=(0,), constrained_director_idx=(0,)
)

# Set the constant magnetic field object
magnetic_field_amplitude = magnetic_field * np.array(
    [np.cos(magnetic_field_angle), np.sin(magnetic_field_angle), 0]
)
magnetic_field_object = ConstantMagneticField(
    magnetic_field_amplitude, ramp_interval=500.0, start_time=0.0, end_time=100000
)

# Apply magnetic forces
magnetic_beam_sim.add_forcing_to(magnetic_rod).using(
    ExternalMagneticFieldForces,
    external_magnetic_field=magnetic_field_object,
)

# Add callbacks
class MagneticBeamCallBack(CallBackBaseClass):
    def __init__(self, step_skip: int, callback_params: dict):
        CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.callback_params = callback_params

    def make_callback(self, system, time, current_step: int):
        if current_step % self.every == 0:
            self.callback_params["time"].append(time)
            self.callback_params["step"].append(current_step)
            self.callback_params["velocity_norm"].append(
                np.linalg.norm(system.velocity_collection)
            )


# Add call back for plotting time history of the rod
post_processing_dict = defaultdict(list)
magnetic_beam_sim.collect_diagnostics(magnetic_rod).using(
    MagneticBeamCallBack, step_skip=500, callback_params=post_processing_dict
)

magnetic_beam_sim.finalize()
timestepper = PositionVerlet()
final_time = 1000.0
dl = base_length / n_elem
dt = 0.01 * dl
total_steps = int(final_time / dt)
integrate(timestepper, magnetic_beam_sim, final_time, total_steps)

if PLOT_FIGURE:
    with plt.style.context("ggplot"):
        fig = plt.figure(figsize=(10, 8), frameon=True, dpi=150)
        ax = fig.add_subplot(111)
        ax.plot(
            magnetic_rod.position_collection[0, ...],
            magnetic_rod.position_collection[1, ...],
            lw=2,
            c=to_rgb("xkcd:bluish"),
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        fig.show()

        fig = plt.figure(figsize=(10, 8), frameon=True, dpi=150)
        ax = fig.add_subplot(111)
        ax.semilogy(
            np.asarray(post_processing_dict["time"]),
            np.asarray(post_processing_dict["velocity_norm"]),
            lw=2,
            c=to_rgb("xkcd:bluish"),
        )
        ax.set_xlabel("t")
        ax.set_ylabel("|v|")
        fig.show()

        plt.show() # block
    if SAVE_FIGURE:
        fig.savefig("Magnetic_beam_profile: N=" + str(magnetic_rod.n_elems) + ".png")
