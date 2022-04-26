import sys

# FIXME without appending sys.path make it more generic
sys.path.append("../")
sys.path.append("../../")
from elastica import *
from post_processing import (
    plot_video_with_surface,
)


class MagneticBeamSimulator(BaseSystemCollection, Constraints, Forcing, CallBacks):
    pass


magnetic_beam_sim = MagneticBeamSimulator()
# setting up test params
n_rods = 10
base_length = 1.5  # m
spacing_between_rods = base_length  # following Gu2020
n_elem = 50
start_collection = np.zeros((n_rods, 3))
for i in range(n_rods):
    start_collection[i, 0] = i * spacing_between_rods
direction = np.array([0.0, 0.0, 1.0])
normal = np.array([0.0, 1.0, 0.0])
base_radius = 0.15  # m
base_area = np.pi * base_radius**2
volume = base_area * base_length
moment_of_inertia = np.pi / 4 * base_radius**4
density = 2.39e3  # kg/m3
nu = 50
E = 1.85e5  # Pa
shear_modulus = 6.16e4  # Pa

# Parameters are from Gu2020
angular_frequency = np.deg2rad(5.0)  # angular frequency of the rotating magnetic field
magnetic_field_strength = 80e-3  # 80mT
# MBAL2_EI is a non-dimensional number from Wang 2019
MBAL2_EI = (
    3.82e-5 * magnetic_field_strength * 4e-3 / (1.85e5 * np.pi / 4 * (0.4e-3) ** 4)
)  # Magnetization magnitude * B * Length/(EI)
magnetization_density = (
    MBAL2_EI * E * moment_of_inertia / (volume * magnetic_field_strength * base_length)
)
carpet_length = spacing_between_rods * (n_rods - 1)
spatial_magnetiation_wavelength = carpet_length
spatial_magnetisation_phase_diff = np.pi
magnetization_angle = spatial_magnetisation_phase_diff + (
    2 * np.pi * start_collection[..., 0] / spatial_magnetiation_wavelength
)
magnetic_rod_list = []

for i in range(n_rods):
    magnetization_direction = np.array(
        [np.sin(magnetization_angle[i]), 0.0, np.cos(magnetization_angle[i])]
    ).reshape(3, 1) * np.ones((n_elem))
    magnetic_rod = MagneticRod.straight_rod(
        n_elem,
        start_collection[i],
        direction,
        normal,
        base_length,
        base_radius,
        density,
        nu,
        E,
        shear_modulus=shear_modulus,
        magnetization_density=magnetization_density,
        magnetization_direction=magnetization_direction,
    )
    magnetic_beam_sim.append(magnetic_rod)
    magnetic_rod_list.append(magnetic_rod)


# Add boundary conditions, one end of rod is clamped
for i in range(n_rods):
    magnetic_beam_sim.constrain(magnetic_rod_list[i]).using(
        OneEndFixedBC, constrained_position_idx=(0,), constrained_director_idx=(0,)
    )

# Set the constant magnetic field object
magnetic_field_object = SingleModeOscillatingMagneticField(
    magnetic_field_amplitude=magnetic_field_strength * np.array([1, 1e-2, 1]),
    magnetic_field_angular_frequency=np.array(
        [angular_frequency, 0, angular_frequency]
    ),
    magnetic_field_phase_difference=np.array([0, np.pi / 2, np.pi / 2]),
    ramp_interval=0.01,
    start_time=0.0,
    end_time=5e3,
)

# Apply magnetic forces
for i in range(n_rods):
    magnetic_beam_sim.add_forcing_to(magnetic_rod_list[i]).using(
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
            self.callback_params["position"].append(system.position_collection.copy())
            self.callback_params["com"].append(system.compute_position_center_of_mass())
            self.callback_params["radius"].append(system.radius.copy())
            self.callback_params["velocity"].append(system.velocity_collection.copy())
            self.callback_params["tangents"].append(system.tangents.copy())


num_cycles = 3
final_time = num_cycles * 2 * np.pi / angular_frequency
dl = base_length / n_elem
dt = 0.01 * dl
total_steps = int(final_time / dt)
rendering_fps = 10
step_skip = int(1.0 / (rendering_fps * dt))

# Add call back for plotting time history of the rod
rod_post_processing_list = []
for idx, rod in enumerate(magnetic_rod_list):
    rod_post_processing_list.append(defaultdict(list))
    magnetic_beam_sim.collect_diagnostics(rod).using(
        MagneticBeamCallBack,
        step_skip=step_skip,
        callback_params=rod_post_processing_list[idx],
    )


timestepper = PositionVerlet()
magnetic_beam_sim.finalize()
integrate(timestepper, magnetic_beam_sim, final_time, total_steps)

# Plot the magnetic rod time history
plot_video_with_surface(
    rod_post_processing_list,
    fps=rendering_fps,
    step=4,
    x_limits=(-2, carpet_length + 2),
    y_limits=(-2, 2),
    z_limits=(-2, 2),
)
