import numpy as np
import elastica as ea
from examples.RodContactCase.post_processing import (
    plot_video_with_surface,
    plot_velocity,
)


class ParallelRodRodContact(
    ea.BaseSystemCollection,
    ea.Constraints,
    ea.Connections,
    ea.Forcing,
    ea.Damping,
    ea.CallBacks,
):
    pass


parallel_rod_rod_contact_sim = ParallelRodRodContact()

# Simulation parameters
dt = 5e-4
final_time = 10
total_steps = int(final_time / dt)
time_step = np.float64(final_time / total_steps)
rendering_fps = 20
step_skip = int(1.0 / (rendering_fps * time_step))

# Rod parameters
base_length = 0.5
base_radius = 0.01
base_area = np.pi * base_radius ** 2
density = 1750
nu = 0.0
E = 3e5
poisson_ratio = 0.5
shear_modulus = E / (poisson_ratio + 1.0)

# Rod orientations
start = np.zeros(
    3,
)
inclination = np.deg2rad(0)
direction = np.array([0.0, np.cos(inclination), np.sin(inclination)])
normal = np.array([0.0, -np.sin(inclination), np.cos(inclination)])


# Rod 1
n_elem_rod_one = 50
start_rod_one = start + normal * 0.2

rod_one = ea.CosseratRod.straight_rod(
    n_elem_rod_one,
    start_rod_one,
    direction,
    normal,
    base_length,
    base_radius,
    density,
    youngs_modulus=E,
    shear_modulus=shear_modulus,
)

rod_one.velocity_collection[:] += 0.05 * -normal.reshape(3, 1)

parallel_rod_rod_contact_sim.append(rod_one)

# Rod 2
n_elem_rod_two = 50

start_rod_two = start

rod_two = ea.CosseratRod.straight_rod(
    n_elem_rod_two,
    start_rod_two,
    direction,
    normal,
    base_length,
    base_radius,
    density,
    youngs_modulus=E,
    shear_modulus=shear_modulus,
)

parallel_rod_rod_contact_sim.append(rod_two)

# Contact between two rods
parallel_rod_rod_contact_sim.connect(rod_one, rod_two).using(
    ea.ExternalContact, k=1e3, nu=0.001
)

# add damping
damping_constant = 2e-4
parallel_rod_rod_contact_sim.dampen(rod_one).using(
    ea.AnalyticalLinearDamper,
    damping_constant=damping_constant,
    time_step=dt,
)
parallel_rod_rod_contact_sim.dampen(rod_two).using(
    ea.AnalyticalLinearDamper,
    damping_constant=damping_constant,
    time_step=dt,
)

# Add call backs
class RodCallBack(ea.CallBackBaseClass):
    """ """

    def __init__(self, step_skip: int, callback_params: dict):
        ea.CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.callback_params = callback_params

    def make_callback(self, system, time, current_step: int):
        if current_step % self.every == 0:
            self.callback_params["time"].append(time)
            self.callback_params["step"].append(current_step)
            self.callback_params["position"].append(system.position_collection.copy())
            self.callback_params["radius"].append(system.radius.copy())
            self.callback_params["com"].append(system.compute_position_center_of_mass())
            self.callback_params["com_velocity"].append(
                system.compute_velocity_center_of_mass()
            )

            total_energy = (
                system.compute_translational_energy()
                + system.compute_rotational_energy()
                + system.compute_bending_energy()
                + system.compute_shear_energy()
            )
            self.callback_params["total_energy"].append(total_energy)

            return


post_processing_dict_rod1 = ea.defaultdict(
    list
)  # list which collected data will be append
# set the diagnostics for rod and collect data
parallel_rod_rod_contact_sim.collect_diagnostics(rod_one).using(
    RodCallBack,
    step_skip=step_skip,
    callback_params=post_processing_dict_rod1,
)

post_processing_dict_rod2 = ea.defaultdict(
    list
)  # list which collected data will be append
# set the diagnostics for rod and collect data
parallel_rod_rod_contact_sim.collect_diagnostics(rod_two).using(
    RodCallBack,
    step_skip=step_skip,
    callback_params=post_processing_dict_rod2,
)

parallel_rod_rod_contact_sim.finalize()
# Do the simulation

timestepper = ea.PositionVerlet()
ea.integrate(timestepper, parallel_rod_rod_contact_sim, final_time, total_steps)

# plotting the videos
filename_video = "parallel_rods_contact.mp4"
plot_video_with_surface(
    [post_processing_dict_rod1, post_processing_dict_rod2],
    video_name=filename_video,
    fps=rendering_fps,
    step=1,
    vis3D=True,
    vis2D=True,
)

filaname = "parallel_rods_velocity.png"
plot_velocity(
    post_processing_dict_rod1,
    post_processing_dict_rod2,
    filename=filaname,
    SAVE_FIGURE=True,
)
