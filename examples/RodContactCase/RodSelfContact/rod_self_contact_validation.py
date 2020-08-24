import sys

sys.path.append("../../../")
from elastica import *
from examples.RodContactCase.post_processing import (
    plot_video_with_surface,
    plot_velocity,
)


class RodSelfContact(
    BaseSystemCollection, Constraints, Connections, Forcing, CallBacks
):
    pass


rod_self_contact = RodSelfContact()

# Simulation parameters
dt = 5e-5
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
nu = 0.001
E = 3e5
poisson_ratio = 0.5

# Rod orientations
start = np.zeros(3,)
inclination = np.deg2rad(0)
direction = np.array([0.0, np.cos(inclination), np.sin(inclination)])
normal = np.array([0.0, -np.sin(inclination), np.cos(inclination)])


# Rod 1
n_elem_rod_one = 50
start_rod_one = start + normal * 0.2

rod_one = CosseratRod.straight_rod(
    n_elem_rod_one,
    start_rod_one,
    direction,
    normal,
    base_length,
    base_radius,
    density,
    nu,
    E,
    poisson_ratio,
)

# rod_one.velocity_collection[:] += 0.05 * -normal.reshape(3,1)

rod_self_contact.append(rod_one)


# Contact between two rods
rod_self_contact.connect(rod_one, rod_one).using(SelfContact, k=1e3, nu=0.001)

# Add call backs
class RodCallBack(CallBackBaseClass):
    """

    """

    def __init__(self, step_skip: int, callback_params: dict):
        CallBackBaseClass.__init__(self)
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


post_processing_dict_rod1 = defaultdict(
    list
)  # list which collected data will be append
# set the diagnostics for rod and collect data
rod_self_contact.collect_diagnostics(rod_one).using(
    RodCallBack, step_skip=step_skip, callback_params=post_processing_dict_rod1,
)


rod_self_contact.finalize()
# Do the simulation

timestepper = PositionVerlet()
integrate(timestepper, rod_self_contact, final_time, total_steps)

# plotting the videos
filename_video = "self_contact.mp4"
plot_video_with_surface(
    [post_processing_dict_rod1],
    video_name=filename_video,
    fps=rendering_fps,
    step=1,
    vis3D=True,
    vis2D=True,
)

# filaname = "parallel_rods_velocity.png"
# plot_velocity(post_processing_dict_rod1, post_processing_dict_rod2, filename=filaname, SAVE_FIGURE=True)
