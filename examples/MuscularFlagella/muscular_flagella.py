__doc__ = """Muscular flagella example from Zhang et. al. Nature Comm 2019 paper."""

import numpy as np
import elastica as ea
from examples.MuscularFlagella.post_processing import (
    plot_video_2D,
    plot_video,
    plot_com_position_vs_time,
    plot_position_vs_time_comparison_cpp,
)
from examples.MuscularFlagella.connection_flagella import (
    MuscularFlagellaConnection,
)
from examples.MuscularFlagella.muscle_forces_flagella import MuscleForces


class MuscularFlagellaSimulator(
    ea.BaseSystemCollection,
    ea.Constraints,
    ea.Connections,
    ea.Forcing,
    ea.CallBacks,
    ea.Damping,
):
    pass


muscular_flagella_sim = MuscularFlagellaSimulator()

# set up the simulation parameters
final_time = 6.5  # s
time_step = 5e-8  # s/step
total_steps = int(final_time / time_step)
rendering_fps = 200
step_skip = int(1.0 / (rendering_fps * time_step))

# setting up the PDMS body parameters
n_elem_body = 18
n_elem_head = 4
density_body = 0.965e-3  # g/mm3
base_length_body = 1.927  # mm
E = 3.86e6  # MPa
poisson_ratio = 0.5
shear_modulus = E / (poisson_ratio + 1.0)

base_radius_head = 0.02  # mm
base_radius_tail = 0.007  # mm
radius = np.ones((n_elem_body))
# First 4 elements are head, rest is tail
radius[:n_elem_head] = base_radius_head
radius[n_elem_head:] = base_radius_tail

start = np.zeros(
    3,
)
start[2] = 0.1
direction = np.array([1.0, 0.0, 0.0])
normal = np.array([0.0, 0.0, 1.0])
binormal = np.cross(direction, normal)
nu_body = 0

flagella_body = ea.CosseratRod.straight_rod(
    n_elem_body,
    start,
    direction,
    normal,
    base_length_body,
    radius,
    density_body,
    youngs_modulus=E,
    shear_modulus=shear_modulus,
)

# In order to match bending stiffness of the tail as given in below reference, recompute and
# change the mass moment of inertia, shear and bending matrices. We are only changing the tail,
# geometric parameters of head is already computed, when flagella_body object is initialized.
# Reference: Aydin, O., Zhang, X., Nuethong, S., Pagan-Diaz, G. J., et al. PNAS (2019).
radius_reference = 0.0053  # mm
area_reference = np.pi * radius_reference ** 2

# Second moment of area for disk cross-section
I0_1 = area_reference * area_reference / (4.0 * np.pi)
I0_2 = I0_1
I0_3 = 2.0 * I0_2
I0 = np.zeros((3, 3))
np.fill_diagonal(I0, np.array([I0_1, I0_2, I0_3]))

rest_lengths = flagella_body.rest_lengths
alpha_c = 4.0 / 3.0
shear_modulus = E / (poisson_ratio + 1.0)

bending_matrix = np.zeros((3, 3, n_elem_body - n_elem_head))

for i in range(n_elem_head, n_elem_body):
    flagella_body.mass_second_moment_of_inertia[..., i] = (
        I0 * density_body * rest_lengths[i]
    )

    flagella_body.inv_mass_second_moment_of_inertia[..., i] = np.linalg.inv(
        flagella_body.mass_second_moment_of_inertia[..., i]
    )

    np.fill_diagonal(
        flagella_body.shear_matrix[..., i],
        [
            alpha_c * shear_modulus * area_reference,
            alpha_c * shear_modulus * area_reference,
            E * area_reference,
        ],
    )

    np.fill_diagonal(
        bending_matrix[..., i - n_elem_head],
        [
            E * I0_1,
            E * I0_2,
            shear_modulus * I0_3,
        ],
    )

flagella_body.bend_matrix[..., n_elem_head:] = (
    bending_matrix[..., 1:] * rest_lengths[n_elem_head + 1 :]
    + bending_matrix[..., :-1] * rest_lengths[n_elem_head:-1]
) / (rest_lengths[n_elem_head + 1 :] + rest_lengths[n_elem_head:-1])


muscular_flagella_sim.append(flagella_body)

# setting up the muscle parameters
n_elem_muscle = 2
density_muscle = 2.6e-4  # g/mm3
base_radius_muscle = 0.01  # mm
base_length_muscle = 0.10756
E_muscle = 0.3e5  # MPa
shear_modulus_muscle = E_muscle / (poisson_ratio + 1.0)
nu_muscle = 1e-6 / density_muscle / (np.pi * base_radius_muscle ** 2)

# Start position of the muscle is the 4th element position of body. Lets use the exact location, because this will
# simplify the connection implementation.
element_pos = 0.5 * (
    flagella_body.position_collection[..., 1:]
    + flagella_body.position_collection[..., :-1]
)
start_muscle = np.array([4.5 * base_length_muscle, 0.0053, 0.1])


flagella_muscle = ea.CosseratRod.straight_rod(
    n_elem_muscle,
    start_muscle,
    direction,
    normal,
    base_length_muscle,
    base_radius_muscle,
    density_muscle,
    youngs_modulus=E_muscle,
    shear_modulus=shear_modulus_muscle,
)

muscular_flagella_sim.append(flagella_muscle)

# add damping
muscular_flagella_sim.dampen(flagella_muscle).using(
    ea.AnalyticalLinearDamper,
    damping_constant=nu_muscle,
    time_step=time_step,
)

# Connect muscle and body
body_connection_idx = (4, 5)
muscle_connection_idx = (0, -1)
k_connection = 3.8e2

muscular_flagella_sim.connect(
    first_rod=flagella_body,
    second_rod=flagella_muscle,
    first_connect_idx=body_connection_idx,
    second_connect_idx=muscle_connection_idx,
).using(
    MuscularFlagellaConnection,
    k=k_connection,
    normal=normal,
)

# Add muscle forces
beat_frequency = 3.6 / 2  # Hz
amplitude = 12  # microN
muscular_flagella_sim.add_forcing_to(flagella_muscle).using(
    MuscleForces,
    amplitude=amplitude,
    frequency=beat_frequency,
)


# Add slender body theory
density_fluid = 1.15e-3  # g/mm3
reynolds_number = 1.8e-2
dynamic_viscosity = 1.2e-3
muscular_flagella_sim.add_forcing_to(flagella_body).using(
    ea.SlenderBodyTheory, dynamic_viscosity=dynamic_viscosity
)

# Add call backs
class MuscularFlagellaCallBack(ea.CallBackBaseClass):
    def __init__(self, step_skip: int, callback_params: dict):
        ea.CallBackBaseClass.__init__(self)
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


post_processing_dict_body = ea.defaultdict(list)
muscular_flagella_sim.collect_diagnostics(flagella_body).using(
    MuscularFlagellaCallBack,
    step_skip=step_skip,
    callback_params=post_processing_dict_body,
)

post_processing_dict_muscle = ea.defaultdict(list)
muscular_flagella_sim.collect_diagnostics(flagella_muscle).using(
    MuscularFlagellaCallBack,
    step_skip=step_skip,
    callback_params=post_processing_dict_muscle,
)


muscular_flagella_sim.finalize()

timestepper = ea.PositionVerlet()
print("Total steps", total_steps)
ea.integrate(timestepper, muscular_flagella_sim, final_time, total_steps)


# Plot the videos
filename_video = "muscular_flagella.mp4"

plot_video(
    [post_processing_dict_body, post_processing_dict_muscle],
    video_name="3d_" + filename_video,
    fps=rendering_fps,
    step=1,
    x_limits=(-2.0, 2.0),
    y_limits=(-0.5, 0.5),
    z_limits=(-0.5, 0.5),
    dpi=100,
)

plot_video_2D(
    [post_processing_dict_body, post_processing_dict_muscle],
    video_name="2d_" + filename_video,
    fps=rendering_fps,
    step=1,
    x_limits=(-2.0, 2.0),
    y_limits=(-0.5, 0.5),
    z_limits=(-0.5, 0.5),
    dpi=100,
)

plot_com_position_vs_time(
    post_processing_dict_body, file_name="muscular_flagella_com_pos_vs_time.png"
)

plot_position_vs_time_comparison_cpp(
    post_processing_dict_body,
    file_name="muscular_flagella_com_pos_vs_time_comparison_with_cpp.png",
)

# Store the data for later use and plotting
import os

save_folder = os.path.join(os.getcwd(), "data")
os.makedirs(save_folder, exist_ok=True)

position_history_body = np.array(post_processing_dict_body["position"])
position_history_muscle = np.array(post_processing_dict_muscle["position"])

com_history_body = np.array(post_processing_dict_body["com"])
com_history_muscle = np.array(post_processing_dict_muscle["com"])

radius_history_body = np.array(post_processing_dict_body["radius"])
radius_history_muscle = np.array(post_processing_dict_muscle["radius"])

velocity_history_body = np.array(post_processing_dict_body["velocity"])
velocity_history_muscle = np.array(post_processing_dict_muscle["velocity"])

tangent_history_body = np.array(post_processing_dict_body["tangents"])
tangent_history_muscle = np.array(post_processing_dict_muscle["tangents"])

time_history = np.array(post_processing_dict_body["time"])

np.savez(
    os.path.join(save_folder, "muscular_flagella_time_history.npz"),
    position_history_body=position_history_body,
    position_history_muscle=position_history_muscle,
    com_history_body=com_history_body,
    com_history_muscle=com_history_muscle,
    radius_history_body=radius_history_body,
    radius_history_muscle=radius_history_muscle,
    velocity_history_body=velocity_history_body,
    velocity_history_muscle=velocity_history_muscle,
    tangent_history_body=tangent_history_body,
    tangent_history_muscle=tangent_history_muscle,
    time_history=time_history,
    body_direction=direction,
    body_normal=normal,
)
