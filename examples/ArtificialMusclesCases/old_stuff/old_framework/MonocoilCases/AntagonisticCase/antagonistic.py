import numpy as np
import os
from elastica import *
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d import proj3d, Axes3D
from tqdm import tqdm
from examples.ArtificialMusclesCases.CoiledMusclesCases.PremadeCases.MuscleForcing import (
    PointSpring,
)

from elastica._linalg import (
    _batch_cross,
    _batch_norm,
    _batch_dot,
    _batch_matvec,
)
from typing import Dict, Sequence
from numba import njit

from examples.ArtificialMusclesCases.post_processing import (
    plot_video_with_surface,
)

from examples.ArtificialMusclesCases.muscle_fiber_init_symbolic import (
    get_fiber_geometry,
)

from examples.ArtificialMusclesCases.CoiledMusclesCases.PremadeCases.TestingBC import (
    IsometricBC,
    IsometricStrainBC,
)


from elastica.experimental.connection_contact_joint.parallel_connection import (
    get_connection_vector_straight_straight_rod,
)

from examples.ArtificialMusclesCases.connect_straight_rods import ContactSurfaceJoint

from examples.ArtificialMusclesCases.artificial_muscle_actuation import (
    ArtficialMuscleActuation,
    ManualArtficialMuscleActuation,
    ArtficialMuscleActuationDecoupled,
)

from examples.ArtificialMusclesCases.memory_block_connections import (
    MemoryBlockConnections,
)


class MuscleCase(
    BaseSystemCollection,
    Constraints,
    MemoryBlockConnections,
    Forcing,
    CallBacks,
    Damping,
):
    pass


# Add callback functions for plotting position of the rod later on
class RodCallBack(CallBackBaseClass):
    """ """

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
            self.callback_params["directors"].append(system.director_collection.copy())
            self.callback_params["internal_force"].append(system.internal_forces.copy())
            self.callback_params["external_force"].append(system.external_forces.copy())

            return


muscle_sim = MuscleCase()
final_time = 20
untwisting_start_time = 10
time_untwisting = 1
length_scale = 1e-3
mass_scale = 1e-3
divide = 80.63 / 40
base_length = 80.63 * length_scale / divide
n_turns_per_length = 34 / (63 * length_scale)
n_turns = n_turns_per_length * base_length
link_scale = 1
initial_link_per_length = link_scale * 2.4166 / (length_scale)  # turns per unit length
E_scale = 1
n_elem_per_turn = 24  # at least 24 for stable coil beyond 30 seconds
n_elem = n_elem_per_turn * int(n_turns)

room_temperature = 25
E = 1925 * E_scale * mass_scale / length_scale  # E at room temperature
mass = 0.22012 * mass_scale / divide
Thompson_model = False
Contraction = True
Isometric = True
Self_Contact = True
save_data = True
povray_viz = True
current_path = os.getcwd()
save_folder = os.path.join(current_path, "data")
os.makedirs(save_folder, exist_ok=True)

force_mag = 1.6  # default is 0
kappa_change = 0.0557691  # default is 0
# kappa_change =  1.31255#0 is no actuation
print(kappa_change)

youngs_modulus_coefficients = [
    2.26758447119,
    -0.00996645676489,
    0.0000323219668553,
    -3.8696662364 * 1e-7,
    -6.3964732027 * 1e-7,
    2.0149695202 * 1e-8,
    -2.5861167614 * 1e-10,
    1.680136396 * 1e-12,
    -5.4956153529 * 1e-15,
    7.2138065668 * 1e-18,
]  # coefficients of youngs modulus interpolation polynomial


dt = 0.3 * base_length / n_elem
total_steps = int(final_time / dt)
time_step = np.float64(final_time / total_steps)
rendering_fps = 20
step_skip = int(1.0 / (rendering_fps * time_step))

# Rest of the rod parameters and construct rod
base_radius = (0.8 / 2) * length_scale
helix_radius = (2.5 / 2) * length_scale
end_temperature = 150
thermal_expansion_coefficient = 8e-5  # 7e-5 to 9e-5
end_radius = (
    (thermal_expansion_coefficient * (end_temperature - 25)) + 1
) * base_radius
temp = min(
    25 + ((end_radius - base_radius) / (thermal_expansion_coefficient * base_radius)),
    180,
)
print(end_radius / base_radius, temp)

base_area = np.pi * base_radius ** 2
I = np.pi / 4 * base_radius ** 4
volume = base_area * base_length
nu = 3e-3
relaxationNu = 0.0

poisson_ratio = 0.5
shear_modulus = E / (poisson_ratio + 1.0)


direction = np.array([0.0, 0.0, 1.0])
normal = np.array([1.0, 0.0, 0.0])


def cross2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.cross(a, b)


binormal = cross2(direction, normal)
start = np.zeros(
    3,
)

# Helix rod structure
start_position_of_helix = start
direction_helical_rod = direction
binormal_helical_rod = binormal
normal_helical_rod = normal
k_m = n_turns_per_length


fiber_length, start_coil, position_collection, director_collection = get_fiber_geometry(
    n_elem=n_elem,
    start_radius_list=[helix_radius],
    taper_slope_list=[0],
    start_position=start_position_of_helix,
    direction=direction,
    normal=normal,
    offset_list=[np.pi / 2],
    length=base_length,
    turns_per_length_list=[k_m],
    initial_link_per_length=initial_link_per_length,
    CCW_list=[False],
)

volume = base_area * fiber_length
density = mass / volume

muscle_rod1 = CosseratRod.straight_rod(
    n_elem,
    start_coil,
    direction_helical_rod,
    normal_helical_rod,
    fiber_length,
    base_radius,
    density,
    youngs_modulus=E,
    shear_modulus=shear_modulus,
    position=position_collection,
    directors=director_collection,
)

fiber_length, start_coil, position_collection, director_collection = get_fiber_geometry(
    n_elem=n_elem,
    start_radius_list=[helix_radius],
    taper_slope_list=[0],
    start_position=start_position_of_helix + direction * base_length,
    direction=direction,
    normal=normal,
    offset_list=[np.pi / 2],
    length=base_length,
    turns_per_length_list=[k_m],
    initial_link_per_length=initial_link_per_length,
    CCW_list=[False],
)

volume = base_area * fiber_length
density = mass / volume

muscle_rod2 = CosseratRod.straight_rod(
    n_elem,
    start_coil,
    direction_helical_rod,
    normal_helical_rod,
    fiber_length,
    base_radius,
    density,
    youngs_modulus=E,
    shear_modulus=shear_modulus,
    position=position_collection,
    directors=director_collection,
)

post_processing_dict_list = []

muscle_rods = [muscle_rod1, muscle_rod2]
constrain_start_positions = {}
constrain_start_directors = {}

for i in range(2):
    muscle_sim.append(muscle_rods[i])

    # Add damping
    muscle_sim.dampen(muscle_rods[i]).using(
        AnalyticalLinearDamper,
        damping_constant=nu,
        time_step=dt,
    )

    constrain_start_positions[i] = np.zeros_like(muscle_rod1.position_collection)
    constrain_start_directors[i] = np.zeros_like(muscle_rod1.director_collection)

    muscle_sim.constrain(muscle_rods[i]).using(
        GeneralConstraint,
        constrained_position_idx=(-1,),
        constrained_director_idx=(-1,),
        translational_constraint_selector=np.array([True, True, False]),
        rotational_constraint_selector=np.array([True, True, True]),
    )

    muscle_sim.constrain(muscle_rods[i]).using(
        GeneralConstraint,
        constrained_position_idx=(0,),
        constrained_director_idx=(0,),
        translational_constraint_selector=np.array([True, True, False]),
        rotational_constraint_selector=np.array([True, True, True]),
    )

    if Isometric:
        muscle_sim.constrain(muscle_rods[i]).using(
            IsometricStrainBC,
            desired_length=1.14 * base_length,
            direction=direction,
            constrain_start_positions=constrain_start_positions[i],
            constrain_start_directors=constrain_start_directors[i],
            constraint_node_idx=[min(0, 2 - 3 * i)],
            length_node_idx=[0, -1],
        )

    # # Add self contact to prevent penetration
    if Self_Contact:
        for elem in range(n_elem - n_elem_per_turn):
            (
                rod_one_direction_vec_in_material_frame,
                rod_two_direction_vec_in_material_frame,
                offset_btw_rods,
            ) = get_connection_vector_straight_straight_rod(
                muscle_rods[i],
                muscle_rods[i],
                (elem, elem + 1),
                (elem + n_elem_per_turn, elem + n_elem_per_turn + 1),
            )
            muscle_sim.connect(
                first_rod=muscle_rods[i],
                second_rod=muscle_rods[i],
                first_connect_idx=elem,
                second_connect_idx=elem + n_elem_per_turn,
            ).using(
                ContactSurfaceJoint,
                k=0,
                nu=0,
                k_repulsive=muscle_rods[i].shear_matrix[2, 2, elem - 1] * 10,
                rod_one_direction_vec_in_material_frame=rod_one_direction_vec_in_material_frame[
                    ..., 0
                ],
                rod_two_direction_vec_in_material_frame=rod_two_direction_vec_in_material_frame[
                    ..., 0
                ],
                offset_btw_rods=offset_btw_rods[0],
            )

    post_processing_dict_list.append(
        defaultdict(list)
    )  # list which collected data will be append
    # set the diagnostics for rod and collect data
    muscle_sim.collect_diagnostics(muscle_rods[i]).using(
        RodCallBack,
        step_skip=step_skip,
        callback_params=post_processing_dict_list[i],
    )


force_scale = force_mag * 1e-3
end_force = force_scale * np.array([0.0, 0.0, E * base_area])
zero_force = np.array([0.0, 0.0, 0.0])
start_force = force_scale * np.array([0.0, 0.0, -E * base_area])
print("Force: " + str(end_force * 1e3 / length_scale))


# Add point force on the rod 1
muscle_sim.add_forcing_to(muscle_rods[0]).using(
    EndpointForces,
    start_force=start_force,
    end_force=zero_force,
    ramp_up_time=time_step,
)
# Add point force on the rod 2
muscle_sim.add_forcing_to(muscle_rods[1]).using(
    EndpointForces, start_force=zero_force, end_force=end_force, ramp_up_time=time_step
)

point1 = np.array([0.0, 0.0, 0.0])
point2 = np.array([0.0, 0.0, 0.0])


muscle_sim.add_forcing_to(muscle_rod1).using(
    PointSpring,
    k=muscle_rods[0].shear_matrix[2, 2, elem - 1] * 10,
    nu=nu,
    point=point1,
    index=-1,
)
muscle_sim.add_forcing_to(muscle_rod2).using(
    PointSpring,
    k=muscle_rods[1].shear_matrix[2, 2, elem - 1] * 10,
    nu=nu,
    point=point2,
    index=0,
)

start_kappa = np.zeros((3, n_elem - 1))
start_bend_matrix = np.zeros((3, 3, n_elem - 1))
start_shear_matrix = np.zeros((3, 3, n_elem))
start_mass_second_moment_of_inertia = np.zeros((3, 3, n_elem))
start_inv_mass_second_moment_of_inertia = np.zeros((3, 3, n_elem))


start_sigma = np.zeros((3, n_elem))


end_kappa = np.zeros((3, n_elem - 1))
end_sigma = np.zeros((3, n_elem))

# Thompson Model Force and Torque
force_thompson = np.zeros((3, n_elem + 1))
torque_thompson = np.zeros((3, n_elem))

if Contraction:
    # muscle_sim.add_forcing_to(muscle_rod).using(
    #     ArtficialMuscleActuation,
    #     start_density=density,
    #     start_kappa=start_kappa,
    #     start_sigma= start_sigma,
    #     start_radius=base_radius,
    #     start_bend_matrix = start_bend_matrix,
    #     start_shear_matrix = start_shear_matrix,
    #     start_mass_second_moment_of_inertia = start_mass_second_moment_of_inertia,
    #     start_inv_mass_second_moment_of_inertia = start_inv_mass_second_moment_of_inertia,
    #     contraction_time=time_untwisting,
    #     start_time = untwisting_start_time,
    #     kappa_change = kappa_change,
    #     thermal_expansion_coefficient = thermal_expansion_coefficient,
    #     room_temperature = room_temperature,
    #     end_temperature = end_temperature,
    #     youngs_modulus_coefficients = youngs_modulus_coefficients,
    # )
    # muscle_sim.add_forcing_to(muscle_rod).using(
    # ManualArtficialMuscleActuation,
    # start_kappa=start_kappa,
    # contraction_time=time_untwisting,
    # start_time = untwisting_start_time,
    # kappa_change = 0.01,
    # )
    muscle_sim.add_forcing_to(muscle_rods[0]).using(
        ManualArtficialMuscleActuation2,
        start_density=density,
        start_kappa=start_kappa,
        start_sigma=start_sigma,
        start_radius=base_radius,
        start_bend_matrix=start_bend_matrix,
        start_shear_matrix=start_shear_matrix,
        start_mass_second_moment_of_inertia=start_mass_second_moment_of_inertia,
        start_inv_mass_second_moment_of_inertia=start_inv_mass_second_moment_of_inertia,
        contraction_time=time_untwisting,
        start_time=untwisting_start_time,
        kappa_change=kappa_change,
        room_temperature=room_temperature,
        end_temperature=end_temperature,
        youngs_modulus_coefficients=youngs_modulus_coefficients,
        thermal_expansion_coefficient=thermal_expansion_coefficient,
    )


# finalize simulation
muscle_sim.finalize()

for i in range(2):
    start_kappa[:] = muscle_rods[i].kappa[:]
    start_sigma[:] = muscle_rods[i].sigma[:]
    start_shear_matrix[:] = muscle_rods[i].shear_matrix[:]
    start_bend_matrix[:] = muscle_rods[i].bend_matrix[:]
    start_mass_second_moment_of_inertia[:] = muscle_rods[
        i
    ].mass_second_moment_of_inertia[:]
    start_inv_mass_second_moment_of_inertia[:] = muscle_rods[
        i
    ].inv_mass_second_moment_of_inertia[:]
    muscle_rods[i].rest_kappa[:] = muscle_rods[i].kappa[:]
    muscle_rods[i].rest_sigma[:] = muscle_rods[i].sigma[:]


# Run the simulation
time_stepper = PositionVerlet()

do_step, stages_and_updates = extend_stepper_interface(time_stepper, muscle_sim)

dt = np.float64(float(final_time) / total_steps)

time = 0
progress_bar = True
for i in tqdm(range(total_steps), disable=(not progress_bar)):
    try:
        point1[:] = muscle_rod2.position_collection[:, 0] * direction
        point2[:] = muscle_rod1.position_collection[:, n_elem] * direction
        time = do_step(time_stepper, stages_and_updates, muscle_sim, time, dt)
    except RuntimeError:
        print("RuntimeError, Exiting Sim")


# plotting the videos
filename_video = "Antgonistic_mono.mp4"
plot_video_with_surface(
    post_processing_dict_list,
    video_name=filename_video,
    fps=rendering_fps,
    step=1,
    vis3D=True,
    vis2D=True,
    x_limits=[-base_length / 2, base_length / 2],
    y_limits=[-base_length / 2, base_length / 2],
    z_limits=[-1.1 * base_length, 3.1 * base_length],
)

if povray_viz:
    import pickle

    filename = "data/monocoiled_muscle.dat"
    file = open(filename, "wb")
    pickle.dump(post_processing_dict_list, file)
    file.close()


positions = np.array(post_processing_dict_list[0]["position"])
# midpoint = int(positions.shape[0]*untwisting_start_time/final_time)
# print(100*(((positions[midpoint,2,-1]-positions[midpoint,2,0])-(positions[0,2,-1]-positions[0,2,0]))/(positions[0,2,-1]-positions[0,2,0])))

internal_force = np.array(post_processing_dict_list[0]["internal_force"])
print((internal_force[-1, 2, 0] * 1e6) - 2.9)


if save_data:
    # Save data as npz file
    time = np.array(post_processing_dict_list[0]["time"])

    active_muscle_rod_position_history = np.zeros((1, time.shape[0], 3, n_elem + 1))
    active_muscle_rod_radius_history = np.zeros((1, time.shape[0], n_elem))

    passive_muscle_rod_position_history = np.zeros((1, time.shape[0], 3, n_elem + 1))
    passive_muscle_rod_radius_history = np.zeros((1, time.shape[0], n_elem))

    active_muscle_rod_position_history[0, :, :, :] = np.array(
        post_processing_dict_list[0]["position"]
    )
    active_muscle_rod_radius_history[0, :, :] = np.array(
        post_processing_dict_list[0]["radius"]
    )

    passive_muscle_rod_position_history[0, :, :, :] = np.array(
        post_processing_dict_list[1]["position"]
    )
    passive_muscle_rod_radius_history[0, :, :] = np.array(
        post_processing_dict_list[1]["radius"]
    )

    marker_position_history = np.zeros((1, time.shape[0], 3, 2))
    marker_radius_history = 1.1 * helix_radius * np.ones((1, time.shape[0], 1))

    marker_position_history[0, :, :, 0] = np.array(
        (post_processing_dict_list[0]["position"])
    )[:, :, -1] * direction.reshape((1, 3))
    marker_position_history[0, :, :, 1] = np.array(
        (post_processing_dict_list[1]["position"])
    )[:, :, 0] * direction.reshape((1, 3))

    np.savez(
        os.path.join(save_folder, "antagonistic.npz"),
        time=time,
        active_muscle_rod_position_history=active_muscle_rod_position_history,
        active_muscle_rod_radius_history=active_muscle_rod_radius_history,
        passive_muscle_rod_position_history=passive_muscle_rod_position_history,
        passive_muscle_rod_radius_history=passive_muscle_rod_radius_history,
        marker_position_history=marker_position_history,
        marker_radius_history=marker_radius_history,
    )
    # np.savez(
    #     os.path.join(save_folder, "mono_position_tangent_force"+str(1000+int(100*force_mag))[1:]+".npz"),
    #     time=time,
    #     positions=muscle_rods_position_history,
    #     tangents=muscle_rods_tangent_history,
    #     radius=muscle_rods_radius_history,
    #     internal_force = muscle_rods_internal_force_history,
    #     external_force = muscle_rods_external_force_history,
    # )
