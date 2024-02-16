import numpy as np
import os
from elastica import *
from examples.ArtificialMusclesCases.MuscleCallBack import MuscleCallBack
from examples.ArtificialMusclesCases.post_processing import (
    plot_video_with_surface,
)

from examples.ArtificialMusclesCases.muscle_fiber_init_symbolic import (
    get_fiber_geometry,
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

from examples.ArtificialMusclesCases.CoiledMusclesCases.PremadeCases.TestingBC import (
    IsometricBC,
    IsometricStrainBC,
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


muscle_sim = MuscleCase()

final_time = 20
untwisting_start_time = 10
time_untwisting = 1
length_scale = 1e-3
mass_scale = 1e-3
divide = 80.63 / 25
base_length = 80.63 * length_scale / divide
n_turns_per_length = 0.732 / length_scale
n_turns = n_turns_per_length * base_length
link_scale = 1
initial_link_per_length = link_scale * 2.4166 / (length_scale)  # turns per unit length
E_scale = 1
n_elem_per_turn = 12  # at least 24 for stable coil beyond 30 seconds
n_elem = n_elem_per_turn * int(n_turns)

room_temperature = 25
E = 1925 * E_scale * mass_scale / length_scale  # E at room temperature
mass = 0.22012 * mass_scale / divide
Contraction = True
Self_Contact = True
save_data = True
povray_viz = True
current_path = os.getcwd()
save_folder = os.path.join(current_path, "data")
os.makedirs(save_folder, exist_ok=True)

force_mag = 0  # default is 0
kappa_change = 1  # default is 1

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
print(dt)

# Rest of the rod parameters and construct rod
base_radius = (0.74 / 2) * length_scale
helix_radius = (2.12 / 2) * length_scale
end_temperature = 120
thermal_expansion_coeficient = 5.4025e-3  #
end_radius = ((thermal_expansion_coeficient * (end_temperature - 25)) + 1) * base_radius
temp = min(
    25 + ((end_radius - base_radius) / (thermal_expansion_coeficient * base_radius)),
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
print(density)

muscle_rod = CosseratRod.straight_rod(
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

muscle_sim.append(muscle_rod)

# Add damping
muscle_sim.dampen(muscle_rod).using(
    AnalyticalLinearDamper,
    damping_constant=nu,
    time_step=dt,
)


constrain_start_positions = np.zeros_like(muscle_rod.position_collection)
constrain_start_directors = np.zeros_like(muscle_rod.director_collection)


muscle_sim.constrain(muscle_rod).using(
    GeneralConstraint,
    constrained_position_idx=(-1,),
    constrained_director_idx=(-1,),
    translational_constraint_selector=np.array([True, True, False]),
    rotational_constraint_selector=np.array([True, True, True]),
)

muscle_sim.constrain(muscle_rod).using(
    GeneralConstraint,
    constrained_position_idx=(0,),
    constrained_director_idx=(0,),
    translational_constraint_selector=np.array([True, True, False]),
    rotational_constraint_selector=np.array([True, True, True]),
)


# # Add self contact to prevent penetration
if Self_Contact:
    for elem in range(n_elem - n_elem_per_turn):
        (
            rod_one_direction_vec_in_material_frame,
            rod_two_direction_vec_in_material_frame,
            offset_btw_rods,
        ) = get_connection_vector_straight_straight_rod(
            muscle_rod,
            muscle_rod,
            (elem, elem + 1),
            (elem + n_elem_per_turn, elem + n_elem_per_turn + 1),
        )
        muscle_sim.connect(
            first_rod=muscle_rod,
            second_rod=muscle_rod,
            first_connect_idx=elem,
            second_connect_idx=elem + n_elem_per_turn,
        ).using(
            ContactSurfaceJoint,
            k=0,
            nu=0,
            k_repulsive=muscle_rod.shear_matrix[2, 2, elem - 1] * 10,
            rod_one_direction_vec_in_material_frame=rod_one_direction_vec_in_material_frame[
                ..., 0
            ],
            rod_two_direction_vec_in_material_frame=rod_two_direction_vec_in_material_frame[
                ..., 0
            ],
            offset_btw_rods=offset_btw_rods[0],
        )


force_scale = force_mag * 1e-3
end_force = force_scale * np.array([0.0, 0.0, E * base_area])
start_force = force_scale * np.array([0.0, 0.0, -E * base_area])
print("Force: " + str(end_force * 1e3 / length_scale))


# Add point torque on the rod
muscle_sim.add_forcing_to(muscle_rod).using(
    EndpointForces, start_force=start_force, end_force=end_force, ramp_up_time=time_step
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
    #     start_radius=base_radius,
    #     start_density=density,
    #     start_kappa=start_kappa,
    #     start_sigma= start_sigma,
    #     start_bend_matrix = start_bend_matrix,
    #     start_shear_matrix = start_shear_matrix,
    #     start_mass_second_moment_of_inertia = start_mass_second_moment_of_inertia,
    #     start_inv_mass_second_moment_of_inertia = start_inv_mass_second_moment_of_inertia,
    #     contraction_time=time_untwisting,
    #     start_time = untwisting_start_time,
    #     kappa_change = kappa_change,
    #     thermal_expansion_coeficient = thermal_expansion_coeficient,
    #     room_temperature = room_temperature,
    #     end_temperature = end_temperature,
    #     youngs_modulus_coefficients = youngs_modulus_coefficients,
    # )
    muscle_sim.add_forcing_to(muscle_rod).using(
        ManualArtficialMuscleActuation,
        start_kappa=start_kappa,
        contraction_time=time_untwisting,
        start_time=untwisting_start_time,
        kappa_change=0.01,
    )


# muscle_sim.constrain(muscle_rod).using(
# IsometricBC,
# constrain_start_time = untwisting_start_time,
# constrain_start_positions=constrain_start_positions,
# constrain_start_directors = constrain_start_directors,
# constrained_nodes = [0,-1]
# )

muscle_sim.constrain(muscle_rod).using(
    IsometricStrainBC,
    desired_length=base_length,
    constraint_node_idx=[0, -1],
    length_node_idx=[0, -1],
    direction=direction,
    constrain_start_time=untwisting_start_time,
    constrain_start_positions=constrain_start_positions,
    constrain_start_directors=constrain_start_directors,
    constrained_nodes=[0, -1],
)


class ElementwiseForcesAndTorques(NoForces):
    """
    Applies torque on rigid body
    """

    def __init__(self, torques, forces):
        super(ElementwiseForcesAndTorques, self).__init__()
        self.torques = torques
        self.forces = forces

    def apply_forces(self, system, time: np.float64 = 0.0):
        system.external_torques -= self.torques
        system.external_forces -= self.forces


# Add point torque on the rod
muscle_sim.add_forcing_to(muscle_rod).using(
    ElementwiseForcesAndTorques, torques=torque_thompson, forces=force_thompson
)


post_processing_dict = defaultdict(list)  # list which collected data will be append
# set the diagnostics for rod and collect data
muscle_sim.collect_diagnostics(muscle_rod).using(
    MuscleCallBack,
    step_skip=step_skip,
    callback_params=post_processing_dict,
)

# finalize simulation
muscle_sim.finalize()

# store initial properties for actuation
start_kappa[:] = muscle_rod.kappa[:]
start_sigma[:] = muscle_rod.sigma[:]
start_shear_matrix[:] = muscle_rod.shear_matrix[:]
start_bend_matrix[:] = muscle_rod.bend_matrix[:]
start_mass_second_moment_of_inertia[:] = muscle_rod.mass_second_moment_of_inertia[:]
start_inv_mass_second_moment_of_inertia[
    :
] = muscle_rod.inv_mass_second_moment_of_inertia[:]


# fix muscle shape/annealing
muscle_rod.rest_kappa[:] = muscle_rod.kappa[:]
muscle_rod.rest_sigma[:] = muscle_rod.sigma[:]


# Run the simulation
time_stepper = PositionVerlet()
integrate(time_stepper, muscle_sim, final_time, total_steps)
post_processing_dict_list = [post_processing_dict]

internal_force = np.array(post_processing_dict_list[0]["internal_force"])
print(internal_force)
print((internal_force[-1, 2, 0] * 1e6) - 2.06945077386273)


# plotting the videos
filename_video = "monocoiled_muscle.mp4"
plot_video_with_surface(
    post_processing_dict_list,
    video_name=filename_video,
    fps=rendering_fps,
    step=1,
    vis3D=True,
    vis2D=True,
    x_limits=[-base_length / 2, base_length / 2],
    y_limits=[-base_length / 2, base_length / 2],
    z_limits=[-1.1 * base_length, 2.1 * base_length],
)

if povray_viz:
    import pickle

    filename = "data/monocoiled_muscle.dat"
    file = open(filename, "wb")
    pickle.dump(post_processing_dict, file)
    file.close()


if save_data:
    # Save data as npz file
    time = np.array(post_processing_dict_list[0]["time"])

    n_muscle_rod = len([muscle_rod])

    muscle_rods_position_history = np.zeros(
        (n_muscle_rod, time.shape[0], 3, n_elem + 1)
    )
    muscle_rods_radius_history = np.zeros((n_muscle_rod, time.shape[0], n_elem))
    muscle_rods_tangent_history = np.zeros((n_muscle_rod, time.shape[0], 3, n_elem))
    muscle_rods_internal_force_history = np.zeros(
        (n_muscle_rod, time.shape[0], 3, n_elem + 1)
    )
    muscle_rods_external_force_history = np.zeros(
        (n_muscle_rod, time.shape[0], 3, n_elem + 1)
    )

    for i in range(n_muscle_rod):
        muscle_rods_position_history[i, :, :, :] = np.array(
            post_processing_dict_list[i]["position"]
        )
        muscle_rods_radius_history[i, :, :] = np.array(
            post_processing_dict_list[i]["radius"]
        )
        directors = np.array(post_processing_dict_list[i]["directors"])
        muscle_rods_tangent_history[i, :, :, :] = np.array(directors[:, 2, :, :])
        muscle_rods_internal_force_history[i, :, :, :] = np.array(
            post_processing_dict_list[i]["internal_force"]
        )
        muscle_rods_external_force_history[i, :, :, :] = np.array(
            post_processing_dict_list[i]["external_force"]
        )

    np.savez(
        os.path.join(
            save_folder,
            "monocoiled_muscle" + str(100 + int(10 * force_mag))[1:] + ".npz",
        ),
        time=time,
        muscle_rods_position_history=muscle_rods_position_history,
        muscle_rods_radius_history=muscle_rods_radius_history,
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
