import os
import numpy as np
from elastica import *
import sys


# sys.path.append("../")
# sys.path.append("../../")
# sys.path.append("../../../")
# sys.path.append("../../../../")
# sys.path.append("../../../../../")
# sys.path.append("../../../../../../")


from typing import Dict, Sequence
from numba import njit

from examples.ArtificialMusclesCases.post_processing import (
    plot_video_with_surface,
)

from examples.ArtificialMusclesCases.muscle_fiber_init_symbolic import (
    get_fiber_geometry,
)


from elastica.experimental.connection_contact_joint.parallel_connection import (
    get_connection_vector_straight_straight_rod,
)

from examples.ArtificialMusclesCases.connect_straight_rods import (
    ContactSurfaceJoint,
    SurfaceJointSideBySide,
    ParallelJointInterior,
)

from examples.ArtificialMusclesCases.artificial_muscle_actuation import (
    ArtficialMuscleActuation,
    ManualArtficialMuscleActuation,
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


muscle_sim = MuscleCase()

final_time = 3
untwisting_start_time = 0
time_untwisting = 1
length_scale = 1e-3
mass_scale = 1e-3
base_length = 20.00 * length_scale
n_coils_per_length = 34 / (63 * length_scale)  # coils/mm
n_coils = n_coils_per_length * base_length
link_scale = 1
n_elem_per_coil = 36  # at least 24 for stable coil beyond 30 seconds
n_elem = n_elem_per_coil * int(n_coils)
initial_link_per_length = 387 * link_scale  # turns per unit length


dt = 0.3 * base_length / n_elem
total_steps = int(final_time / dt)
time_step = np.float64(final_time / total_steps)
rendering_fps = 20
step_skip = int(1.0 / (rendering_fps * time_step))

# Rest of the rod parameters and construct rod
base_radius = (0.47 / 2) * length_scale  # (0.47/2)*length_scale
ply_radius = 0.265 * length_scale  # 0.265+0.47/2 = 0.5
helix_radius = (2.5 / 2) * length_scale
end_temperature = 150
# end_radius = 1.02*base_radius
thermal_expansion_coeficient = (
    5.4025e-3  # ((end_radius/base_radius)-1)/(end_temperature-25)
)

base_area = np.pi * base_radius ** 2
I = np.pi / 4 * base_radius ** 4
volume = base_area * base_length
nu = 3e-3
relaxationNu = 0.0
E_scale = 1
poisson_ratio = 0.5


room_temperature = 25
E = 1925 * E_scale * mass_scale / length_scale  # E at room temperature
shear_modulus = E / (poisson_ratio + 1.0)
density = 1090
Thompson_model = False
Contraction = True
Isometric = False
Self_Contact = False
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


direction = np.array([0.0, 0.0, 1.0])
normal = np.array([0.0, 1.0, 0.0])


def cross2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.cross(a, b)


binormal = cross2(direction, normal)
start = np.zeros(
    3,
)
start_position_of_helix = start
direction_helical_rod = direction
binormal_helical_rod = binormal
normal_helical_rod = normal
k_m = n_coils / base_length  # coils per unit height


def helix_length_per_coil(k, r):
    return np.sqrt((2 * np.pi * r * k) ** 2 + 1) / k


supercoils_per_coil_length = 1 / (6 * length_scale)  # 1 coil per 6 mm
# supercoils_per_coil_length = 0./length_scale
k_s = (
    supercoils_per_coil_length * helix_length_per_coil(k_m, helix_radius) * k_m
)  # supercoils per unit height
start_kappa = {}
start_sigma = {}
end_kappa = {}
end_sigma = {}
start_bend_matrix = {}
start_shear_matrix = {}
start_mass_second_moment_of_inertia = {}
start_inv_mass_second_moment_of_inertia = {}
total_number_of_helical_rods = 3


# create supercoil
muscle_rods = {}
for fiber in range(total_number_of_helical_rods):
    start_kappa[fiber] = np.zeros((3, n_elem - 1))
    start_sigma[fiber] = np.zeros((3, n_elem))
    end_kappa[fiber] = np.zeros((3, n_elem - 1))
    end_sigma[fiber] = np.zeros((3, n_elem))
    start_bend_matrix[fiber] = np.zeros((3, 3, n_elem - 1))
    start_shear_matrix[fiber] = np.zeros((3, 3, n_elem))
    start_mass_second_moment_of_inertia[fiber] = np.zeros((3, 3, n_elem))
    start_inv_mass_second_moment_of_inertia[fiber] = np.zeros((3, 3, n_elem))

    # get element positions and directors
    fiber_length, start, position_collection, director_collection = get_fiber_geometry(
        n_elem=n_elem,
        start_radius_list=[helix_radius, ply_radius],
        taper_slope_list=[0, 0],
        start_position=start_position_of_helix,
        direction=direction,
        normal=normal,
        offset_list=[0, 2 * fiber * np.pi / total_number_of_helical_rods],
        length=base_length,
        turns_per_length_list=[k_m, k_s],
        initial_link_per_length=initial_link_per_length,
        CCW_list=[False, False],
    )

    # create muscle rod
    muscle_rods[fiber] = CosseratRod.straight_rod(
        n_elem,
        start,
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

    # append to sim
    muscle_sim.append(muscle_rods[fiber])

    # Add damping
    muscle_sim.dampen(muscle_rods[fiber]).using(
        AnalyticalLinearDamper,
        damping_constant=nu,
        time_step=dt,
    )

    # Artificial muscle actuation
    muscle_sim.add_forcing_to(muscle_rods[fiber]).using(
        ArtficialMuscleActuation,
        start_density=density,
        start_kappa=start_kappa[fiber],
        start_sigma=start_sigma[fiber],
        start_radius=base_radius,
        start_bend_matrix=start_bend_matrix[fiber],
        start_shear_matrix=start_shear_matrix[fiber],
        start_mass_second_moment_of_inertia=start_mass_second_moment_of_inertia[fiber],
        start_inv_mass_second_moment_of_inertia=start_inv_mass_second_moment_of_inertia[
            fiber
        ],
        contraction_time=time_untwisting,
        start_time=untwisting_start_time,
        kappa_change=kappa_change,
        thermal_expansion_coeficient=thermal_expansion_coeficient,
        room_temperature=room_temperature,
        end_temperature=end_temperature,
        youngs_modulus_coefficients=youngs_modulus_coefficients,
    )

    # muscle_sim.add_forcing_to(muscle_rods[fiber]).using(
    # ManualArtficialMuscleActuation,
    # start_kappa=start_kappa[fiber],
    # contraction_time=time_untwisting,
    # start_time = untwisting_start_time,
    # kappa_change = 0.2,
    # )

    muscle_sim.constrain(muscle_rods[fiber]).using(
        GeneralConstraint,
        constrained_position_idx=(-1,),
        constrained_director_idx=(-1,),
        translational_constraint_selector=np.array([True, True, False]),
        rotational_constraint_selector=np.array([True, True, True]),
    )

    muscle_sim.constrain(muscle_rods[fiber]).using(
        GeneralConstraint,
        constrained_position_idx=(0,),
        constrained_director_idx=(0,),
        translational_constraint_selector=np.array([True, True, False]),
        rotational_constraint_selector=np.array([True, True, True]),
    )

    # force at end
    # muscle_sim.add_forcing_to(muscle_rods[fiber]).using(
    # EndpointForces, end_force, 0, ramp_up_time=5)

    # # Add self contact to prevent penetration
    if Self_Contact:
        for elem in range(n_elem - n_elem_per_coil):
            for n_elem_above in range(1, n_elem_per_coil):
                (
                    rod_one_direction_vec_in_material_frame,
                    rod_two_direction_vec_in_material_frame,
                    offset_btw_rods,
                ) = get_connection_vector_straight_straight_rod(
                    muscle_rods[fiber],
                    muscle_rods[fiber],
                    (elem, elem + 1),
                    (elem + n_elem_above, elem + n_elem_above + 1),
                )
                muscle_sim.connect(
                    first_rod=muscle_rods[fiber],
                    second_rod=muscle_rods[fiber],
                    first_connect_idx=elem,
                    second_connect_idx=elem + n_elem_above,
                ).using(
                    ContactSurfaceJoint,
                    k=0,
                    nu=0,
                    k_repulsive=muscle_rods[fiber].shear_matrix[2, 2, elem - 1] * 10,
                    rod_one_direction_vec_in_material_frame=rod_one_direction_vec_in_material_frame[
                        ..., 0
                    ],
                    rod_two_direction_vec_in_material_frame=rod_two_direction_vec_in_material_frame[
                        ..., 0
                    ],
                    offset_btw_rods=offset_btw_rods[0],
                )


res = [
    (a, b)
    for idx, a in enumerate(range(total_number_of_helical_rods))
    for b in range(total_number_of_helical_rods)[idx + 1 :]
]

# #Connect the three rods
for pair in res:
    fiber1 = pair[0]
    fiber2 = pair[1]
    (
        rod_one_direction_vec_in_material_frame,
        rod_two_direction_vec_in_material_frame,
        offset_btw_rods,
    ) = get_connection_vector_straight_straight_rod(
        muscle_rods[fiber1], muscle_rods[fiber2], (0, n_elem), (0, n_elem)
    )
    for i in range(n_elem):
        muscle_sim.connect(
            first_rod=muscle_rods[fiber1],
            second_rod=muscle_rods[fiber2],
            first_connect_idx=i,
            second_connect_idx=i,
        ).using(
            SurfaceJointSideBySide,
            k=muscle_rods[fiber1].shear_matrix[2, 2, i] * 100,
            nu=0,
            k_repulsive=muscle_rods[fiber1].shear_matrix[2, 2, i] * 100,
            rod_one_direction_vec_in_material_frame=rod_one_direction_vec_in_material_frame[
                :, i
            ],
            rod_two_direction_vec_in_material_frame=rod_two_direction_vec_in_material_frame[
                :, i
            ],
            offset_btw_rods=offset_btw_rods[i],
        )


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

            return


post_processing_dict_list = []

for fiber in range(total_number_of_helical_rods):
    post_processing_dict_list.append(
        defaultdict(list)
    )  # list which collected data will be append

    # set the diagnostics for rod and collect data
    muscle_sim.collect_diagnostics(muscle_rods[fiber]).using(
        RodCallBack,
        step_skip=step_skip,
        callback_params=post_processing_dict_list[fiber],
    )


# finalize simulation
muscle_sim.finalize()


for fiber in range(total_number_of_helical_rods):
    start_kappa[fiber][:] = muscle_rods[fiber].kappa[:]
    start_shear_matrix[fiber][:] = muscle_rods[fiber].shear_matrix[:]
    start_bend_matrix[fiber][:] = muscle_rods[fiber].bend_matrix[:]
    start_mass_second_moment_of_inertia[fiber][:] = muscle_rods[
        fiber
    ].mass_second_moment_of_inertia[:]
    muscle_rods[fiber].rest_kappa[:] = muscle_rods[fiber].kappa[:]
    muscle_rods[fiber].rest_sigma[:] = muscle_rods[fiber].sigma[:]


# Run the simulation
time_stepper = PositionVerlet()
integrate(time_stepper, muscle_sim, final_time, total_steps)


# plotting the videos
filename_video = "muscle_supercoil.mp4"
plot_video_with_surface(
    post_processing_dict_list,
    video_name=filename_video,
    fps=rendering_fps,
    step=1,
    vis3D=True,
    vis2D=True,
    # x_limits=[-2*helix_radius, 2*helix_radius],
    # y_limits=[-2*helix_radius,2*helix_radius],
    x_limits=[-0.5 * base_length, 0.5 * base_length],
    y_limits=[-0.5 * base_length, 0.5 * base_length],
    z_limits=[0, base_length],
)


save_data = True
if save_data:
    # Save data as npz file
    import os

    current_path = os.getcwd()
    save_folder = os.path.join(current_path, "data")
    os.makedirs(save_folder, exist_ok=True)
    time = np.array(post_processing_dict_list[0]["time"])

    n_muscle_rod = len(muscle_rods)

    muscle_rods_position_history = np.zeros(
        (n_muscle_rod, time.shape[0], 3, n_elem + 1)
    )
    muscle_rods_radius_history = np.zeros((n_muscle_rod, time.shape[0], n_elem))

    for i in range(n_muscle_rod):
        muscle_rods_position_history[i, :, :, :] = np.array(
            post_processing_dict_list[i]["position"]
        )
        muscle_rods_radius_history[i, :, :] = np.array(
            post_processing_dict_list[i]["radius"]
        )

    np.savez(
        os.path.join(save_folder, "supercoiled_muscle_symbolic.npz"),
        time=time,
        muscle_rods_position_history=muscle_rods_position_history,
        muscle_rods_radius_history=muscle_rods_radius_history,
    )

    if povray_viz:
        import pickle

        filename = "data/supercoiled_muscle.dat"
        file = open(filename, "wb")
        pickle.dump(post_processing_dict_list, file)
        file.close()
