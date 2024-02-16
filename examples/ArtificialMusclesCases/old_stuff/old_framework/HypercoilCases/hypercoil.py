import numpy as np
from elastica import *
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d import proj3d, Axes3D
from tqdm import tqdm
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
    plot_snake_velocity,
)

from examples.ArtificialMusclesCases.artificial_muscle_actuation import (
    ArtficialMuscleActuation,
)


from examples.ArtificialMusclesCases.muscle_fiber_init_symbolic import (
    get_fiber_geometry,
)
from elastica.experimental.connection_contact_joint.parallel_connection import *


class MuscleCase(
    BaseSystemCollection, Constraints, Connections, Forcing, CallBacks, Damping
):
    pass


muscle_sim = MuscleCase()


final_time = 1
scale = 100
divide = 63.55 / 20
base_length = 63.55 / (scale * divide)
pitch_divide = 1
n_turns = 0.3462 * base_length * scale / pitch_divide
resolution_factor = 10
n_elem = resolution_factor * 12 * int(n_turns) * pitch_divide


dt = resolution_factor * 0.03 * base_length / (n_elem)
total_steps = int(final_time / dt)
time_step = np.float64(final_time / total_steps)
rendering_fps = 20
step_skip = int(1.0 / (rendering_fps * time_step))

# Rest of the rod parameters and construct rod
base_radius = (0.74 / scale) / 4
end_radius = 1.2 * base_radius
base_area = np.pi * base_radius ** 2
I = np.pi / 4 * base_radius ** 4
nu = 5.0 / 100
relaxationNu = 0.0
E = 422.7e6 / 100000
poisson_ratio = 0.5
shear_modulus = E / (poisson_ratio + 1.0)

initialTotalTwist = 0.0

direction = np.array([0.0, 0.0, 1.0])
normal = np.array([1.0, 0.0, 0.0])
binormal = np.cross(direction, normal)
start = np.zeros(
    3,
)

F_pulling_scalar = 300


# Helix rod structure
start_position_of_helix = start
direction_helical_rod = direction
binormal_helical_rod = binormal
normal_helical_rod = normal
start_director = np.vstack(
    (
        normal,
        binormal,
        direction,
    )
)
# Offset angle changes the start angle of the helix
k_m = n_turns / base_length
k_s = 3 * k_m
k_h = 3 * k_s
k = [k_m, k_s, k_h]
start_kappa = {}
start_sigma = {}
end_kappa = {}
end_sigma = {}
start_bend_matrix = {}
start_shear_matrix = {}
start_mass_second_moment_of_inertia = {}


time_untwisting = 5

end_force = np.array([0.0, 0.0, -0.04])
initial_link_per_length = 0

# create supercoil
muscle_rods = {}
for supercoil in range(3):
    for fiber in range(3):
        start_kappa[(fiber, supercoil)] = np.zeros((3, n_elem - 1))
        start_sigma[(fiber, supercoil)] = np.zeros((3, n_elem))
        end_kappa[(fiber, supercoil)] = np.zeros((3, n_elem - 1))
        end_sigma[(fiber, supercoil)] = np.zeros((3, n_elem))
        start_bend_matrix[(fiber, supercoil)] = np.zeros((3, 3, n_elem - 1))
        start_shear_matrix[(fiber, supercoil)] = np.zeros((3, 3, n_elem))
        start_mass_second_moment_of_inertia[(fiber, supercoil)] = np.zeros(
            (3, 3, n_elem)
        )

        # get element positions and directors
        (
            fiber_length,
            start,
            position_collection,
            director_collection,
        ) = get_fiber_geometry(
            n_elem=n_elem,
            start_radius_list=[
                base_radius * 4 * (4 + 2 * np.sqrt(3)) / 3,
                base_radius * (4 + 2 * np.sqrt(3)) / 3,
                base_radius * 2 / np.sqrt(3),
            ],
            taper_slope_list=[0, 0, 0],
            start_position=start_position_of_helix,
            direction=direction,
            normal=normal,
            binormal=binormal,
            offset_list=[0, 2 * fiber * np.pi / 3, 2 * supercoil * np.pi / 3],
            length=base_length,
            turns_per_length_list=k,
            initial_link_per_length=initial_link_per_length,
            CCW_list=[False, False, False],
        )

        volume = base_area * fiber_length
        mass = 0.43006 * 3 / divide
        density = mass / (9 * volume)
        # create muscle rod
        muscle_rods[(fiber, supercoil)] = CosseratRod.straight_rod(
            n_elem,
            start,
            direction_helical_rod,
            normal_helical_rod,
            fiber_length,
            base_radius,
            density,
            0.0,  # internal damping constant, deprecated in v0.3.0
            E,
            shear_modulus=shear_modulus,
            position=position_collection,
            directors=director_collection,
        )

        # append to sim
        muscle_sim.append(muscle_rods[(fiber, supercoil)])

        # Add damping
        muscle_sim.dampen(muscle_rods[(fiber, supercoil)]).using(
            AnalyticalLinearDamper,
            damping_constant=nu,
            time_step=dt,
        )

        # add boundary constraints

#         #free Z
#         muscle_sim.constrain(muscle_rods[(fiber,supercoil)]).using(
#         GeneralConstraint,
#         constrained_position_idx=(0,),
#         constrained_director_idx=(0,),
#         translational_constraint_selector=np.array([True, True, False]),
#         rotational_constraint_selector=np.array([True, True, True]),)

#         #fixed
#         muscle_sim.constrain(muscle_rods[(fiber,supercoil)]).using(
#         GeneralConstraint,
#         constrained_position_idx=(-1,),
#         constrained_director_idx=(-1,),
#         translational_constraint_selector=np.array([True, True, True]),
#         rotational_constraint_selector=np.array([True, True, True]),)

#         # Add self contact to prevent penetration
#         muscle_sim.connect(muscle_rods[(fiber,supercoil)], muscle_rods[(fiber,supercoil)]).using(SelfContact, k=1e4, nu=10)

#         #Artificial muscle actuation
#         muscle_sim.add_forcing_to(muscle_rods[(fiber,supercoil)]).using(
#         ArtficialMuscleActuation,
#         start_radius=base_radius,
#         start_density=density,
#         end_radius=end_radius,
#         start_kappa=start_kappa[(fiber,supercoil)],
#         start_bend_matrix = start_bend_matrix[(fiber,supercoil)],
#         start_shear_matrix = start_shear_matrix[(fiber,supercoil)],
#         start_mass_second_moment_of_inertia = start_mass_second_moment_of_inertia[(fiber,supercoil)],
#         ramp_up_time=time_untwisting,)


#         # #force at end
#         # muscle_sim.add_forcing_to(muscle_rods[fiber]).using(
#         # EndpointForces, end_force, 0, ramp_up_time=1e-2)

# res = [(a, b) for idx, a in enumerate(range(9)) for b in range(9)[idx + 1:]]

# #Connect the three fibers in each supercoil
# for pair in res:
#     fiber1 = pair[0]%3
#     supercoil1 = (pair[0]- pair[0]%3)/3
#     fiber2 = pair[1]%3
#     supercoil2 = (pair[1]- pair[1]%3)/3

#     rod_one_direction_vec_in_material_frame,rod_two_direction_vec_in_material_frame,offset_btw_rods = get_connection_vector_straight_straight_rod(muscle_rods[(fiber1,supercoil1)], muscle_rods[(fiber2,supercoil2)],(0, n_elem),(0, n_elem))
#     for i in range(n_elem):
#         if abs(offset_btw_rods[i]) > 1e-1/scale:
#             continue
#         muscle_sim.connect(
#             first_rod= muscle_rods[(fiber1,supercoil1)], second_rod=muscle_rods[(fiber2,supercoil2)], first_connect_idx=i, second_connect_idx=i
#         ).using(
#             SurfaceJointSideBySide,
#             k=1e2,
#             nu=1e-5,
#             k_repulsive=1e3,
#             rod_one_direction_vec_in_material_frame=rod_one_direction_vec_in_material_frame[
#                 :, i
#             ],
#             rod_two_direction_vec_in_material_frame=rod_two_direction_vec_in_material_frame[
#                 :, i
#             ],
#             offset_btw_rods=offset_btw_rods[i],
#         )
#         print(pair)


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

for supercoil in range(3):
    for fiber in range(3):

        post_processing_dict_list.append(
            defaultdict(list)
        )  # list which collected data will be append

        # set the diagnostics for rod and collect data
        muscle_sim.collect_diagnostics(muscle_rods[(fiber, supercoil)]).using(
            RodCallBack,
            step_skip=step_skip,
            callback_params=post_processing_dict_list[3 * supercoil + fiber],
        )


# finalize simulation
muscle_sim.finalize()

untwist_ratio = 0.8

for supercoil in range(3):
    for fiber in range(3):
        start_kappa[(fiber, supercoil)][:] = muscle_rods[(fiber, supercoil)].kappa[:]
        start_shear_matrix[(fiber, supercoil)][:] = muscle_rods[
            (fiber, supercoil)
        ].shear_matrix[:]
        start_bend_matrix[(fiber, supercoil)][:] = muscle_rods[
            (fiber, supercoil)
        ].bend_matrix[:]
        start_mass_second_moment_of_inertia[(fiber, supercoil)][:] = muscle_rods[
            (fiber, supercoil)
        ].mass_second_moment_of_inertia[:]
        # start_sigma[(fiber,supercoil)][:] = muscle_rods[(fiber,supercoil)].sigma[:]
        # end_kappa[(fiber,supercoil)][:] = muscle_rods[(fiber,supercoil)].kappa[:]
        # end_sigma[(fiber,supercoil)][:] = muscle_rods[(fiber,supercoil)].sigma[:]
        # end_kappa[(fiber,supercoil)][2,:] = muscle_rods[(fiber,supercoil)].kappa[2,:] - np.sign(k_m)*untwist_ratio*abs(muscle_rods[(fiber,supercoil)].kappa[2,:])
        muscle_rods[(fiber, supercoil)].rest_kappa[:] = muscle_rods[
            (fiber, supercoil)
        ].kappa[:]
        muscle_rods[(fiber, supercoil)].rest_sigma[:] = muscle_rods[
            (fiber, supercoil)
        ].sigma[:]


# Run the simulation
time_stepper = PositionVerlet()
integrate(time_stepper, muscle_sim, final_time, total_steps)


# plotting the videos
filename_video = "muscle_hypercoil.mp4"
plot_video_with_surface(
    post_processing_dict_list,
    video_name=filename_video,
    fps=rendering_fps,
    step=1,
    vis3D=True,
    vis2D=True,
    x_limits=[-base_length / 2, base_length / 2],
    y_limits=[-base_length / 2, base_length / 2],
    z_limits=[0, 1.1 * base_length],
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
        os.path.join(save_folder, "hypercoiled_muscle_symbolic.npz"),
        time=time,
        muscle_rods_position_history=muscle_rods_position_history,
        muscle_rods_radius_history=muscle_rods_radius_history,
    )
