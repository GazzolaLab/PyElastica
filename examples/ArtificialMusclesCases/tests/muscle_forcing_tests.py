import numpy as np
from elastica import *
from MuscleForcing import PointSpring

from examples.ArtificialMusclesCases.post_processing import (
    plot_video_with_surface,
)


class TestCase(
    BaseSystemCollection, Constraints, Connections, Forcing, CallBacks, Damping
):
    pass


from tqdm import tqdm

from elastica.timestepper.__init__ import extend_stepper_interface


test_sim = TestCase()

final_time = 10
dt = 1e-4
total_steps = int(final_time / dt)
time_step = np.float64(final_time / total_steps)
rendering_fps = 20
step_skip = int(1.0 / (rendering_fps * time_step))
n_elem = 50
start = np.zeros((3,))
direction = np.array([1.0, 0.0, 0.0])
normal = np.array([0.0, 1.0, 0.0])
base_length = 1
base_radius = base_length / 20
density = 1000
E = 1e5
poisson_ratio = 0.5
shear_modulus = E / (1 + poisson_ratio)
nu = 3e-3

test_rod1 = CosseratRod.straight_rod(
    n_elem,
    start,
    direction,
    normal,
    base_length,
    base_radius,
    density,
    youngs_modulus=E,
    shear_modulus=shear_modulus,
)

test_rod2 = CosseratRod.straight_rod(
    n_elem,
    start + 1.1 * direction + 0.1 * normal,
    direction,
    normal,
    base_length,
    base_radius,
    density,
    youngs_modulus=E,
    shear_modulus=shear_modulus,
)
test_rod3 = CosseratRod.straight_rod(
    n_elem,
    start + 1.1 * direction - 0.1 * normal,
    direction,
    normal,
    base_length,
    base_radius,
    density,
    youngs_modulus=E,
    shear_modulus=shear_modulus,
)

test_sim.append(test_rod1)
test_sim.append(test_rod2)
test_sim.append(test_rod3)


test_sim.dampen(test_rod1).using(
    AnalyticalLinearDamper,
    damping_constant=nu,
    time_step=dt,
)
test_sim.dampen(test_rod2).using(
    AnalyticalLinearDamper,
    damping_constant=nu,
    time_step=dt,
)
test_sim.dampen(test_rod3).using(
    AnalyticalLinearDamper,
    damping_constant=nu,
    time_step=dt,
)


point1 = np.array([0.0, 0.0, 0.0])
point2 = np.array([0.0, 0.0, 0.0])


test_sim.add_forcing_to(test_rod1).using(
    PointSpring,
    k=100,
    nu=100,
    point=point1,
    index=-1,
)
test_sim.add_forcing_to(test_rod2).using(
    PointSpring,
    k=100,
    nu=100,
    point=point2,
    index=0,
)
test_sim.add_forcing_to(test_rod3).using(
    PointSpring,
    k=100,
    nu=100,
    point=point2,
    index=0,
)

test_sim.constrain(test_rod2).using(
    GeneralConstraint,
    constrained_position_idx=(0,),
    constrained_director_idx=(0,),
    translational_constraint_selector=np.array([False, True, True]),
    rotational_constraint_selector=np.array([True, True, True]),
)

test_sim.constrain(test_rod3).using(
    GeneralConstraint,
    constrained_position_idx=(0,),
    constrained_director_idx=(0,),
    translational_constraint_selector=np.array([False, True, True]),
    rotational_constraint_selector=np.array([True, True, True]),
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
            self.callback_params["internal_force"].append(system.internal_forces.copy())
            self.callback_params["external_force"].append(system.external_forces.copy())

            return


post_processing_dict_list = []
test_rods = [test_rod1, test_rod2, test_rod3]
total_number_of_rods = 3

for i in range(total_number_of_rods):
    post_processing_dict_list.append(
        defaultdict(list)
    )  # list which collected data will be append

    # set the diagnostics for rod and collect data
    test_sim.collect_diagnostics(test_rods[i]).using(
        RodCallBack,
        step_skip=step_skip,
        callback_params=post_processing_dict_list[i],
    )


# finalize simulation
test_sim.finalize()
point1[:] = (
    test_rod2.position_collection[:, 0] + test_rod3.position_collection[:, 0]
) / 2
point2[:] = test_rod1.position_collection[:, n_elem]


# Run the simulation
time_stepper = PositionVerlet()

do_step, stages_and_updates = extend_stepper_interface(time_stepper, test_sim)

dt = np.float64(float(final_time) / total_steps)

time = 0
progress_bar = True
for i in tqdm(range(total_steps), disable=(not progress_bar)):
    point1[:] = (
        test_rod2.position_collection[:, 0] + test_rod3.position_collection[:, 0]
    ) / 2
    point2[:] = test_rod1.position_collection[:, n_elem]
    time = do_step(time_stepper, stages_and_updates, test_sim, time, dt)


positions1 = np.array(post_processing_dict_list[0]["position"])
positions2 = np.array(post_processing_dict_list[1]["position"])

print(positions1[:, :, -1])
print(len(post_processing_dict_list))

filename_video = "TestPointSpring.mp4"
plot_video_with_surface(
    post_processing_dict_list,
    video_name=filename_video,
    fps=rendering_fps,
    step=1,
    vis3D=True,
    vis2D=True,
    x_limits=[0, 3 * base_length],
    y_limits=[-0.5 * base_length, 0.5 * base_length],
    z_limits=[-0.5 * base_length, 0.5 * base_length],
)
