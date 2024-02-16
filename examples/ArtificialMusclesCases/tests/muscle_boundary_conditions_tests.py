import numpy as np
from elastica import *
from TestingBC import IsometricStrainBC

from examples.ArtificialMusclesCases.post_processing import (
    plot_video_with_surface,
)


class TestCase(
    BaseSystemCollection, Constraints, Connections, Forcing, CallBacks, Damping
):
    pass


from examples.ArtificialMusclesCases.muscle_fiber_init_symbolic import (
    get_fiber_geometry,
)


test_sim = TestCase()

final_time = 20
dt = 1e-4
total_steps = int(final_time / dt)
time_step = np.float64(final_time / total_steps)
rendering_fps = 20
step_skip = int(1.0 / (rendering_fps * time_step))
n_elem = 5 * 24
start = np.zeros((3,))
direction = np.array([1.0, 0.0, 0.0])
normal = np.array([0.0, 1.0, 0.0])
base_length = 1
base_radius = base_length / 20
density = 1000
E = 1e3
poisson_ratio = 0.5
shear_modulus = E / (1 + poisson_ratio)
nu = 3e-2

fiber_length, start_coil, position_collection, director_collection = get_fiber_geometry(
    n_elem=n_elem,
    start_radius_list=[base_radius * 2.5],
    taper_slope_list=[0],
    start_position=start,
    direction=direction,
    normal=normal,
    offset_list=[np.pi / 2],
    length=base_length,
    turns_per_length_list=[10],
    initial_link_per_length=0,
    CCW_list=[False],
)


test_rod1 = CosseratRod.straight_rod(
    n_elem,
    start_coil,
    direction,
    normal,
    fiber_length,
    base_radius,
    density,
    youngs_modulus=E,
    shear_modulus=shear_modulus,
    position=position_collection,
    directors=director_collection,
)

test_sim.append(test_rod1)

test_sim.dampen(test_rod1).using(
    AnalyticalLinearDamper,
    damping_constant=nu,
    time_step=dt,
)

constrain_start_positions = np.zeros_like(test_rod1.position_collection)
constrain_start_directors = np.zeros_like(test_rod1.director_collection)


test_sim.constrain(test_rod1).using(
    IsometricStrainBC,
    desired_length=1.5 * base_length,
    direction=direction,
    constrain_start_positions=constrain_start_positions,
    constrain_start_directors=constrain_start_directors,
    length_node_idx=[0, -1],
    constraint_node_idx=[-1],
)

test_sim.constrain(test_rod1).using(
    GeneralConstraint,
    constrained_position_idx=(0,),
    constrained_director_idx=(0,),
    translational_constraint_selector=np.array([True, True, True]),
    rotational_constraint_selector=np.array([True, True, True]),
)


force_scale = 1e-2
base_area = np.pi * base_radius ** 2
end_force = force_scale * E * base_area * direction
zero_force = np.array([0.0, 0.0, 0.0])
start_force = -force_scale * E * base_area * direction
test_sim.add_forcing_to(test_rod1).using(
    EndpointForces, start_force=zero_force, end_force=end_force, ramp_up_time=time_step
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
test_rods = [test_rod1]
total_number_of_rods = 1

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
test_rod1.rest_kappa[:] = test_rod1.kappa[:]
test_rod1.rest_sigma[:] = test_rod1.sigma[:]

# Run the simulation
time_stepper = PositionVerlet()
integrate(time_stepper, test_sim, final_time, total_steps)


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
