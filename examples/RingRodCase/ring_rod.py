import numpy as np
import elastica as ea

from examples.RingRodCase.ring_rod_post_processing import plot_video


class RingSimulator(
    ea.BaseSystemCollection, ea.Constraints, ea.Forcing, ea.Damping, ea.CallBacks
):
    pass


ring_sim = RingSimulator()

# Simulation parameters
final_time = 15

# setting up test params
n_elem = 50
ring_center_position = np.zeros((3,))
direction = np.array([0.0, 0.0, 1.0])
normal = np.array([0.0, 1.0, 0.0])
base_length = 0.35
base_radius = base_length * 0.011
density = 1000
E = 2e5
poisson_ratio = 0.5
shear_modulus = E / (poisson_ratio + 1.0)

ring_rod = ea.CosseratRod.ring_rod(
    n_elem,
    ring_center_position,
    direction,
    normal,
    base_length,
    base_radius,
    density,
    youngs_modulus=E,
    shear_modulus=shear_modulus,
)

ring_sim.append(ring_rod)

# Add gravitational forces
gravitational_acc = -9.80665
ring_sim.add_forcing_to(ring_rod).using(
    ea.GravityForces, acc_gravity=np.array([0.0, gravitational_acc, 0.0])
)


# Add constraints
ring_sim.constrain(ring_rod).using(
    ea.OneEndFixedBC, constrained_position_idx=(0,), constrained_director_idx=(0,)
)

# Add damping
damping_constant = 4e-3
time_step = 1e-4
ring_sim.dampen(ring_rod).using(
    ea.AnalyticalLinearDamper,
    damping_constant=damping_constant,
    time_step=time_step,
)

total_steps = int(final_time / time_step)
rendering_fps = 60
step_skip = int(1.0 / (rendering_fps * time_step))

# Add call backs
class RingRodCallBack(ea.CallBackBaseClass):
    """
    Call back function for ring rod
    """

    def __init__(self, step_skip: int, callback_params: dict):
        ea.CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.callback_params = callback_params

    def make_callback(self, system, time, current_step: int):

        if current_step % self.every == 0:

            self.callback_params["time"].append(time)
            self.callback_params["step"].append(current_step)
            self.callback_params["position"].append(system.position_collection.copy())
            self.callback_params["length"].append(system.rest_lengths.copy())
            self.callback_params["radius"].append(system.radius.copy())
            self.callback_params["velocity"].append(system.velocity_collection.copy())
            self.callback_params["avg_velocity"].append(
                system.compute_velocity_center_of_mass()
            )

            self.callback_params["com"].append(system.compute_position_center_of_mass())
            self.callback_params["curvature"].append(system.kappa.copy())

            return


pp_list = ea.defaultdict(list)
ring_sim.collect_diagnostics(ring_rod).using(
    RingRodCallBack, step_skip=step_skip, callback_params=pp_list
)

ring_sim.finalize()

timestepper = ea.PositionVerlet()
ea.integrate(timestepper, ring_sim, final_time, total_steps)


filename_video = "ring_rod.mp4"
plot_video(
    [pp_list],
    video_name=filename_video,
    fps=rendering_fps,
    xlim=(-1, 1),
    ylim=(-1, 1),
)
