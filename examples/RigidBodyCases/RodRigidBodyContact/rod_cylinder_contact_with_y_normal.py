import numpy as np
import elastica as ea
from post_processing import plot_video, plot_cylinder_rod_position


class SingleRodSingleCylinderInteractionSimulator(
    ea.BaseSystemCollection,
    ea.Constraints,
    ea.Connections,
    ea.Forcing,
    ea.CallBacks,
    ea.Damping,
):
    pass


# Options
PLOT_FIGURE = True

single_rod_sim = SingleRodSingleCylinderInteractionSimulator()
# setting up test params
n_elem = 50

inclination = np.deg2rad(30)
direction = np.array([0.0, np.sin(inclination), -np.cos(inclination)])
normal = np.array([0.0, np.cos(inclination), np.sin(inclination)])

base_length = 0.5
base_radius = 0.01
base_area = np.pi * base_radius ** 2
density = 1750
E = 3e5
poisson_ratio = 0.5
shear_modulus = E / (1 + poisson_ratio)

cylinder_start = np.array([0.3, 0.0, 0.0])
cylinder_direction = np.array([0.0, 1.0, 0.0])
cylinder_normal = np.array([1.0, 0.0, 0.0])

cylinder_height = 0.4
cylinder_radius = 10.0 * base_radius

# can be set to True, checks collision at tips of rod
TIP_COLLISION = False

# Cylinder surface starts at 0.2
tip_offset = 0.0

TIP_CHOICE = 1
if TIP_COLLISION:
    # The random choice decides which tip of the rod intersects with cylinder
    TIP_CHOICE = np.random.choice([1, -1])
    tip_offset = 0.5 * TIP_CHOICE * base_length * np.cos(inclination)

start_rod_1 = np.array(
    [
        0.15,
        0.5 * cylinder_height - 0.5 * base_length * np.sin(inclination),
        0.5 * base_length * np.cos(inclination) + tip_offset,
    ]
)
# start_rod_1[2] = cylinder_radius + base_length

rod1 = ea.CosseratRod.straight_rod(
    n_elem,
    start_rod_1,
    direction,
    normal,
    base_length,
    base_radius,
    density,
    youngs_modulus=E,
    shear_modulus=shear_modulus,
)
# Give it an initial push
rod1.velocity_collection[0, ...] = 0.05
single_rod_sim.append(rod1)


cylinder = ea.Cylinder(
    cylinder_start,
    cylinder_direction,
    cylinder_normal,
    cylinder_height,
    cylinder_radius,
    density,
)
single_rod_sim.append(cylinder)

single_rod_sim.connect(rod1, cylinder).using(ea.ExternalContact, 1e2, 0.1)


# Add call backs
class PositionCollector(ea.CallBackBaseClass):
    """
    Call back function for continuum snake
    """

    def __init__(self, step_skip: int, callback_params: dict):
        ea.CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.callback_params = callback_params

    def make_callback(self, system, time, current_step: int):
        if current_step % self.every == 0:
            self.callback_params["time"].append(time)
            # Collect only x
            self.callback_params["position"].append(system.position_collection.copy())
            self.callback_params["com"].append(system.compute_position_center_of_mass())
            return


recorded_rod_history = ea.defaultdict(list)
single_rod_sim.collect_diagnostics(rod1).using(
    PositionCollector, step_skip=200, callback_params=recorded_rod_history
)
recorded_cyl_history = ea.defaultdict(list)
single_rod_sim.collect_diagnostics(cylinder).using(
    PositionCollector, step_skip=200, callback_params=recorded_cyl_history
)

# add damping
damping_constant = 1e-3
dt = 1e-4
single_rod_sim.dampen(rod1).using(
    ea.AnalyticalLinearDamper,
    damping_constant=damping_constant,
    time_step=dt,
)

single_rod_sim.finalize()
timestepper = ea.PositionVerlet()
final_time = 2.0
dl = base_length / n_elem
total_steps = int(final_time / dt)
print("Total steps", total_steps)

ea.integrate(timestepper, single_rod_sim, final_time, total_steps)

if PLOT_FIGURE:
    plot_video(
        recorded_rod_history,
        recorded_cyl_history,
        "cylinder_rod_collision.mp4",
        cylinder_direction=cylinder_direction,
        cylinder_height=cylinder_height,
        cylinder_radius=cylinder_radius,
    )

    plot_cylinder_rod_position(
        recorded_rod_history,
        recorded_cyl_history,
        cylinder_radius=cylinder_radius,
        rod_base_radius=base_radius,
        TIP_COLLISION=TIP_COLLISION,
        TIP_CHOICE=TIP_CHOICE,
    )
