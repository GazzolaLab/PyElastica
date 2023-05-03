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
direction = np.array([0.0, np.cos(inclination), np.sin(inclination)])
normal = np.array([0.0, -np.sin(inclination), np.cos(inclination)])

# can be y or z too, meant for testing purposes of rod-body contact in different planes
action_plane_key = "x"

# can be set to True, checks collision at tips of rod
TIP_COLLISION = True
TIP_CHOICE = 1

_roll_key = 0 if action_plane_key == "x" else (1 if action_plane_key == "y" else 2)
if action_plane_key == "x":
    global_rot_mat = np.eye(3)
elif action_plane_key == "y":
    # Rotate +ve 90 about z
    global_rot_mat = np.zeros((3, 3))
    global_rot_mat[0, 1] = -1.0
    global_rot_mat[1, 0] = 1.0
    global_rot_mat[2, 2] = 1.0
else:
    # Rotate -ve 90 abuot y
    global_rot_mat = np.zeros((3, 3))
    global_rot_mat[1, 1] = 1.0
    global_rot_mat[0, 2] = 1.0
    global_rot_mat[2, 0] = 1.0


direction = global_rot_mat @ direction
normal = global_rot_mat @ normal

base_length = 0.5
base_radius = 0.01
base_area = np.pi * base_radius ** 2
density = 1750
E = 3e5
poisson_ratio = 0.5
shear_modulus = E / (1 + poisson_ratio)

cylinder_start = global_rot_mat @ np.array([0.3, 0.0, 0.0])
cylinder_direction = global_rot_mat @ np.array([0.0, 0.0, 1.0])
cylinder_normal = global_rot_mat @ np.array([0.0, 1.0, 0.0])

cylinder_height = 0.4
cylinder_radius = 10.0 * base_radius

# Cylinder surface starts at 0.2
tip_offset = 0.0
if TIP_COLLISION:
    # The random choice decides which tip of the rod intersects with cylinder
    TIP_CHOICE = np.random.choice([1, -1])
    tip_offset = 0.5 * TIP_CHOICE * base_length * np.cos(inclination)

start_rod_1 = np.array(
    [
        0.15,
        -0.5 * base_length * np.cos(inclination) + tip_offset,
        0.5 * cylinder_height - 0.5 * base_length * np.sin(inclination),
    ]
)
start_rod_1 = global_rot_mat @ start_rod_1

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
rod1.velocity_collection[_roll_key, ...] = 0.05
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
        _roll_key=_roll_key,
    )
