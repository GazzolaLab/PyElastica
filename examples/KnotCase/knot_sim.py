__doc__ = """Knot simulation using a set of targets to guide the rod end to form a knot. A few nodes along the rod are also pulled to intermediate targets to aid in forming the knot."""

import numpy as np
from IPython.display import Video
from knot_forcing import MultiTargetForce, SnapForce
from knot_visualization import plot_video_2D, plot_video3D
import elastica as ea


class StretchingBeamSimulator(
    ea.BaseSystemCollection, 
    ea.Constraints, 
    ea.Forcing, 
    ea.Damping, 
    ea.CallBacks, 
    ea.Contact,
):
    pass

# Options
GENERATE_2D_VIDEO = True
GENERATE_3D_VIDEO = True

stretch_sim = StretchingBeamSimulator()
final_time = 60
dt = 0.0005

# setting up test params
n_elem = 50
start = np.zeros((3,))
direction = np.array([1.0, 0.0, 0.0])
normal = np.array([0.0, 1.0, 0.0])
base_length = 1.5
base_radius = 0.025
base_area = np.pi * base_radius ** 2
density = 1000
youngs_modulus =  2e3 
poisson_ratio = 0.5
shear_modulus = youngs_modulus / (poisson_ratio + 1.0)

stretchable_rod = ea.CosseratRod.straight_rod(
    n_elem,
    start,
    direction,
    normal,
    base_length,
    base_radius,
    density,
    youngs_modulus=youngs_modulus,
    shear_modulus=shear_modulus,
)
stretch_sim.append(stretchable_rod)

#Boundary conditions
stretch_sim.constrain(stretchable_rod).using(ea.FixedConstraint, constrained_position_idx=(0,))

#Endpoint target forces
force_mag = 1.0
targets = np.array([[.85,0.03,.85],
                    [0.0,0.06,1.2],
                    [-0.60,0.09,.70],
                    [-0.4,0.12,-0.4],
                    [0.1,0.15,-0.7],
                    [0.15,-0.5,-0.5],
                    [0.2,-0.3,0.1],
                    [0.2,0.1,0.12],
                    [0.2,0.3,0.12],
                    [0.0,5.0,0.0]])  # Target positions for the end node

stretch_sim.add_forcing_to(stretchable_rod).using(MultiTargetForce, force_mag,ramp_up_time=1,targets=targets)

#Snap forces - certain nodes along the rod are pulled to intermediate targets once in range to aid in forming the knot
snap_force_mag = 2.0
snap_targets = np.array([[0.27,0.0,0.0],
                         [0.3,0.05,0.2],
                         [0.2,0.1,-0.05]])
snap_nodes = np.array([10,17,33])

stretch_sim.add_forcing_to(stretchable_rod).using(SnapForce, snap_force_mag, snap_nodes, snap_targets, ramp_up_time=3, distance=.05, stop_target = targets[-3], stop_distance = 0.04)

#Damping
damping_constant = .4
stretch_sim.dampen(stretchable_rod).using(
    ea.AnalyticalLinearDamper,
    damping_constant=damping_constant,
    time_step=dt,
)

stretch_sim.dampen(stretchable_rod).using(
    ea.LaplaceDissipationFilter,
    filter_order=4
)

#Self contact
stretch_sim.detect_contact_between(stretchable_rod, stretchable_rod).using(
    ea.RodSelfContact, k=5e3, nu=3
)


# Add call backs
class AxialStretchingCallBack(ea.CallBackBaseClass):
    """
    Records the position of the rod
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
            return

recorded_history = ea.defaultdict(list)
stretch_sim.collect_diagnostics(stretchable_rod).using(AxialStretchingCallBack, step_skip=1, callback_params=recorded_history)

# Finalize and run the simulation
stretch_sim.finalize()
timestepper = ea.PositionVerlet()
total_steps = int(final_time / dt)
print("Total steps", total_steps)
ea.integrate(timestepper, stretch_sim, final_time, total_steps)

if GENERATE_2D_VIDEO:
    filename_video = "knot2D.mp4"
    plot_video_2D(recorded_history, video_name=filename_video, margin=0.2, fps=10,targets=targets)
    Video("knot2D.mp4")

if GENERATE_3D_VIDEO:
    filename_video = "knot3D.mp4"
    plot_video3D(recorded_history, video_name=filename_video, margin=0.2, fps=10)
    Video("knot3D.mp4")