import numpy as np

# FIXME without appending sys.path make it more generic
import sys

sys.path.append("../../../")
from elastica.wrappers import (
    BaseSystemCollection,
    Forcing,
    CallBacks,
)
from elastica.rigidbody import Sphere
from elastica.timestepper.symplectic_steppers import PositionVerlet
from elastica.timestepper import integrate
from elastica.callback_functions import CallBackBaseClass
from elastica.external_forces import GravityForces
from elastica.interaction import (
    InteractionPlaneRigidBody,
    AnistropicFrictionalPlaneRigidBody,
)
from collections import defaultdict


class SphereBouncingOnPlaneSumulator(BaseSystemCollection, Forcing, CallBacks):
    pass


sphere_bouncing_sim = SphereBouncingOnPlaneSumulator()
sphere_radius = 0.05
sphere_intial_displacement = 1.0
sphere = Sphere([0.0, sphere_intial_displacement, 0.0], sphere_radius, 1000)
sphere_initial_velocity = 1.0
sphere.velocity_collection[..., 0] = [0.0, sphere_initial_velocity, 0.0]
sphere_bouncing_sim.append(sphere)


# Add call backs
class PositionCollector(CallBackBaseClass):
    """
    Call back function for continuum snake
    """

    def __init__(self, step_skip: int, callback_params: dict):
        CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.callback_params = callback_params

    def make_callback(self, system, time, current_step: int):
        if current_step % self.every == 0:
            self.callback_params["time"].append(time)
            # Collect only x
            self.callback_params["position"].append(system.position_collection.copy())
            self.callback_params["com"].append(system.compute_position_center_of_mass())
            return


recorded_sphere_history = defaultdict(list)
sphere_bouncing_sim.collect_diagnostics(sphere).using(
    PositionCollector, step_skip=20, callback_params=recorded_sphere_history,
)

# Add gravitational forces
gravitational_acc = -9.80665
sphere_bouncing_sim.add_forcing_to(sphere).using(
    GravityForces, acc_gravity=np.array([0.0, gravitational_acc, 0.0])
)

# Add a plane with interactino forces
origin_plane = np.zeros((3,))
normal_plane = np.array([0.0, 1.0, 0.0])
slip_velocity_tol = 1e-4
static_mu_array = np.array([0.8, 0.4, 0.4])  # [forward, backward, sideways]
kinetic_mu_array = np.array([0.4, 0.2, 0.2])  # [forward, backward, sideways]
sphere_bouncing_sim.add_forcing_to(sphere).using(
    AnistropicFrictionalPlaneRigidBody,
    k=1e4,
    nu=0,
    plane_origin=origin_plane,
    plane_normal=normal_plane,
    slip_velocity_tol=slip_velocity_tol,
    static_mu_array=static_mu_array,
    kinetic_mu_array=kinetic_mu_array,
)

sphere_bouncing_sim.finalize()

timestepper = PositionVerlet()

final_time = 2.0
dt = 1.0e-3
total_steps = int(final_time / dt)
print("Total steps", total_steps)
integrate(timestepper, sphere_bouncing_sim, final_time, total_steps)

VIS = True
if VIS == True:
    from matplotlib import pyplot as plt
    from matplotlib.patches import Circle

    fig = plt.figure(3, figsize=(8, 5))
    ax = fig.add_subplot(111)

    com = np.array(recorded_sphere_history["com"])
    sim_time = np.array(recorded_sphere_history["time"])
    ax.plot(sim_time, com[:, 1], c="r")

    for sim_time_val, y in zip(sim_time, com[:, 1]):
        circle = Circle((sim_time_val, y), sphere_radius, color="b")
        ax.add_artist(circle)
    ax.hlines(
        origin_plane[1], sim_time[0], sim_time[-1], "k", linestyle="dashed",
    )
    # From newton's second equation of motion
    analytical_max_height = (
        sphere_intial_displacement
        + (sphere_initial_velocity ** 2) * 0.5 / -gravitational_acc
    )
    ax.hlines(
        analytical_max_height, sim_time[0], sim_time[-1], "g", linestyle="dashed",
    )
    ax.set_aspect("equal")
    ax.set_xlabel("t")
    ax.set_ylabel("y")
    plt.show()
