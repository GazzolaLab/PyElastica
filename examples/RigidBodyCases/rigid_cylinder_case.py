import numpy as np
import sys

sys.path.append("../../")

import os
from collections import defaultdict
from elastica.wrappers import (
    BaseSystemCollection,
    Constraints,
    Forcing,
    CallBacks,
)
from elastica.rod.rigid_body import RigidBodyCylinder
from elastica.external_forces import GravityForces
from elastica.interaction import AnistropicFrictionalPlaneRigidBody
from elastica.callback_functions import CallBackBaseClass
from elastica.timestepper.symplectic_steppers import PositionVerlet, PEFRL
from elastica.timestepper import integrate


class RigidCylinderSimulator(BaseSystemCollection, Constraints, Forcing, CallBacks):
    pass


rigid_cylinder_sim = RigidCylinderSimulator()


# setting up test params
start = np.zeros((3,))
direction = np.array([0.0, 1.0, 0.0])
normal = np.array([0.0, 0.0, 1.0])
binormal = np.cross(direction, normal)
base_length = 1.0
base_radius = 0.05
base_area = np.pi * base_radius ** 2
density = 1000


rigid_rod = RigidBodyCylinder(
    start, direction, normal, base_length, base_radius, density,
)

rigid_cylinder_sim.append(rigid_rod)

# Add gravitational forces
gravitational_acc = -9.80665
rigid_cylinder_sim.add_forcing_to(rigid_rod).using(
    GravityForces, acc_gravity=np.array([0.0, gravitational_acc, 0.0])
)

# Add friction forces
origin_plane = np.array([0.0, 0.0, 0.0])
normal_plane = np.array([0.0, 1.0, 0.0])
slip_velocity_tol = 1e-8
froude = 0.1
period = 1.0
mu = base_length / (period * period * np.abs(gravitational_acc) * froude)
kinetic_mu_array = np.array([mu, mu, mu])  # [forward, backward, sideways]
static_mu_array = 2 * kinetic_mu_array
rigid_cylinder_sim.add_forcing_to(rigid_rod).using(
    AnistropicFrictionalPlaneRigidBody,
    k=1.0,
    nu=1e-0,
    plane_origin=origin_plane,
    plane_normal=normal_plane,
    slip_velocity_tol=slip_velocity_tol,
    static_mu_array=static_mu_array,
    kinetic_mu_array=kinetic_mu_array,
)


# Add call backs
class RigidCylinderCallBack(CallBackBaseClass):
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
            self.callback_params["step"].append(current_step)
            self.callback_params["position"].append(system.position_collection.copy())
            self.callback_params["velocity"].append(system.velocity_collection.copy())

        return


step_skip = 200
pp_list = defaultdict(list)
rigid_cylinder_sim.collect_diagnostics(rigid_rod).using(
    RigidCylinderCallBack, step_skip=step_skip, callback_params=pp_list,
)

rigid_cylinder_sim.finalize()
timestepper = PositionVerlet()

rigid_rod.velocity_collection[2] += 0.1

final_time = 1.0  # 11.0 + 0.01)
dt = 4.0e-5
total_steps = int(final_time / dt)
print("Total steps", total_steps)
integrate(timestepper, rigid_cylinder_sim, final_time, total_steps)
