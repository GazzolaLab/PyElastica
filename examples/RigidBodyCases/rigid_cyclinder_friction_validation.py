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
from elastica.rod.rigid_body import RigidBodyCyclinder
from elastica.external_forces import GravityForces, NoForces
from elastica.interaction import AnistropicFrictionalPlaneRigidBody
from elastica.callback_functions import CallBackBaseClass
from elastica.timestepper.symplectic_steppers import PositionVerlet, PEFRL
from elastica.timestepper import integrate
from examples.FrictionValidationCases.friction_validation_postprocessing import (
    plot_axial_friction_validation,
)


class RigidCyclinderSimulator(BaseSystemCollection, Constraints, Forcing, CallBacks):
    pass


# Options
PLOT_FIGURE = True
SAVE_FIGURE = True
SAVE_RESULTS = False


def rigid_cyclinder_friction_validation(force=0.0):
    """
    This test case is for validating friction calculation for rigid body. Here cyclinder direction
    and normal directions are parallel and base of the cylinder is touching the ground. We are validating
    our friction model for different forces.
    :param force:
    :return:
    """
    rigid_cyclinder_sim = RigidCyclinderSimulator()

    # setting up test params
    start = np.zeros((3,))
    direction = np.array([0.0, 1.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    binormal = np.cross(direction, normal)
    base_length = 1.0
    base_radius = 0.025
    base_area = np.pi * base_radius ** 2
    mass = 1.0
    density = mass / (base_length * base_area)

    rigid_rod = RigidBodyCyclinder(
        start, direction, normal, base_length, base_radius, density,
    )

    rigid_cyclinder_sim.append(rigid_rod)

    class PointForceToCenter(NoForces):
        """
        Applies uniform forces to entire rod
        """

        def __init__(self, force, direction=np.array([0.0, 0.0, 0.0])):
            super(PointForceToCenter, self).__init__()
            self.force = (force * direction).reshape(3, 1)

        def apply_forces(self, system, time: np.float = 0.0):
            system.external_forces += self.force

    # Add point force on the rod
    rigid_cyclinder_sim.add_forcing_to(rigid_rod).using(
        PointForceToCenter, force=force, direction=normal
    )

    # Add gravitational forces
    gravitational_acc = -9.80665
    rigid_cyclinder_sim.add_forcing_to(rigid_rod).using(
        GravityForces, acc_gravity=np.array([0.0, gravitational_acc, 0.0])
    )

    # Add friction forces
    origin_plane = np.array([0.0, 0.0, 0.0])
    normal_plane = np.array([0.0, 1.0, 0.0])
    slip_velocity_tol = 1e-4
    static_mu_array = np.array([0.8, 0.4, 0.4])  # [forward, backward, sideways]
    kinetic_mu_array = np.array([0.4, 0.2, 0.2])  # [forward, backward, sideways]
    rigid_cyclinder_sim.add_forcing_to(rigid_rod).using(
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
    class RigidCyclinderCallBack(CallBackBaseClass):
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
                self.callback_params["position"].append(
                    system.position_collection.copy()
                )
                self.callback_params["velocity"].append(
                    system.velocity_collection.copy()
                )

            return

    step_skip = 200
    pp_list = defaultdict(list)
    rigid_cyclinder_sim.collect_diagnostics(rigid_rod).using(
        RigidCyclinderCallBack, step_skip=step_skip, callback_params=pp_list,
    )

    rigid_cyclinder_sim.finalize()
    timestepper = PositionVerlet()

    final_time = 0.25  # 11.0 + 0.01)
    dt = 4.0e-5
    total_steps = int(final_time / dt)
    print("Total steps", total_steps)
    integrate(timestepper, rigid_cyclinder_sim, final_time, total_steps)

    # compute translational and rotational energy
    translational_energy = rigid_rod.compute_translational_energy()
    rotational_energy = 0.0
    # compute translational and rotational energy using analytical equations

    if force >= 0.0:  # forward friction force
        if np.abs(force) <= np.abs(static_mu_array[0] * mass * gravitational_acc):
            analytical_translational_energy = 0.0
        else:
            analytical_translational_energy = (
                final_time ** 2
                / (2.0 * mass)
                * (
                    np.abs(force)
                    - kinetic_mu_array[0] * mass * np.abs(gravitational_acc)
                )
                ** 2
            )

    else:  # backward friction force
        if np.abs(force) <= np.abs(static_mu_array[1] * mass * gravitational_acc):
            analytical_translational_energy = 0.0
        else:
            analytical_translational_energy = (
                final_time ** 2
                / (2.0 * mass)
                * (
                    np.abs(force)
                    - kinetic_mu_array[1] * mass * np.abs(gravitational_acc)
                )
                ** 2
            )

    analytical_rotational_energy = 0.0

    return {
        "rod": rigid_rod,
        "sweep": force,
        "translational_energy": translational_energy,
        "rotational_energy": rotational_energy,
        "analytical_translational_energy": analytical_translational_energy,
        "analytical_rotational_energy": analytical_rotational_energy,
    }


if __name__ == "__main__":
    import multiprocessing as mp

    force = list([float(x) / 10.0 for x in range(0, 110, 5)])

    # across jump
    force.extend([float(x) * -1.0 / 10.0 for x in range(1, 110, 5)])

    force.sort()
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(rigid_cyclinder_friction_validation, force)

    if PLOT_FIGURE:
        filename = "axial_friction.png"
        plot_axial_friction_validation(results, SAVE_FIGURE, filename)

    if SAVE_RESULTS:
        import pickle

        filename = "axial_friction.dat"
        file = open(filename, "wb")
        pickle.dump([results], file)
        file.close()
