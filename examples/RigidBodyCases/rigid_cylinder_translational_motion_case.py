import numpy as np
from elastica import *
from examples.FrictionValidationCases.friction_validation_postprocessing import (
    plot_friction_validation,
)


class RigidCylinderSimulator(BaseSystemCollection, Constraints, Forcing, CallBacks):
    pass


# Options
PLOT_FIGURE = True
SAVE_FIGURE = True
SAVE_RESULTS = False


def rigid_cylinder_translational_motion_verification(force=0.0):
    """
    This test case is for validating translational motion of
    rigid cylinder. Here we are applying point for on the cylinder
    and compare the kinetic energy of the cylinder after T=0.25s
    with the analytical calculation.
    :param force:
    :return:
    """
    rigid_cylinder_sim = RigidCylinderSimulator()

    # setting up test params
    # setting up test params
    start = np.zeros((3,))
    direction = np.array([0.0, 1.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    binormal = np.cross(direction, normal)
    base_length = 1.0
    base_radius = 0.05
    base_area = np.pi * base_radius ** 2
    density = 1000

    cylinder = Cylinder(start, direction, normal, base_length, base_radius, density)

    rigid_cylinder_sim.append(cylinder)

    class PointForceToCenter(NoForces):
        """
        Applies force on rigid body
        """

        def __init__(self, force, direction=np.array([0.0, 0.0, 0.0])):
            super(PointForceToCenter, self).__init__()
            self.force = (force * direction).reshape(3, 1)

        def apply_forces(self, system, time: np.float = 0.0):
            system.external_forces += self.force

    # Add point force on the rod
    rigid_cylinder_sim.add_forcing_to(cylinder).using(
        PointForceToCenter, force=force, direction=normal.reshape(3, 1)
    )

    # Add call backs
    class RigidSphereCallBack(CallBackBaseClass):
        """
        Call back function
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
    rigid_cylinder_sim.collect_diagnostics(cylinder).using(
        RigidSphereCallBack, step_skip=step_skip, callback_params=pp_list
    )

    rigid_cylinder_sim.finalize()
    timestepper = PositionVerlet()

    final_time = 0.25
    dt = 4.0e-5
    total_steps = int(final_time / dt)
    print("Total steps", total_steps)
    integrate(timestepper, rigid_cylinder_sim, final_time, total_steps)

    # compute translational and rotational energy
    translational_energy = cylinder.compute_translational_energy()
    rotational_energy = cylinder.compute_rotational_energy()
    # compute translational and rotational energy using analytical equations
    mass = np.pi * base_radius ** 2 * base_length * density

    analytical_translational_energy = 0.5 * mass * (force / mass * final_time) ** 2
    analytical_rotational_energy = 0.0

    return {
        "rod": cylinder,
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
        results = pool.map(rigid_cylinder_translational_motion_verification, force)

    if PLOT_FIGURE:
        filename = "translational_energy_test_for_cylinder.png"
        plot_friction_validation(results, SAVE_FIGURE, filename)

    if SAVE_RESULTS:
        import pickle

        filename = "translational_energy_test_for_cylinder.dat"
        file = open(filename, "wb")
        pickle.dump([results], file)
        file.close()
