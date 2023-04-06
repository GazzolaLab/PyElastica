import numpy as np
import elastica as ea
from examples.FrictionValidationCases.friction_validation_postprocessing import (
    plot_friction_validation,
)


class RigidCylinderSimulator(
    ea.BaseSystemCollection, ea.Constraints, ea.Forcing, ea.CallBacks
):
    pass


# Options
PLOT_FIGURE = True
SAVE_FIGURE = True
SAVE_RESULTS = False


def rigid_cylinder_rotational_motion_verification(torque=0.0):
    """
    This test case is for validating rotational motion of
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

    cylinder = ea.Cylinder(start, direction, normal, base_length, base_radius, density)

    rigid_cylinder_sim.append(cylinder)

    class PointCoupleToCenter(ea.NoForces):
        """
        Applies torque on rigid body
        """

        def __init__(self, torque, direction=np.array([0.0, 0.0, 0.0])):
            super(PointCoupleToCenter, self).__init__()
            self.torque = (torque * direction).reshape(3, 1)

        def apply_forces(self, system, time: np.float = 0.0):
            system.external_torques += np.einsum(
                "ijk, jk->ik", system.director_collection, self.torque
            )

    # Add point torque on the rod
    rigid_cylinder_sim.add_forcing_to(cylinder).using(
        PointCoupleToCenter, torque=torque, direction=direction
    )

    # Add call backs
    class RigidSphereCallBack(ea.CallBackBaseClass):
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
                self.callback_params["step"].append(current_step)
                self.callback_params["position"].append(
                    system.position_collection.copy()
                )
                self.callback_params["velocity"].append(
                    system.velocity_collection.copy()
                )

            return

    step_skip = 200
    pp_list = ea.defaultdict(list)
    rigid_cylinder_sim.collect_diagnostics(cylinder).using(
        RigidSphereCallBack, step_skip=step_skip, callback_params=pp_list
    )

    rigid_cylinder_sim.finalize()
    timestepper = ea.PositionVerlet()

    final_time = 0.25
    dt = 4.0e-5
    total_steps = int(final_time / dt)
    print("Total steps", total_steps)
    ea.integrate(timestepper, rigid_cylinder_sim, final_time, total_steps)

    # compute translational and rotational energy
    translational_energy = cylinder.compute_translational_energy()
    rotational_energy = cylinder.compute_rotational_energy()
    # compute translational and rotational energy using analytical equations
    mass = np.pi * base_radius ** 2 * base_length * density
    mass_moment_of_inertia = 0.5 * mass * base_radius ** 2

    analytical_translational_energy = 0.0
    analytical_rotational_energy = (
        0.5 * (torque * final_time) ** 2 / mass_moment_of_inertia
    )

    return {
        "rod": cylinder,
        "sweep": torque,
        "translational_energy": translational_energy,
        "rotational_energy": rotational_energy,
        "analytical_translational_energy": analytical_translational_energy,
        "analytical_rotational_energy": analytical_rotational_energy,
    }


if __name__ == "__main__":
    import multiprocessing as mp

    torque = list([float(x) / 10.0 for x in range(0, 110, 5)])

    # across jump
    torque.extend([float(x) * -1.0 / 10.0 for x in range(1, 110, 5)])

    torque.sort()
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(rigid_cylinder_rotational_motion_verification, torque)

    if PLOT_FIGURE:
        filename = "rotational_energy_test_for_cylinder.png"
        plot_friction_validation(results, SAVE_FIGURE, filename)

    if SAVE_RESULTS:
        import pickle

        filename = "rotational_energy_test_for_cylinder.dat"
        file = open(filename, "wb")
        pickle.dump([results], file)
        file.close()
