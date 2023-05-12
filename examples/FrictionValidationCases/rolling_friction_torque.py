__doc__ = """Rolling friction validation, for detailed explanation refer to Gazzola et. al. R. Soc. 2018
section 4.1.4 and Appendix G """

import numpy as np
import elastica as ea
from examples.FrictionValidationCases.friction_validation_postprocessing import (
    plot_friction_validation,
)


class RollingFrictionTorqueSimulator(
    ea.BaseSystemCollection, ea.Constraints, ea.Forcing
):
    pass


# Options
PLOT_FIGURE = True
SAVE_FIGURE = True
SAVE_RESULTS = True


def simulate_rolling_friction_torque_with(C_s=0.0):

    rolling_friction_torque_sim = RollingFrictionTorqueSimulator()

    # setting up test params
    n_elem = 50
    start = np.zeros((3,))
    direction = np.array([0.0, 0.0, 1.0])
    normal = np.array([0.0, 1.0, 0.0])
    base_length = 1.0
    base_radius = 0.025
    base_area = np.pi * base_radius ** 2
    mass = 1.0
    density = mass / (base_length * base_area)
    E = 1e9
    # For shear modulus of 2E/3
    poisson_ratio = 0.5
    shear_modulus = E / (poisson_ratio + 1.0)

    # Set shear matrix
    shear_matrix = np.repeat(1e4 * np.identity((3))[:, :, np.newaxis], n_elem, axis=2)

    shearable_rod = ea.CosseratRod.straight_rod(
        n_elem,
        start,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        youngs_modulus=E,
        shear_modulus=shear_modulus,
    )

    # TODO: CosseratRod has to be able to take shear matrix as input, we should change it as done below
    shearable_rod.shear_matrix = shear_matrix

    rolling_friction_torque_sim.append(shearable_rod)
    rolling_friction_torque_sim.constrain(shearable_rod).using(ea.FreeBC)

    # Add gravitational forces
    gravitational_acc = -9.80665
    rolling_friction_torque_sim.add_forcing_to(shearable_rod).using(
        ea.GravityForces, acc_gravity=np.array([0.0, gravitational_acc, 0.0])
    )

    # Add Uniform torque on the rod
    rolling_friction_torque_sim.add_forcing_to(shearable_rod).using(
        ea.UniformTorques, torque=1.0 * C_s, direction=direction
    )

    # Add friction forces
    origin_plane = np.array([0.0, -base_radius, 0.0])
    normal_plane = np.array([0.0, 1.0, 0.0])
    slip_velocity_tol = 1e-4
    static_mu_array = np.array([0.4, 0.4, 0.4])  # [forward, backward, sideways]
    kinetic_mu_array = np.array([0.2, 0.2, 0.2])  # [forward, backward, sideways]

    rolling_friction_torque_sim.add_forcing_to(shearable_rod).using(
        ea.AnisotropicFrictionalPlane,
        k=10.0,
        nu=1e-4,
        plane_origin=origin_plane,
        plane_normal=normal_plane,
        slip_velocity_tol=slip_velocity_tol,
        static_mu_array=static_mu_array,
        kinetic_mu_array=kinetic_mu_array,
    )

    rolling_friction_torque_sim.finalize()
    timestepper = ea.PositionVerlet()

    final_time = 1.0
    dt = 1e-6
    total_steps = int(final_time / dt)
    print("Total steps", total_steps)
    ea.integrate(timestepper, rolling_friction_torque_sim, final_time, total_steps)

    # compute translational and rotational energy
    translational_energy = shearable_rod.compute_translational_energy()
    rotational_energy = shearable_rod.compute_rotational_energy()

    # compute translational and rotational energy using analytical equations
    force_slip = static_mu_array[2] * mass * gravitational_acc
    force_noslip = 2.0 * C_s / (3.0 * base_radius)

    mass_moment_of_inertia = 0.5 * mass * base_radius ** 2

    if np.abs(force_noslip) <= np.abs(force_slip):
        analytical_translational_energy = (
            2.0 / mass * (final_time * C_s / (3.0 * base_radius)) ** 2
        )
        analytical_rotational_energy = (
            2.0
            * mass_moment_of_inertia
            * (final_time * C_s / (3.0 * base_radius ** 2 * mass)) ** 2
        )
    else:
        analytical_translational_energy = (
            mass * (kinetic_mu_array[2] * gravitational_acc * final_time) ** 2 / 2.0
        )
        analytical_rotational_energy = (
            (C_s - kinetic_mu_array[2] * mass * np.abs(gravitational_acc) * base_radius)
            ** 2
            * final_time ** 2
            / (2.0 * mass_moment_of_inertia)
        )

    return {
        "rod": shearable_rod,
        "sweep": C_s,
        "translational_energy": translational_energy,
        "rotational_energy": rotational_energy,
        "analytical_translational_energy": analytical_translational_energy,
        "analytical_rotational_energy": analytical_rotational_energy,
    }


if __name__ == "__main__":
    import multiprocessing as mp

    C_s = list([float(x) / 1000.0 for x in range(0, 140, 10)])

    # across jump
    C_s.extend([float(x) / 1000.0 for x in range(140, 190, 10)])

    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(simulate_rolling_friction_torque_with, C_s)

    if PLOT_FIGURE:
        filename = "rolling_friction_torque.png"
        plot_friction_validation(results, SAVE_FIGURE, filename)

    if SAVE_RESULTS:
        import pickle

        filename = "rolling_friction_torque.dat"
        file = open(filename, "wb")
        pickle.dump([results], file)
        file.close()
