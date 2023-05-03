__doc__ = """Rolling friction validation, for detailed explanation refer to Gazzola et. al. R. Soc. 2018
section 4.1.4 and Appendix G """

import numpy as np
import elastica as ea
from examples.FrictionValidationCases.friction_validation_postprocessing import (
    plot_friction_validation,
)


class RollingFrictionOnInclinedPlaneSimulator(
    ea.BaseSystemCollection, ea.Constraints, ea.Forcing
):
    pass


# Options
PLOT_FIGURE = True
SAVE_FIGURE = True
SAVE_RESULTS = True


def simulate_rolling_friction_on_inclined_plane_with(alpha_s=0.0):

    rolling_friction_on_inclined_plane_sim = RollingFrictionOnInclinedPlaneSimulator()

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

    rolling_friction_on_inclined_plane_sim.append(shearable_rod)
    rolling_friction_on_inclined_plane_sim.constrain(shearable_rod).using(ea.FreeBC)

    gravitational_acc = -9.80665
    rolling_friction_on_inclined_plane_sim.add_forcing_to(shearable_rod).using(
        ea.GravityForces, acc_gravity=np.array([0.0, gravitational_acc, 0.0])
    )

    alpha = alpha_s * np.pi
    origin_plane = np.array(
        [-base_radius * np.sin(alpha), -base_radius * np.cos(alpha), 0.0]
    )
    normal_plane = np.array([np.sin(alpha), np.cos(alpha), 0.0])
    normal_plane = normal_plane / np.sqrt(np.dot(normal_plane, normal_plane))
    slip_velocity_tol = 1e-4
    static_mu_array = np.array([0.4, 0.4, 0.4])  # [forward, backward, sideways]
    kinetic_mu_array = np.array([0.2, 0.2, 0.2])  # [forward, backward, sideways]

    rolling_friction_on_inclined_plane_sim.add_forcing_to(shearable_rod).using(
        ea.AnisotropicFrictionalPlane,
        k=10.0,
        nu=1e-4,
        plane_origin=origin_plane,
        plane_normal=normal_plane,
        slip_velocity_tol=slip_velocity_tol,
        static_mu_array=static_mu_array,
        kinetic_mu_array=kinetic_mu_array,
    )

    rolling_friction_on_inclined_plane_sim.finalize()
    timestepper = ea.PositionVerlet()

    final_time = 0.5
    dt = 1e-6
    total_steps = int(final_time / dt)
    print("Total steps", total_steps)
    ea.integrate(
        timestepper, rolling_friction_on_inclined_plane_sim, final_time, total_steps
    )

    # compute translational and rotational energy
    translational_energy = shearable_rod.compute_translational_energy()
    rotational_energy = shearable_rod.compute_rotational_energy()

    # compute translational and rotational energy using analytical equations
    force_slip = static_mu_array[0] * mass * gravitational_acc * np.cos(alpha)
    force_noslip = -mass * gravitational_acc * np.sin(alpha) / 3.0

    mass_moment_of_inertia = 0.5 * mass * base_radius ** 2

    if np.abs(force_noslip) <= np.abs(force_slip):
        analytical_translational_energy = (
            2.0 * mass * (gravitational_acc * final_time * np.sin(alpha)) ** 2 / 9.0
        )
        analytical_rotational_energy = (
            2.0
            * mass_moment_of_inertia
            * (gravitational_acc * final_time * np.sin(alpha) / (3 * base_radius)) ** 2
        )
    else:
        analytical_translational_energy = (
            mass
            * (
                gravitational_acc
                * final_time
                * (np.sin(alpha) - kinetic_mu_array[0] * np.cos(alpha))
            )
            ** 2
            / 2.0
        )
        analytical_rotational_energy = (
            kinetic_mu_array[0]
            * mass
            * gravitational_acc
            * base_radius
            * final_time
            * np.cos(alpha)
        ) ** 2 / (2.0 * mass_moment_of_inertia)

    return {
        "rod": shearable_rod,
        "sweep": alpha_s,
        "translational_energy": translational_energy,
        "rotational_energy": rotational_energy,
        "analytical_translational_energy": analytical_translational_energy,
        "analytical_rotational_energy": analytical_rotational_energy,
    }


if __name__ == "__main__":
    import multiprocessing as mp

    # 0.05, 0.1, 0.2, 0.25
    alpha_s = list([float(x) / 100.0 for x in range(5, 26, 5)])

    # across jump 0.26 0.29
    alpha_s.extend([float(x) / 100.0 for x in range(26, 30)])

    # 0.3, 0.35, ..., 0.5
    alpha_s.extend([float(x) / 100.0 for x in range(30, 51, 5)])

    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(simulate_rolling_friction_on_inclined_plane_with, alpha_s)

    if PLOT_FIGURE:
        filename = "rolling_friction_on_inclined_plane.png"
        plot_friction_validation(results, SAVE_FIGURE, filename)

    if SAVE_RESULTS:
        import pickle

        filename = "rolling_friction_on_inclined_plane.dat"
        file = open(filename, "wb")
        pickle.dump([results], file)
        file.close()
