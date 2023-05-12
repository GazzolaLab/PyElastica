__doc__ = """Axial friction validation, for detailed explanation refer to Gazzola et. al. R. Soc. 2018
section 4.1.4 and Appendix G """
import numpy as np
import elastica as ea
from examples.FrictionValidationCases.friction_validation_postprocessing import (
    plot_axial_friction_validation,
)


class AxialFrictionSimulator(ea.BaseSystemCollection, ea.Constraints, ea.Forcing):
    pass


# Options
PLOT_FIGURE = True
SAVE_FIGURE = False
SAVE_RESULTS = False


def simulate_axial_friction_with(force=0.0):

    axial_friction_sim = AxialFrictionSimulator()

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
    E = 1e5
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

    axial_friction_sim.append(shearable_rod)
    axial_friction_sim.constrain(shearable_rod).using(ea.FreeBC)

    # Add gravitational forces
    gravitational_acc = -9.80665
    axial_friction_sim.add_forcing_to(shearable_rod).using(
        ea.GravityForces, acc_gravity=np.array([0.0, gravitational_acc, 0.0])
    )

    # Add Uniform force on the rod
    axial_friction_sim.add_forcing_to(shearable_rod).using(
        ea.UniformForces, force=1.0 * force, direction=direction
    )

    # Add friction forces
    origin_plane = np.array([0.0, -base_radius, 0.0])
    normal_plane = np.array([0.0, 1.0, 0.0])
    slip_velocity_tol = 1e-4
    static_mu_array = np.array([0.8, 0.4, 0.4])  # [forward, backward, sideways]
    kinetic_mu_array = np.array([0.4, 0.2, 0.2])  # [forward, backward, sideways]

    axial_friction_sim.add_forcing_to(shearable_rod).using(
        ea.AnisotropicFrictionalPlane,
        k=10.0,
        nu=1e-4,
        plane_origin=origin_plane,
        plane_normal=normal_plane,
        slip_velocity_tol=slip_velocity_tol,
        static_mu_array=static_mu_array,
        kinetic_mu_array=kinetic_mu_array,
    )

    axial_friction_sim.finalize()
    timestepper = ea.PositionVerlet()

    final_time = 0.25
    dt = 1e-5
    total_steps = int(final_time / dt)
    print("Total steps", total_steps)
    ea.integrate(timestepper, axial_friction_sim, final_time, total_steps)

    # compute translational and rotational energy
    translational_energy = shearable_rod.compute_translational_energy()
    rotational_energy = shearable_rod.compute_rotational_energy()

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
        "rod": shearable_rod,
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
        results = pool.map(simulate_axial_friction_with, force)

    if PLOT_FIGURE:
        filename = "axial_friction.png"
        plot_axial_friction_validation(results, SAVE_FIGURE, filename)

    if SAVE_RESULTS:
        import pickle

        filename = "axial_friction.dat"
        file = open(filename, "wb")
        pickle.dump([results], file)
        file.close()
