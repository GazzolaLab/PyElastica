__doc__ = """Helical buckling convergence study, for detailed explanation refer to Gazzola et. al. R. Soc. 2018
  section 3.4.1 """

import numpy as np
import elastica as ea
from examples.HelicalBucklingCase.helicalbuckling_postprocessing import (
    analytical_solution,
    envelope,
    plot_helicalbuckling,
)
from examples.convergence_functions import plot_convergence, calculate_error_norm


class HelicalBucklingSimulator(
    ea.BaseSystemCollection, ea.Constraints, ea.Forcing, ea.Damping
):
    pass


# Options
PLOT_FIGURE = True
SAVE_FIGURE = True
SAVE_RESULTS = False


def simulate_helicalbucklin_beam_with(
    elements=10, SAVE_FIGURE=False, PLOT_FIGURE=False
):
    helicalbuckling_sim = HelicalBucklingSimulator()

    # setting up test params
    n_elem = elements
    start = np.zeros((3,))
    direction = np.array([0.0, 0.0, 1.0])
    normal = np.array([0.0, 1.0, 0.0])
    base_length = 100.0
    base_radius = 0.35
    base_area = np.pi * base_radius ** 2
    density = 1.0 / (base_area)
    nu = 0.01 / density / base_area
    E = 1e6
    slack = 3
    number_of_rotations = 27
    # For shear modulus of 1e4, nu is 99!
    poisson_ratio = 99
    shear_matrix = np.repeat(1e5 * np.identity((3))[:, :, np.newaxis], n_elem, axis=2)
    temp_bend_matrix = np.zeros((3, 3))
    np.fill_diagonal(temp_bend_matrix, [1.345, 1.345, 0.789])
    bend_matrix = np.repeat(temp_bend_matrix[:, :, np.newaxis], n_elem - 1, axis=2)

    shearable_rod = ea.CosseratRod.straight_rod(
        n_elem,
        start,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        youngs_modulus=E,
    )
    # TODO: CosseratRod has to be able to take shear matrix as input, we should change it as done below

    shearable_rod.shear_matrix[:] = shear_matrix
    shearable_rod.bend_matrix[:] = bend_matrix

    helicalbuckling_sim.append(shearable_rod)
    # add damping
    dl = base_length / n_elem
    dt = 1e-3 * dl
    helicalbuckling_sim.dampen(shearable_rod).using(
        ea.AnalyticalLinearDamper,
        damping_constant=nu,
        time_step=dt,
    )

    helicalbuckling_sim.constrain(shearable_rod).using(
        ea.HelicalBucklingBC,
        constrained_position_idx=(0, -1),
        constrained_director_idx=(0, -1),
        twisting_time=500,
        slack=slack,
        number_of_rotations=number_of_rotations,
    )

    helicalbuckling_sim.finalize()
    timestepper = ea.PositionVerlet()
    shearable_rod.velocity_collection[..., int((n_elem) / 2)] += np.array(
        [0, 1e-6, 0.0]
    )
    # timestepper = PEFRL()

    final_time = 10500
    total_steps = int(final_time / dt)
    print("Total steps", total_steps)
    ea.integrate(timestepper, helicalbuckling_sim, final_time, total_steps)

    # calculate errors and norms
    # Since we need to evaluate analytical solution only on nodes, n_nodes = n_elems+1
    phi_analytical_envelope = analytical_solution(base_length, n_elem + 1)
    phi_computed_envelope = envelope(shearable_rod.position_collection)

    error, l1, l2, linf = calculate_error_norm(
        phi_analytical_envelope[1], phi_computed_envelope[1], n_elem
    )

    if PLOT_FIGURE:
        plot_helicalbuckling(shearable_rod, SAVE_FIGURE)

    return {"rod": shearable_rod, "error": error, "l1": l1, "l2": l2, "linf": linf}


if __name__ == "__main__":
    import multiprocessing as mp

    convergence_elements = list([100, 200, 400, 800])

    # Convergence study
    # for n_elem in [5, 6, 7, 8, 9, 10]
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(simulate_helicalbucklin_beam_with, convergence_elements)

    if PLOT_FIGURE:
        filename = "HelicalBuckling_convergence_test.png"
        plot_convergence(results, SAVE_FIGURE, filename)

    if SAVE_RESULTS:
        import pickle

        filename = "HelicalBuckling_convergence_test_data.dat"
        file = open(filename, "wb")
        pickle.dump([results], file)
        file.close()
