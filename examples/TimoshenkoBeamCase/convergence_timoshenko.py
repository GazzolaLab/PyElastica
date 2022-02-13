__doc__ = """Timoshenko beam convergence study, for detailed explanation refer to 
Gazzola et. al. R. Soc. 2018  section 3.4.3 """

import numpy as np
import sys

# FIXME without appending sys.path make it more generic
sys.path.append("../../")
from elastica import *
from examples.TimoshenkoBeamCase.timoshenko_postprocessing import (
    plot_timoshenko,
    analytical_shearable,
)
from examples.convergence_functions import calculate_error_norm, plot_convergence


class TimoshenkoBeamSimulator(BaseSystemCollection, Constraints, Forcing):
    pass


# Options
PLOT_FIGURE = True
SAVE_FIGURE = False
SAVE_RESULTS = False
ADD_UNSHEARABLE_ROD = False


def simulate_timoshenko_beam_with(
    elements=10, PLOT_FIGURE=False, ADD_UNSHEARABLE_ROD=False
):
    timoshenko_sim = TimoshenkoBeamSimulator()
    final_time = 5000
    # setting up test params
    n_elem = elements
    start = np.zeros((3,))
    direction = np.array([0.0, 0.0, 1.0])
    normal = np.array([0.0, 1.0, 0.0])
    base_length = 3.0
    base_radius = 0.25
    density = 5000
    nu = 0.1
    E = 1e6
    # For shear modulus of 1e4, nu is 99!
    poisson_ratio = 99

    shearable_rod = CosseratRod.straight_rod(
        n_elem,
        start,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        nu,
        E,
        poisson_ratio,
    )

    timoshenko_sim.append(shearable_rod)
    timoshenko_sim.constrain(shearable_rod).using(
        OneEndFixedBC, constrained_position_idx=(0,), constrained_director_idx=(0,)
    )
    end_force = np.array([-15.0, 0.0, 0.0])
    timoshenko_sim.add_forcing_to(shearable_rod).using(
        EndpointForces, 0.0 * end_force, end_force, ramp_up_time=final_time / 2
    )

    if ADD_UNSHEARABLE_ROD:
        # Start into the plane
        unshearable_start = np.array([0.0, -1.0, 0.0])
        unshearable_rod = CosseratRod.straight_rod(
            n_elem,
            unshearable_start,
            direction,
            normal,
            base_length,
            base_radius,
            density,
            nu,
            E,
            # Unshearable rod needs G -> inf, which is achievable with -ve poisson ratio
            poisson_ratio=-0.7,
        )

        timoshenko_sim.append(unshearable_rod)
        timoshenko_sim.constrain(unshearable_rod).using(
            OneEndFixedBC, constrained_position_idx=(0,), constrained_director_idx=(0,)
        )
        timoshenko_sim.add_forcing_to(unshearable_rod).using(
            EndpointForces, 0.0 * end_force, end_force, ramp_up_time=final_time / 2
        )

    timoshenko_sim.finalize()
    timestepper = PositionVerlet()
    # timestepper = PEFRL()

    dl = base_length / n_elem
    dt = 0.01 * dl
    total_steps = int(final_time / dt)
    print("Total steps", total_steps)
    integrate(timestepper, timoshenko_sim, final_time, total_steps)

    if PLOT_FIGURE:
        plot_timoshenko(shearable_rod, end_force, SAVE_FIGURE, ADD_UNSHEARABLE_ROD)

    # calculate errors and norms
    # Since we need to evaluate analytical solution only on nodes, n_nodes = n_elems+1
    error, l1, l2, linf = calculate_error_norm(
        analytical_shearable(shearable_rod, end_force, n_elem + 1)[1],
        shearable_rod.position_collection[0, ...],
        n_elem,
    )
    return {"rod": shearable_rod, "error": error, "l1": l1, "l2": l2, "linf": linf}


if __name__ == "__main__":
    import multiprocessing as mp

    # 5, 6, ... 9
    convergence_elements = list(range(5, 10))
    # 10, 20, ... , 100
    convergence_elements.extend([10 * x for x in range(1, 11)])
    convergence_elements.extend([200])

    # Convergence study
    # for n_elem in [5, 6, 7, 8, 9, 10]
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(simulate_timoshenko_beam_with, convergence_elements)

    if PLOT_FIGURE:
        filename = "Timoshenko_convergence_test.png"
        plot_convergence(results, SAVE_FIGURE, filename)

    if SAVE_RESULTS:
        import pickle

        filename = "Timoshenko_convergence_test_data.dat"
        file = open(filename, "wb")
        pickle.dump([results], file)
        file.close()
