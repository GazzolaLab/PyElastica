__doc__ = """ Validating phase space of dynamic cantilever beam analytical_cantilever_soln with  respect to varying densities.
The theoretical dynamic response is obtained via Euler-Bernoulli beam theory."""

from dynamic_cantilever_post_processing import plot_phase_space_with
from dynamic_cantilever import simulate_dynamic_cantilever_with

if __name__ == "__main__":
    import multiprocessing as mp

    PLOT_FIGURE = True
    SAVE_FIGURE = False

    # density = 500, 1000, 2000, 3000, 4000, 5000 kg/m3
    densities = [500]
    densities.extend(range(1000, 6000, 1000))

    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(simulate_dynamic_cantilever_with, densities)

    theory_frequency = []
    sim_frequency = []
    theory_amplitude = []
    sim_amplitude = []

    for result in results:
        theory_frequency.append(result["theoretical_frequency"])
        sim_frequency.append(result["simulated_frequency"])
        theory_amplitude.append(result["theoretical_amplitude"])
        sim_amplitude.append(result["simulated_amplitude"])

    # Plot frequencies and amplitudes vs densities
    plot_phase_space_with(
        densities,
        theory_frequency,
        sim_frequency,
        theory_amplitude,
        sim_amplitude,
        PLOT_FIGURE=PLOT_FIGURE,
        SAVE_FIGURE=SAVE_FIGURE,
    )
