import matplotlib.pyplot as plt
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

    fig = plt.figure(figsize=(20, 8), frameon=True, dpi=150)

    ax_freq = fig.add_subplot(121)
    ax_freq.grid(visible=True, which="both", color="k", linestyle="-")
    ax_freq.plot(
        densities,
        theory_frequency,
        color="k",
        marker="o",
        ms=8,
        lw=2,
        label="theoretical_frequency",
    )
    ax_freq.plot(
        densities,
        sim_frequency,
        color="r",
        marker="o",
        ms=8,
        lw=2,
        label="simulated_frequency",
    )
    ax_freq.set_ylim([0, 0.6])
    ax_freq.set_xlabel(r"Density [$kg/m^3$]")
    ax_freq.set_ylabel(r"Frequency [$rad/s$]")
    ax_freq.legend()

    ax_amp = fig.add_subplot(122)
    ax_amp.grid(visible=True, which="both", color="k", linestyle="-")
    ax_amp.plot(
        densities,
        theory_amplitude,
        color="k",
        marker="o",
        ms=8,
        lw=2,
        label="theoretical_amplitude",
    )
    ax_amp.plot(
        densities,
        sim_amplitude,
        color="r",
        marker="o",
        ms=8,
        lw=2,
        label="simulated_amplitude",
    )
    ax_amp.set_ylim([0, 0.05])
    ax_amp.set_xlabel(r"Density [$kg/m^3$]")
    ax_amp.set_ylabel(r"Amplitude [$m$]")
    ax_amp.legend()

    if PLOT_FIGURE:
        plt.show()

    if SAVE_FIGURE:
        fig.savefig("Dynamic_cantilever_phase_plot.png")
