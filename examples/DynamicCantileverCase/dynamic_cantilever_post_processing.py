import matplotlib.pyplot as plt
import numpy as np


# Plotting frequency and amplitudes against densities
def plot_phase_space_with(
    densities,
    theory_frequency,
    sim_frequency,
    theory_amplitude,
    sim_amplitude,
    PLOT_FIGURE=True,
    SAVE_FIGURE=False,
):

    fig = plt.figure(figsize=(20, 8), frameon=True, dpi=150)

    ax_freq = fig.add_subplot(121)
    ax_freq.grid(which="both", color="k", linestyle="-")
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


def plot_end_position_with(
    recorded_history,
    analytical_cantilever_soln,
    omegas,
    amplitudes,
    peak,
    PLOT_FIGURE,
    SAVE_FIGURE,
):
    fig = plt.figure(figsize=(20, 8), frameon=True, dpi=150)
    ax = fig.add_subplot(121)
    ax.plot(
        recorded_history["time"],
        recorded_history["deflection"],
        lw=2.0,
        label="Cosserat rod model",
    )
    ax.set_xlabel("Time [s]", fontsize=16)
    ax.set_ylabel("Displacement [m]", fontsize=16)
    ax.grid(which="both", color="k", linestyle="--")

    time = np.array(recorded_history["time"])
    positions = analytical_cantilever_soln.get_time_dependent_positions(1, time)

    ax.plot(time, positions, lw=2.0, label="Euler-Bernoulli beam theory")
    ax.legend()

    # FFT plot
    fft_ax = fig.add_subplot(122)
    fft_ax.plot(omegas, amplitudes, lw=2, color="k")
    fft_ax.set_xlim([0, analytical_cantilever_soln.get_omega() * 2])

    fft_ax.set_xlabel("Frequency [rad/s]", fontsize=16)
    fft_ax.set_ylabel("Amplitude [m]", fontsize=16)

    fft_ax.plot(omegas[peak], amplitudes[peak], "*", color="red", ms=8)

    if PLOT_FIGURE:
        plt.show()

    if SAVE_FIGURE:
        fig.savefig("Dynamic_cantilever_visualization.png")


def plot_dynamic_cantilever_video_with(
    mode,
    recorded_history,
    rendering_fps,
):
    print("Plotting video ...")
    video_name = f"Dynamic_cantilever_mode_{mode + 1}.mp4"

    import matplotlib.animation as manimation
    from tqdm import tqdm

    positions_over_time = np.array(recorded_history["position"])

    FFMpegWriter = manimation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=rendering_fps, metadata=metadata)
    mvfig = plt.figure(figsize=(10, 8), frameon=True, dpi=150)
    ax = mvfig.add_subplot(111)
    ax.set_xlim([0, 1.25])
    ax.set_ylim([-0.02, 0.02])
    ax.set_xlabel("x [m]", fontsize=16)
    ax.set_ylabel("z [m]", fontsize=16)
    rod_lines_2d = ax.plot(
        positions_over_time[0][0], positions_over_time[0][2], lw=8, color="k"
    )[0]

    with writer.saving(mvfig, video_name, dpi=150):
        for t in tqdm(range(1, len(recorded_history["time"]))):
            rod_lines_2d.set_xdata(positions_over_time[t][0])
            rod_lines_2d.set_ydata(positions_over_time[t][2])
            writer.grab_frame()

    plt.close(plt.gcf())
