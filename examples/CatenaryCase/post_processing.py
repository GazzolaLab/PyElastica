import numpy as np
import matplotlib

from matplotlib import pyplot as plt
import matplotlib.animation as manimation
from tqdm import tqdm
from scipy import optimize


def plot_video(
    plot_params: dict,
    video_name: str = "video.mp4",
    fps: int = 15,
    xlim: tuple[float, float] = (0, 4),
    ylim: tuple[float, float] = (-1, 1),
) -> None:
    positions_over_time = np.array(plot_params["position"])

    print("plot video")
    FFMpegWriter = manimation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    fig = plt.figure(figsize=(10, 8), frameon=True, dpi=150)
    ax = fig.add_subplot(111)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("x [m]", fontsize=16)
    ax.set_ylabel("y [m]", fontsize=16)
    # plt.axis("equal")
    with writer.saving(fig, video_name, dpi=150):
        rod_lines_2d = ax.plot(positions_over_time[0][0], positions_over_time[0][2])[0]
        for time_idx in tqdm(range(1, len(plot_params["time"]))):
            rod_lines_2d.set_xdata(positions_over_time[time_idx][0])
            rod_lines_2d.set_ydata(positions_over_time[time_idx][2])
            writer.grab_frame()

    # Be a good boy and close figures
    # https://stackoverflow.com/a/37451036
    # plt.close(fig) alone does not suffice
    # See https://github.com/matplotlib/matplotlib/issues/8560/
    plt.close(plt.gcf())


def plot_catenary(
    plot_params: dict,
    xlim: tuple[float, float] = (0, 1),
    ylim: tuple[float, float] = (-0.5, 0.5),
    base_length: float = 1.0,
) -> None:
    """
    Catenary analytical solution from Routh, Edward John (1891). "Chapter X: On Strings". A Treatise on Analytical Statics. University Press.
    """
    position = np.array(plot_params["position"])
    lowest_point = np.min(position[-1][2])
    x_catenary = np.linspace(0, base_length, 100)

    def f_non_elastic_catenary(x: float) -> float:
        return x * (1 - np.cosh(1 / (2 * x))) - lowest_point

    a = optimize.fsolve(f_non_elastic_catenary, x0=1.0)  # solve for a
    y_catenary = a * np.cosh((x_catenary - 0.5) / a) - a * np.cosh(1 / (2 * a))
    plt.plot(position[-1][0], position[-1][2], label="Simulation", linewidth=3)
    plt.plot(
        x_catenary,
        y_catenary,
        label="Catenary (Analytical Solution)",
        linewidth=3,
        linestyle="dashed",
    )
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.title("Catenary Final Shape")
    plt.grid()
    plt.legend()
    plt.xlabel("x [m]", fontsize=16)
    plt.ylabel("y [m]", fontsize=16)
    plt.savefig("plot.png", dpi=300)
    plt.show()
