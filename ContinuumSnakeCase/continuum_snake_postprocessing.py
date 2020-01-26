import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb


def plot_snake_velocity(
    list_input, period, filename="slithering_snake_velocity.png", SAVE_FIGURE=False
):

    time_per_period = np.array(list_input["time"]) / period
    avg_velocity = np.array(list_input["avg_velocity"])

    fig = plt.figure(figsize=(10, 8), frameon=True, dpi=150)
    ax = fig.add_subplot(111)
    ax.grid(b=True, which="minor", color="k", linestyle="--")
    ax.grid(b=True, which="major", color="k", linestyle="-")
    ax.plot(time_per_period[:], avg_velocity[:, 2], "r-", label="forward")
    ax.plot(
        time_per_period[:],
        avg_velocity[:, 0],
        c=to_rgb("xkcd:bluish"),
        label="lateral",
    )
    ax.plot(time_per_period[:], avg_velocity[:, 1], "k-", label="normal")
    fig.legend(prop={"size": 20})
    plt.show()

    if SAVE_FIGURE:
        fig.savefig(filename)


def plot_video(
    list_input, video_name="video.mp4", margin=0.2, fps=15
):  # (time step, x/y/z, node)
    import matplotlib.animation as manimation

    positions_over_time = np.array(list_input["position"])

    print("plot video")
    FFMpegWriter = manimation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    fig = plt.figure()
    plt.axis("equal")
    with writer.saving(fig, video_name, 100):
        for time in range(1, len(list_input["time"])):
            x = positions_over_time[time][2]
            y = positions_over_time[time][0]
            fig.clf()
            plt.plot(x, y, "o")
            plt.xlim([0 - margin, 10 + margin])
            plt.ylim([-5 - margin, 5 + margin])
            writer.grab_frame()
