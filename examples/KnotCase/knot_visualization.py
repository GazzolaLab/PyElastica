from matplotlib import pyplot as plt
import matplotlib.animation as manimation
import numpy as np

def plot_video_2D(plot_params: dict, video_name="video.mp4", margin=0.2, fps=15,targets=None):
    t = np.array(plot_params["time"])
    positions_over_time = np.array(plot_params["position"])
    total_time = int(np.around(t[..., -1], 1))
    total_frames = fps * total_time
    step = round(len(t) / total_frames)

    print("creating video -- this can take a few minutes")
    FFMpegWriter = manimation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=fps, metadata=metadata)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.axis("equal")
    rod_lines_2d = ax.plot(
        positions_over_time[0][2], positions_over_time[0][0], linewidth=3
    )[0]
    
    if targets is not None:
        c = np.arange(0,len(targets))/len(targets)
        ax.scatter(targets[:, 2], targets[:, 0], cmap = "inferno" ,c = c, label="Target")
    ax.set_xlim([-1 - margin, 1 + margin])
    ax.set_ylim([-1 - margin, 1 + margin])
    with writer.saving(fig, video_name, dpi=100):
        with plt.style.context("seaborn-v0_8-whitegrid"):
            for time in range(1, len(t), step):
                rod_lines_2d.set_xdata(positions_over_time[time][2])
                rod_lines_2d.set_ydata(positions_over_time[time][0])

                writer.grab_frame()
    plt.close(fig)
    print("Video saved as", video_name)

def plot_video3D(plot_params: dict, video_name="video.mp4", margin=0.2, fps=15):
    t = np.array(plot_params["time"])
    positions_over_time = np.array(plot_params["position"])
    total_time = int(np.around(t[..., -1], 1))
    total_frames = fps * total_time
    print("Total time:", total_time, "Total frames:", total_frames)
    step = round(len(t) / total_frames)
    print("creating video -- this can take a few minutes")
    FFMpegWriter = manimation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(-1 - margin, 1 + margin)
    ax.set_ylim(-1 - margin, 1 + margin)
    ax.set_zlim(-1 - margin, 1 + margin)
    ax.view_init(elev=10, azim=-40)
    rod_lines_3d = ax.plot(
        positions_over_time[0][2],
        positions_over_time[0][0],
        positions_over_time[0][1],
        linewidth=3,
    )[0]
    with writer.saving(fig, video_name, dpi=100):
        with plt.style.context("seaborn-v0_8-whitegrid"):
            for time in range(1, len(t), step):
                rod_lines_3d.set_xdata(positions_over_time[time][2])
                rod_lines_3d.set_ydata(positions_over_time[time][0])
                rod_lines_3d.set_3d_properties(positions_over_time[time][1])

                writer.grab_frame()
    plt.close(fig)
    print("Video saved as", video_name)
