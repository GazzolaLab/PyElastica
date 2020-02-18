import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb


def plot_video(
    plot_params: dict, video_name="video.mp4", margin=0.2, fps=15
):  # (time step, x/y/z, node)
    import matplotlib.animation as manimation

    positions_over_time = np.array(plot_params["position"])

    print("plot video")
    FFMpegWriter = manimation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    fig = plt.figure()
    plt.axis("equal")
    with writer.saving(fig, video_name, 100):
        for time in range(1, len(plot_params["time"])):
            x = positions_over_time[time][2]
            y = positions_over_time[time][1]
            fig.clf()
            plt.plot(x, y, "o")
            plt.xlim([0 - margin, 2.5 + margin])
            plt.ylim([-1.25 - margin, 1.25 + margin])
            writer.grab_frame()


def plot_video_actiavation_muscle(
    activation_list: dict,
    torque_list: dict,
    video_name="video.mp4",
    margin=0.2,
    fps=15,
    step=100,
):
    import matplotlib.animation as manimation

    time = np.array(activation_list["time"])
    activation = np.array(activation_list["activation_signal"])
    torque = np.array(torque_list["torque"])
    element_position = np.array(torque_list["element_position"])

    print("plot video")
    FFMpegWriter = manimation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    fig = plt.figure()
    plt.subplot(211)
    plt.axis("equal")
    with writer.saving(fig, video_name, 100):
        for time in range(1, time.shape[0], int(step)):
            # ax1 = plt.subplot(2, 2, 1)
            # ax2 = plt.subplot(222, frameon=False)
            # x = activation[time][2]
            actv = activation[time]
            torq = torque[time][1]
            pos = element_position[time]
            fig.clf()
            plt.subplot(2, 1, 1)
            plt.plot(actv, "-")
            plt.ylim([-1 - margin, 1 + margin])
            plt.subplot(2, 1, 2)
            plt.plot(pos, torq, "-")
            # plt.xlim([0 - margin, 2.5 + margin])

            writer.grab_frame()

    # plt.subplot(221)
    #
    # # equivalent but more general
    # ax1 = plt.subplot(2, 2, 1)
    #
    # # add a subplot with no frame
    # ax2 = plt.subplot(222, frameon=False)
    #
    # # add a polar subplot
    # plt.subplot(223, projection='polar')
    #
    # # add a red subplot that shares the x-axis with ax1
    # plt.subplot(224, sharex=ax1, facecolor='red')
    #
    # # delete ax2 from the figure
    # plt.delaxes(ax2)
    #
    # # add ax2 to the figure again
    # plt.subplot(ax2)
