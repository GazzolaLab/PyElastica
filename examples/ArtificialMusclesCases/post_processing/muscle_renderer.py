""" Rendering Script using POVray

This script reads simulated data file to render POVray animation movie.
The data file should contain dictionary of positions vectors and times.

The script supports multiple camera position where a video is generated
for each camera view.

Notes
-----
    The module requires POVray installed.
"""

import multiprocessing
import os
from functools import partial
from multiprocessing import Pool
from moviepy.editor import ImageSequenceClip


import numpy as np
from scipy import interpolate
from tqdm import tqdm


from examples.ArtificialMusclesCases.post_processing._povmacros import (
    Stages,
    pyelastica_rod,
    render,
    sphere,
)

# Setup (USER DEFINE)
def muscle_renderer(data, OUTPUT_FILENAME, camera_location, look_at_location):
    # Rendering Configuration (USER DEFINE)
    OUTPUT_IMAGES_DIR = "frames"
    FPS = 20.0
    WIDTH = 1920  # 400
    HEIGHT = 1080  # 250
    DISPLAY_FRAMES = "Off"  # Display povray images during the rendering. ['On', 'Off']

    # Camera/Light Configuration (USER DEFINE)
    stages = Stages()
    stages.add_camera(
        # Add top viewpoint
        location=camera_location,
        angle=30,
        look_at=look_at_location,
        sky=[-1, 0, 0],
        name="top",
    )
    stages.add_light(
        # Sun light
        position=[0, 2500, 10],
        color="White",
        camera_id=-1,
    )
    stage_scripts = stages.generate_scripts()

    # Externally Including Files (USER DEFINE)
    # If user wants to include other POVray objects such as grid or coordinate axes,
    # objects can be defined externally and included separately.
    included = ["default.inc"]

    # Multiprocessing Configuration (USER DEFINE)
    MULTIPROCESSING = True
    THREAD_PER_AGENT = 4  # Number of thread use per rendering process.
    NUM_AGENT = multiprocessing.cpu_count() // 2  # number of parallel rendering.

    # Execute
    # Convert data to numpy array
    times = np.array(data[0]["time"])  # shape: (timelength)
    xs = {}
    base_radius = {}

    # Interpolate Data
    # Interpolation step serves two purposes. If simulated frame rate is lower than
    # the video frame rate, the intermediate frames are linearly interpolated to
    # produce smooth video. Otherwise if simulated frame rate is higher than
    # the video frame rate, interpolation reduces the number of frame to reduce
    # the rendering time.
    runtime = times.max()  # Physical run time
    total_frame = int(runtime * FPS)  # Number of frames for the video
    recorded_frame = times.shape[0]  # Number of simulated frames
    times_true = np.linspace(0, runtime, total_frame)  # Adjusted timescale

    for i in range(len(data)):
        xs[i] = np.array(data[i]["position"])  # shape: (timelength, 3, num_element)
        base_radius[i] = np.array(data[i]["radius"])[:, 0:1] * np.ones_like(
            xs[i][:, 0, :]
        )  # shape: (timelength, num_element-1)
        xs[i] = interpolate.interp1d(times, xs[i], axis=0)(times_true)
        base_radius[i] = interpolate.interp1d(times, base_radius[i], axis=0)(times_true)

    # tracking_sphere_center = np.zeros((xs[0].shape[0],xs[0].shape[1]))
    # first_muscle_end = np.zeros((xs[0].shape[0],xs[0].shape[1]))
    # second_muscle_start = np.zeros((xs[0].shape[0],xs[0].shape[1]))
    # n_fibers = int(len(data)/2)
    # for i in range(n_fibers):
    #     first_muscle_end += xs[i][:,:,-1]/n_fibers
    #     second_muscle_start += xs[i+n_fibers][:,:,0]/n_fibers
    # tracking_sphere_center = 0.5*(first_muscle_end+second_muscle_start)

    times = interpolate.interp1d(times, times, axis=0)(times_true)
    # colors = [[31,119,180],[255,127,14],[44,160,44]]
    colors = [[192, 190, 175]]

    # Rendering
    # For each frame, a 'pov' script file is generated in OUTPUT_IMAGE_DIR directory.
    batch = []
    for view_name in stage_scripts.keys():  # Make Directory
        output_path = os.path.join(OUTPUT_IMAGES_DIR, view_name)
        os.makedirs(output_path, exist_ok=True)
    for frame_number in tqdm(range(total_frame), desc="Scripting"):
        for view_name, stage_script in stage_scripts.items():
            output_path = os.path.join(OUTPUT_IMAGES_DIR, view_name)

            # Colect povray scripts
            script = []
            script.extend(['#include "{}"'.format(s) for s in included])
            script.append(stage_script)

            for i in range(len(data)):
                color = colors[i % len(colors)]
                rod_object = pyelastica_rod(
                    x=xs[i][frame_number],
                    r=base_radius[i][frame_number],
                    color="rgb<{}/255,{}/255,{}/255>".format(
                        color[0], color[1], color[2]
                    ),
                    transmit=0.5,
                )
                script.append(rod_object)
            # tracking_sphere = sphere(
            #     tracking_sphere_center[frame_number],
            #     r = 6*base_radius[0][0,0],
            #     color ="rgb<1,0,0>",
            #     transmit = 0,
            # )
            # print(tracking_sphere)
            # script.append(tracking_sphere)
            pov_script = "\n".join(script)

            # Write .pov script file
            file_path = os.path.join(output_path, "frame_{:04d}".format(frame_number))
            with open(file_path + ".pov", "w+") as f:
                f.write(pov_script)
            batch.append(file_path)

    # Process POVray
    # For each frames, a 'png' image file is generated in OUTPUT_IMAGE_DIR directory.
    pbar = tqdm(total=len(batch), desc="Rendering")  # Progress Bar
    if MULTIPROCESSING:
        func = partial(
            render,
            width=WIDTH,
            height=HEIGHT,
            display=DISPLAY_FRAMES,
            pov_thread=THREAD_PER_AGENT,
        )
        with Pool(NUM_AGENT) as p:
            for message in p.imap_unordered(func, batch):
                # (TODO) POVray error within child process could be an issue
                pbar.update()
    else:
        for filename in batch:
            render(
                filename,
                width=WIDTH,
                height=HEIGHT,
                display=DISPLAY_FRAMES,
                pov_thread=multiprocessing.cpu_count(),
            )
            pbar.update()

    # Create Video using moviepy
    for view_name in stage_scripts.keys():
        imageset_path = os.path.join(OUTPUT_IMAGES_DIR, view_name)
        imageset = [
            os.path.join(imageset_path, path)
            for path in os.listdir(imageset_path)
            if path[-3:] == "png"
        ]
        imageset.sort()
        filename = OUTPUT_FILENAME + "_" + view_name + ".mp4"
        clip = ImageSequenceClip(imageset, fps=FPS)
        clip.write_videofile(filename, fps=FPS)


if __name__ == "__main__":
    DATA_PATH = "ArtificialMusclesCases/SingleMuscleCases/PureContraction/Samuel_supercoil/data/PureContractionSamuel_supercoil.dat"
    OUTPUT_FILENAME = "ArtificialMusclesCases/SingleMuscleCases/PureContraction/Samuel_supercoil/PureContractionSamuel_supercoil"
    camera_location = [0, 60e-3, 10e-3]
    look_at_location = [0, 0, 10e-3]
    SAVE_PICKLE = True
    assert os.path.exists(DATA_PATH), "File does not exists"
    try:
        if SAVE_PICKLE:
            import pickle as pk

            with open(DATA_PATH, "rb") as fptr:
                data = pk.load(fptr)
        else:
            # (TODO) add importing npz file format
            raise NotImplementedError("Only pickled data is supported")
    except OSError as err:
        print("Cannot open the datafile {}".format(DATA_PATH))
        print(str(err))
        raise
    muscle_renderer(data, OUTPUT_FILENAME, camera_location, look_at_location)
