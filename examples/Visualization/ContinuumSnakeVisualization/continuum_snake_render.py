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

import numpy as np
from scipy import interpolate
from tqdm import tqdm

from examples.Visualization._povmacros import Stages, pyelastica_rod, render

# Setup (USER DEFINE)
DATA_PATH = "continuum_snake.dat"  # Path to the simulation data
SAVE_PICKLE = True

# Rendering Configuration (USER DEFINE)
OUTPUT_FILENAME = "pov_snake"
OUTPUT_IMAGES_DIR = "frames"
FPS = 20.0
WIDTH = 1920  # 400
HEIGHT = 1080  # 250
DISPLAY_FRAMES = "Off"  # Display povray images during the rendering. ['On', 'Off']

# Camera/Light Configuration (USER DEFINE)
stages = Stages()
stages.add_camera(
    # Add diagonal viewpoint
    location=[15.0, 10.5, -15.0],
    angle=30,
    look_at=[4.0, 2.7, 2.0],
    name="diag",
)
stages.add_camera(
    # Add top viewpoint
    location=[0, 15, 3],
    angle=30,
    look_at=[0.0, 0, 3],
    sky=[-1, 0, 0],
    name="top",
)
stages.add_light(
    # Sun light
    position=[1500, 2500, -1000],
    color="White",
    camera_id=-1,
)
stages.add_light(
    # Flash light for camera 0
    position=[15.0, 10.5, -15.0],
    color=[0.09, 0.09, 0.1],
    camera_id=0,
)
stages.add_light(
    # Flash light for camera 1
    position=[0.0, 8.0, 5.0],
    color=[0.09, 0.09, 0.1],
    camera_id=1,
)
stage_scripts = stages.generate_scripts()

# Externally Including Files (USER DEFINE)
# If user wants to include other POVray objects such as grid or coordinate axes,
# objects can be defined externally and included separately.
included = ["../default.inc"]

# Multiprocessing Configuration (USER DEFINE)
MULTIPROCESSING = True
THREAD_PER_AGENT = 4  # Number of thread use per rendering process.
NUM_AGENT = multiprocessing.cpu_count() // 2  # number of parallel rendering.

# Execute
if __name__ == "__main__":
    # Load Data
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

    # Convert data to numpy array
    times = np.array(data["time"])  # shape: (timelength)
    xs = np.array(data["position"])  # shape: (timelength, 3, num_element)

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

    xs = interpolate.interp1d(times, xs, axis=0)(times_true)
    times = interpolate.interp1d(times, times, axis=0)(times_true)
    base_radius = np.ones_like(xs[:, 0, :]) * 0.050  # (TODO) radius could change

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

            # If the data contains multiple rod, this part can be modified to include
            # multiple rods.
            rod_object = pyelastica_rod(
                x=xs[frame_number],
                r=base_radius[frame_number],
                color="rgb<0.45,0.39,1>",
            )
            script.append(rod_object)
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

    # Create Video using ffmpeg
    for view_name in stage_scripts.keys():
        imageset_path = os.path.join(OUTPUT_IMAGES_DIR, view_name)

        filename = OUTPUT_FILENAME + "_" + view_name + ".mp4"

        os.system(f"ffmpeg -r {FPS} -i {imageset_path}/frame_%04d.png {filename}")
