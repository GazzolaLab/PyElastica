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
from moviepy.editor import ImageSequenceClip
from scipy import interpolate
from tqdm import tqdm
from matplotlib import pyplot as plt


from _povmacros import Stages, pyelastica_rod, render, surface_mesh


def mesh_rod_render(data, output_file_name):

    # Rendering Configuration (USER DEFINE)
    # OUTPUT_FILENAME = "pov_antagonistic"
    OUTPUT_FILENAME = "pov_antagonistic_arm"

    OUTPUT_IMAGES_DIR = "frames"
    if os.path.exists(OUTPUT_IMAGES_DIR):
        os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)

    FPS = 20.0
    WIDTH = 1920  # 400
    HEIGHT = 1080  # 250
    DISPLAY_FRAMES = "Off"  # Display povray images during the rendering. ['On', 'Off']

    # Camera/Light Configuration (USER DEFINE)
    stages = Stages()
    stages.add_camera(
        # Add front viewpoint
        location=[0, 40e-2, 10e-2],
        angle=30,
        look_at=[0.0, 0, 0],
        sky=[0, -40e-2, 0],
        name="front",
    )
    stages.add_light(
        # Sun light
        position=[0.0, 0.0, 1000],
        color=[1, 1, 1],
        camera_id=-1,
    )
    stages.add_light(
        # camera light
        position=[0, 40.0e-2, 10e-2],
        color=[0.5, 0.5, 0.5],
        camera_id=0,
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
    if __name__ == "__main__":
        # Convert data to numpy array
        times = np.array(data[0]["time"])  # shape: (timelength)
        faces_rigid_body = np.array(
            data[0]["faces"]
        )  # shape: (timelength, 3 ,3, num_element)
        # from elastica import Mesh
        # mesh = Mesh(r"lower_arm.stl")
        # mesh.translate(-np.array(mesh.mesh_center))
        # mesh.rotate(np.array([1,0,0]),180)
        # mesh.scale(np.array([1e-3,1e-3,1e-3]))
        # print(np.array(data[0]["faces"])-mesh.faces)
        x = []
        r = []
        for i in range(len(data) - 1):
            x.append(
                np.array(data[i + 1]["position"])
            )  # shape: (timelength, 3, num_element)
            r.append(
                np.array(data[i + 1]["radius"])
            )  # shape: (timelength, 3, num_element)

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
        for i in range(len(data) - 1):
            x[i] = interpolate.interp1d(times, x[i], axis=0)(times_true)
            r[i] = interpolate.interp1d(times, r[i], axis=0)(times_true)
        faces_rigid_body = interpolate.interp1d(times, faces_rigid_body, axis=0)(
            times_true
        )
        times = interpolate.interp1d(times, times, axis=0)(times_true)
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
                for i in range(len(data) - 1):
                    xi = x[i]
                    ri = r[i]
                    rod_object = pyelastica_rod(
                        x=xi[frame_number],
                        r=ri[frame_number],
                        color="rgb<192/255,190/255,175/255>",
                        transmit=0.5,
                    )
                    script.append(rod_object)

                current_facets = faces_rigid_body[frame_number]
                surface_object = surface_mesh(
                    current_facets,
                    color=[1, 1, 1],
                    transmit=0,
                )
                script.append(surface_object)
                pov_script = "\n".join(script)

                # Write .pov script file
                file_path = os.path.join(
                    output_path, "frame_{:04d}".format(frame_number)
                )
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
