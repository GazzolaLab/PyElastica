import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from tqdm import tqdm
from typing import Dict, Sequence
import logging
from elastica.utils import MaxDimension, Tolerance
from elastica.external_forces import NoForces, inplace_addition, SystemType


class NonconserativeForce(NoForces):
    def __init__(self, load=1):
        super(NonconserativeForce, self).__init__()
        self.load = load

    def apply_forces(self, system: SystemType, time=0.0):
        self.compute_nonconserative_forces(
            self.load, system.mass, system.director_collection, system.external_forces
        )

    def compute_nonconserative_forces(self, load, mass, direction, external_forces):
        NCforce_direction = direction[0]
        NCforce_direction = NCforce_direction / np.linalg.norm(
            NCforce_direction, axis=0
        )
        NCforce = NCforce_direction * self.load
        for i in range(len(mass) - 1):
            NCforce[0, i] = NCforce[0, i] * mass[i]
            NCforce[1, i] = NCforce[1, i] * mass[i]
            NCforce[2, i] = NCforce[2, i] * mass[i]

        inplace_addition(external_forces, NCforce)


def find_tip_position(rod, n_elem):
    x_tip = rod.position_collection[0][n_elem]
    y_tip = rod.position_collection[1][n_elem]
    z_tip = rod.position_collection[2][n_elem]

    return x_tip, y_tip, z_tip


def adjust_square_cross_section(
    rod, youngs_modulus: float, length: float, ring_rod_flag: bool = False
):
    n_elements = rod.n_elems
    n_voronoi_elements = n_elements if ring_rod_flag else n_elements - 1

    log = logging.getLogger()

    side_length = np.zeros(n_elements)
    side_length.fill(length)

    new_area = np.pi * rod.radius * rod.radius

    new_moi_1 = (side_length**4) / 12
    new_moi_2 = (side_length**4) / 12
    new_moi_3 = new_moi_2 * 2

    new_moi = np.array([new_moi_1, new_moi_2, new_moi_3]).transpose()

    mass_second_moment_of_inertia_temp = np.einsum(
        "ij,i->ij", new_moi, rod.density * rod.rest_lengths
    )

    for i in range(n_elements):
        np.fill_diagonal(
            rod.mass_second_moment_of_inertia[..., i],
            mass_second_moment_of_inertia_temp[i, :],
        )
    # sanity check of mass second moment of inertia
    if (rod.mass_second_moment_of_inertia < Tolerance.atol()).all():
        message = "Mass moment of inertia matrix smaller than tolerance, please check provided radius, density and length."
        log.warning(message)

    for i in range(n_elements):
        # Check rank of mass moment of inertia matrix to see if it is invertible
        assert (
            np.linalg.matrix_rank(rod.mass_second_moment_of_inertia[..., i])
            == MaxDimension.value()
        )
        rod.inv_mass_second_moment_of_inertia[..., i] = np.linalg.inv(
            rod.mass_second_moment_of_inertia[..., i]
        )

    # Shear/Stretch matrix
    shear_modulus = youngs_modulus / (2.0 * (1.0 + 0.5))

    # Value taken based on best correlation for Poisson ratio = 0.5, from
    # "On Timoshenko's correction for shear in vibrating beams" by Kaneko, 1975
    alpha_c = 27.0 / 28.0
    rod.shear_matrix *= 0.0
    for i in range(n_elements):
        np.fill_diagonal(
            rod.shear_matrix[..., i],
            [
                alpha_c * shear_modulus * new_area[i],
                alpha_c * shear_modulus * new_area[i],
                youngs_modulus * new_area[i],
            ],
        )

    # Bend/Twist matrix
    bend_matrix = np.zeros(
        (MaxDimension.value(), MaxDimension.value(), n_voronoi_elements + 1), np.float64
    )
    for i in range(n_elements):
        np.fill_diagonal(
            bend_matrix[..., i],
            [
                youngs_modulus * new_moi_1[i],
                youngs_modulus * new_moi_2[i],
                shear_modulus * new_moi_3[i],
            ],
        )
    if ring_rod_flag:  # wrap around the value in the last element
        bend_matrix[..., -1] = bend_matrix[..., 0]
    for i in range(0, MaxDimension.value()):
        assert np.all(
            bend_matrix[i, i, :] > Tolerance.atol()
        ), " Bend matrix has to be greater than 0."

    # Compute bend matrix in Voronoi Domain
    rest_lengths_temp_for_voronoi = (
        np.hstack((rod.rest_lengths, rod.rest_lengths[0]))
        if ring_rod_flag
        else rod.rest_lengths
    )
    rod.bend_matrix = (
        bend_matrix[..., 1:] * rest_lengths_temp_for_voronoi[1:]
        + bend_matrix[..., :-1] * rest_lengths_temp_for_voronoi[0:-1]
    ) / (rest_lengths_temp_for_voronoi[1:] + rest_lengths_temp_for_voronoi[:-1])


def plot_video_with_surface(
    rods_history: Sequence[Dict],
    video_name="video.mp4",
    fps=60,
    step=1,
    vis2D=True,
    **kwargs,
):
    plt.rcParams.update({"font.size": 22})

    folder_name = kwargs.get("folder_name", "")

    # 2d case <always 2d case for now>
    import matplotlib.animation as animation

    # simulation time
    sim_time = np.array(rods_history[0]["time"])

    # Rod
    n_visualized_rods = len(rods_history)  # should be one for now
    # Rod info
    rod_history_unpacker = lambda rod_idx, t_idx: (
        rods_history[rod_idx]["position"][t_idx],
        rods_history[rod_idx]["radius"][t_idx],
    )
    # Rod center of mass
    com_history_unpacker = lambda rod_idx, t_idx: rods_history[rod_idx]["com"][time_idx]

    # Generate target sphere data
    sphere_flag = False
    if kwargs.__contains__("sphere_history"):
        sphere_flag = True
        sphere_history = kwargs.get("sphere_history")
        n_visualized_spheres = len(sphere_history)  # should be one for now
        sphere_history_unpacker = lambda sph_idx, t_idx: (
            sphere_history[sph_idx]["position"][t_idx],
            sphere_history[sph_idx]["radius"][t_idx],
        )
        # color mapping
        sphere_cmap = cm.get_cmap("Spectral", n_visualized_spheres)

    # video pre-processing
    print("plot scene visualization video")
    FFMpegWriter = animation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    dpi = kwargs.get("dpi", 100)

    xlim = kwargs.get("x_limits", (-1.0, 1.0))
    ylim = kwargs.get("y_limits", (-1.0, 1.0))
    zlim = kwargs.get("z_limits", (-0.05, 1.0))

    difference = lambda x: x[1] - x[0]
    max_axis_length = max(difference(xlim), difference(ylim))
    # The scaling factor from physical space to matplotlib space
    scaling_factor = (2 * 0.1) / max_axis_length  # Octopus head dimension
    scaling_factor *= 2.6e3  # Along one-axis

    if kwargs.get("vis3D", True):
        fig = plt.figure(1, figsize=(10, 8), frameon=True, dpi=dpi)
        ax = plt.axes(projection="3d")

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)

        ax.view_init(elev=90, azim=-90)

        time_idx = 0
        rod_lines = [None for _ in range(n_visualized_rods)]
        rod_com_lines = [None for _ in range(n_visualized_rods)]
        rod_scatters = [None for _ in range(n_visualized_rods)]

        for rod_idx in range(n_visualized_rods):
            inst_position, inst_radius = rod_history_unpacker(rod_idx, time_idx)
            if not inst_position.shape[1] == inst_radius.shape[0]:
                inst_position = 0.5 * (inst_position[..., 1:] + inst_position[..., :-1])

            rod_scatters[rod_idx] = ax.scatter(
                inst_position[0],
                inst_position[1],
                inst_position[2],
                s=np.pi * (scaling_factor * inst_radius) ** 2,
            )

        if sphere_flag:
            sphere_artists = [None for _ in range(n_visualized_spheres)]
            for sphere_idx in range(n_visualized_spheres):
                sphere_position, sphere_radius = sphere_history_unpacker(
                    sphere_idx, time_idx
                )
                sphere_artists[sphere_idx] = ax.scatter(
                    sphere_position[0],
                    sphere_position[1],
                    sphere_position[2],
                    s=np.pi * (scaling_factor * sphere_radius) ** 2,
                )
                # sphere_radius,
                # color=sphere_cmap(sphere_idx),)
                ax.add_artist(sphere_artists[sphere_idx])

        # ax.set_aspect("equal")
        video_name_3D = folder_name + "3D_" + video_name

        with writer.saving(fig, video_name_3D, dpi):
            with plt.style.context("seaborn-v0_8-whitegrid"):
                for time_idx in tqdm(range(0, sim_time.shape[0], int(step))):

                    for rod_idx in range(n_visualized_rods):
                        inst_position, inst_radius = rod_history_unpacker(
                            rod_idx, time_idx
                        )
                        if not inst_position.shape[1] == inst_radius.shape[0]:
                            inst_position = 0.5 * (
                                inst_position[..., 1:] + inst_position[..., :-1]
                            )

                        rod_scatters[rod_idx]._offsets3d = (
                            inst_position[0],
                            inst_position[1],
                            inst_position[2],
                        )

                        # rod_scatters[rod_idx].set_offsets(inst_position[:2].T)
                        rod_scatters[rod_idx].set_sizes(
                            np.pi * (scaling_factor * inst_radius) ** 2
                        )

                    if sphere_flag:
                        for sphere_idx in range(n_visualized_spheres):
                            sphere_position, _ = sphere_history_unpacker(
                                sphere_idx, time_idx
                            )
                            sphere_artists[sphere_idx]._offsets3d = (
                                sphere_position[0],
                                sphere_position[1],
                                sphere_position[2],
                            )

                    writer.grab_frame()

        # Be a good boy and close figures
        # https://stackoverflow.com/a/37451036
        # plt.close(fig) alone does not suffice
        # See https://github.com/matplotlib/matplotlib/issues/8560/
        # plt.close(plt.gcf())
