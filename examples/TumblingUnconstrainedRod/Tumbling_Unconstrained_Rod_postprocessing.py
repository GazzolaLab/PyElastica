import logging
from elastica.utils import MaxDimension, Tolerance
from elastica.external_forces import NoForces
from elastica.typing import SystemType
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from tqdm import tqdm
from typing import Dict, Sequence


class EndpointForces_with_time_factor(NoForces):

    def __init__(self, start_force, end_force, time_factor):

        super(EndpointForces_with_time_factor, self).__init__()
        self.start_force = start_force
        self.end_force = end_force
        self.time_factor = time_factor

    def apply_forces(self, system: SystemType, time=0.0):

        factor = self.time_factor(time)

        system.external_forces[..., 0] += self.start_force * factor
        system.external_forces[..., -1] += self.end_force * factor


class EndPointTorque(NoForces):
    def __init__(self, torque, direction=np.array([0.0, 0.0, 0.0])):
        super(EndPointTorque, self).__init__()
        self.torque = torque * direction

    def apply_torques(self, system: SystemType, time: np.float64 = 0.0):
        n_elems = system.n_elems
        if time < 1:
            factor = 1
        else:
            factor = 0
        system.external_torques[..., -1] += self.torque * factor


class EndPointTorque_with_time_factor(NoForces):
    def __init__(self, torque, time_factor, direction=np.array([0.0, 0.0, 0.0])):
        super(EndPointTorque_with_time_factor, self).__init__()
        self.torque = torque * direction
        self.time_factor = time_factor

    def apply_torques(self, system: SystemType, time: np.float64 = 0.0):
        n_elems = system.n_elems

        factor = self.time_factor(time)

        system.external_torques[..., -1] += self.torque * factor


def lamda_t_function(time):
    if time < 2.5:
        factor = time * (1 / 2.5)
    elif time > 2.5 and time < 5.0:
        factor = -time * (1 / 2.5) + 2
    else:
        factor = 0

    return factor


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

        ax.view_init(elev=0, azim=0)

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
                            1000 * np.pi * (scaling_factor * inst_radius) ** 2
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


def adjust_square_cross_section(
    rod, youngs_modulus: float, length: float, ring_rod_flag: bool = False
):
    n_elements = rod.n_elems
    n_voronoi_elements = n_elements if ring_rod_flag else n_elements - 1

    log = logging.getLogger()

    side_length = np.zeros(n_elements)
    side_length.fill(length)

    new_area = np.pi * rod.radius * rod.radius

    new_moi_1 = ((side_length**4) / 12) * (1200000)
    new_moi_2 = ((side_length**4) / 12) * (1200000)
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
                youngs_modulus * new_area[i] * 100,
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
                youngs_modulus * new_moi_1[i] * (1 / 20),
                youngs_modulus * new_moi_2[i] * (1 / 20),
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
