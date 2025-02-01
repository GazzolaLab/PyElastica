from typing import Optional
import logging

from numpy.testing import assert_allclose
from elastica.utils import MaxDimension, Tolerance
from elastica._linalg import _batch_cross, _batch_norm, _batch_dot
from elastica.rod.factory_function import (
    _assert_dim,
    _position_validity_checker,
    _directors_validity_checker,
    _position_validity_checker_ring_rod,
)
import numpy as np
from matplotlib import pyplot as plt

from matplotlib import cm
from tqdm import tqdm

from typing import Dict, Sequence


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


def adjust_parameter(
    n_elements,
    direction,
    normal,
    base_length,
    length,
    base_radius,
    density,
    youngs_modulus: float,
    *,
    rod_origin_position: np.ndarray,
    ring_rod_flag: bool,
    shear_modulus: Optional[float] = None,
    position: Optional[np.ndarray] = None,
    directors: Optional[np.ndarray] = None,
    **kwargs,
):
    log = logging.getLogger()

    if "poisson_ratio" in kwargs:
        # Deprecation warning for poission_ratio
        raise NameError(
            "Poisson's ratio is deprecated for Cosserat Rod for clarity. Please provide shear_modulus instead."
        )

    # sanity checks here
    assert n_elements > 2 if ring_rod_flag else n_elements > 1
    assert base_length > Tolerance.atol()
    assert np.sqrt(np.dot(normal, normal)) > Tolerance.atol()
    assert np.sqrt(np.dot(direction, direction)) > Tolerance.atol()

    # define the number of nodes and voronoi elements based on if rod is
    # straight and open or closed and ring shaped
    n_nodes = n_elements if ring_rod_flag else n_elements + 1
    n_voronoi_elements = n_elements if ring_rod_flag else n_elements - 1

    # check if position is given.
    if position is None:  # Generate straight and uniform rod
        # Set the position array
        position = np.zeros((MaxDimension.value(), n_nodes))
        if not ring_rod_flag:  # i.e. a straight open rod

            start = rod_origin_position
            end = start + direction * base_length

            for i in range(0, 3):
                position[i, ...] = np.linspace(start[i], end[i], n_elements + 1)

            _position_validity_checker(position, start, n_elements)
        else:  # i.e a closed ring rod
            ring_center_position = rod_origin_position
            binormal = np.cross(direction, normal)
            for i in range(n_elements):
                position[..., i] = (
                    base_length
                    / (2 * np.pi)
                    * (
                        np.cos(2 * np.pi / n_elements * i) * binormal
                        + np.sin(2 * np.pi / n_elements * i) * direction
                    )
                ) + ring_center_position
            _position_validity_checker_ring_rod(
                position, ring_center_position, n_elements
            )

    # Compute rest lengths and tangents
    position_for_difference = (
        np.hstack((position, position[..., 0].reshape(3, 1)))
        if ring_rod_flag
        else position
    )
    position_diff = position_for_difference[..., 1:] - position_for_difference[..., :-1]
    rest_lengths = _batch_norm(position_diff)
    tangents = position_diff / rest_lengths
    normal /= np.linalg.norm(normal)

    if directors is None:  # Generate straight uniform rod
        print("Directors not specified")
        # Set the directors matrix
        directors = np.zeros((MaxDimension.value(), MaxDimension.value(), n_elements))
        # Construct directors using tangents and normal
        normal_collection = np.repeat(normal[:, np.newaxis], n_elements, axis=1)
        # Check if rod normal and rod tangent are perpendicular to each other otherwise
        # directors will be wrong!!
        assert_allclose(
            _batch_dot(normal_collection, tangents),
            0,
            atol=Tolerance.atol(),
            err_msg=(" Rod normal and tangent are not perpendicular to each other!"),
        )
        directors[0, ...] = normal_collection
        directors[1, ...] = _batch_cross(tangents, normal_collection)
        directors[2, ...] = tangents
    _directors_validity_checker(directors, tangents, n_elements)

    # Set radius array
    radius = np.zeros((n_elements))
    # Check if the user input radius is valid
    radius_temp = np.array(base_radius)
    _assert_dim(radius_temp, 2, "radius")
    radius[:] = radius_temp
    # Check if the elements of radius are greater than tolerance
    assert np.all(radius > Tolerance.atol()), " Radius has to be greater than 0."

    # Set density array
    density_array = np.zeros((n_elements))
    # Check if the user input density is valid
    density_temp = np.array(density)
    _assert_dim(density_temp, 2, "density")
    density_array[:] = density_temp
    # Check if the elements of density are greater than tolerance
    assert np.all(
        density_array > Tolerance.atol()
    ), " Density has to be greater than 0."

    # Second moment of inertia

    side_length = np.zeros(n_elements)
    side_length.fill(length)

    A0 = np.pi * radius * radius

    I0_1 = ((side_length**4) / 12) * (1200000)
    I0_2 = ((side_length**4) / 12) * (1200000)
    I0_3 = I0_2 * 2

    I0 = np.array([I0_1, I0_2, I0_3]).transpose()

    # Mass second moment of inertia for disk cross-section
    mass_second_moment_of_inertia = np.zeros(
        (MaxDimension.value(), MaxDimension.value(), n_elements), np.float64
    )

    mass_second_moment_of_inertia_temp = np.einsum(
        "ij,i->ij", I0, density * rest_lengths
    )

    for i in range(n_elements):
        np.fill_diagonal(
            mass_second_moment_of_inertia[..., i],
            mass_second_moment_of_inertia_temp[i, :],
        )
    # sanity check of mass second moment of inertia
    if (mass_second_moment_of_inertia < Tolerance.atol()).all():
        message = "Mass moment of inertia matrix smaller than tolerance, please check provided radius, density and length."
        log.warning(message)

    # Inverse of second moment of inertia
    inv_mass_second_moment_of_inertia = np.zeros(
        (MaxDimension.value(), MaxDimension.value(), n_elements)
    )
    for i in range(n_elements):
        # Check rank of mass moment of inertia matrix to see if it is invertible
        assert (
            np.linalg.matrix_rank(mass_second_moment_of_inertia[..., i])
            == MaxDimension.value()
        )
        inv_mass_second_moment_of_inertia[..., i] = np.linalg.inv(
            mass_second_moment_of_inertia[..., i]
        )

    # Shear/Stretch matrix
    if not shear_modulus:
        log.info(
            """Shear modulus is not explicitly given.\n
            In such case, we compute shear_modulus assuming poisson's ratio of 0.5"""
        )
        shear_modulus = youngs_modulus / (2.0 * (1.0 + 0.5))

    # Value taken based on best correlation for Poisson ratio = 0.5, from
    # "On Timoshenko's correction for shear in vibrating beams" by Kaneko, 1975
    alpha_c = 27.0 / 28.0
    shear_matrix = np.zeros(
        (MaxDimension.value(), MaxDimension.value(), n_elements), np.float64
    )
    for i in range(n_elements):
        np.fill_diagonal(
            shear_matrix[..., i],
            [
                alpha_c * shear_modulus * A0[i],
                alpha_c * shear_modulus * A0[i],
                youngs_modulus * A0[i] * 100,
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
                youngs_modulus * I0_1[i] * (1 / 20),
                youngs_modulus * I0_2[i] * (1 / 20),
                shear_modulus * I0_3[i],
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
        np.hstack((rest_lengths, rest_lengths[0])) if ring_rod_flag else rest_lengths
    )
    bend_matrix = (
        bend_matrix[..., 1:] * rest_lengths_temp_for_voronoi[1:]
        + bend_matrix[..., :-1] * rest_lengths_temp_for_voronoi[0:-1]
    ) / (rest_lengths_temp_for_voronoi[1:] + rest_lengths_temp_for_voronoi[:-1])

    return (
        mass_second_moment_of_inertia,
        inv_mass_second_moment_of_inertia,
        bend_matrix,
    )
