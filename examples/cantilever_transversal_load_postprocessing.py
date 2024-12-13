import numpy as np
import matplotlib
#matplotlib.use("Agg")  # Must be before importing matplotlib.pyplot or pylab!
from matplotlib import pyplot as plt
from matplotlib import cm
from tqdm import tqdm
from typing import Dict, Sequence
from typing import Optional
import logging
from numpy.testing import assert_allclose
from elastica.utils import MaxDimension, Tolerance
from elastica._linalg import _batch_cross, _batch_norm, _batch_dot
from elastica.rod.factory_function import _assert_dim,_position_validity_checker,_directors_validity_checker,_position_validity_checker_ring_rod


def Find_Tip_Position(rod, n_elem):
    x_tip = rod.position_collection[0][n_elem]
    y_tip = rod.position_collection[1][n_elem]
    z_tip = rod.position_collection[2][n_elem]

    return x_tip, y_tip, z_tip

def plot_video_with_surface(
    rods_history: Sequence[Dict],
    video_name="video.mp4",
    fps=60,
    step=1,
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

        ax.view_init(elev=20, azim=20)

        time_idx = 0
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
        #plt.close(plt.gcf())

def adjust_square_cross_section(
    n_elements,
    direction,
    normal,
    base_length,
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
    side_length.fill(0.01)

    A0 = np.pi * radius * radius


    I0_1 = (side_length**4)/12
    I0_2 = (side_length**4)/12
    I0_3 = I0_2*2

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
                youngs_modulus * A0[i],
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
                youngs_modulus * I0_1[i],
                youngs_modulus * I0_2[i],
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
#This list is used to do the convergence study. This problem of cantilever subjected to a transversal load doesn't have an analytical,
#So we used the numerical solution that set n_elem as 512 as the approximate analytical solution
def analytical_results (index):
    analytical_results=[
        0.00000000e+00, 1.22204628e-07, 1.35200810e-05, 4.01513712e-05,
        7.99730376e-05, 1.32941281e-04, 1.99011559e-04, 2.78138600e-04,
        3.70276423e-04, 4.75378357e-04, 5.93397051e-04, 7.24284499e-04,
        8.67992049e-04, 1.02447043e-03, 1.19366975e-03, 1.37553953e-03,
        1.57002873e-03, 1.77708572e-03, 1.99665834e-03, 2.22869391e-03,
        2.47313923e-03, 2.72994059e-03, 2.99904381e-03, 3.28039424e-03,
        3.57393679e-03, 3.87961590e-03, 4.19737562e-03, 4.52715957e-03,
        4.86891100e-03, 5.22257275e-03, 5.58808731e-03, 5.96539683e-03,
        6.35444310e-03, 6.75516760e-03, 7.16751149e-03, 7.59141563e-03,
        8.02682062e-03, 8.47366675e-03, 8.93189407e-03, 9.40144239e-03,
        9.88225128e-03, 1.03742601e-02, 1.08774079e-02, 1.13916338e-02,
        1.19168763e-02, 1.24530742e-02, 1.30001658e-02, 1.35580893e-02,
        1.41267830e-02, 1.47061848e-02, 1.52962324e-02, 1.58968638e-02,
        1.65080166e-02, 1.71296282e-02, 1.77616363e-02, 1.84039782e-02,
        1.90565913e-02, 1.97194129e-02, 2.03923802e-02, 2.10754305e-02,
        2.17685010e-02, 2.24715289e-02, 2.31844514e-02, 2.39072057e-02,
        2.46397290e-02, 2.53819584e-02, 2.61338313e-02, 2.68952850e-02,
        2.76662567e-02, 2.84466838e-02, 2.92365039e-02, 3.00356543e-02,
        3.08440727e-02, 3.16616967e-02, 3.24884640e-02, 3.33243125e-02,
        3.41691802e-02, 3.50230049e-02, 3.58857250e-02, 3.67572786e-02,
        3.76376041e-02, 3.85266402e-02, 3.94243253e-02, 4.03305983e-02,
        4.12453981e-02, 4.21686639e-02, 4.31003348e-02, 4.40403504e-02,
        4.49886500e-02, 4.59451736e-02, 4.69098609e-02, 4.78826522e-02,
        4.88634876e-02, 4.98523077e-02, 5.08490530e-02, 5.18536646e-02,
        5.28660834e-02, 5.38862507e-02, 5.49141079e-02, 5.59495968e-02,
        5.69926592e-02, 5.80432372e-02, 5.91012732e-02, 6.01667098e-02,
        6.12394897e-02, 6.23195560e-02, 6.34068518e-02, 6.45013207e-02,
        6.56029064e-02, 6.67115529e-02, 6.78272044e-02, 6.89498053e-02,
        7.00793003e-02, 7.12156343e-02, 7.23587527e-02, 7.35086008e-02,
        7.46651243e-02, 7.58282693e-02, 7.69979819e-02, 7.81742086e-02,
        7.93568962e-02, 8.05459917e-02, 8.17414424e-02, 8.29431959e-02,
        8.41511998e-02, 8.53654024e-02, 8.65857520e-02, 8.78121972e-02,
        8.90446868e-02, 9.02831702e-02, 9.15275966e-02, 9.27779159e-02,
        9.40340781e-02, 9.52960333e-02, 9.65637321e-02, 9.78371254e-02,
        9.91161643e-02, 1.00400800e-01, 1.01690984e-01, 1.02986669e-01,
        1.04287807e-01, 1.05594350e-01, 1.06906251e-01, 1.08223463e-01,
        1.09545939e-01, 1.10873633e-01, 1.12206499e-01, 1.13544491e-01,
        1.14887564e-01, 1.16235671e-01, 1.17588769e-01, 1.18946812e-01,
        1.20309756e-01, 1.21677557e-01, 1.23050171e-01, 1.24427554e-01,
        1.25809663e-01, 1.27196455e-01, 1.28587887e-01, 1.29983916e-01,
        1.31384501e-01, 1.32789599e-01, 1.34199168e-01, 1.35613168e-01,
        1.37031556e-01, 1.38454293e-01, 1.39881337e-01, 1.41312648e-01,
        1.42748186e-01, 1.44187911e-01, 1.45631783e-01, 1.47079764e-01,
        1.48531814e-01, 1.49987894e-01, 1.51447966e-01, 1.52911991e-01,
        1.54379932e-01, 1.55851751e-01, 1.57327410e-01, 1.58806872e-01,
        1.60290100e-01, 1.61777057e-01, 1.63267707e-01, 1.64762014e-01,
        1.66259941e-01, 1.67761453e-01, 1.69266514e-01, 1.70775089e-01,
        1.72287144e-01, 1.73802642e-01, 1.75321550e-01, 1.76843834e-01,
        1.78369458e-01, 1.79898390e-01, 1.81430597e-01, 1.82966043e-01,
        1.84504697e-01, 1.86046526e-01, 1.87591496e-01, 1.89139575e-01,
        1.90690732e-01, 1.92244934e-01, 1.93802149e-01, 1.95362346e-01,
        1.96925493e-01, 1.98491560e-01, 2.00060515e-01, 2.01632328e-01,
        2.03206968e-01, 2.04784405e-01, 2.06364609e-01, 2.07947550e-01,
        2.09533198e-01, 2.11121524e-01, 2.12712499e-01, 2.14306094e-01,
        2.15902280e-01, 2.17501027e-01, 2.19102309e-01, 2.20706096e-01,
        2.22312361e-01, 2.23921075e-01, 2.25532211e-01, 2.27145741e-01,
        2.28761639e-01, 2.30379877e-01, 2.32000427e-01, 2.33623265e-01,
        2.35248361e-01, 2.36875692e-01, 2.38505229e-01, 2.40136948e-01,
        2.41770821e-01, 2.43406825e-01, 2.45044932e-01, 2.46685118e-01,
        2.48327358e-01, 2.49971626e-01, 2.51617898e-01, 2.53266148e-01,
        2.54916353e-01, 2.56568488e-01, 2.58222528e-01, 2.59878451e-01,
        2.61536231e-01, 2.63195845e-01, 2.64857269e-01, 2.66520481e-01,
        2.68185455e-01, 2.69852171e-01, 2.71520603e-01, 2.73190730e-01,
        2.74862529e-01, 2.76535977e-01, 2.78211052e-01, 2.79887730e-01,
        2.81565991e-01, 2.83245812e-01, 2.84927171e-01, 2.86610046e-01,
        2.88294416e-01, 2.89980258e-01, 2.91667553e-01, 2.93356277e-01,
        2.95046410e-01, 2.96737931e-01, 2.98430819e-01, 3.00125053e-01,
        3.01820613e-01, 3.03517477e-01, 3.05215625e-01, 3.06915037e-01,
        3.08615693e-01, 3.10317572e-01, 3.12020653e-01, 3.13724919e-01,
        3.15430347e-01, 3.17136919e-01, 3.18844614e-01, 3.20553414e-01,
        3.22263299e-01, 3.23974248e-01, 3.25686244e-01, 3.27399267e-01,
        3.29113297e-01, 3.30828315e-01, 3.32544303e-01, 3.34261242e-01,
        3.35979112e-01, 3.37697895e-01, 3.39417573e-01, 3.41138127e-01,
        3.42859538e-01, 3.44581788e-01, 3.46304859e-01, 3.48028732e-01,
        3.49753389e-01, 3.51478811e-01, 3.53204982e-01, 3.54931882e-01,
        3.56659495e-01, 3.58387801e-01, 3.60116783e-01, 3.61846424e-01,
        3.63576705e-01, 3.65307609e-01, 3.67039118e-01, 3.68771215e-01,
        3.70503882e-01, 3.72237102e-01, 3.73970858e-01, 3.75705131e-01,
        3.77439905e-01, 3.79175162e-01, 3.80910885e-01, 3.82647057e-01,
        3.84383660e-01, 3.86120679e-01, 3.87858094e-01, 3.89595890e-01,
        3.91334049e-01, 3.93072555e-01, 3.94811390e-01, 3.96550537e-01,
        3.98289980e-01, 4.00029701e-01, 4.01769684e-01, 4.03509911e-01,
        4.05250367e-01, 4.06991034e-01, 4.08731895e-01, 4.10472933e-01,
        4.12214132e-01, 4.13955475e-01, 4.15696946e-01, 4.17438527e-01,
        4.19180201e-01, 4.20921952e-01, 4.22663764e-01, 4.24405619e-01,
        4.26147501e-01, 4.27889393e-01, 4.29631278e-01, 4.31373140e-01,
        4.33114961e-01, 4.34856726e-01, 4.36598416e-01, 4.38340016e-01,
        4.40081509e-01, 4.41822878e-01, 4.43564106e-01, 4.45305176e-01,
        4.47046072e-01, 4.48786776e-01, 4.50527272e-01, 4.52267543e-01,
        4.54007571e-01, 4.55747340e-01, 4.57486833e-01, 4.59226034e-01,
        4.60964923e-01, 4.62703486e-01, 4.64441704e-01, 4.66179561e-01,
        4.67917039e-01, 4.69654122e-01, 4.71390791e-01, 4.73127029e-01,
        4.74862820e-01, 4.76598146e-01, 4.78332989e-01, 4.80067332e-01,
        4.81801158e-01, 4.83534448e-01, 4.85267185e-01, 4.86999352e-01,
        4.88730931e-01, 4.90461903e-01, 4.92192252e-01, 4.93921959e-01,
        4.95651006e-01, 4.97379375e-01, 4.99107049e-01, 5.00834008e-01,
        5.02560235e-01, 5.04285711e-01, 5.06010418e-01, 5.07734338e-01,
        5.09457451e-01, 5.11179741e-01, 5.12901186e-01, 5.14621770e-01,
        5.16341473e-01, 5.18060276e-01, 5.19778160e-01, 5.21495106e-01,
        5.23211094e-01, 5.24926107e-01, 5.26640124e-01, 5.28353125e-01,
        5.30065092e-01,
    ]
    return analytical_results[index]
