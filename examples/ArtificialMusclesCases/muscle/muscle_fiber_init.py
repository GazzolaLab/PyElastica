import numpy as np
from elastica._linalg import (
    _batch_cross,
    _batch_norm,
    _batch_dot,
    _batch_matvec,
    _batch_matmul,
)

from elastica.rod.knot_theory import _compute_writhe, _compute_twist


def get_fiber_geometry(
    n_elem,
    radius_list,
    start_position,
    direction,
    normal_m,
    binormal_m,
    offset_angles,
    length,
    k_list,
    initial_link_per_length,
    CCW=(False, False, False),
):

    # Compute normal, binormal and director collection
    # We need to compute directors because helix is not a straight rod and it has a complex shape.
    position_collection = np.zeros((3, n_elem + 1))
    director_collection = np.zeros((3, 3, n_elem))
    normal_collection = np.zeros((3, n_elem))
    binormal_collection = np.zeros((3, n_elem))

    r_m, r_s, r_h = radius_list
    k_m, k_s, k_h = k_list
    offset_m, offset_s, offset_h = offset_angles
    k_m = (2 * int(CCW[0]) - 1) * k_m
    k_s = (2 * int(CCW[1]) - 1) * k_s
    k_h = (2 * int(CCW[2]) - 1) * k_h
    h = 1 / (2 * np.pi * k_m)

    curve_angle = np.arange(0, (n_elem + 1)) / (n_elem + 1)

    for i in range(n_elem + 1):
        theta_m = curve_angle[i] * 2 * np.pi * k_m * length
        theta_s = curve_angle[i] * 2 * np.pi * k_s * length
        theta_h = curve_angle[i] * 2 * np.pi * k_h * length

        normal_s = (-np.cos(theta_m + offset_m) * normal_m) - (
            np.sin(theta_m + offset_m) * binormal_m
        )

        dnormal_s = (
            np.sin(theta_m + offset_m) * normal_m
            - np.cos(theta_m + offset_m) * binormal_m
        )
        ddnormal_s = -normal_s

        binormal_s = (
            (h * np.sin(theta_m + offset_m) * normal_m)
            - (h * np.cos(theta_m + offset_m) * binormal_m)
            + r_m * direction
        ) / np.sqrt(r_m ** 2 + h ** 2)

        dbinormal_s = (
            h * np.cos(theta_m + offset_m) * normal_m
            + h * np.sin(theta_m + offset_m) * binormal_m
        ) / np.sqrt(r_m ** 2 + h ** 2)

        ddbinormal_s = (
            -h * np.sin(theta_m + offset_m) * normal_m
            + h * np.cos(theta_m + offset_m) * binormal_m
        ) / np.sqrt(r_m ** 2 + h ** 2)

        t_s = (
            (-r_m * np.sin(theta_m + offset_m) * normal_m)
            + (r_m * np.cos(theta_m + offset_m) * binormal_m)
            + h * direction
        )

        dt_s = (
            -r_m * np.cos(theta_m + offset_m) * normal_m
            - r_m * np.sin(theta_m + offset_m) * binormal_m
        )

        t_h = (
            t_s
            + r_s * (dnormal_s + k_s * binormal_s) * np.cos((theta_s) + offset_s)
            + r_s * (dbinormal_s - k_s * normal_s) * np.sin((theta_s) + offset_s)
        )
        dt_h = (
            dt_s
            + r_s
            * (ddnormal_s + 2 * k_s * dbinormal_s - (k_s ** 2) * normal_s)
            * np.cos((theta_s) + offset_s)
            + r_s
            * (ddbinormal_s - 2 * k_s * dnormal_s - (k_s ** 2) * binormal_s)
            * np.sin((theta_s) + offset_s)
        )
        normal_h = dt_h / np.linalg.norm(dt_h)
        binormal_h = np.cross(t_h, normal_h) / np.linalg.norm(t_h)

        position_collection[..., i] = (
            start_position
            + r_m
            * (
                np.cos(theta_m + offset_m) * normal_m
                + np.sin(theta_m + offset_m) * binormal_m
            )
            + r_s
            * (
                np.cos((theta_s) + offset_s) * normal_s
                + np.sin((theta_s) + offset_s) * binormal_s
            )
            + r_h
            * (
                np.cos((theta_h) + offset_h) * normal_h
                + np.sin((theta_h) + offset_h) * binormal_h
            )
            + abs(h * theta_m) * direction
        )

    start = position_collection[..., 0]

    # Compute rod tangents using positions
    position_for_difference = position_collection
    position_diff = position_for_difference[..., 1:] - position_for_difference[..., :-1]
    rest_lengths = _batch_norm(position_diff)
    tangents = position_diff / rest_lengths

    for i in range(n_elem):
        # Compute the normal vector at each element. Since we allow helix radius to vary, we need to compute
        # vectors creating normal for each element.
        normal_collection[0, i] = -tangents[1, i]
        normal_collection[1, i] = tangents[0, i]
        normal_collection[..., i] /= np.linalg.norm(normal_collection[..., i])

        binormal_collection[..., i] = np.cross(
            tangents[..., i], normal_collection[..., i]
        )
        director_collection[..., i] = np.vstack(
            (
                normal_collection[..., i],
                binormal_collection[..., i],
                tangents[..., i],
            )
        )

    fiber_length = rest_lengths.sum()
    centerline = position_collection.reshape(
        (1, position_collection.shape[0], position_collection.shape[-1])
    )
    coil_writhe = _compute_writhe(centerline)
    normal_collection_twist_calc = normal_collection.reshape(
        (1, normal_collection.shape[0], normal_collection.shape[-1])
    )
    total_intrinsic_twist, local_intrinsic_twist = _compute_twist(
        centerline, normal_collection_twist_calc
    )
    coil_twist = (
        initial_link_per_length * length - coil_writhe[0] - total_intrinsic_twist[0]
    )
    print("Writhe = " + str(coil_writhe[0]))
    print("Link = " + str(initial_link_per_length * length))
    print("Intrinsic Twist = " + str(total_intrinsic_twist[0]))
    print("Coil Twist = " + str(coil_twist))

    rotation_angle = (coil_twist * 2 * np.pi) * (curve_angle / curve_angle[-1])

    def _get_rotation_matrix2(scale, axis_collection):
        blocksize = axis_collection.shape[1]
        rot_mat = np.empty((3, 3, blocksize))

        for k in range(blocksize):
            v0 = axis_collection[0, k]
            v1 = axis_collection[1, k]
            v2 = axis_collection[2, k]

            theta = np.sqrt(v0 * v0 + v1 * v1 + v2 * v2)

            v0 /= theta + 1e-14
            v1 /= theta + 1e-14
            v2 /= theta + 1e-14

            theta *= scale[k]
            u_prefix = np.sin(theta)
            u_sq_prefix = 1.0 - np.cos(theta)

            rot_mat[0, 0, k] = 1.0 - u_sq_prefix * (v1 * v1 + v2 * v2)
            rot_mat[1, 1, k] = 1.0 - u_sq_prefix * (v0 * v0 + v2 * v2)
            rot_mat[2, 2, k] = 1.0 - u_sq_prefix * (v0 * v0 + v1 * v1)

            rot_mat[0, 1, k] = u_prefix * v2 + u_sq_prefix * v0 * v1
            rot_mat[1, 0, k] = -u_prefix * v2 + u_sq_prefix * v0 * v1
            rot_mat[0, 2, k] = -u_prefix * v1 + u_sq_prefix * v0 * v2
            rot_mat[2, 0, k] = u_prefix * v1 + u_sq_prefix * v0 * v2
            rot_mat[1, 2, k] = u_prefix * v0 + u_sq_prefix * v1 * v2
            rot_mat[2, 1, k] = -u_prefix * v0 + u_sq_prefix * v1 * v2

        return rot_mat

    def _rotate2(director_collection, scale, axis_collection):
        """
        Does alibi rotations
        https://en.wikipedia.org/wiki/Rotation_matrix#Ambiguities

        Parameters
        ----------
        director_collection
        scale
        axis_collection

        Returns
        -------

        TODO Finish documentation
        """
        return _batch_matmul(
            director_collection, _get_rotation_matrix2(scale, axis_collection)
        )

    director_collection[:] = _rotate2(
        director_collection, scale=rotation_angle, axis_collection=tangents
    )
    normal_collection_twist_calc2 = director_collection[0, :, :].reshape(
        (1, normal_collection.shape[0], normal_collection.shape[-1])
    )
    post_rotation_twist, local_new_twist = _compute_twist(
        centerline, normal_collection_twist_calc2
    )
    print("New Twist = " + str(post_rotation_twist[0]))

    assert (
        abs(post_rotation_twist[0] - coil_twist - total_intrinsic_twist[0]) < 1.0
    ), "Not Enough Elements To Capture Twist"

    return fiber_length, start, position_collection, director_collection


def get_tapered_geometry(
    n_elem,
    start_radius_list,
    taper_slope_list,
    start_position,
    direction_m,
    normal_m,
    binormal_m,
    offset_list,
    length,
    turns_per_length_list,
    initial_link_per_length,
    CCW=(False, False),
):

    # Compute normal, binormal and director collection
    # We need to compute directors because helix is not a straight rod and it has a complex shape.
    position_collection = np.zeros((3, n_elem + 1))
    director_collection = np.zeros((3, 3, n_elem))
    normal_collection = np.zeros((3, n_elem))
    binormal_collection = np.zeros((3, n_elem))

    start_radius_m, start_radius_s = start_radius_list
    taper_slope_m, taper_slope_s = taper_slope_list
    offset_m, offset_s = offset_list
    turns_per_length_m, turns_per_length_s = turns_per_length_list
    turns_per_length_m = (2 * int(CCW[0]) - 1) * turns_per_length_m
    turns_per_length_s = (2 * int(CCW[1]) - 1) * turns_per_length_s

    curve_angle = np.arange(0, (n_elem + 1)) * length / (n_elem + 1)

    for i in range(n_elem + 1):
        theta = curve_angle[i]
        k_m = 2 * np.pi * turns_per_length_m
        k_s = 2 * np.pi * turns_per_length_s
        r_m = start_radius_m + (theta * taper_slope_m)
        r_s = start_radius_s + (theta * taper_slope_s)

        # r_m* (np.cos(k_m*theta + offset_m) * normal_m+ np.sin(k_m*theta + offset_m) * binormal_m)+  theta * direction_m
        t_m = direction_m + r_m * k_m * (
            (-np.sin(k_m * theta + offset_m) * normal_m)
            + (np.cos(k_m * theta + offset_m) * binormal_m)
        )
        +taper_slope_m * (
            (np.cos(k_m * theta + offset_m) * normal_m)
            + (np.sin(k_m * theta + offset_m) * binormal_m)
        )

        dtdt_m = (
            -r_m
            * (k_m ** 2)
            * (
                (np.cos(k_m * theta + offset_m) * normal_m)
                + (np.sin(k_m * theta + offset_m) * binormal_m)
            )
        )
        +2 * taper_slope_m * k_m * (
            (-np.sin(k_m * theta + offset_m) * normal_m)
            + (np.cos(k_m * theta + offset_m) * binormal_m)
        )

        normal_s = dtdt_m / np.linalg.norm(dtdt_m)
        direction_s = t_m / np.linalg.norm(t_m)
        binormal_s = np.cross(direction_s, normal_s)

        position_collection[..., i] = (
            start_position
            + r_m
            * (
                np.cos(k_m * theta + offset_m) * normal_m
                + np.sin(k_m * theta + offset_m) * binormal_m
            )
            + theta * direction_m
            + r_s
            * (
                np.cos(k_s * theta + offset_s) * normal_s
                + np.sin(k_s * theta + offset_s) * binormal_s
            )
        )

    start = position_collection[..., 0]

    # Compute rod tangents using positions
    position_for_difference = position_collection
    position_diff = position_for_difference[..., 1:] - position_for_difference[..., :-1]
    rest_lengths = _batch_norm(position_diff)
    tangents = position_diff / rest_lengths

    for i in range(n_elem):
        # Compute the normal vector at each element. Since we allow helix radius to vary, we need to compute
        # vectors creating normal for each element.
        normal_collection[0, i] = -tangents[1, i]
        normal_collection[1, i] = tangents[0, i]
        normal_collection[..., i] /= np.linalg.norm(normal_collection[..., i])

        binormal_collection[..., i] = np.cross(
            tangents[..., i], normal_collection[..., i]
        )
        director_collection[..., i] = np.vstack(
            (
                normal_collection[..., i],
                binormal_collection[..., i],
                tangents[..., i],
            )
        )

    fiber_length = rest_lengths.sum()
    centerline = position_collection.reshape(
        (1, position_collection.shape[0], position_collection.shape[-1])
    )
    coil_writhe = _compute_writhe(centerline)
    normal_collection_twist_calc = normal_collection.reshape(
        (1, normal_collection.shape[0], normal_collection.shape[-1])
    )
    total_intrinsic_twist, local_intrinsic_twist = _compute_twist(
        centerline, normal_collection_twist_calc
    )
    coil_twist = (
        initial_link_per_length * length - coil_writhe[0] - total_intrinsic_twist[0]
    )
    # print("Writhe = "+str(coil_writhe[0]))
    # print("Link = "+str(initial_link_per_length*length))
    # print("Intrinsic Twist = "+str(total_intrinsic_twist[0]))
    # print("Coil Twist = "+str(coil_twist))

    rotation_angle = (coil_twist * 2 * np.pi) * (curve_angle / curve_angle[-1])

    def _get_rotation_matrix2(scale, axis_collection):
        blocksize = axis_collection.shape[1]
        rot_mat = np.empty((3, 3, blocksize))

        for k in range(blocksize):
            v0 = axis_collection[0, k]
            v1 = axis_collection[1, k]
            v2 = axis_collection[2, k]

            theta = np.sqrt(v0 * v0 + v1 * v1 + v2 * v2)

            v0 /= theta + 1e-14
            v1 /= theta + 1e-14
            v2 /= theta + 1e-14

            theta *= scale[k]
            u_prefix = np.sin(theta)
            u_sq_prefix = 1.0 - np.cos(theta)

            rot_mat[0, 0, k] = 1.0 - u_sq_prefix * (v1 * v1 + v2 * v2)
            rot_mat[1, 1, k] = 1.0 - u_sq_prefix * (v0 * v0 + v2 * v2)
            rot_mat[2, 2, k] = 1.0 - u_sq_prefix * (v0 * v0 + v1 * v1)

            rot_mat[0, 1, k] = u_prefix * v2 + u_sq_prefix * v0 * v1
            rot_mat[1, 0, k] = -u_prefix * v2 + u_sq_prefix * v0 * v1
            rot_mat[0, 2, k] = -u_prefix * v1 + u_sq_prefix * v0 * v2
            rot_mat[2, 0, k] = u_prefix * v1 + u_sq_prefix * v0 * v2
            rot_mat[1, 2, k] = u_prefix * v0 + u_sq_prefix * v1 * v2
            rot_mat[2, 1, k] = -u_prefix * v0 + u_sq_prefix * v1 * v2

        return rot_mat

    def _rotate2(director_collection, scale, axis_collection):
        """
        Does alibi rotations
        https://en.wikipedia.org/wiki/Rotation_matrix#Ambiguities

        Parameters
        ----------
        director_collection
        scale
        axis_collection

        Returns
        -------

        TODO Finish documentation
        """
        return _batch_matmul(
            director_collection, _get_rotation_matrix2(scale, axis_collection)
        )
        # return _batch_matmul(
        #     _get_rotation_matrix2(scale, axis_collection), director_collection
        # )

    director_collection[:] = _rotate2(
        director_collection, scale=rotation_angle, axis_collection=tangents
    )
    normal_collection_twist_calc2 = director_collection[0, :, :].reshape(
        (1, normal_collection.shape[0], normal_collection.shape[-1])
    )
    post_rotation_twist, local_new_twist = _compute_twist(
        centerline, normal_collection_twist_calc2
    )
    # print("New Twist = "+str(post_rotation_twist[0]))

    assert (
        abs(post_rotation_twist[0] - coil_twist - total_intrinsic_twist[0]) < 1.0
    ), "Not Enough Elements To Capture Twist"

    return fiber_length, start, position_collection, director_collection
