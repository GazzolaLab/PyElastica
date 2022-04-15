import numpy as np
# Join the two rods
from elastica._linalg import (
    _batch_norm,
    _batch_matvec,
)



def get_connection_vector_straight_straight_rod(
    rod_one,
    rod_two,
    rod_one_start_idx,
    rod_one_end_idx,
):

    # Compute rod element positions
    rod_one_element_position = 0.5 * (
        rod_one.position_collection[..., 1:] + rod_one.position_collection[..., :-1]
    )
    rod_one_element_position = rod_one_element_position[:, rod_one_start_idx:rod_one_end_idx]
    rod_two_element_position = 0.5 * (
        rod_two.position_collection[..., 1:] + rod_two.position_collection[..., :-1]
    )

    # Lets get the distance between rod elements
    distance_vector_rod_one_to_rod_two = (
        rod_two_element_position - rod_one_element_position
    )
    distance_vector_rod_one_to_rod_two_norm = _batch_norm(
        distance_vector_rod_one_to_rod_two
    )
    distance_vector_rod_one_to_rod_two /= distance_vector_rod_one_to_rod_two_norm

    distance_vector_rod_two_to_rod_one = -distance_vector_rod_one_to_rod_two

    rod_one_direction_vec_in_material_frame = _batch_matvec(
        rod_one.director_collection[:,:,rod_one_start_idx:rod_one_end_idx], distance_vector_rod_one_to_rod_two
    )
    rod_two_direction_vec_in_material_frame = _batch_matvec(
        rod_two.director_collection, distance_vector_rod_two_to_rod_one
    )

    offset_btw_rods = distance_vector_rod_one_to_rod_two_norm - (
        rod_one.radius[rod_one_start_idx:rod_one_end_idx] + rod_two.radius
    )

    return (
        rod_one_direction_vec_in_material_frame,
        rod_two_direction_vec_in_material_frame,
        offset_btw_rods,
    )


