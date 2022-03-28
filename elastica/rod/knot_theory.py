__doc__ = """
This script is for computing the link-writhe-twist (LWT) of a rod using the method from Klenin & Langowski 2000 paper.
Algorithms are adapted from section S2 of Charles et. al. PRL 2019 paper.

Following example cases includes computing LWT quantities to study the bifurcation:

- `Example case (PlectonemesCase) <https://github.com/GazzolaLab/PyElastica/blob/master/examples/RodContactCase/RodSelfContact/PlectonemesCase/plectoneme_case.py>`_
- `Example case (SolenoidCase) <https://github.com/GazzolaLab/PyElastica/blob/master/examples/RodContactCase/RodSelfContact/SolenoidsCase/solenoid_case.py>`_

The details discussion is included in `N Charles et. al. PRL (2019) <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.123.208003>`_.
"""
__all__ = [
    "KnotTheoryCompatibleProtocol",
    "KnotTheory",
    "compute_twist",
    "compute_link",
    "compute_writhe",
]

import sys

if sys.version_info.minor >= 8:
    # typing Protocol is introduced in python 3.8
    from typing import Protocol
elif sys.version_info.minor < 8:
    # Protocol is implemented in typing_extensions for previous pythons
    from typing_extensions import Protocol

from typing import Union

from numba import njit
import numpy as np

from elastica.rod.rod_base import RodBase
from elastica._linalg import _batch_norm, _batch_dot, _batch_cross


class KnotTheoryCompatibleProtocol(Protocol):
    """KnotTheoryCompatibleProtocol

    Required properties to use KnotTheory mixin
    """

    @property
    def position_collection(self) -> np.ndarray:
        ...

    @property
    def director_collection(self) -> np.ndarray:
        ...

    @property
    def radius(self) -> np.ndarray:
        ...

    @property
    def base_length(self) -> np.ndarray:
        ...


class KnotTheory:
    """
    This mixin should be used in RodBase-derived class that satisfies KnotCompatibleProtocol.
    The theory behind this module is based on  the method from Klenin & Langowski 2000 paper.

    KnotTheory can be mixed with any rod-class based on RodBase::

        class MyRod(RodBase, KnotTheory):
            def __init__(self):
                super().__init__()
        rod = MyRod(...)

        total_twist = rod.compute_twist()
        total_link = rod.compute_link()

    There are few alternative way of handling edge-condition in computing Link and Writhe.
    Here, we provide three methods: "next_tangent", "end_to_end", and "net_tangent".
    The default *type_of_additional_segment* is set to "next_tangent."

    ========================== =====================================
    type_of_additional_segment Description
    ========================== =====================================
    next_tangent               | Adds a two new point at the begining and end of the center line.
                               | Distance of these points are given in segment_length.
                               | Direction of these points are computed using the rod tangents at
                               | the begining and end.
    end_to_end                 | Adds a two new point at the begining and end of the center line.
                               | Distance of these points are given in segment_length.
                               | Direction of these points are computed using the rod node end
                               | positions.
    net_tangent                | Adds a two new point at the begining and end of the center line.
                               | Distance of these points are given in segment_length. Direction of
                               | these points are point wise avarege of nodes at the first and
                               | second half of the rod.
    ========================== =====================================

    """

    MIXIN_PROTOCOL = Union[RodBase, KnotTheoryCompatibleProtocol]

    def compute_twist(self: MIXIN_PROTOCOL):
        """
        See :ref:`api/rods:Knot Theory (Mixin)` for the detail.
        """
        total_twist, local_twist = compute_twist(
            self.position_collection[None, ...],
            self.director_collection[0][None, ...],
        )
        return total_twist[0]

    def compute_writhe(
        self: MIXIN_PROTOCOL, type_of_additional_segment: str = "next_tangent"
    ):
        """
        See :ref:`api/rods:Knot Theory (Mixin)` for the detail.

        Parameters
        ----------
        type_of_additional_segment : str
            Determines the method to compute new segments (elements) added to the rod.
            Valid inputs are "next_tangent", "end_to_end", "net_tangent", otherwise program uses the center line.
        """
        return compute_writhe(
            self.position_collection[None, ...],
            self.rest_lengths.sum(),
            type_of_additional_segment,
        )[0]

    def compute_link(
        self: MIXIN_PROTOCOL, type_of_additional_segment: str = "next_tangent"
    ):
        """
        See :ref:`api/rods:Knot Theory (Mixin)` for the detail.

        Parameters
        ----------
        type_of_additional_segment : str
            Determines the method to compute new segments (elements) added to the rod.
            Valid inputs are "next_tangent", "end_to_end", "net_tangent", otherwise program uses the center line.
        """
        print(self.rest_lengths.sum())
        return compute_link(
            self.position_collection[None, ...],
            self.director_collection[0][None, ...],
            self.radius[None, ...],
            self.rest_lengths.sum(),
            type_of_additional_segment,
        )[0]


def compute_twist(center_line, normal_collection):
    """
    Compute the twist of a rod, using center_line and normal collection.

    Methods used in this function is adapted from method 2a Klenin & Langowski 2000 paper.

    .. warning:: If center line is straight, although the normals of each element is pointing different direction computed twist will be zero.

    Typical runtime of this function is longer than simulation steps. While we provide a function to compute
    topological quantities at every timesteps, **we highly recommend** to compute LWT during the post-processing
    stage.::

        import elastica
        ...
        normal_collection = director_collection[:,0,...] # shape of director (time, 3, 3, n_elems)
        elastica.compute_twist(
            center_line,                                 # shape (time, 3, n_nodes)
            normal_collection                            # shape (time, 3, n_elems)
        )

    Parameters
    ----------
    center_line : numpy.ndarray
        3D (time, 3, n_nodes) array containing data with 'float' type.
        Time history of rod node positions.
    normal_collection : numpy.ndarray
        3D (time, 3, n_elems) array containing data with 'float' type.
        Time history of rod elements normal direction.

    Returns
    -------
    total_twist : numpy.ndarray
    local_twist : numpy.ndarray

    """
    # fmt: off
    # Format is turned off because I want the assertion message to display the line.
    assert center_line.shape[2] == normal_collection.shape[2] + 1, \
        "Please check the shape (axis-2) of center_line(n_node=n_elems+1) or normal_collection(n_elems)."
    assert center_line.shape[0] == normal_collection.shape[0], \
        "The number of timesteps (axis-0) must be equal"
    assert center_line.shape[1] == normal_collection.shape[1] == 3, \
        "The dimension (axis-1) must be 3"
    # fmt: on

    total_twist, local_twist = _compute_twist(center_line, normal_collection)

    return total_twist, local_twist


@njit(cache=True)
def _compute_twist(center_line, normal_collection):
    """
    Parameters
    ----------
    center_line : numpy.ndarray
        3D (time, 3, n_nodes) array containing data with 'float' type.
        Time history of rod node positions.
    normal_collection : numpy.ndarray
        3D (time, 3, n_elems) array containing data with 'float' type.
        Time history of rod elements normal direction.

    Returns
    -------
    total_twist : numpy.ndarray
    local_twist : numpy.ndarray
    """

    timesize, _, blocksize = center_line.shape

    total_twist = np.zeros((timesize))
    local_twist = np.zeros((timesize, blocksize - 2))  # Only consider interior nodes

    # compute s vector
    for k in range(timesize):
        # s is a secondary vector field.
        s = center_line[k, :, 1:] - center_line[k, :, :-1]
        # Compute tangents
        tangent = s / _batch_norm(s)

        # Compute the projection of normal collection (d1) on normal-binormal plane.
        projection_of_normal_collection = (
            normal_collection[k, :, :]
            - _batch_dot(tangent, normal_collection[k, :, :]) * tangent
        )
        projection_of_normal_collection /= _batch_norm(projection_of_normal_collection)

        # Eq 27 in Klenin & Langowski 2000
        # p is defined on interior nodes
        p = _batch_cross(s[:, :-1], s[:, 1:])
        p /= _batch_norm(p)

        # Compute the angle we need to turn d1 around s to get p
        # sign part tells whether d1 must be rotated ccw(+) or cw(-) around s
        alpha = np.sign(
            _batch_dot(
                _batch_cross(projection_of_normal_collection[:, :-1], p), s[:, :-1]
            )
        ) * np.arccos(_batch_dot(projection_of_normal_collection[:, :-1], p))

        gamma = np.sign(
            _batch_dot(
                _batch_cross(p, projection_of_normal_collection[:, 1:]), s[:, 1:]
            )
        ) * np.arccos(_batch_dot(projection_of_normal_collection[:, 1:], p))

        # An angle 1 is a full rotation, 0.5 is rotation by pi, 0.25 is pi/2 etc.
        alpha /= 2 * np.pi
        gamma /= 2 * np.pi
        twist_temp = alpha + gamma
        # Make sure twist is between (-1/2 to 1/2) as defined in pg 313 Klenin & Langowski 2000
        idx = np.where(twist_temp > 0.5)[0]
        twist_temp[idx] -= 1
        idx = np.where(twist_temp < -0.5)[0]
        twist_temp[idx] += 1

        # Check if there is any Nan. Nan's appear when rod tangents are parallel to each other.
        idx = np.where(np.isnan(twist_temp))[0]
        twist_temp[idx] = 0.0

        local_twist[k, :] = twist_temp
        total_twist[k] = np.sum(twist_temp)

    return total_twist, local_twist


def compute_writhe(center_line, segment_length, type_of_additional_segment):
    """
    This function computes the total writhe history of a rod.

    Equations used are from method 1a from Klenin & Langowski 2000 paper.

    Typical runtime of this function is longer than simulation steps. While we provide a function to compute
    topological quantities at every timesteps, **we highly recommend** to compute LWT during the post-processing
    stage.::

        import elastica
        ...
        elastica.compute_writhe(
            center_line,                               # shape (time, 3, n_nodes)
            segment_length,
            type_of_additional_segment="next_tangent"
        )

    Parameters
    ----------
    center_line : numpy.ndarray
        3D (time, 3, n_nodes) array containing data with 'float' type.
        Time history of rod node positions.
    segment_length : float
        Length of added segments.
    type_of_additional_segment : str
        Determines the method to compute new segments (elements) added to the rod.
        Valid inputs are "next_tangent", "end_to_end", "net_tangent", otherwise program uses the center line.

    Returns
    -------
    total_writhe : numpy.ndarray

    """
    # fmt: off
    # Format is turned off because I want the assertion message to display the line.
    assert center_line.shape[1] == 3, \
        "The dimension (axis-1) must be 3"
    # fmt: on

    center_line_with_added_segments, _, _ = _compute_additional_segment(
        center_line, segment_length, type_of_additional_segment
    )

    total_writhe = _compute_writhe(center_line_with_added_segments)

    return total_writhe


@njit(cache=True)
def _compute_writhe(center_line):
    """
    Parameters
    ----------
    center_line : numpy.ndarray
        3D (time, 3, n_nodes) array containing data with 'float' type.
        Time history of rod node positions.

    Returns
    -------
    total_writhe : numpy.ndarray
    """

    time, _, blocksize = center_line.shape

    omega_star = np.zeros((blocksize - 2, blocksize - 1))
    segment_writhe = np.zeros((blocksize - 2, blocksize - 1))
    total_writhe = np.zeros((time))

    # Compute the writhe between each pair first.
    for k in range(time):
        for i in range(blocksize - 2):
            for j in range(i + 1, blocksize - 1):

                point_one = center_line[k, :, i]
                point_two = center_line[k, :, i + 1]
                point_three = center_line[k, :, j]
                point_four = center_line[k, :, j + 1]

                # Eq 15 in Klenin & Langowski 2000
                r12 = point_two - point_one
                r34 = point_four - point_three
                r14 = point_four - point_one
                r13 = point_three - point_one
                r23 = point_three - point_two
                r24 = point_four - point_two

                n1 = np.cross(r13, r14)
                n1 /= np.linalg.norm(n1)
                n2 = np.cross(r14, r24)
                n2 /= np.linalg.norm(n2)
                n3 = np.cross(r24, r23)
                n3 /= np.linalg.norm(n3)
                n4 = np.cross(r23, r13)
                n4 /= np.linalg.norm(n4)

                # Eq 16a in Klenin & Langowski 2000
                omega_star[i, j] = (
                    np.arcsin(np.dot(n1, n2))
                    + np.arcsin(np.dot(n2, n3))
                    + np.arcsin(np.dot(n3, n4))
                    + np.arcsin(np.dot(n4, n1))
                )

                if np.isnan(omega_star[i, j]):
                    omega_star[i, j] = 0

                # Eq 16b in Klenin & Langowski 2000
                segment_writhe[i, j] = (
                    omega_star[i, j]
                    * np.sign(np.dot(np.cross(r34, r12), r13))
                    / (4 * np.pi)
                )

        # Compute the total writhe
        # Eq 13 in Klenin & Langowski 2000
        total_writhe[k] = 2 * np.sum(segment_writhe)

    return total_writhe


def compute_link(
    center_line: np.ndarray,
    normal_collection: np.ndarray,
    radius: np.ndarray,
    segment_length: float,
    type_of_additional_segment: str,
):
    """
    This function computes the total link history of a rod.

    Equations used are from method 1a from Klenin & Langowski 2000 paper.

    Typical runtime of this function is longer than simulation steps. While we provide a function to compute
    topological quantities at every timesteps, **we highly recommend** to compute LWT during the post-processing
    stage.::

        import elastica
        ...
        normal_collection = director_collection[:,0,...] # shape of director (time, 3, 3, n_elems)
        elastica.compute_link(
            center_line,                                 # shape (time, 3, n_nodes)
            normal_collection,                           # shape (time 3, n_elems)
            radius,                                      # shape (time, n_elems)
            segment_length,
            type_of_additional_segment="next_tangent"
        )

    Parameters
    ----------
    center_line : numpy.ndarray
        3D (time, 3, n_nodes) array containing data with 'float' type.
        Time history of rod node positions.
    normal_collection : numpy.ndarray
        3D (time, 3, n_elems) array containing data with 'float' type.
        Time history of rod elements normal direction.
    radius : numpy.ndarray
        2D (time, n_elems) array containing data with 'float' type.
        Time history of rod element radius.
    segment_length : float
        Length of added segments.
    type_of_additional_segment : str
        Determines the method to compute new segments (elements) added to the rod.
        Valid inputs are "next_tangent", "end_to_end", "net_tangent", otherwise program uses the center line.

    Returns
    -------
    total_link : numpy.ndarray

    """
    # fmt: off
    # Format is turned off because I want the assertion message to display the line.
    assert center_line.shape[2] == normal_collection.shape[2] + 1 == radius.shape[1] + 1, \
        "Please check the shape (axis-2) of center_line(n_node=n_elems+1) or normal_collection(n_elems)."
    assert center_line.shape[0] == normal_collection.shape[0] == radius.shape[0], \
        "The number of timesteps (axis-0) must be equal"
    assert center_line.shape[1] == normal_collection.shape[1] == 3, \
        "The dimension (axis-1) for center_line and normal_collection must be 3"
    # fmt: on

    # Compute auxiliary line
    auxiliary_line = _compute_auxiliary_line(center_line, normal_collection, radius)

    # Add segments at the beginning and end of the rod center line and auxiliary line.
    (
        center_line_with_added_segments,
        beginning_direction,
        end_direction,
    ) = _compute_additional_segment(
        center_line, segment_length, type_of_additional_segment
    )
    auxiliary_line_with_added_segments = _compute_auxiliary_line_added_segments(
        beginning_direction, end_direction, auxiliary_line, segment_length
    )

    """
    Total link of a rod is computed using the method 1a from Klenin & Langowski 2000
    """
    total_link = _compute_link(
        center_line_with_added_segments, auxiliary_line_with_added_segments
    )

    return total_link


@njit(cache=True)
def _compute_auxiliary_line(center_line, normal_collection, radius):
    """
    This function computes the auxiliary line using rod center line and normal collection.

    Parameters
    ----------
    center_line : numpy.ndarray
        3D (time, 3, n_nodes) array containing data with 'float' type.
        Time history of rod node positions.
    normal_collection : numpy.ndarray
        3D (time, 3, n_elems) array containing data with 'float' type.
        Time history of rod elements normal direction.
    radius : numpy.ndarray
        2D (time, n_elems) array containing data with 'float' type.
        Time history of rod element radius.

    Returns
    -------
    auxillary_line : numpy.ndarray

    """
    time, _, blocksize = center_line.shape
    auxiliary_line = np.zeros(center_line.shape)
    projection_of_normal_collection = np.zeros((3, blocksize))
    radius_on_nodes = np.zeros((blocksize))

    for i in range(time):
        tangent = center_line[i, :, 1:] - center_line[i, :, :-1]
        tangent /= _batch_norm(tangent)
        # Compute the projection of normal collection (d1) on normal-binormal plane.
        projection_of_normal_collection_temp = (
            normal_collection[i, :, :]
            - _batch_dot(tangent, normal_collection[i, :, :]) * tangent
        )
        projection_of_normal_collection_temp /= _batch_norm(
            projection_of_normal_collection_temp
        )

        # First node have the same direction with second node. They share the same element.
        # TODO: Instead of this maybe we should use the trapezoidal rule or averaging operator for normal and radius.
        projection_of_normal_collection[:, 0] = projection_of_normal_collection_temp[
            :, 0
        ]
        projection_of_normal_collection[:, 1:] = projection_of_normal_collection_temp[:]
        radius_on_nodes[0] = radius[i, 0]
        radius_on_nodes[1:] = radius[i, :]

        auxiliary_line[i, :, :] = (
            radius_on_nodes * projection_of_normal_collection + center_line[i, :, :]
        )

    return auxiliary_line


@njit(cache=True)
def _compute_link(center_line, auxiliary_line):
    """

    Parameters
    ----------
    center_line : numpy.ndarray
        3D (time, 3, n_nodes) array containing data with 'float' type.
        Time history of rod node positions.
    auxiliary_line : numpy.ndarray
        3D (time, 3, n_nodes) array containing data with 'float' type.
        Time history of auxiliary line.

    Returns
    -------
    total_link : numpy.ndarray

    """
    timesize, _, blocksize_center_line = center_line.shape
    blocksize_auxiliary_line = auxiliary_line.shape[-1]

    omega_star = np.zeros((blocksize_center_line - 1, blocksize_auxiliary_line - 1))
    segment_link = np.zeros((blocksize_center_line - 1, blocksize_auxiliary_line - 1))
    total_link = np.zeros((timesize))

    # Compute the writhe between each pair first.
    for k in range(timesize):
        for i in range(blocksize_center_line - 1):
            for j in range(blocksize_auxiliary_line - 1):

                point_one = center_line[k, :, i]
                point_two = center_line[k, :, i + 1]
                point_three = auxiliary_line[k, :, j]
                point_four = auxiliary_line[k, :, j + 1]

                # Eq 15 in Klenin & Langowski 2000
                r12 = point_two - point_one
                r34 = point_four - point_three
                r14 = point_four - point_one
                r13 = point_three - point_one
                r23 = point_three - point_two
                r24 = point_four - point_two

                n1 = np.cross(r13, r14)
                n1 /= np.linalg.norm(n1)
                n2 = np.cross(r14, r24)
                n2 /= np.linalg.norm(n2)
                n3 = np.cross(r24, r23)
                n3 /= np.linalg.norm(n3)
                n4 = np.cross(r23, r13)
                n4 /= np.linalg.norm(n4)

                # Eq 16a in Klenin & Langowski 2000
                omega_star[i, j] = (
                    np.arcsin(np.dot(n1, n2))
                    + np.arcsin(np.dot(n2, n3))
                    + np.arcsin(np.dot(n3, n4))
                    + np.arcsin(np.dot(n4, n1))
                )

                if np.isnan(omega_star[i, j]):
                    omega_star[i, j] = 0

                # Eq 16b in Klenin & Langowski 2000
                segment_link[i, j] = (
                    omega_star[i, j]
                    * np.sign(np.dot(np.cross(r34, r12), r13))
                    / (4 * np.pi)
                )

        # Compute the total writhe
        # Eq 6 in Klenin & Langowski 2000
        # Unlike the writhe, link computed using two curves so we do not multiply by 2
        total_link[k] = np.sum(segment_link)

    return total_link


@njit(cache=True)
def _compute_auxiliary_line_added_segments(
    beginning_direction, end_direction, auxiliary_line, segment_length
):
    """
    This code is for computing position of added segments to the auxiliary line.

    Parameters
    ----------
    beginning_direction : numpy.ndarray
        2D (time, 3) array containing data with 'float' type.
        Time history of center line tangent at the beginning.
    end_direction : numpy.ndarray
        2D (time, 3) array containing data with 'float' type.
        Time history of center line tangent at the end.
    auxiliary_line : numpy.ndarray
        3D (time, 3, n_nodes) array containing data with 'float' type.
        Time history of auxiliary line.
    segment_length : float
        Length of added segments.

    Returns
    -------
    new_auxiliary_line : numpy.ndarray

    """
    timesize, _, blocksize = auxiliary_line.shape

    new_auxiliary_line = np.zeros((timesize, 3, blocksize + 2))

    new_auxiliary_line[:, :, 1:-1] = auxiliary_line

    new_auxiliary_line[:, :, 0] = (
        auxiliary_line[:, :, 0] + beginning_direction * segment_length
    )

    new_auxiliary_line[:, :, -1] = (
        auxiliary_line[:, :, -1] + end_direction * segment_length
    )

    return new_auxiliary_line


@njit(cache=True)
def _compute_additional_segment(
    center_line, segment_length, type_of_additional_segment
):
    """
    This function adds two points at the end of center line. Distance from the center line is given by segment_length.
    Direction from center line to the new point locations can be computed using 3 methods, which can be selected by
    type. For more details section S2 of Charles et. al. PRL 2019 paper.

    next_tangent:
        This function adds a two new point at the begining and end of the center line. Distance of these points are
        given in segment_length. Direction of these points are computed using the rod tangents at the begining and end.
    end_to_end:
        This function adds a two new point at the begining and end of the center line. Distance of these points are
        given in segment_length. Direction of these points are computed using the rod node end positions.
    net_tangent:
        This function adds a two new point at the begining and end of the center line. Distance of these points are
        given in segment_length. Direction of these points are point wise avarege of nodes at the first and second half
        of the rod.

    Parameters
    ----------
    center_line : numpy.ndarray
        3D (time, 3, n_nodes) array containing data with 'float' type.
        Time history of rod node positions.
    segment_length : float
        Length of added segments.
    type_of_additional_segment : str
        Determines the method to compute new segments (elements) added to the rod.
        Valid inputs are "next_tangent", "end_to_end", "net_tangent". If None, returns the center line.

    Returns
    -------
    center_line : numpy.ndarray
    beginning_direction : numpy.ndarray
    end_direction : numpy.ndarray

    """
    if type_of_additional_segment is None:
        beginning_direction = np.zeros((center_line.shape[0], 3))
        end_direction = np.zeros((center_line.shape[0], 3))
        return center_line, beginning_direction, end_direction

    timesize, _, blocksize = center_line.shape
    new_center_line = np.zeros(
        (timesize, 3, blocksize + 2)
    )  # +2 is for added two new points
    beginning_direction = np.zeros((timesize, 3))
    end_direction = np.zeros((timesize, 3))

    if type_of_additional_segment == "next_tangent":
        for i in range(timesize):
            # Direction of the additional point at the beginning of the rod
            direction_of_rod_begin = center_line[i, :, 0] - center_line[i, :, 1]
            direction_of_rod_begin /= np.linalg.norm(direction_of_rod_begin)

            # Direction of the additional point at the end of the rod
            direction_of_rod_end = center_line[i, :, -1] - center_line[i, :, -2]
            direction_of_rod_end /= np.linalg.norm(direction_of_rod_end)
    elif type_of_additional_segment == "end_to_end":
        for i in range(timesize):
            # Direction of the additional point at the beginning of the rod
            direction_of_rod_begin = center_line[i, :, 0] - center_line[i, :, -1]
            direction_of_rod_begin /= np.linalg.norm(direction_of_rod_begin)

            # Direction of the additional point at the end of the rod
            direction_of_rod_end = -direction_of_rod_begin
    elif type_of_additional_segment == "net_tangent":
        for i in range(timesize):
            # Direction of the additional point at the beginning of the rod
            n_nodes_begin = int(np.floor(blocksize / 2))
            average_begin = (
                np.sum(center_line[i, :, :n_nodes_begin], axis=1) / n_nodes_begin
            )
            n_nodes_end = int(np.ceil(blocksize / 2))
            average_end = np.sum(center_line[i, :, n_nodes_end:], axis=1) / (
                blocksize - n_nodes_end + 1
            )
            direction_of_rod_begin = average_begin - average_end
            direction_of_rod_begin /= np.linalg.norm(direction_of_rod_begin)
            direction_of_rod_end = -direction_of_rod_begin
    else:
        raise NotImplementedError("unavailable type_of_additional_segment is given")

    # Compute new centerline and beginning/end direction
    for i in range(timesize):
        first_point = center_line[i, :, 0] + segment_length * direction_of_rod_begin
        last_point = center_line[i, :, -1] + segment_length * direction_of_rod_end

        new_center_line[i, :, 1:-1] = center_line[i, :, :]
        new_center_line[i, :, 0] = first_point
        new_center_line[i, :, -1] = last_point

        beginning_direction[i, :] = direction_of_rod_begin
        end_direction[i, :] = direction_of_rod_end

    return new_center_line, beginning_direction, end_direction
