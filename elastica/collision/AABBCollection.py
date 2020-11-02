""" Axis Aligned Bounding Boxes for coarse collision detection
"""
import numpy as np
from elastica.utils import MaxDimension


class AABBCollection:
    def __init__(
        self,
        elemental_position_collection,
        dimension_collection,
        elements_per_aabb: int,
    ):
        """
        Doesn't differentiate tangent direction from the rest : potentially harmful as
        maybe you don't need to expand to radius amount in tangential direction

        :param position_collection:
        :param dimension_collection:
        :param elements_per_aabb:
        """
        n_positions = elemental_position_collection.shape[1]  # n_pos
        self.n_aabb = n_positions // elements_per_aabb
        self.n_aabb += (
            1 if (n_positions % elements_per_aabb) else 0
        )  # extra if not perfectly divisible
        self.elements_per_aabb = elements_per_aabb
        self.aabb = np.empty(
            (MaxDimension.value(), 2, self.n_aabb)
        )  # 2 for min and max

        assert dimension_collection.shape[1] == n_positions, "bad"

        # Initialize the aabbs posthaste
        self.update(elemental_position_collection, dimension_collection)

    @classmethod
    def make_from_aabb(cls, aabb_collection, scale_factor=4):
        # Make position collection and dimension collection arrays from aabb_collection
        # Wasted effort, but only once during construction
        n_aabb_from_lower_level = len(aabb_collection)
        elemental_position_collection = np.zeros(
            (MaxDimension.value(), n_aabb_from_lower_level)
        )
        # (r,r,dl) in (d1,d2,d3) coordinates
        dimension_collection = np.zeros((MaxDimension.value(), n_aabb_from_lower_level))

        for idx, aabb in enumerate(aabb_collection):
            # By design in the bottom level, there's only one AABB. So the last index is always 1
            # Also asserting herre
            assert aabb.n_aabb == 1, "Number of aabbs not 1"
            elemental_position_collection[..., idx] = 0.5 * (
                aabb.aabb[..., 0, 0] + aabb.aabb[..., 1, 0]
            )
            dimension_collection[..., idx] = 0.5 * (
                aabb.aabb[..., 1, 0] - aabb.aabb[..., 0, 0]
            )

        return cls(elemental_position_collection, dimension_collection, scale_factor)

    def _update(self, aabb_collection):
        # Updates internal state from another aabb

        """
        for i in range(self.n_aabb):
            start = i * self.elements_per_aabb
            stop = (i + 1) * self.elements_per_aabb

            self.aabb[..., 0, i] = np.amin(aabb_collection[..., 0, start:stop], axis=1)
            self.aabb[..., 1, i] = np.amax(aabb_collection[..., 1, start:stop], axis=1)
        """

        # Guaranteed that self.n_aabb = 1 and start, stop corresponds to aabb_collection
        # so directly do it
        temp = np.array([aabb.aabb[..., 0, 0] for aabb in aabb_collection])
        self.aabb[..., 0, 0] = np.amin(temp, axis=0)
        temp = np.array([aabb.aabb[..., 1, 0] for aabb in aabb_collection])
        self.aabb[..., 1, 0] = np.amax(temp, axis=0)

    def update(self, elemental_position_collection, dimension_collection):
        # Initialize the boxes
        for i in range(self.n_aabb):
            start = i * self.elements_per_aabb
            stop = (i + 1) * self.elements_per_aabb
            self.aabb[..., 0, i] = np.amin(
                elemental_position_collection[..., start:stop], axis=1
            ) - np.amax(dimension_collection[..., start:stop], axis=1)
            self.aabb[..., 1, i] = np.amax(
                elemental_position_collection[..., start:stop], axis=1
            ) + np.amax(dimension_collection[..., start:stop], axis=1)


def find_nearest_integer_square_root(x: int):
    from math import sqrt

    return round(sqrt(x))


class AABBHierarchy:
    """Simple hierarchy for handling cylinder collisions alone, meant for a rod"""

    def __init__(
        self, position_collection, dimension_collection, avg_n_dofs_in_final_level
    ):
        """
        scaling is always set to 4, so that theres' 1 major AABBCollection, then scaling_factor
        smaller AABBs, then scaling factor even smaller AABBs (which cover the elements
        basically)
        :param position_collection:
        """
        # Determine empirical scaling factor based on how many dofs are available in the
        # system
        n_positions = position_collection.shape[1]
        assert dimension_collection.shape[1] == n_positions, "bad"

        # First determine the number of levels according to 1,4,16,64...
        potential_n_aabbs_in_final_level = n_positions // avg_n_dofs_in_final_level
        potential_n_aabbs_in_final_level += (
            1 if n_positions % avg_n_dofs_in_final_level else 0
        )

        # nearest power of 4 that is less than the number
        n_levels_bound_below = np.int(
            np.floor(0.5 * np.log2(potential_n_aabbs_in_final_level))
        )
        n_levels_bound_above = np.int(
            np.ceil(0.5 * np.log2(potential_n_aabbs_in_final_level))
        )
        # Check which is the closest and use that as the number of levels

        # Else check differences
        if (4 ** n_levels_bound_above - potential_n_aabbs_in_final_level) > (
            potential_n_aabbs_in_final_level - 4 ** n_levels_bound_below
        ):
            self.n_levels = n_levels_bound_below + 1
        else:
            # If they are the same, then its an exact power of four which is good
            # the same code works
            self.n_levels = n_levels_bound_above + 1

        n_aabbs_in_final_level = 4 ** (self.n_levels - 1)
        self.avg_n_dofs_in_final_level = n_positions // n_aabbs_in_final_level
        # Needs to be distributed across aabbs
        self.extra_n_dofs_in_final_level = n_positions % n_aabbs_in_final_level

        assert self.extra_n_dofs_in_final_level < n_aabbs_in_final_level

        # self.n_aabbs_in_first_level = find_nearest_integer_square_root(n_positions)
        # self.avg_n_dofs_in_second_level = n_positions // self.n_aabbs_in_first_level
        # self.extra_n_dofs_in_second_level = n_positions % self.n_aabbs_in_first_level

        # We now need to construct all AABBs in the hierarchy sequentially from bottom
        # up
        self.aabb = []

        # Bottom level, already updated
        # Need to slice position collections and dimension_collections
        # elements_per_aabb is 1 as we are going to have the finest discretization
        stop = 0
        for i in range(n_aabbs_in_final_level - self.extra_n_dofs_in_final_level):
            start = stop
            stop = (i + 1) * self.avg_n_dofs_in_final_level
            self.aabb.append(
                AABBCollection(
                    position_collection[..., start:stop],
                    dimension_collection[..., start:stop],
                    self.avg_n_dofs_in_final_level,
                )
            )

        # Slice with an extra element (+1) to accommodate extra dofs
        # Done extra_n_dofs times
        for i in range(
            n_aabbs_in_final_level - self.extra_n_dofs_in_final_level,
            n_aabbs_in_final_level,
        ):
            start = stop
            stop = start + (self.avg_n_dofs_in_final_level + 1)
            self.aabb.append(
                AABBCollection(
                    position_collection[..., start:stop],
                    dimension_collection[..., start:stop],
                    self.avg_n_dofs_in_final_level + 1,
                )
            )

        # Now for all levels above, add an AABB taking cues from the previously constructed AABBs
        # Depends on the current level, start from 0
        count_elapsed_n_aabbs = 0
        for level in range(self.n_levels - 1):
            # Traverse bottom down
            # If 0, traverse n_level - 1 hierarchy, 1 traverse n_level -2 hierarchy and so on...
            # if levle = self.n_levels - 2, correct_level is 0
            correct_level = self.n_levels - level - 2
            n_aabbs_in_current_level = 4 ** (correct_level)
            n_aabbs_in_next_level = 4 ** (correct_level + 1)
            for i_aabb in range(n_aabbs_in_current_level):
                # Get start, stop indices of n_aabbs from previous level
                start_idx_in_aabb_list = i_aabb * 4 + count_elapsed_n_aabbs
                stop_idx_in_aabb_list = (i_aabb + 1) * 4 + count_elapsed_n_aabbs
                # n_aabb from prev levels
                assert (
                    stop_idx_in_aabb_list
                    <= count_elapsed_n_aabbs + n_aabbs_in_next_level
                )
                self.aabb.append(
                    AABBCollection.make_from_aabb(
                        self.aabb[start_idx_in_aabb_list:stop_idx_in_aabb_list]
                    )
                )
            count_elapsed_n_aabbs += n_aabbs_in_next_level

        # Add one for the middle level
        # self.aabb.append(AABBCollection(position_collection, dimension_collection, self.n_aabbs_in_first_level))

    def n_aabbs_at_level(self, i: int):
        assert i < self.n_levels
        return 4 ** (i)

    def update(self, position_collection, dimension_collection):
        # Update bottom level first, the first level entries
        n_aabbs_in_final_level = self.n_aabbs_at_level(self.n_levels - 1)
        stop = 0
        for i in range(n_aabbs_in_final_level - self.extra_n_dofs_in_final_level):
            start = stop
            stop = (i + 1) * self.avg_n_dofs_in_final_level
            self.aabb[i].update(
                position_collection[..., start:stop],
                dimension_collection[..., start:stop],
            )

        # Slice with an extra element (+1) to accommodate extra dofs
        # Done extra_n_dofs times
        for i in range(
            n_aabbs_in_final_level - self.extra_n_dofs_in_final_level,
            n_aabbs_in_final_level,
        ):
            start = stop
            stop = start + (self.avg_n_dofs_in_final_level + 1)
            self.aabb[i].update(
                position_collection[..., start:stop],
                dimension_collection[..., start:stop],
            )

        # Update all other levels middle level from bottom level's aabbs
        count_elapsed_n_aabbs = 0
        for level in range(self.n_levels - 2, 0 - 1, -1):
            # Traverse bottom down
            # If 0, traverse n_level - 1 hierarchy, 1 traverse n_level -2 hierarchy and so on...
            # if levle = self.n_levels - 2, correct_level is 0
            n_aabbs_in_current_level = self.n_aabbs_at_level(level)
            n_aabbs_in_next_level = self.n_aabbs_at_level(level + 1)
            for i_aabb in range(n_aabbs_in_current_level):
                # Get start, stop indices of n_aabbs from previous level
                start_idx_in_aabb_list = i_aabb * 4 + count_elapsed_n_aabbs
                stop_idx_in_aabb_list = (i_aabb + 1) * 4 + count_elapsed_n_aabbs
                self.aabb[
                    count_elapsed_n_aabbs + n_aabbs_in_next_level + i_aabb
                ]._update(self.aabb[start_idx_in_aabb_list:stop_idx_in_aabb_list])
            count_elapsed_n_aabbs += n_aabbs_in_next_level


def are_aabb_intersecting(first_aabb_collection, second_aabb_collection):
    return True
