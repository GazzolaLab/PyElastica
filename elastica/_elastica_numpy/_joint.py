__doc__ = """ Joint between rods module of Elastica Numpy implementation """

import numpy as np
from elastica.utils import Tolerance, MaxDimension
from elastica.joint import FreeJoint


class ExternalContact(FreeJoint):
    """
    Assumes that the second entity is a rigid body for now, can be
    changed at a later time

    Most of the cylinder-cylinder contact SHOULD be implemented
    as given in this paper:
    http://larochelle.sdsmt.edu/publications/2005-2009/Collision%20Detection%20of%20Cylindrical%20Rigid%20Bodies%20Using%20Line%20Geometry.pdf

    but, it isn't (the elastica-cpp kernels are implented)!
    This is maybe to speed-up the kernel, but it's
    potentially dangerous as it does not deal with "end" conditions
    correctly.
    """

    def __init__(self, k, nu):
        super().__init__(k, nu)
        # 0 is min, 1 is max
        self.aabb_rod = np.empty((MaxDimension.value(), 2))
        self.aabb_cylinder = np.empty((MaxDimension.value(), 2))

    # Should be a free function, can be jitted
    def __aabbs_not_intersecting(self, aabb_one, aabb_two):
        # FIXME : Not memory friendly, Not early exit
        first_max_second_min = aabb_one[..., 1] < aabb_two[..., 0]
        first_min_second_max = aabb_two[..., 1] < aabb_one[..., 0]
        # Returns true if aabbs not intersecting, else if it intersects returns false
        return np.any(np.logical_or(first_max_second_min, first_min_second_max))

    def __find_min_dist(self, x1, e1, x2, e2):
        """Assumes x2, e2 is one elment for now

        Will definitely get speedup from numba
        """

        def _batch_inner(first, second):
            return np.einsum("ij,ij->j", first, second)

        def _batch_inner_spec(first, second):
            return np.einsum("ij,i->j", first, second)

        # Compute distances for all cases
        def _difference_impl_for_multiple_st(ax1, ae1, at1, ax2, ae2, as2):
            return (
                ax2.reshape(3, 1, -1)
                + np.einsum("i,jk->ijk", ae2, as2)
                - ax1.reshape(3, 1, -1)
                - np.einsum("ik,jk->ijk", ae1, at1)
            )

        def _difference_impl(ax1, ae1, at1, ax2, ae2, as2):
            return (
                ax2.reshape(3, -1)
                + np.einsum("i,j->ij", ae2, as2)
                - ax1.reshape(3, -1)
                - np.einsum("ik,k->ik", ae1, at1)
            )

        e1e1 = _batch_inner(e1, e1)
        e1e2 = _batch_inner_spec(e1, e2)
        e2e2 = np.inner(e2, e2)

        x1e1 = _batch_inner(x1, e1)
        x1e2 = _batch_inner_spec(x1, e2)
        x2e1 = _batch_inner_spec(e1, x2)
        x2e2 = np.inner(x2, e2)

        # Parameteric representation of line
        # First line is x_1 + e_1 * t
        # Second line is x_2 + e_2 * s
        s = np.empty_like(e1e1)
        t = np.empty_like(e1e1)

        parallel_condition = np.abs(1.0 - e1e2 ** 2 / (e1e1 * e2e2))
        parallel_idx = parallel_condition < 1e-5
        not_parallel_idx = np.bitwise_not(parallel_idx)
        anything_not_parallel = np.any(not_parallel_idx)

        if np.any(parallel_idx):
            # Some are parallel, so do processing
            t[parallel_idx] = (x2e1[parallel_idx] - x1e1[parallel_idx]) / e1e1[
                parallel_idx
            ]  # Comes from taking dot of e1 with a normal
            t[parallel_idx] = np.clip(
                t[parallel_idx], 0.0, 1.0
            )  # Note : the out version doesn't work
            s[parallel_idx] = (
                x1e2[parallel_idx] + t[parallel_idx] * e1e2[parallel_idx] - x2e2
            ) / e2e2  # Same as before
            s[parallel_idx] = np.clip(s[parallel_idx], 0.0, 1.0)

        if anything_not_parallel:
            # Using the Cauchy-Binet formula on eq(7) in docstring referenc
            s[not_parallel_idx] = (
                e1e1[not_parallel_idx] * (x1e2[not_parallel_idx] - x2e2)
                + e1e2[not_parallel_idx]
                * (x2e1[not_parallel_idx] - x1e1[not_parallel_idx])
            ) / (e1e1[not_parallel_idx] * e2e2 - (e1e2[not_parallel_idx]) ** 2)
            t[not_parallel_idx] = (
                e1e2[not_parallel_idx] * s[not_parallel_idx]
                + x2e1[not_parallel_idx]
                - x1e1[not_parallel_idx]
            ) / e1e1[not_parallel_idx]

        if anything_not_parallel:
            # Done here rather than the other loop to avoid
            # creating copies by selection of selections such as s[not_parallel_idx][idx]
            # which creates copies and bor

            # Remnants for non-parallel indices
            # as parallel selections are always clipped
            idx1 = s < 0.0
            idx2 = s > 1.0
            idx3 = t < 0.0
            idx4 = t > 1.0
            idx = idx1 | idx2 | idx3 | idx4

            if np.any(idx):
                local_e1e1 = e1e1[idx]
                local_e1e2 = e1e2[idx]

                local_x1e1 = x1e1[idx]
                local_x1e2 = x1e2[idx]
                local_x2e1 = x2e1[idx]

                potential_t = np.empty((4, local_e1e1.shape[0]))
                potential_s = np.empty_like(potential_t)
                potential_dist = np.empty_like(potential_t)

                # Fill in the possibilities
                potential_t[0, ...] = (local_x2e1 - local_x1e1) / local_e1e1
                potential_s[0, ...] = 0.0

                potential_t[1, ...] = (
                    local_x2e1 + local_e1e2 - local_x1e1
                ) / local_e1e1
                potential_s[1, ...] = 1.0

                potential_t[2, ...] = 0.0
                potential_s[2, ...] = (local_x1e2 - x2e2) / e2e2

                potential_t[3, ...] = 1.0
                potential_s[3, ...] = (local_x1e2 + local_e1e2 - x2e2) / e2e2

                np.clip(potential_t, 0.0, 1.0, out=potential_t)
                np.clip(potential_s, 0.0, 1.0, out=potential_s)

                potential_difference = _difference_impl_for_multiple_st(
                    x1[..., idx], e1[..., idx], potential_t, x2, e2, potential_s
                )
                np.sqrt(
                    np.einsum(
                        "ijk,ijk->jk", potential_difference, potential_difference
                    ),
                    out=potential_dist,
                )
                min_idx = np.expand_dims(np.argmin(potential_dist, axis=0), axis=0)
                s[idx] = np.take_along_axis(potential_s, min_idx, axis=0)[
                    0
                ]  # [0] at the end reduces the dimension, you can also squeeze
                t[idx] = np.take_along_axis(potential_t, min_idx, axis=0)[
                    0
                ]  # [0] at the end reduces the dimension, you can also squeeze

        return _difference_impl(x1, e1, t, x2, e2, s)

    def apply_forces(self, rod_one, index_one, cylinder_two, index_two):
        del index_one, index_two

        # First, check for a global AABB bounding box, and see whether that
        # intersects

        # FIXME : Optimization : multiple passes over same array to do min/max
        max_possible_dimension = np.zeros((3,))
        max_possible_dimension[...] = np.amax(rod_one.radius) + np.amax(rod_one.lengths)
        self.aabb_rod[..., 0] = (
            np.amin(rod_one.position_collection, axis=1) - max_possible_dimension
        )
        self.aabb_rod[..., 1] = (
            np.amax(rod_one.position_collection, axis=1) + max_possible_dimension
        )

        cylinder_dimensions_in_world_FOR = cylinder_two.director_collection[
            ..., 0
        ].T @ np.array(
            [[cylinder_two.radius], [cylinder_two.radius], [0.5 * cylinder_two.length]]
        )
        np.amax(
            np.abs(cylinder_dimensions_in_world_FOR), axis=1, out=max_possible_dimension
        )
        self.aabb_cylinder[..., 0] = (
            cylinder_two.position_collection[..., 0] - max_possible_dimension
        )
        self.aabb_cylinder[..., 1] = (
            cylinder_two.position_collection[..., 0] + max_possible_dimension
        )

        if self.__aabbs_not_intersecting(self.aabb_cylinder, self.aabb_rod):
            return

        x_rod = rod_one.position_collection[..., :-1]  # Discount last node
        r_rod = rod_one.radius
        l_rod = rod_one.lengths
        # We need at start of the element
        x_cyl = (
            cylinder_two.position_collection[..., 0]
            - 0.5 * cylinder_two.length * cylinder_two.director_collection[2, :, 0]
        )
        r_cyl = cylinder_two.radius  # scalar
        l_cyl = cylinder_two.length
        sum_r = r_rod + r_cyl

        # Element-wise bounding box, if outside then don't worry
        del_x = x_rod - np.expand_dims(x_cyl, axis=1)
        norm_del_x = np.sqrt(np.einsum("ij,ij->j", del_x, del_x))
        idx = norm_del_x <= (sum_r + l_rod + l_cyl)

        # If everything is out, don't bother to process
        if not np.any(idx):
            # idx[0] = True
            return

        # Process only the selected idx elements
        x_selected = x_rod[..., idx]
        edge_selected = l_rod[idx] * rod_one.tangents[..., idx]
        edge_cylinder = l_cyl * cylinder_two.director_collection[2, :, 0]

        # find the shortest line segment between the two centerline
        # segments : differs from normal cylinder-cylinder intersection
        distance_vector_collection = self.__find_min_dist(
            x_selected, edge_selected, x_cyl, edge_cylinder
        )
        distance_vector_lengths = np.sqrt(
            np.einsum(
                "ij,ij->j", distance_vector_collection, distance_vector_collection
            )
        )
        distance_vector_collection /= distance_vector_lengths

        gamma = sum_r[idx] - distance_vector_lengths
        interacting_idx = gamma > -1e-5
        # This step is necessary to scatter the masked results back to the original array
        # With pure masking, information regarding where in the original array we had True/False is lost
        # For example if idx = a < 0.2 => idx = [True False True]
        # b = a[idx]. nidx = b < -0.2 => nidx = [True False].
        # We now need selections of a such that we somehow do and AND with idx & nidx
        # But there's a shape mismatch. To overcome that find where in the idx vector we have Trues
        # widx = np.where(idx)[0] => widx = [0, 2] => idx[widx] = [True True]
        # Thus idx[widx] &= nidx gives now idx = [True False False]
        # Final [0] needed because where returns a one-tuple
        w_idx = np.where(idx)[0]

        # Get subset of interacting indices from original as explained above
        idx[w_idx] &= interacting_idx

        rod_elemental_forces = rod_one.external_forces + rod_one.internal_forces
        # No better way to do this?
        padded_idx = np.hstack((idx, False))  # Because at the end there's one more node
        r_padded_idx = np.roll(padded_idx, 1)
        rod_elemental_forces = 0.5 * (
            rod_elemental_forces[..., padded_idx]
            + rod_elemental_forces[..., r_padded_idx]
        )
        equilibrium_forces = -rod_elemental_forces + cylinder_two.external_forces
        pruned_distance_vector_collection = distance_vector_collection[
            ..., interacting_idx
        ]

        normal_force = np.einsum(
            "ij,ij->j", equilibrium_forces, pruned_distance_vector_collection
        )
        # Following line same as np.where(normal_force < 0.0, -normal_force, 0.0)
        np.abs(np.minimum(normal_force, 0.0), out=normal_force)

        # CHECK FOR GAMMA > 0.0?
        # Make it a float array of 1's and 0's
        # mask = (gamma[interacting_idx] > 0.0) * 1.0
        mask = np.heaviside(gamma[interacting_idx], 0.0)

        contact_force = self.k * gamma[interacting_idx]
        interpenetration_velocity = (
            0.5
            * (
                rod_one.velocity_collection[..., padded_idx]
                + rod_one.velocity_collection[..., r_padded_idx]
            )
            - cylinder_two.velocity_collection
        )
        contact_damping_force = self.nu * np.einsum(
            "ij,ij->j", interpenetration_velocity, pruned_distance_vector_collection
        )

        # magnitude* direction
        net_contact_force = (
            normal_force + 0.5 * mask * (contact_damping_force + contact_force)
        ) * pruned_distance_vector_collection

        ##  Equivalent statments
        # padded_idx_mask = padded_idx * 1.0
        # r_padded_idx_mask = r_padded_idx * 1.0
        padded_idx_mask = np.heaviside(padded_idx, 0.0)
        r_padded_idx_mask = np.heaviside(r_padded_idx, 0.0)

        padded_idx_mask[0] = 0.5
        r_padded_idx_mask[-1] = 0.5
        padded_idx_mask = padded_idx_mask[padded_idx]
        r_padded_idx_mask = r_padded_idx_mask[r_padded_idx]

        # Add it to the rods at the end of the day
        force_on_ith_element = net_contact_force * padded_idx_mask
        force_on_i_plus_oneth_element = net_contact_force * r_padded_idx_mask
        rod_one.external_forces[..., padded_idx] -= force_on_ith_element
        rod_one.external_forces[..., r_padded_idx] -= force_on_i_plus_oneth_element
        cylinder_two.external_forces[..., 0] += np.sum(
            force_on_ith_element + force_on_i_plus_oneth_element, axis=1
        )
