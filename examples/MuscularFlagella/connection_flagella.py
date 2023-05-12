__doc__ = """Muscular flagella connection class Numba implementation. """
__all__ = ["MuscularFlagellaConnection"]
import numpy as np
from numba import njit
from elastica.joint import FreeJoint
from elastica._linalg import _batch_matvec


class MuscularFlagellaConnection(FreeJoint):
    """
    This connection class is for Muscular Flagella and it is not generalizable. Since our goal is to
    replicate the experimental data. We assume muscular flagella is not moving out of plane.

    """

    def __init__(
        self,
        k,
        normal,
    ):
        """

        Parameters
        ----------
        k : float
            The spring constant at the connection.
        normal : np.ndarray
            1D array of floats. Normal direction of the rods.
        """
        super().__init__(k, nu=0)

        self.normal = normal

    def apply_forces(self, rod_one, index_one, rod_two, index_two):
        self.torque = self._apply_forces(
            self.k,
            self.normal,
            index_one,
            index_two,
            rod_one.tangents,
            rod_one.position_collection,
            rod_two.position_collection,
            rod_two.external_forces,
        )

    @staticmethod
    @njit(cache=True)
    def _apply_forces(
        k,
        normal,
        index_one,
        index_two,
        rod_one_tangents,
        rod_one_position_collection,
        rod_two_position_collection,
        rod_two_external_forces,
    ):
        # This connection routine is not generalizable. Our goal here is to replicate the experiment data.
        # Thus below code is hard codded. Torques are computed along the centerline of the muscle
        # and transfered to the body.
        start_idx = index_one[0]
        end_idx = index_one[-1]

        armlength = 0.0053
        armdirection1 = np.cross(rod_one_tangents[..., start_idx], normal)
        armposition1 = armlength * (armdirection1) / np.linalg.norm(armdirection1)
        startposition = (
            rod_one_position_collection[..., start_idx]
            + rod_one_position_collection[..., start_idx + 1]
        ) / 2 + armposition1

        armdirection2 = np.cross(rod_one_tangents[..., end_idx], normal)
        armposition2 = armlength * (armdirection2) / np.linalg.norm(armdirection2)
        endposition = (
            rod_one_position_collection[..., end_idx]
            + rod_one_position_collection[..., end_idx + 1]
        ) / 2 + armposition2

        forcestart = k * (
            rod_two_position_collection[..., index_two[0]] - startposition
        )
        forceend = k * (rod_two_position_collection[..., index_two[-1]] - endposition)

        rod_two_external_forces[..., index_two[0]] -= forcestart
        rod_two_external_forces[..., index_two[-1]] -= forceend

        Torque1 = np.cross(
            armposition1, (forcestart - np.array([0.0, 0.0, forcestart[2]]))
        )
        Torque2 = np.cross(armposition2, (forceend - np.array([0.0, 0.0, forceend[2]])))

        # We are taking the average torques to prevent any numerical issues.
        # Torque has to have only one component, thus remove the other components, because motion is in plane.
        Torqueaverage2 = 0.5 * (Torque1[2] - Torque2[2])

        Torqueaverage = np.array([0.0, 0.0, Torqueaverage2]).reshape(3, 1)

        return Torqueaverage

    def apply_torques(self, rod_one, index_one, rod_two, index_two):

        self._apply_torques(
            index_one,
            self.torque,
            rod_one.director_collection,
            rod_one.external_torques,
        )

    @staticmethod
    @njit(cache=True)
    def _apply_torques(
        index_one, torque, rod_one_director_collection, rod_one_external_torques
    ):
        start_idx = index_one[0]
        end_idx = index_one[-1]
        rod_one_external_torques[..., start_idx] += 0.5 * _batch_matvec(
            rod_one_director_collection[..., start_idx : start_idx + 1], torque
        ).reshape(3)
        rod_one_external_torques[..., end_idx] -= 0.5 * _batch_matvec(
            rod_one_director_collection[..., end_idx : end_idx + 1], torque
        ).reshape(3)
