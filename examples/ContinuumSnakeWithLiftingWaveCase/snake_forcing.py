__doc__ = """ External forces and Actions (in learning) of the snake terrain case."""

import numpy as np
from elastica._linalg import _batch_matvec
from elastica.utils import _bspline
from numba import njit
from elastica._linalg import (
    _batch_norm,
    _batch_product_k_ik_to_ik,
    _batch_vec_oneD_vec_cross,
)
from elastica.external_forces import NoForces
from elastica.external_forces import (
    inplace_addition,
    inplace_substraction,
)


class MuscleTorquesLifting(NoForces):
    """
    This class applies muscle torques along the body. The applied muscle torques are treated
    as applied external forces. This class can apply lifting
    muscle torques as a traveling wave with a beta spline or only
    as a traveling wave. For implementation details refer to X. Zhang et. al. Nat. Comm. 2021

            Attributes
            ----------
            direction: numpy.ndarray
                    2D (dim, 1) array containing data with 'float' type. Muscle torque direction.
            angular_frequency: float
                    Angular frequency of traveling wave.
            wave_number: float
                    Wave number of traveling wave.
            phase_shift: float
                    Phase shift of traveling wave.
            ramp_up_time: float
                    Applied muscle torques are ramped up until ramp up time.
            my_spline: numpy.ndarray
                    1D (blocksize) array containing data with 'float' type. Generated spline.
            switch_on_time: float
                    time to switch on the muscle activation.
    """

    def __init__(
        self,
        b_coeff,
        period,
        wave_number,
        phase_shift,
        direction,
        rest_lengths,
        ramp_up_time=0.0,
        with_spline=False,
        switch_on_time=0.0,
    ):
        """

        Parameters
        ----------
        b_coeff: nump.ndarray
                1D array containing data with 'float' type.
                Beta coefficients for beta-spline.
        period: float
                Period of traveling wave.
        wave_number: float
                Wave number of traveling wave.
        phase_shift: float
                Phase shift of traveling wave.
        direction: numpy.ndarray
           1D (dim) array containing data with 'float' type.
        ramp_up_time: float
                Applied muscle torques are ramped up until ramp up time.
        with_spline: boolean
                Option to use beta-spline.
        switch_on_time: float
                time to switch on the muscle activation.

        """
        super().__init__()

        self.direction = direction  # Direction torque applied
        self.angular_frequency = 2.0 * np.pi / period
        self.wave_number = wave_number
        self.phase_shift = phase_shift

        assert ramp_up_time >= 0.0
        self.ramp_up_time = ramp_up_time
        assert switch_on_time >= 0.0
        self.switch_on_time = switch_on_time

        # s is the position of nodes on the rod, we go from node=1 to node=nelem-1, because there is no
        # torques applied by first and last node on elements. Reason is that we cannot apply torque in an
        # infinitesimal segment at the beginning and end of rod, because there is no additional element
        # (at element=-1 or element=n_elem+1) to provide internal torques to cancel out an external
        # torque. This coupled with the requirement that the sum of all muscle torques has
        # to be zero results in this condition.
        self.s = np.cumsum(rest_lengths)
        self.s /= self.s[-1]

        if with_spline:
            assert b_coeff.size != 0, "Beta spline coefficient array (t_coeff) is empty"
            spline, ctr_pts, ctr_coeffs = _bspline(b_coeff)
            self.spline = spline(self.s)

        else:
            self.spline = np.full_like(self.s)

    def apply_torques(self, system, time: np.float64 = 0.0):
        self.compute_muscle_torques(
            time,
            self.spline,
            self.s,
            self.angular_frequency,
            self.wave_number,
            self.phase_shift,
            self.ramp_up_time,
            self.direction,
            self.switch_on_time,
            system.tangents,
            system.director_collection,
            system.external_torques,
        )

    @staticmethod
    @njit(cache=True)
    def compute_muscle_torques(
        time,
        spline,
        s,
        angular_frequency,
        wave_number,
        phase_shift,
        ramp_up_time,
        direction,
        switch_on_time,
        tangents,
        director_collection,
        external_torques,
    ):
        if time > switch_on_time:
            # Ramp up the muscle torque
            factor = min(1.0, (time - switch_on_time) / ramp_up_time)
            # From the node 1 to node nelem-1
            # Magnitude of the torque. Am = beta(s) * sin(2pi*t/T + 2pi*s/lambda + phi)
            # There is an inconsistency with paper and Elastica cpp implementation. In paper sign in
            # front of wave number is positive, in Elastica cpp it is negative.
            torque_mag = (
                factor
                * spline
                * np.sin(
                    angular_frequency * (time - switch_on_time - phase_shift)
                    - wave_number * s
                )
            )
            # Head and tail of the snake is opposite compared to elastica cpp. We need to iterate torque_mag
            # from last to first element.
            # compute torque direction for lifting wave.
            # Here, direction of each element is computed separately
            # based on the rod tangent and normal direction. This is implemented to
            # correct the binormal direction when snake undergoes lateral bending
            avg_element_direction = 0.5 * (tangents[..., :-1] + tangents[..., 1:])
            torque_direction = _batch_vec_oneD_vec_cross(
                avg_element_direction, direction
            )
            torque_direction_unit = _batch_product_k_ik_to_ik(
                1 / (_batch_norm(torque_direction) + 1e-14),
                torque_direction,
            )
            torque = _batch_product_k_ik_to_ik(
                torque_mag[-2::-1], torque_direction_unit
            )

            inplace_addition(
                external_torques[..., 1:],
                _batch_matvec(director_collection[..., 1:], torque),
            )
            inplace_substraction(
                external_torques[..., :-1],
                _batch_matvec(director_collection[..., :-1], torque),
            )
