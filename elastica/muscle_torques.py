import numpy as np
from elastica._linalg import _batch_matvec
from elastica._spline import _bspline
from elastica.external_forces import NoForces


class MuscleTorques(NoForces):
    """
    Applies muscle torques on the body. It can apply muscle torques
    as travelling wave with beta spline or only as travelling wave.
    """

    def __init__(
        self,
        base_length,
        b_coeff,
        period,
        wave_length,
        phase_shift,
        rampupTime,
        direction,
        WithSpline=False,
    ):
        super(MuscleTorques, self).__init__()

        self.direction = direction.reshape(3, 1)  # Direction torque applied
        self.angular_frequency = 2.0 * np.pi / period
        self.wave_number = 2.0 * np.pi / (wave_length)
        self.phase_shift = phase_shift

        assert rampupTime >= 0.0
        if rampupTime == 0:
            self.rampupTime = 1e-14
        else:
            self.rampupTime = rampupTime

        if WithSpline:
            assert b_coeff.size != 0, "Beta spline coefficient array (t_coeff) is empty"
            self.my_spline, ctr_pts, ctr_coeffs = _bspline(
                b_coeff, base_length, keep_pts=True
            )

        else:

            def constant_function(input):
                """
                Return array of ones same as the size of the input array. This
                function is called when Beta spline function is not used.
                Parameters
                ----------
                input

                Returns
                -------

                """
                return np.ones(input.shape)

            self.my_spline = constant_function

    def apply_torques(self, system, time: np.float = 0.0):

        # Ramp up the muscle torque
        factor = min(1.0, time / self.rampupTime)

        # From the node 1 to node nelem-1
        # s is the position of nodes on the rod, we go from node=1 to node=nelem-1, because there is no
        # torques applied by first and last node on elements.
        s = np.cumsum(system.rest_lengths)[:-1]

        # Magnitude of the torque. Am = beta(s) * sin(2pi*t/T + 2pi*s/lambda + phi)
        # There is an inconsistency with paper and Elastica cpp implementation. In paper sign in
        # front of wave number is positive, in Elastica cpp it is negative.
        torque_mag = (
            factor
            * self.my_spline(s)
            * np.sin(
                self.angular_frequency * time - self.wave_number * s + self.phase_shift
            )
        )

        torque = np.einsum("j,ij->ij", torque_mag, self.direction)

        # TODO: Find a way without doing tow batch_matvec product
        system.external_torques[..., :-1] -= _batch_matvec(
            system.director_collection[..., :-1], torque
        )
        system.external_torques[..., 1:] += _batch_matvec(
            system.director_collection[..., 1:], torque
        )
