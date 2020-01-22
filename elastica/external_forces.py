__doc__ = """ External forcing for rod """

import numpy as np
from elastica._linalg import _batch_matvec
from elastica._spline import _bspline


class NoForces:
    """ Base class for external forcing for Rods

    Can make this an abstract class, but its inconvenient
    for the user to keep on defining apply_forces and
    apply_torques object over and over.
    """

    def __init__(self):
        pass

    def apply_forces(self, system, time: np.float = 0.0):
        """ Apply forces to a system object.

        In NoForces, this routine simply passes.

        Parameters
        ----------
        system : system that is Rod-like
        time : np.float, the time of simulation

        Returns
        -------
        None

        """

        pass

    def apply_torques(self, system, time: np.float = 0.0):
        """ Apply torques to a Rod-like object.

        In NoForces, this routine simply passes.

        Parameters
        ----------
        system : system that is Rod-like
        time : np.float, the time of simulation

        Returns
        -------
        None
        """
        pass


class GravityForces(NoForces):
    """ Applies a constant gravity on the entire rod
    """

    def __init__(self, acc_gravity=np.array([0.0, -9.80665, 0.0])):
        super(GravityForces, self).__init__()
        self.acc_gravity = acc_gravity

    def apply_forces(self, system, time=0.0):
        system.external_forces += np.outer(self.acc_gravity, system.mass)


class EndpointForces(NoForces):
    """ Applies constant forces on endpoints
    """

    def __init__(self, start_force, end_force, rampupTime=0.0):
        super(EndpointForces, self).__init__()
        self.start_force = start_force
        self.end_force = end_force
        assert rampupTime >= 0.0
        if rampupTime == 0:
            self.rampupTime = 1e-14
        else:
            self.rampupTime = rampupTime

    def apply_forces(self, system, time=0.0):
        factor = min(1.0, time / self.rampupTime)

        system.external_forces[..., 0] += self.start_force * factor
        system.external_forces[..., -1] += self.end_force * factor


class UniformTorques(NoForces):
    """
    Applies uniform torque to entire rod
    """

    def __init__(self, torque, direction=np.array([0.0, 0.0, 0.0])):
        super(UniformTorques, self).__init__()
        self.torque = (torque * direction).reshape(3, 1)

    def apply_torques(self, system, time: np.float = 0.0):
        torque_on_one_element = self.torque / system.n_elems
        system.external_torques += _batch_matvec(
            system.director_collection, torque_on_one_element
        )


class UniformForces(NoForces):
    """
    Applies uniform forces to entire rod
    """

    def __init__(self, force, direction=np.array([0.0, 0.0, 0.0])):
        super(UniformForces, self).__init__()
        self.force = (force * direction).reshape(3, 1)

    def apply_forces(self, system, time: np.float = 0.0):
        force_on_one_element = self.force / system.n_elems

        system.external_forces += force_on_one_element

        # Because mass of first and last node is half
        system.external_forces[..., 0] -= 0.5 * force_on_one_element[:, 0]
        system.external_forces[..., -1] -= 0.5 * force_on_one_element[:, 0]


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
