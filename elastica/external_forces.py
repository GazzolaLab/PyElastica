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

    def __init__(self, start_force, end_force, ramp_up_time=0.0):
        super(EndpointForces, self).__init__()
        self.start_force = start_force
        self.end_force = end_force
        assert ramp_up_time >= 0.0
        if ramp_up_time == 0:
            self.ramp_up_time = 1e-14
        else:
            self.ramp_up_time = ramp_up_time

    def apply_forces(self, system, time=0.0):
        factor = min(1.0, time / self.ramp_up_time)

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
        wave_number,
        phase_shift,
        ramp_up_time,
        direction,
        with_spline=False,
    ):
        super(MuscleTorques, self).__init__()

        self.direction = direction.reshape(3, 1)  # Direction torque applied
        self.angular_frequency = 2.0 * np.pi / period
        self.wave_number = wave_number
        self.phase_shift = phase_shift

        assert ramp_up_time >= 0.0
        if ramp_up_time == 0:
            self.ramp_up_time = 1e-14
        else:
            self.ramp_up_time = ramp_up_time

        if with_spline:
            assert b_coeff.size != 0, "Beta spline coefficient array (t_coeff) is empty"
            self.my_spline, ctr_pts, ctr_coeffs = _bspline(b_coeff, base_length)

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
        factor = min(1.0, time / self.ramp_up_time)
        # From the node 1 to node nelem-1
        # s is the position of nodes on the rod, we go from node=1 to node=nelem-1, because there is no
        # torques applied by first and last node on elements. Reason is that we cannot apply torque in an
        # infinitesimal segment at the beginning and end of rod, because there is no additional element
        # (at element=-1 or element=n_elem+1) to provide internal torques to cancel out an external
        # torque. This coupled with the requirement that the sum of all muscle torques has
        # to be zero results in this condition.
        s = np.cumsum(system.rest_lengths)
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
        # Head and tail of the snake is opposite compared to elastica cpp. We need to iterate torque_mag
        # from last to first element.
        torque = np.einsum("j,ij->ij", torque_mag[::-1], self.direction)
        # TODO: Find a way without doing tow batch_matvec product
        system.external_torques[..., 1:] += _batch_matvec(
            system.director_collection, torque
        )[..., 1:]
        system.external_torques[..., :-1] -= _batch_matvec(
            system.director_collection[..., :-1], torque[..., 1:]
        )


class EndpointForcesSinusoidal(NoForces):
    """ Applies sinusoidal forces on endpoints
    """

    def __init__(
        self,
        start_force_mag,
        end_force_mag,
        ramp_up_time=0.0,
        tangent_direction=np.array([0, 0, 1]),
        normal_direction=np.array([0, 1, 0]),
    ):
        super(EndpointForcesSinusoidal, self).__init__()
        # Start force
        self.start_force_mag = start_force_mag
        self.end_force_mag = end_force_mag

        # Applied force directions
        self.normal_direction = normal_direction
        # self.roll_direction = np.cross(tangent_direction, normal_direction)
        self.roll_direction = np.cross(normal_direction,tangent_direction)

        assert ramp_up_time >= 0.0
        if ramp_up_time == 0:
            self.ramp_up_time = 1e-14
        else:
            self.ramp_up_time = ramp_up_time

    def apply_forces(self, system, time=0.0):

        if time < self.ramp_up_time:
            # When time smaller than ramp up time apply the force in normal direction
            # First pull the rod upward or downward direction some time.
            start_force = -2.0*self.start_force_mag * self.normal_direction
            end_force = -2.0*self.end_force_mag * self.normal_direction

            system.external_forces[..., 0] += start_force
            system.external_forces[..., -1] += end_force

        else:
            # When time is greater than ramp up time, forces are applied in normal
            # and roll direction or forces are in a plane perpendicular to the
            # direction.

            # First force applied to start of the rod
            roll_forces_start = (
                self.start_force_mag
                * np.cos(0.5 * np.pi * (time - self.ramp_up_time))
                * self.roll_direction
            )
            normal_forces_start = (
                self.start_force_mag
                * np.sin(0.5 * np.pi * (time - self.ramp_up_time))
                * self.normal_direction
            )
            start_force = roll_forces_start + normal_forces_start
            # Now force applied to end of the rod
            roll_forces_end = (
                self.end_force_mag
                * np.cos(0.5 * np.pi * (time - self.ramp_up_time))
                * self.roll_direction
            )
            normal_forces_end = (
                self.end_force_mag
                * np.sin(0.5 * np.pi * (time - self.ramp_up_time))
                * self.normal_direction
            )
            end_force = roll_forces_end + normal_forces_end
            # Update external forces
            system.external_forces[..., 0] += start_force
            system.external_forces[..., -1] += end_force
