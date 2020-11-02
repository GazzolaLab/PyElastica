__doc__ = """ Numba implementation module for boundary condition implementations that apply external forces to the rod."""

import numpy as np
from elastica._linalg import _batch_matvec
from elastica.utils import _bspline

import numba
from numba import njit
from elastica._elastica_numba._linalg import _batch_product_i_k_to_ik


class NoForces:
    """
    This is the base class for external forcing boundary conditions applied to rod-like objects.

    Note
    ----
    Every new external forcing class must be derived
    from NoForces class.

    """

    def __init__(self):
        """
        NoForces class does not need any input parameters.
        """
        pass

    def apply_forces(self, system, time: np.float = 0.0):
        """Apply forces to a rod-like object.

        In NoForces class, this routine simply passes.

        Parameters
        ----------
        system : object
            System that is Rod-like.
        time : float
            The time of simulation.

        Returns
        -------


        """

        pass

    def apply_torques(self, system, time: np.float = 0.0):
        """Apply torques to a rod-like object.

        In NoForces class, this routine simply passes.

        Parameters
        ----------
        system : object
            System that is Rod-like.
        time : float
            The time of simulation.

        Returns
        -------

        """
        pass


class GravityForces(NoForces):
    """
    This class applies a constant gravitational force to the entire rod.

        Attributes
        ----------
        acc_gravity: numpy.ndarray
            1D (dim) array containing data with 'float' type. Gravitational acceleration vector.

    """

    def __init__(self, acc_gravity=np.array([0.0, -9.80665, 0.0])):
        """

        Parameters
        ----------
        acc_gravity: numpy.ndarray
            1D (dim) array containing data with 'float' type. Gravitational acceleration vector.

        """
        super(GravityForces, self).__init__()
        self.acc_gravity = acc_gravity

    def apply_forces(self, system, time=0.0):
        self.compute_gravity_forces(
            self.acc_gravity, system.mass, system.external_forces
        )

    @staticmethod
    @njit(cache=True)
    def compute_gravity_forces(acc_gravity, mass, external_forces):
        """
        This function add gravitational forces on the nodes. We are
        using njit decorated function to increase the speed.

        Parameters
        ----------
        acc_gravity: numpy.ndarray
            1D (dim) array containing data with 'float' type. Gravitational acceleration vector.
        mass: numpy.ndarray
            1D (blocksize) array containing data with 'float' type. Mass on the nodes.
        external_forces: numpy.ndarray
            2D (dim, blocksize) array containing data with 'float' type. External force vector.

        Returns
        -------

        """
        inplace_addition(external_forces, _batch_product_i_k_to_ik(acc_gravity, mass))


class EndpointForces(NoForces):
    """
    This class applies constant forces on the endpoint nodes.

        Attributes
        ----------
        start_force: numpy.ndarray
            2D (dim, 1) array containing data with 'float' type. Force applied to first node of the rod-like object.
        end_force: numpy.ndarray
            2D (dim, 1) array containing data with 'float' type. Force applied to last node of the rod-like object.
        ramp_up_time: float
            Applied forces are ramped up until ramp up time.

    """

    def __init__(self, start_force, end_force, ramp_up_time=0.0):
        """

        Parameters
        ----------
        start_force: numpy.ndarray
            2D (dim, 1) array containing data with 'float' type.
            Force applied to first node of the rod-like object.
        end_force: numpy.ndarray
            2D (dim, 1) array containing data with 'float' type.
            Force applied to last node of the rod-like object.
        ramp_up_time: float
            Applied forces are ramped up until ramp up time.

        """
        super(EndpointForces, self).__init__()
        self.start_force = start_force
        self.end_force = end_force
        assert ramp_up_time >= 0.0
        self.ramp_up_time = ramp_up_time

    def apply_forces(self, system, time=0.0):
        # factor = min(1.0, time / self.ramp_up_time)
        #
        # system.external_forces[..., 0] += self.start_force * factor
        # system.external_forces[..., -1] += self.end_force * factor

        self.compute_end_point_forces(
            system.external_forces,
            self.start_force,
            self.end_force,
            time,
            self.ramp_up_time,
        )

    @staticmethod
    @njit(cache=True)
    def compute_end_point_forces(
        external_forces, start_force, end_force, time, ramp_up_time
    ):
        """
        Compute end point forces that are applied on the rod using numba njit decorator.

        Parameters
        ----------
        external_forces: numpy.ndarray
            2D (dim, blocksize) array containing data with 'float' type. External force vector.
        start_force: numpy.ndarray
            2D (dim, 1) array containing data with 'float' type.
        end_force: numpy.ndarray
            2D (dim, 1) array containing data with 'float' type.
            Force applied to last node of the rod-like object.
        time: float
        ramp_up_time: float
            Applied forces are ramped up until ramp up time.

        Returns
        -------

        """
        factor = min(1.0, time / ramp_up_time)
        external_forces[..., 0] += start_force * factor
        external_forces[..., -1] += end_force * factor


class UniformTorques(NoForces):
    """
    This class applies a uniform torque to the entire rod.

        Attributes
        ----------
        torque: numpy.ndarray
            2D (dim, 1) array containing data with 'float' type. Total torque applied to a rod-like object.

    """

    def __init__(self, torque, direction=np.array([0.0, 0.0, 0.0])):
        """

        Parameters
        ----------
        torque: float
            Torque magnitude applied to a rod-like object.
        direction: numpy.ndarray
            1D (dim) array containing data with 'float' type.
            Direction in which torque applied.
        """
        super(UniformTorques, self).__init__()
        self.torque = torque * direction

    def apply_torques(self, system, time: np.float = 0.0):
        n_elems = system.n_elems
        torque_on_one_element = (
            _batch_product_i_k_to_ik(self.torque, np.ones((n_elems))) / n_elems
        )
        system.external_torques += _batch_matvec(
            system.director_collection, torque_on_one_element
        )


class UniformForces(NoForces):
    """
    This class applies a uniform force to the entire rod.

        Attributes
        ----------
        force:  numpy.ndarray
            2D (dim, 1) array containing data with 'float' type. Total force applied to a rod-like object.
    """

    def __init__(self, force, direction=np.array([0.0, 0.0, 0.0])):
        """

        Parameters
        ----------
        force: float
            Force magnitude applied to a rod-like object.
        direction: numpy.ndarray
            1D (dim) array containing data with 'float' type.
            Direction in which force applied.
        """
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
    This class applies muscle torques along the body. The applied muscle torques are treated
    as applied external forces. This class can apply
    muscle torques as a traveling wave with a beta spline or only
    as a traveling wave. For implementation details refer to Gazzola et. al.
    RSoS. (2018).

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

    """

    def __init__(
        self,
        base_length,
        b_coeff,
        period,
        wave_number,
        phase_shift,
        direction,
        rest_lengths,
        ramp_up_time=0.0,
        with_spline=False,
    ):
        """

        Parameters
        ----------
        base_length: float
            Rest length of the rod-like object.
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
           1D (dim) array containing data with 'float' type. Muscle torque direction.
        ramp_up_time: float
            Applied muscle torques are ramped up until ramp up time.
        with_spline: boolean
            Option to use beta-spline.

        """
        super(MuscleTorques, self).__init__()

        self.direction = direction  # Direction torque applied
        self.angular_frequency = 2.0 * np.pi / period
        self.wave_number = wave_number
        self.phase_shift = phase_shift

        assert ramp_up_time >= 0.0
        self.ramp_up_time = ramp_up_time

        # s is the position of nodes on the rod, we go from node=1 to node=nelem-1, because there is no
        # torques applied by first and last node on elements. Reason is that we cannot apply torque in an
        # infinitesimal segment at the beginning and end of rod, because there is no additional element
        # (at element=-1 or element=n_elem+1) to provide internal torques to cancel out an external
        # torque. This coupled with the requirement that the sum of all muscle torques has
        # to be zero results in this condition.
        self.s = np.cumsum(rest_lengths)

        if with_spline:
            assert b_coeff.size != 0, "Beta spline coefficient array (t_coeff) is empty"
            my_spline, ctr_pts, ctr_coeffs = _bspline(b_coeff, base_length)
            self.my_spline = my_spline(self.s)

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

            self.my_spline = constant_function(self.s)

    def apply_torques(self, system, time: np.float = 0.0):
        self.compute_muscle_torques(
            time,
            self.my_spline,
            self.s,
            self.angular_frequency,
            self.wave_number,
            self.phase_shift,
            self.ramp_up_time,
            self.direction,
            system.director_collection,
            system.external_torques,
        )

    @staticmethod
    @njit(cache=True)
    def compute_muscle_torques(
        time,
        my_spline,
        s,
        angular_frequency,
        wave_number,
        phase_shift,
        ramp_up_time,
        direction,
        director_collection,
        external_torques,
    ):
        # Ramp up the muscle torque
        factor = min(1.0, time / ramp_up_time)
        # From the node 1 to node nelem-1
        # Magnitude of the torque. Am = beta(s) * sin(2pi*t/T + 2pi*s/lambda + phi)
        # There is an inconsistency with paper and Elastica cpp implementation. In paper sign in
        # front of wave number is positive, in Elastica cpp it is negative.
        torque_mag = (
            factor
            * my_spline
            * np.sin(angular_frequency * time - wave_number * s + phase_shift)
        )
        # Head and tail of the snake is opposite compared to elastica cpp. We need to iterate torque_mag
        # from last to first element.
        torque = _batch_product_i_k_to_ik(direction, torque_mag[::-1])
        inplace_addition(
            external_torques[..., 1:],
            _batch_matvec(director_collection, torque)[..., 1:],
        )
        inplace_substraction(
            external_torques[..., :-1],
            _batch_matvec(director_collection[..., :-1], torque[..., 1:]),
        )


@njit(cache=True)
def inplace_addition(external_force_or_torque, force_or_torque):
    """
    This function does inplace addition. First argument
    `external_force_or_torque` is the system.external_forces
    or system.external_torques. Second argument force or torque
    vector to be added.

    Parameters
    ----------
    external_force_or_torque: numpy.ndarray
        2D (dim, blocksize) array containing data with 'float' type.
    force_or_torque: numpy.ndarray
        2D (dim, blocksize) array containing data with 'float' type.

    Returns
    -------

    """
    blocksize = force_or_torque.shape[1]
    for i in range(3):
        for k in range(blocksize):
            external_force_or_torque[i, k] += force_or_torque[i, k]


@njit(cache=True)
def inplace_substraction(external_force_or_torque, force_or_torque):
    """
    This function does inplace substraction. First argument
    `external_force_or_torque` is the system.external_forces
    or system.external_torques. Second argument force or torque
    vector to be substracted.
    Parameters
    ----------
    external_force_or_torque: numpy.ndarray
        2D (dim, blocksize) array containing data with 'float' type.
    force_or_torque: numpy.ndarray
        2D (dim, blocksize) array containing data with 'float' type.

    Returns
    -------

    """
    blocksize = force_or_torque.shape[1]
    for i in range(3):
        for k in range(blocksize):
            external_force_or_torque[i, k] -= force_or_torque[i, k]
