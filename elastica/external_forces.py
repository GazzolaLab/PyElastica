__doc__ = """ Numba implementation module for boundary condition implementations that apply
external forces to the system."""

from typing import TypeVar, Generic

import numpy as np
from numpy.typing import NDArray

from elastica._linalg import _batch_matvec
from elastica.typing import SystemType, RodType, RigidBodyType
from elastica.utils import _bspline

from numba import njit
from elastica._linalg import _batch_product_i_k_to_ik


S = TypeVar("S")


class NoForces(Generic[S]):
    """
    This is the base class for external forcing boundary conditions applied to rod-like objects.

    Notes
    -----
    Every new external forcing class must be derived
    from NoForces class.

    """

    def __init__(self) -> None:
        """
        NoForces class does not need any input parameters.
        """
        pass

    def apply_forces(self, system: S, time: np.float64 = np.float64(0.0)) -> None:
        """Apply forces to a rod-like object.

        In NoForces class, this routine simply passes.

        Parameters
        ----------
        system : SystemType
            Rod or rigid-body object
        time : float
            The time of simulation.

        """
        pass

    def apply_torques(self, system: S, time: np.float64 = np.float64(0.0)) -> None:
        """Apply torques to a rod-like object.

        In NoForces class, this routine simply passes.

        Parameters
        ----------
        system : SystemType
            Rod or rigid-body object
        time : float
            The time of simulation.

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

    def __init__(
        self,
        acc_gravity: NDArray[np.float64] = np.array(
            [0.0, -9.80665, 0.0]
        ),  # FIXME: avoid mutable default
    ) -> None:
        """

        Parameters
        ----------
        acc_gravity: numpy.ndarray
            1D (dim) array containing data with 'float' type. Gravitational acceleration vector.

        """
        super(GravityForces, self).__init__()
        self.acc_gravity = acc_gravity

    def apply_forces(
        self, system: "RodType | RigidBodyType", time: np.float64 = np.float64(0.0)
    ) -> None:
        self.compute_gravity_forces(
            self.acc_gravity, system.mass, system.external_forces
        )

    @staticmethod
    @njit(cache=True)  # type: ignore
    def compute_gravity_forces(
        acc_gravity: NDArray[np.float64],
        mass: NDArray[np.float64],
        external_forces: NDArray[np.float64],
    ) -> None:
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

        """
        inplace_addition(external_forces, _batch_product_i_k_to_ik(acc_gravity, mass))


class EndpointForces(NoForces):
    """
    This class applies constant forces on the endpoint nodes.

        Attributes
        ----------
        start_force: numpy.ndarray
            1D (dim) array containing data with 'float' type. Force applied to first node of the system.
        end_force: numpy.ndarray
            1D (dim) array containing data with 'float' type. Force applied to last node of the system.
        ramp_up_time: float
            Applied forces are ramped up until ramp up time.

    """

    def __init__(
        self,
        start_force: NDArray[np.float64],
        end_force: NDArray[np.float64],
        ramp_up_time: float,
    ) -> None:
        """

        Parameters
        ----------
        start_force: numpy.ndarray
            1D (dim) array containing data with 'float' type.
            Force applied to first node of the system.
        end_force: numpy.ndarray
            1D (dim) array containing data with 'float' type.
            Force applied to last node of the system.
        ramp_up_time: float
            Applied forces are ramped up until ramp up time.

        """
        super(EndpointForces, self).__init__()
        self.start_force = start_force
        self.end_force = end_force
        assert ramp_up_time > 0.0
        self.ramp_up_time = np.float64(ramp_up_time)

    def apply_forces(
        self, system: "RodType | RigidBodyType", time: np.float64 = np.float64(0.0)
    ) -> None:
        self.compute_end_point_forces(
            system.external_forces,
            self.start_force,
            self.end_force,
            time,
            self.ramp_up_time,
        )

    @staticmethod
    @njit(cache=True)  # type: ignore
    def compute_end_point_forces(
        external_forces: NDArray[np.float64],
        start_force: NDArray[np.float64],
        end_force: NDArray[np.float64],
        time: np.float64,
        ramp_up_time: np.float64,
    ) -> None:
        """
        Compute end point forces that are applied on the rod using numba njit decorator.

        Parameters
        ----------
        external_forces: numpy.ndarray
            2D (dim, blocksize) array containing data with 'float' type. External force vector.
        start_force: numpy.ndarray
            1D (dim) array containing data with 'float' type.
        end_force: numpy.ndarray
            1D (dim) array containing data with 'float' type.
            Force applied to last node of the system.
        time: float
        ramp_up_time: float
            Applied forces are ramped up until ramp up time.

        """
        factor = min(1.0, float(time / ramp_up_time))
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

    def __init__(
        self,
        torque: np.float64,
        direction: NDArray[np.float64] = np.array(
            [0.0, 0.0, 0.0]
        ),  # FIXME: avoid mutable default
    ) -> None:
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

    def apply_torques(
        self, system: "RodType | RigidBodyType", time: np.float64 = np.float64(0.0)
    ) -> None:
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

    def __init__(
        self,
        force: np.float64,
        direction: NDArray[np.float64] = np.array(
            [0.0, 0.0, 0.0]
        ),  # FIXME: avoid mutable default
    ) -> None:
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

    def apply_forces(
        self, system: "RodType | RigidBodyType", time: np.float64 = np.float64(0.0)
    ) -> None:
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
        base_length: float,  # TODO: Is this necessary?
        b_coeff: NDArray[np.float64],
        period: float,
        wave_number: float,
        phase_shift: float,
        direction: NDArray[np.float64],
        rest_lengths: NDArray[np.float64],
        ramp_up_time: float,
        with_spline: bool = False,
    ) -> None:
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
        ramp_up_time: np.float64
            Applied muscle torques are ramped up until ramp up time.
        with_spline: boolean
            Option to use beta-spline.

        """
        super(MuscleTorques, self).__init__()

        self.direction = direction  # Direction torque applied
        self.angular_frequency = np.float64(2.0 * np.pi / period)
        self.wave_number = np.float64(wave_number)
        self.phase_shift = np.float64(phase_shift)

        assert ramp_up_time > 0.0
        self.ramp_up_time = np.float64(ramp_up_time)

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
            my_spline, ctr_pts, ctr_coeffs = _bspline(b_coeff)
            self.my_spline = my_spline(self.s)

        else:
            self.my_spline = np.full_like(self.s, fill_value=1.0)

    def apply_torques(
        self, system: "RodType | RigidBodyType", time: np.float64 = np.float64(0.0)
    ) -> None:
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
    @njit(cache=True)  # type: ignore
    def compute_muscle_torques(
        time: float,
        my_spline: NDArray[np.float64],
        s: np.float64,
        angular_frequency: np.float64,
        wave_number: np.float64,
        phase_shift: np.float64,
        ramp_up_time: np.float64,
        direction: NDArray[np.float64],
        director_collection: NDArray[np.float64],
        external_torques: NDArray[np.float64],
    ) -> None:
        # Ramp up the muscle torque
        factor = min(1.0, float(time / ramp_up_time))
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


@njit(cache=True)  # type: ignore
def inplace_addition(
    external_force_or_torque: NDArray[np.float64],
    force_or_torque: NDArray[np.float64],
) -> None:
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


    """
    blocksize = force_or_torque.shape[1]
    for i in range(3):
        for k in range(blocksize):
            external_force_or_torque[i, k] += force_or_torque[i, k]


@njit(cache=True)  # type: ignore
def inplace_substraction(
    external_force_or_torque: NDArray[np.float64],
    force_or_torque: NDArray[np.float64],
) -> None:
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


    """
    blocksize = force_or_torque.shape[1]
    for i in range(3):
        for k in range(blocksize):
            external_force_or_torque[i, k] -= force_or_torque[i, k]


class EndpointForcesSinusoidal(NoForces):
    """
    This class applies sinusoidally varying forces to the ends of a rod.
    Forces are applied in a plane, which is defined by the tangent_direction and normal_direction.

        Attributes
        ----------
        start_force_mag: float
            Magnitude of the force that is applied to the start of the rod (node 0).
        end_force_mag: float
            Magnitude of the force that is applied to the end of the rod (node -1).
        ramp_up_time: float
            Applied forces are applied in the normal direction until time reaches ramp_up_time.
        normal_direction: np.ndarray
            An array (3,) contains type float.
            This is the normal direction of the rod.
        roll_direction: np.ndarray
            An array (3,) contains type float.
            This is the direction perpendicular to rod tangent, and rod normal.

        Notes
        -----
        In order to see example how to use this class, see joint examples.

    """

    def __init__(
        self,
        start_force_mag: float,
        end_force_mag: float,
        ramp_up_time: float = 0.0,
        tangent_direction: NDArray[np.floating] = np.array(
            [0.0, 0.0, 1.0]
        ),  # FIXME: avoid mutable default
        normal_direction: NDArray[np.floating] = np.array(
            [0.0, 1.0, 0.0]
        ),  # FIXME: avoid mutable default
    ) -> None:
        """

        Parameters
        ----------
        start_force_mag: float
            Magnitude of the force that is applied to the start of the system (node 0).
        end_force_mag: float
            Magnitude of the force that is applied to the end of the system (node -1).
        ramp_up_time: float
            Applied muscle torques are ramped up until ramp up time.
        tangent_direction: np.ndarray
            An array (3,) contains type float.
            This is the tangent direction of the system, or normal of the plane that forces applied.
        normal_direction: np.ndarray
            An array (3,) contains type float.
            This is the normal direction of the system.
        """
        super(EndpointForcesSinusoidal, self).__init__()
        # Start force
        self.start_force_mag = np.float64(start_force_mag)
        self.end_force_mag = np.float64(end_force_mag)

        # Applied force directions
        self.normal_direction = normal_direction
        self.roll_direction = np.cross(normal_direction, tangent_direction)

        assert ramp_up_time >= 0.0
        self.ramp_up_time = np.float64(ramp_up_time)

    def apply_forces(
        self, system: "RodType | RigidBodyType", time: np.float64 = np.float64(0.0)
    ) -> None:

        if time < self.ramp_up_time:
            # When time smaller than ramp up time apply the force in normal direction
            # First pull the rod upward or downward direction some time.
            start_force = -2.0 * self.start_force_mag * self.normal_direction
            end_force = -2.0 * self.end_force_mag * self.normal_direction

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
