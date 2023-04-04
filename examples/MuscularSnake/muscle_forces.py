__doc__ = """ Muscular snake muscle forces NumPy implementation """
__all__ = ["MuscleForces"]
import numpy as np
from numba import njit
from elastica import NoForces
from elastica._calculus import difference_kernel


class MuscleForces(NoForces):
    """
    This class is for computing muscle forces. Detailed formulation is given in Eq. 2
    Zhang et. al. Nature Comm 2019 paper

    Attributes
    ----------
        amplitude :  float
            Amplitude of muscle forces.
        wave_number : float
            Wave number for muscle actuation.
        side_of_body : int
            Depending on which side of body, left or right this variable becomes +1 or -1.
            This variable determines the sin wave direction.
        time_delay : float
            Delay time for muscles.
        muscle_start_end_index : numpy.ndarray
            1D (2) array containing data with 'int' type.
            Element start and end index of muscle forces.
    """

    def __init__(
        self,
        amplitude,
        wave_number,
        arm_length,
        time_delay,
        side_of_body,
        muscle_start_end_index,
        step,
        post_processing,
    ):
        """

        Parameters
        ----------
        amplitude :  float
            Amplitude of muscle forces.
        wave_number : float
            Wave number for muscle actuation.
        arm_length : float
            Used to map the torques optimized by CMA into the contraction forces of our muscles.
        time_delay : float
            Delay time for muscles.
        side_of_body : int
            Depending on which side of body, left or right this variable becomes +1 or -1.
        muscle_start_end_index : numpy.ndarray
            1D (2) array containing data with 'int' type.
            Element start and end index of muscle forces.
        """

        self.amplitude = amplitude
        self.wave_number = wave_number
        self.side_of_body = side_of_body
        self.time_delay = time_delay
        self.muscle_start_end_index = muscle_start_end_index

        """
        For legacy purposes, the input from the optimizer is given in terms of
        torque amplitudes (see Gazzola et al. Royal Society Open Source, 2018).
        We then use the same set up and map those values into muscle
        contractile forces by dividing them by the arm length. This is captured
        through the parameter "factor". This also enables a meaningful
        comparison with the continuum snake case of the above reference.
        """
        self.amplitude /= arm_length

        self.post_processing = post_processing
        self.step = step
        self.counter = 0

    def apply_forces(self, system, time: np.float = 0.0):
        forces = self._apply_forces(
            self.amplitude,
            self.wave_number,
            self.side_of_body,
            time,
            self.time_delay,
            self.muscle_start_end_index,
            system.tangents,
            system.external_forces,
        )

        if self.counter % self.step == 0:
            self.post_processing["time"].append(time)
            self.post_processing["step"].append(self.counter)
            self.post_processing["external_forces"].append(forces.copy())
            self.post_processing["element_position"].append(
                np.cumsum(system.lengths).copy()
            )

        self.counter += 1

    @staticmethod
    @njit(cache=True)
    def _apply_forces(
        amplitude,
        wave_number,
        side_of_body,
        time,
        time_delay,
        muscle_start_end_index,
        tangents,
        external_forces,
    ):

        real_time = time - time_delay
        ramp = real_time if real_time <= 1.0 else 1.0
        factor = 0.0 if real_time <= 0.0 else ramp  # max(0.0, ramp)

        forces = np.zeros(external_forces.shape)

        muscle_forces = (
            factor
            * amplitude
            * (side_of_body * 0.5 * np.sin(wave_number * real_time) + 0.5)
            * tangents[..., muscle_start_end_index[0] : muscle_start_end_index[1]]
        )

        forces[
            ..., muscle_start_end_index[0] : muscle_start_end_index[1] + 1
        ] += difference_kernel(muscle_forces)

        external_forces += forces
        return forces
