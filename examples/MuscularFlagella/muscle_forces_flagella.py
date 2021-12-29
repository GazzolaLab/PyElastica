__doc__ = """Muscular flagella, muscle forces class Numba implementation."""
import numpy as np
from elastica.external_forces import NoForces
from elastica._calculus import difference_kernel
from numba import njit


class MuscleForces(NoForces):
    """
    This class is for generating cyclic muscle forces.

    Attributes
    ----------
    amplitude : float
        Amplitude of muscle forces.
    wave_number : float
        Wave number of cyclic muscle contraction.
    """

    def __init__(self, amplitude, frequency):
        """

        Parameters
        ----------
        amplitude :  float
            Amplitude of muscle forces.
        frequency : float
            Beat frequency of the muscle.
        """
        self.amplitude = amplitude
        self.wave_number = 2 * np.pi * frequency

    def apply_forces(self, system, time: np.float = 0.0):
        # muscle_force = (
        #     system.tangents * self.amplitude * np.abs(np.sin(self.wave_number * time))
        # )
        # system.external_forces += difference_kernel(muscle_force)

        self._apply_forces(
            self.amplitude,
            self.wave_number * time,
            system.tangents,
            system.external_forces,
        )

    @staticmethod
    @njit(cache=True)
    def _apply_forces(amplitude, wt, tangents, external_forces):

        muscle_force = tangents * amplitude * np.abs(np.sin(wt))
        external_forces += difference_kernel(muscle_force)
