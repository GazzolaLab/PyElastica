import numpy as np


class AnalyticalDynamicCantilever:
    """
    This class computes the analytical solution to a cantilever's dynamic response
    to an initial velocity profile based on the Euler-Bernoulli beam theory.

    Given the generic dynamic beam equation and boundary conditions imposed on a cantilever,
    nontrivial solutions exist only when

        cosh(beta * base_length) * cos(beta * base_length) + 1 = 0

    where beta can be used to determine the natural frequencies of the cantilever beam:

        omega = beta^2 * sqrt(E * I / mu)

    The first four roots to beta are given as

        betas = [0.596864*pi, 1.49418*pi, 2.50025*pi, 3.49999*pi] / base_length

    The class solves for the analytical solution at a single natural frequency by computing
    the non-parametrized mode shapes as well as a mode parameter. The parameter is determined
    by the initial condition of the rod, namely, the end_velocity input. The free vibration
    equation can then be applied to determine beam deflection at a given time at any position
    of the rod:

        w(x, t) = Re[mode_shape * exp(-1j * omega * t)]

    For details, refer to
    https://en.wikipedia.org/wiki/Euler%E2%80%93Bernoulli_beam_theory#Dynamic_beam_equation
    or
    Han et al (1998), Dynamics of Transversely Vibrating Beams
    Using Four Engineering Theories

        Attributes
        ----------
        base_length: float
            Total length of the rod
        base_area: float
            Cross-sectional area of the rod
        moment_of_inertia: float
            Second moment of area of the rod's cross-section
        young's_modulus: float
            Young's modulus of the rod
        density: float
            Density of the rod
        mode: int
            Index of the first 'mode' th natural frequency.
            Up to the first four modes are supported.
        end_velocity: float
            Initial oscillatory velocity at the end of the rod
    """

    def __init__(
        self,
        base_length,
        base_area,
        moment_of_inertia,
        youngs_modulus,
        density,
        mode,
        end_velocity=0.0,
    ):
        self.base_length = base_length
        self.end_velocity = end_velocity

        if not (isinstance(mode, int) and mode >= 0 and mode < 5):
            raise ValueError(
                "Unsupported mode value, please provide a integer value from 0-4"
            )

        self.mode = mode

        betas = np.array([0.596864, 1.49418, 2.50025, 3.49999]) * np.pi / base_length
        self.beta = betas[mode]
        self.omega = (self.beta ** 2) * np.sqrt(
            youngs_modulus
            * moment_of_inertia
            / (density * base_area * base_length ** 4)
        )

        nonparametrized_mode_at_end = self._compute_nonparametrized_mode(
            base_length, base_length, self.beta
        )
        self.mode_param = end_velocity / (
            -1j * self.omega * nonparametrized_mode_at_end
        )

    def get_initial_velocity_profile(self, positions):
        initial_velocities = []

        for x in positions:
            initial_velocities.append(
                np.real(
                    (-1j * self.omega * self.mode_param)
                    * self._compute_nonparametrized_mode(
                        self.base_length * x, self.base_length, self.beta
                    )
                )
            )
        return np.array(initial_velocities)

    def get_time_dependent_positions(self, position, time_array):
        time_dependent_positions = (
            self.mode_param
            * self._compute_nonparametrized_mode(
                self.base_length * position, self.base_length, self.beta
            )
            * np.exp(-1j * self.omega * time_array)
        )
        return np.real(time_dependent_positions)

    def get_time_dependent_velocities(self, position, time_array):
        time_dependent_velocities = (
            (-1j * self.omega * self.mode_param)
            * self._compute_nonparametrized_mode(
                self.base_length * position, self.base_length, self.beta
            )
            * np.exp(-1j * self.omega * time_array)
        )
        return np.real(time_dependent_velocities)

    def get_omega(self):
        return self.omega

    def get_amplitude(self):
        return self.end_velocity / self.omega

    @staticmethod
    def _compute_nonparametrized_mode(x, base_length, beta):
        a = np.cosh(beta * x) - np.cos(beta * x)
        b = np.cos(beta * base_length) + np.cosh(beta * base_length)
        c = np.sin(beta * base_length) + np.sinh(beta * base_length)
        d = np.sin(beta * x) - np.sinh(beta * x)
        return a + b / c * d
