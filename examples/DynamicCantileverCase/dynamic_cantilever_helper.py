import numpy as np


class DynamicCantileverVibration:
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

        assert isinstance(mode, int)
        assert mode >= 0
        assert mode < 5
        self.mode = mode

        # First four roots to
        # cosh(beta * base_length) * cos(beta * base_length) + 1 = 0
        #
        # Beta values are used to determine natural frequencies of cantilever beams
        # based on dynamic beam equations
        # omega = beta^2 * sqrt(E * I / mu)
        #
        # For details, refer to
        # https://en.wikipedia.org/wiki/Euler%E2%80%93Bernoulli_beam_theory#Dynamic_beam_equation
        # or
        # Han et al (1998), Dynamics of Transversely Vibrating Beams
        # Using Four Engineering Theories
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
