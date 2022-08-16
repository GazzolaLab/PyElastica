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
        end_velocity_z=0.0,
    ):
        self.base_length = base_length
        self.end_velocity_z = end_velocity_z

        assert isinstance(mode, int)
        assert mode >= 0
        assert mode < 5
        self.mode = mode

        betas = np.array([0.596864, 1.49418, 2.50025, 3.49999]) * np.pi / base_length
        self.beta = betas[mode]
        self.omega = (self.beta ** 2) * np.sqrt(
            youngs_modulus
            * moment_of_inertia
            / (density * base_area * base_length ** 4)
        )

        np_mode_l = self._compute_nonparametrized_mode(
            base_length, base_length, self.beta
        )
        self.param = end_velocity_z / (-1j * self.omega * np_mode_l)

    def get_initial_position_profile(self, x_coords):
        positions = []

        for x in x_coords:
            positions.append(
                np.real(
                    self.param
                    * self._compute_nonparametrized_mode(
                        self.base_length * x, self.base_length, self.beta
                    )
                )
            )
        return np.array(positions)

    def get_initial_velocity_profile(self, x_coords):
        velocities = []

        for x in x_coords:
            velocities.append(
                np.real(
                    (-1j * self.omega * self.param)
                    * self._compute_nonparametrized_mode(
                        self.base_length * x, self.base_length, self.beta
                    )
                )
            )
        return np.array(velocities)

    def get_positions(self, x_coord, time_array):
        positions = (
            self.param
            * self._compute_nonparametrized_mode(
                self.base_length * x_coord, self.base_length, self.beta
            )
            * np.exp(-1j * self.omega * time_array)
        )
        return np.real(positions)

    def get_velocities(self, x_coord, time_array):
        velocities = (
            (-1j * self.omega * self.param)
            * self._compute_nonparametrized_mode(
                self.base_length * x_coord, self.base_length, self.beta
            )
            * np.exp(-1j * self.omega * time_array)
        )
        return np.real(velocities)

    def get_omega(self):
        return self.omega

    def get_amplitude(self):
        return self.end_velocity_z / self.omega

    @staticmethod
    def _compute_nonparametrized_mode(x, base_length, betas):
        a = np.cosh(betas * x) - np.cos(betas * x)
        b = np.cos(betas * base_length) + np.cosh(betas * base_length)
        c = np.sin(betas * base_length) + np.sinh(betas * base_length)
        d = np.sin(betas * x) - np.sinh(betas * x)
        return a + b / c * d
