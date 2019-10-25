__doc__ = """Symplectic timesteppers and concepts"""
import numpy as np

from . import TimeStepper


class SymplecticStepper(TimeStepper):
    def __init__(self):
        super(SymplecticStepper, self).__init__()
        self.__steps = [
            v for (k, v) in self.__class__.__dict__.items() if k.endswith("step")
        ]

    @property
    def n_stages(self):
        return len(self.__steps)

    def do_step(self, System, time: np.float64, dt: np.float64):
        for step in self.__steps:
            time = step(self, System, time, dt)
        return time


class PositionVerlet(SymplecticStepper):
    """
    """

    def __init__(self):
        super(PositionVerlet, self).__init__()

    def _first_kinematic_step(self, System, time: np.float64, dt: np.float64):
        prefac = 0.5 * dt
        System.kinematic_states += prefac * System.dynamic_states
        return time + prefac

    def _first_dynamic_step(self, System, time: np.float64, dt: np.float64):
        System.dynamic_states += dt * System(time, dt)
        return time

    def _second_kinematic_step(self, System, time: np.float64, dt: np.float64):
        return self._first_kinematic_step(System, time, dt)


class PEFRL(SymplecticStepper):
    def __init__(self):
        super(PEFRL, self).__init__()
        # xi and chi are confusing, but be careful!
        self.xi_coeff = np.float64(0.1786178958448091e0)  # ξ
        self.lambda_coeff = -np.float64(0.2123418310626054e0)  # λ
        self.chi_coeff = -np.float64(0.6626458266981849e-1)  # χ

        # Pre-calculate other coefficients
        self.lambda_dash_coeff = 0.5 * (1.0 - 2.0 * self.lambda_coeff)
        self.xi_chi_dash_coeff = 1.0 - 2.0 * (self.xi_coeff + self.chi_coeff)

    def _first_kinematic_step(self, System, time: np.float64, dt: np.float64):
        prefac = self.xi_coeff * dt
        System.kinematic_states += prefac * System.dynamic_states
        return time + prefac

    def _first_dynamic_step(self, System, time: np.float64, dt: np.float64):
        System.dynamic_states += self.lambda_dash_coeff * dt * System(time, dt)
        return time

    def _second_kinematic_step(self, System, time: np.float64, dt: np.float64):
        prefac = self.chi_coeff * dt
        System.kinematic_states += prefac * System.dynamic_states
        return time + prefac

    def _second_dynamic_step(self, System, time: np.float64, dt: np.float64):
        System.dynamic_states += self.lambda_coeff * dt * System(time, dt)
        return time

    def _third_kinematic_step(self, System, time: np.float64, dt: np.float64):
        # Need to fill in
        prefac = self.xi_chi_dash_coeff * dt
        System.kinematic_states += prefac * System.dynamic_states
        return time + prefac

    def _third_dynamic_step(self, System, time: np.float64, dt: np.float64):
        return self._second_dynamic_step(System, time, dt)

    def _fourth_kinematic_step(self, System, time: np.float64, dt: np.float64):
        return self._second_kinematic_step(System, time, dt)

    def _fourth_dynamic_step(self, System, time: np.float64, dt: np.float64):
        return self._first_dynamic_step(System, time, dt)

    def _fifth_kinematic_step(self, System, time: np.float64, dt: np.float64):
        return self._first_kinematic_step(System, time, dt)
        # return time + dt # To avoid numerical precision errors
