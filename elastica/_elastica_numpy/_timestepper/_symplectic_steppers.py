__doc__ = """Symplectic timesteppers and concepts of Elastica Numpy implementation"""
import numpy as np


class SymplecticStepperTag:
    def __init__(self):
        pass


class PositionVerlet:
    """
    Position Verlet symplectic time stepper class, which
    includes methods for second-order position Verlet.
    """

    Tag = SymplecticStepperTag()

    def __init__(self):
        pass

    def _first_prefactor(self, dt):
        return 0.5 * dt

    def _first_kinematic_step(self, System, time: np.float64, dt: np.float64):
        prefac = self._first_prefactor(dt)
        System.kinematic_states += prefac * System.kinematic_rates(time, prefac)

    def _first_dynamic_step(self, System, time: np.float64, dt: np.float64):
        System.dynamic_states += dt * System.dynamic_rates(
            time, dt
        )  # TODO : Why should we pass dt into System again?


class PEFRL:
    """
    Position Extended Forest-Ruth Like Algorithm of
    I.M. Omelyan, I.M. Mryglod and R. Folk, Computer Physics Communications 146, 188 (2002),
    http://arxiv.org/abs/cond-mat/0110585
    """

    # xi and chi are confusing, but be careful!
    ξ = np.float64(0.1786178958448091e0)  # ξ
    λ = -np.float64(0.2123418310626054e0)  # λ
    χ = -np.float64(0.6626458266981849e-1)  # χ

    # Pre-calculate other coefficients
    lambda_dash_coeff = 0.5 * (1.0 - 2.0 * λ)
    xi_chi_dash_coeff = 1.0 - 2.0 * (ξ + χ)

    Tag = SymplecticStepperTag()

    def __init__(self):
        pass

    def _first_kinematic_prefactor(self, dt):
        return self.ξ * dt

    def _first_kinematic_step(self, System, time: np.float64, dt: np.float64):
        prefac = self._first_kinematic_prefactor(dt)
        System.kinematic_states += prefac * System.kinematic_rates(time, prefac)

    def _first_dynamic_step(self, System, time: np.float64, dt: np.float64):
        prefac = self.lambda_dash_coeff * dt
        System.dynamic_states += prefac * System.dynamic_rates(time, prefac)

    def _second_kinematic_prefactor(self, dt):
        return self.χ * dt

    def _second_kinematic_step(self, System, time: np.float64, dt: np.float64):
        prefac = self._second_kinematic_prefactor(dt)
        System.kinematic_states += prefac * System.kinematic_rates(time, prefac)

    def _second_dynamic_step(self, System, time: np.float64, dt: np.float64):
        prefac = self.λ * dt
        System.dynamic_states += prefac * System.dynamic_rates(time, prefac)

    def _third_kinematic_prefactor(self, dt):
        return self.xi_chi_dash_coeff * dt

    def _third_kinematic_step(self, System, time: np.float64, dt: np.float64):
        prefac = self._third_kinematic_prefactor(dt)
        # Need to fill in
        System.kinematic_states += prefac * System.kinematic_rates(time, prefac)
