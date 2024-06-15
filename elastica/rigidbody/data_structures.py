__doc__ = "Data structure wrapper for rod components"

from elastica.rod.data_structures import _RodSymplecticStepperMixin
from typing import Any

"""
# FIXME : Explicit Stepper doesn't work as States lose the
# views they initially had when working with a timestepper.
class _RigidRodExplicitStepperMixin:
    def __init__(self):
        (
            self.state,
            self.__deriv_state,
            self.position_collection,
            self.director_collection,
            self.velocity_collection,
            self.omega_collection,
            self.acceleration_collection,
            self.alpha_collection,  # angular acceleration
        ) = _bootstrap_from_data(
            "explicit", self.n_elems, self._vector_states, self._matrix_states
        )

    # def __setattr__(self, name, value):
    #     np.copy(self.__dict__[name], value)

    def __call__(self, time, *args, **kwargs):
        self.update_accelerations(time)  # Internal, external

        # print("KRC", self.state.kinematic_rate_collection)
        # print("DEr", self.__deriv_state.rate_collection)
        if np.shares_memory(
            self.state.kinematic_rate_collection,
            self.velocity_collection
            # self.__deriv_state.rate_collection
        ):
            print("Shares memory")
        else:
            print("Explicit states does not share memory")
        return self.__deriv_state
"""


class _RigidRodSymplecticStepperMixin(_RodSymplecticStepperMixin):
    def __init__(self) -> None:
        super(_RigidRodSymplecticStepperMixin, self).__init__()
        # Expose rate returning functions in the interface
        # to be used by the time-stepping algorithm
        # dynamic rates needs to call update_accelerations and henc
        # is another function

    def update_internal_forces_and_torques(self, *args: Any, **kwargs: Any) -> None:
        pass
