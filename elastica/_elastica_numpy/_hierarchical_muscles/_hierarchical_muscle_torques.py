__doc__ = """Hierarchical muscle torques of Elastica Numpy implementation"""
import numpy as np

from elastica.external_forces import NoForces

from elastica._elastica_numpy._linalg import _batch_matvec
from elastica._elastica_numpy._hierarchical_muscles._hierarchical_bases import (
    SplineHierarchySegments,
)

from elastica._calculus import _clip_array


class HierarchicalMuscleTorques(NoForces):
    def __init__(
        self,
        hierarchy_mapper: SplineHierarchySegments,
        activation_func_or_array,
        direction,
        # ramp_up_time=0.0,
        step_skip=200,
        **kwargs
    ):
        super().__init__()
        # takes in rod lengths, returns outpute torques...
        self.torque_generating_hierarchy = hierarchy_mapper
        # function (time) that generates \sum n_bases amount of signals
        # comes from the controller
        self.activation = (
            activation_func_or_array
            if hasattr(activation_func_or_array, "__call__")
            else lambda time_v: activation_func_or_array
        )
        # self.activation = activation_func
        self.direction = (
            direction  # direction.reshape(3, 1)  # Direction in which torque is applied
        )
        self.activation_function_recorder = kwargs.get(
            "activation_function_recorder", None
        )
        self.torque_profile_recorder = kwargs.get("torque_profile_recorder", None)
        self.step_skip = step_skip
        self.counter = 0  # for recording data from the muscles

    # @profile
    def apply_torques(self, system, time: np.float = 0.0):
        # Compute the torque profile for this time-step, controller might change
        # the active and deactive splines.
        instantaneous_activation = np.hstack((self.activation(time)))
        # instantaneous_activation = np.hstack((self.activation))
        # activation_list = self.activation(time)
        # instantaneous_activation = activation_list[0]
        # for k in range(1, len(activation_list)):
        #     instantaneous_activation = np.hstack(
        #         (instantaneous_activation, activation_list[k])
        #     )
        # restrict the values of activation
        instantaneous_activation = _clip_array(instantaneous_activation, -1.0, 1.0)

        torque_magnitude = self.torque_generating_hierarchy(
            system.lengths, instantaneous_activation
        )

        torque = np.einsum("j,ij->ij", torque_magnitude, self.direction.reshape(3, 1))

        # TODO: Find a way without doing tow batch_matvec product
        system.external_torques[..., 1:] += _batch_matvec(
            system.director_collection, torque
        )[..., 1:]
        system.external_torques[..., :-1] -= _batch_matvec(
            system.director_collection[..., :-1], torque[..., 1:]
        )

        self.counter += 1
        if self.counter % self.step_skip == 0:
            if self.activation_function_recorder is not None:
                self.activation_function_recorder["time"].append(time)
                self.activation_function_recorder["second_activation_signal"].append(
                    instantaneous_activation[:][::-1]
                )
                self.activation_function_recorder["first_activation_signal"].append(
                    instantaneous_activation[:][::-1]
                )
            if self.torque_profile_recorder is not None:
                self.torque_profile_recorder["time"].append(time)
                filter = np.zeros(torque_magnitude.shape)
                second_filter = 0.0 * filter
                second_filter[:] = 1.0
                # second_filter = np.ones(torque_magnitude.shape)
                self.torque_profile_recorder["second_torque_mag"].append(
                    torque_magnitude * second_filter
                )
                filter[:] = 1.0
                # filter = np.ones(torque_magnitude.shape)
                self.torque_profile_recorder["first_torque_mag"].append(
                    torque_magnitude * filter
                )
                self.torque_profile_recorder["torque"].append(
                    system.external_torques.copy()
                )
                self.torque_profile_recorder["element_position"].append(
                    np.cumsum(system.lengths)
                )
