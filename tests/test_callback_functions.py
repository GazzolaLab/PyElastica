__doc__ = """ Call back functions for rod test module """
import sys

# System imports
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from elastica.callback_functions import CallBackBaseClass, MyCallBack
from elastica.utils import Tolerance
import pytest


def mock_rod_init(self):
    self.n_elems = 0.0
    self.position_collection = 0.0
    self.velocity_collection = 0.0
    self.director_collection = 0.0
    self.external_forces = 0.0
    self.external_torques = 0.0


MockRod = type("MockRod", (object,), {"__init__": mock_rod_init})


class TestCallBackBaseClass:
    def test_call_back_base_class(self):
        """
        This test case tests, base class for call functions. make_callback
        does not do anything, but for completion this test case is here.
        Returns
        -------

        """

        mock_rod = MockRod()

        time = 0.0
        current_step = 0
        callbackbase = CallBackBaseClass()
        callbackbase.make_callback(mock_rod, time, current_step)

        assert_allclose(mock_rod.position_collection, 0.0, atol=Tolerance.atol())
        assert_allclose(mock_rod.velocity_collection, 0.0, atol=Tolerance.atol())
        assert_allclose(mock_rod.director_collection, 0.0, atol=Tolerance.atol())
        assert_allclose(mock_rod.external_forces, 0.0, atol=Tolerance.atol())
        assert_allclose(mock_rod.external_torques, 0.0, atol=Tolerance.atol())


class TestMyCallBackClass:
    @pytest.mark.parametrize("n_elem", [2, 4, 16])
    def test_my_call_back_base_class(self, n_elem):
        """
        This test case is for testing MyCallBack function.
        Parameters
        ----------
        n_elem

        Returns
        -------

        """

        mock_rod = MockRod()

        time = np.random.rand(10)
        current_step = list(range(10))
        position_collection = np.random.rand(3, 10)
        velocity_collection = np.random.rand(3, 10)
        director_collection = np.random.rand(3, 3, 10)

        # set arrays in mock rod
        mock_rod.n_elems = n_elem
        mock_rod.position_collection = position_collection
        mock_rod.velocity_collection = velocity_collection
        mock_rod.director_collection = director_collection

        step_skip = 1
        list_test = {
            "time": [],
            "step": [],
            "position": [],
            "velocity": [],
            "directors": [],
        }
        list_correct = {
            "time": [],
            "step": [],
            "position": [],
            "velocity": [],
            "directors": [],
        }

        mycallback = MyCallBack(step_skip, list_test)
        for i in range(10):
            mycallback.make_callback(mock_rod, time[i], current_step[i])

            list_correct["time"].append(time[i])
            list_correct["step"].append(current_step[i])
            list_correct["position"].append(position_collection)
            list_correct["velocity"].append(velocity_collection)
            list_correct["directors"].append(director_collection)

        assert_allclose(list_test["time"], list_correct["time"], atol=Tolerance.atol())
        assert_allclose(list_test["step"], list_correct["step"], atol=Tolerance.atol())
        assert_allclose(
            list_test["position"], list_correct["position"], atol=Tolerance.atol()
        )
        assert_allclose(
            list_test["velocity"], list_correct["velocity"], atol=Tolerance.atol()
        )
        assert_allclose(
            list_test["directors"], list_correct["directors"], atol=Tolerance.atol()
        )
