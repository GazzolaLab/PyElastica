__doc__ = """ Call back functions for rod test module """
import os, sys

# System imports
import logging
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from elastica.callback_functions import CallBackBaseClass, MyCallBack, ExportCallBack
from elastica.utils import Tolerance
import tempfile
import pytest


class MockRod:
    def __init__(self):
        self.n_elems = 0.0
        self.position_collection = 0.0
        self.velocity_collection = 0.0
        self.director_collection = 0.0
        self.external_forces = 0.0
        self.external_torques = 0.0


class MockRodWithElements:
    def __init__(self, n_elems):
        self.n_elems = n_elems
        self.position_collection = np.random.rand(3, n_elems)
        self.velocity_collection = np.random.rand(3, n_elems)
        self.director_collection = np.random.rand(3, 3, n_elems)
        self.external_forces = np.random.rand(3, n_elems)
        self.external_torques = np.random.rand(3, n_elems)


class TestCallBackBaseClass:
    def test_call_back_base_class(self):
        """
        This test case tests, base class for call functions. make_callback
        does not do anything, but for completion this test case is here.
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
    @pytest.mark.parametrize("n_elems", [2, 4, 16])
    def test_my_call_back_base_class(self, n_elems):
        """
        This test case is for testing MyCallBack function.
        """
        mock_rod = MockRodWithElements(n_elems)

        time = np.random.rand(10)
        current_step = list(range(10))

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
            list_correct["position"].append(mock_rod.position_collection.copy())
            list_correct["velocity"].append(mock_rod.velocity_collection.copy())
            list_correct["directors"].append(mock_rod.director_collection.copy())

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


class TestExportCallBackClass:
    @pytest.mark.parametrize("method", ["0", 1, "numba", "test", "some string", None])
    def test_export_call_back_unavailable_save_methods(self, method):
        with pytest.raises(AssertionError) as excinfo:
            callback = ExportCallBack(1, "rod", "tempdir", method)

    @pytest.mark.parametrize("method", ExportCallBack.AVAILABLE_METHOD)
    def test_export_call_back_available_save_methods(self, method):
        try:
            callback = ExportCallBack(1, "rod", "tempdir", method)
        except Error:
            pytest.fail(
                f"Could not create callback module with available method {method}"
            )

    @pytest.mark.parametrize("step_skip", [2, 5, 20, 50, 99])
    def test_export_call_back_small_stepsize_warning(self, caplog, step_skip):
        mock_rod = MockRodWithElements(5)
        with tempfile.TemporaryDirectory() as temp_dir_path:
            ExportCallBack(step_skip, "rod", temp_dir_path, "npz")
        record_tuple = caplog.record_tuples[0]
        assert record_tuple[0] == "root"
        assert record_tuple[1] == logging.WARNING
        assert str(100) in record_tuple[2]
        assert f"recommend (step_skip={step_skip}) at least" in record_tuple[2]

    def test_export_call_back_file_recreate_warning(self, caplog):
        mock_rod = MockRodWithElements(5)
        with tempfile.TemporaryDirectory() as temp_dir_path:
            ExportCallBack(1000, "rod", temp_dir_path, "npz")
            ExportCallBack(1000, "rod", temp_dir_path, "npz")
        record_tuple = caplog.record_tuples[0]
        assert record_tuple[0] == "root"
        assert record_tuple[1] == logging.WARNING
        assert "already exists" in record_tuple[2]

    def test_export_call_back_interval_by_filesize(self):
        mock_rod = MockRodWithElements(5)
        with tempfile.TemporaryDirectory() as temp_dir_path:
            callback = ExportCallBack(1, "rod", temp_dir_path, "npz")
            callback.FILE_SIZE_CUTOFF = 1
            callback.make_callback(mock_rod, 1, 1)

            saved_path_name = callback.save_path.format(0, "npz")
            assert os.path.exists(saved_path_name), "File is not saved."

    @pytest.mark.parametrize("file_save_interval", [1, 5, 10, 15])
    def test_export_call_back_file_save_interval_param(self, file_save_interval):
        mock_rod = MockRodWithElements(5)
        with tempfile.TemporaryDirectory() as temp_dir_path:
            callback = ExportCallBack(
                1, "rod", temp_dir_path, "npz", file_save_interval=file_save_interval
            )
            for step in range(file_save_interval):
                callback.make_callback(mock_rod, 1, step)

            saved_path_name = callback.get_last_saved_path()
            assert os.path.exists(saved_path_name), "File is not saved."

    @pytest.mark.parametrize("step_skip", [2, 5, 10, 15])
    def test_export_call_back_step_skip_param(self, step_skip):
        mock_rod = MockRodWithElements(5)
        with tempfile.TemporaryDirectory() as temp_dir_path:
            callback = ExportCallBack(step_skip, "rod", temp_dir_path, "npz")
            callback.make_callback(mock_rod, 1, step_skip - 1)
            # Check empty
            callback.clear()
            saved_path_name = callback.get_last_saved_path()
            assert saved_path_name is None, "No file should be saved."

            # Check saved
            callback.make_callback(mock_rod, 1, step_skip)
            callback.clear()
            saved_path_name = callback.get_last_saved_path()
            assert saved_path_name is not None, "File should be saved."
            assert os.path.exists(saved_path_name), "File should be saved"

            # Check saved file number
            callback.make_callback(mock_rod, 1, step_skip * 2)
            callback.clear()
            callback.make_callback(mock_rod, 1, step_skip * 5)
            callback.clear()
            saved_path_name = callback.get_last_saved_path()
            assert (
                str(2) in saved_path_name
            ), f"Total 3 file should be saved: {saved_path_name}"

    @pytest.mark.parametrize("file_save_interval", [5, 10])
    def test_export_call_back_file_save_interval_param_ext(self, file_save_interval):
        mock_rod = MockRodWithElements(5)
        n_repeat = 3
        with tempfile.TemporaryDirectory() as temp_dir_path:
            callback = ExportCallBack(
                1, "rod", temp_dir_path, "npz", file_save_interval=file_save_interval
            )
            for step in range(file_save_interval * n_repeat):
                callback.make_callback(mock_rod, 1, step)

            saved_path_name = callback.get_last_saved_path()
            assert os.path.exists(saved_path_name), "File is not saved."

    def test_export_call_back_file_not_saved(self):
        with tempfile.TemporaryDirectory() as temp_dir_path:
            callback = ExportCallBack(
                1, "rod", temp_dir_path, "npz", file_save_interval=10
            )
            saved_path_name = callback.get_last_saved_path()
            assert saved_path_name is None, f"{saved_path_name} should be None"

    def test_export_call_back_close_test(self):
        mock_rod = MockRodWithElements(5)
        with tempfile.TemporaryDirectory() as temp_dir_path:
            callback = ExportCallBack(
                1, "rod", temp_dir_path, "npz", file_save_interval=50
            )
            for step in range(10):
                callback.make_callback(mock_rod, 1, step)
            callback.close()
            saved_path_name = callback.get_last_saved_path()
            assert os.path.exists(saved_path_name), "File is not saved."

    def test_export_call_back_clear_test(self):
        mock_rod = MockRodWithElements(5)
        with tempfile.TemporaryDirectory() as temp_dir_path:
            callback = ExportCallBack(
                1, "rod", temp_dir_path, "npz", file_save_interval=50
            )
            for step in range(10):
                callback.make_callback(mock_rod, 1, step)
            callback.clear()
            saved_path_name = callback.get_last_saved_path()
            assert os.path.exists(saved_path_name), "File is not saved."

    @pytest.mark.parametrize("n_elems", [2, 4, 16])
    def test_export_call_back_class_tempfile_option(self, n_elems):
        """
        This test case is for testing ExportCallBack function, saving into temporary files.
        """
        import pickle

        mock_rod = MockRodWithElements(n_elems)
        time = np.random.rand(10)
        current_step = list(range(10))

        step_skip = 1
        list_correct = {
            "time": [],
            "step": [],
            "position": [],
            "velocity": [],
            "directors": [],
        }

        callback = ExportCallBack(
            step_skip, "rod", "tempdir", "tempfile", file_save_interval=10
        )
        for i in range(10):
            callback.make_callback(mock_rod, time[i], current_step[i])

            list_correct["time"].append(time[i])
            list_correct["step"].append(current_step[i])
            list_correct["position"].append(mock_rod.position_collection)
            list_correct["velocity"].append(mock_rod.velocity_collection)
            list_correct["directors"].append(mock_rod.director_collection)

        file = open(callback._tempfile.name, "rb")
        list_test = pickle.load(file)
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

        callback._tempfile.close()

    @pytest.mark.parametrize("n_elems", [2, 4, 16])
    def test_export_call_back_class_npz_option(self, n_elems):
        """
        This test case is for testing ExportCallBack function, saving into numpy files.
        """
        filename = "test_rod"
        mock_rod = MockRodWithElements(n_elems)
        time = np.random.rand(10)
        current_step = list(range(10))

        step_skip = 1
        list_correct = {
            "time": [],
            "step": [],
            "position": [],
            "velocity": [],
            "directors": [],
        }

        with tempfile.TemporaryDirectory() as temp_dir_path:
            callback = ExportCallBack(
                step_skip, filename, temp_dir_path, "npz", file_save_interval=10
            )
            for i in range(10):
                callback.make_callback(mock_rod, time[i], current_step[i])

                list_correct["time"].append(time[i])
                list_correct["step"].append(current_step[i])
                list_correct["position"].append(mock_rod.position_collection)
                list_correct["velocity"].append(mock_rod.velocity_collection)
                list_correct["directors"].append(mock_rod.director_collection)

            saved_path_name = callback.get_last_saved_path()
            assert os.path.exists(saved_path_name), "File does not exist"
            list_test = np.load(saved_path_name)

            assert_allclose(
                list_test["time"], list_correct["time"], atol=Tolerance.atol()
            )
            assert_allclose(
                list_test["step"], list_correct["step"], atol=Tolerance.atol()
            )
            assert_allclose(
                list_test["position"], list_correct["position"], atol=Tolerance.atol()
            )
            assert_allclose(
                list_test["velocity"], list_correct["velocity"], atol=Tolerance.atol()
            )
            assert_allclose(
                list_test["directors"], list_correct["directors"], atol=Tolerance.atol()
            )

    @pytest.mark.parametrize("n_elems", [2, 4, 16])
    def test_export_call_back_class_pickle_option(self, n_elems):
        """
        This test case is for testing ExportCallBack function, saving into pickle files.
        """
        import pickle

        filename = "test_rod"
        mock_rod = MockRodWithElements(n_elems)
        time = np.random.rand(10)
        current_step = list(range(10))

        step_skip = 1
        list_correct = {
            "time": [],
            "step": [],
            "position": [],
            "velocity": [],
            "directors": [],
        }

        with tempfile.TemporaryDirectory() as temp_dir_path:
            callback = ExportCallBack(
                step_skip, filename, temp_dir_path, "pickle", file_save_interval=10
            )
            for i in range(10):
                callback.make_callback(mock_rod, time[i], current_step[i])

                list_correct["time"].append(time[i])
                list_correct["step"].append(current_step[i])
                list_correct["position"].append(mock_rod.position_collection)
                list_correct["velocity"].append(mock_rod.velocity_collection)
                list_correct["directors"].append(mock_rod.director_collection)

            saved_path_name = callback.get_last_saved_path()
            assert os.path.exists(saved_path_name), "File does not exist"
            file = open(saved_path_name, "rb")
            list_test = pickle.load(file)

            assert_allclose(
                list_test["time"], list_correct["time"], atol=Tolerance.atol()
            )
            assert_allclose(
                list_test["step"], list_correct["step"], atol=Tolerance.atol()
            )
            assert_allclose(
                list_test["position"], list_correct["position"], atol=Tolerance.atol()
            )
            assert_allclose(
                list_test["velocity"], list_correct["velocity"], atol=Tolerance.atol()
            )
            assert_allclose(
                list_test["directors"], list_correct["directors"], atol=Tolerance.atol()
            )
