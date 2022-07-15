__doc__ = """ Module contains callback classes to save simulation data for rod-like objects """
__all__ = ["CallBackBaseClass", "MyCallBack", "ExportCallBack"]

import os
import sys
import numpy as np
import logging

from collections import defaultdict


class CallBackBaseClass:
    """
    This is the base class for callbacks for rod-like objects.

    Notes
    -----
    Every new callback class must be derived from
    CallBackBaseClass.

    """

    def __init__(self):
        """
        CallBackBaseClass does not need any input parameters.
        """
        pass

    def make_callback(self, system, time, current_step: int):
        """
        This method is called every time step. Users can define
        which parameters are called back and recorded. Also users
        can define the sampling rate of these parameters inside the
        method function.

        Parameters
        ----------
        system : object
            System is a rod-like object.
        time : float
            The time of the simulation.
        current_step : int
            Simulation step.

        """
        pass


class MyCallBack(CallBackBaseClass):
    """
    MyCallBack class is derived from the base callback class.
    This is just an example of a callback class, this class as an example/template to write
    new call back classes in your client file.

        Attributes
        ----------
        sample_every: int
            Collect data using make_callback method every sampling step.
        callback_params: dict
            Collected callback data is saved in this dictionary.
    """

    def __init__(self, step_skip: int, callback_params):
        """

        Parameters
        ----------
        step_skip: int
            Collect data using make_callback method every step_skip step.
        callback_params: dict
            Collected data is saved in this dictionary.
        """
        CallBackBaseClass.__init__(self)
        self.sample_every = step_skip
        self.callback_params = callback_params

    def make_callback(self, system, time, current_step: int):

        if current_step % self.sample_every == 0:

            self.callback_params["time"].append(time)
            self.callback_params["step"].append(current_step)
            self.callback_params["position"].append(system.position_collection.copy())
            self.callback_params["directors"].append(system.director_collection.copy())
            self.callback_params["velocity"].append(system.velocity_collection.copy())

            return


class ExportCallBack(CallBackBaseClass):
    """
    ExportCallback is an example callback class to demonstrate
    how to export rod-data into data file.

    If one wants to customize the saving data, we recommend to
    override `make_callback` method.

        Attributes
        ----------
        AVAILABLE_METHOD
            Supported method to save the file. We recommend
            binary save to maintain the tensor structure of
            data.
        FILE_SIZE_CUTOFF
            Maximum buffer size for each file. If the buffer
            size exceed, new file is created. Actual size of
            the file is expected to be marginally larger.
    """

    AVAILABLE_METHOD = ["pickle", "npz", "tempfile"]
    FILE_SIZE_CUTOFF = 32 * 1e6  # mB

    def __init__(
        self,
        step_skip: int,
        filename: str,
        directory: str,
        method: str,
        initial_file_count: int = 0,
        file_save_interval: int = 1e8,
    ):
        """
        Parameters
        ----------
        step_skip : int
            Interval to collect simulation data into buffer.
            The data will be collected at every `dt * step_skip`
            interval.
        filename : str
            Name of the file without extension. The extension will be
            determined depend on the method. File will be saved with the
            name <filename>_<number>.<extension>.
        directory : str
            Directory to save the file. If directory doesn't exist, it will
            be created. During the save, any existing files in this directory
            could be overwritten.
        method : str
            Method name. Only the name in AVAILABLE_METHOD is
            allowed.
        initial_file_count : int
            Initial file count index that will be appended
        file_save_interval : int
            Interval, in steps, to export/save collected buffer
            as file. (default = 1e8)
        """
        # Assertions
        MIN_STEP_SKIP = 100
        if step_skip <= MIN_STEP_SKIP:
            logging.warning(
                f"We recommend (step_skip={step_skip}) at least {MIN_STEP_SKIP}"
            )
        assert (
            method in ExportCallBack.AVAILABLE_METHOD
        ), f"The exporting method ({method}) is not supported. Please use one of {ExportCallBack.AVAILABLE_METHOD}."

        # Create directory
        if os.path.exists(directory):
            logging.warning(
                f"The directory ({directory}) already exists. Previously saved data could be overwritten."
            )
        os.makedirs(directory, exist_ok=True)

        # Argument Parameters
        self.step_skip = step_skip
        self.save_path = os.path.join(directory, filename) + "_{:02d}.{}"
        self.method = method
        self.file_count = initial_file_count
        self.file_save_interval = file_save_interval

        # Data collector
        self.buffer = defaultdict(list)
        self.buffer_size = 0

        # Module
        if method == ExportCallBack.AVAILABLE_METHOD[0]:
            import pickle

            self._pickle = pickle
            self._ext = "pkl"
        elif method == ExportCallBack.AVAILABLE_METHOD[1]:
            from numpy import savez

            self._savez = savez
            self._ext = "npz"
        elif method == ExportCallBack.AVAILABLE_METHOD[2]:
            import tempfile
            import pickle

            self._tempfile = tempfile.NamedTemporaryFile(delete=False)
            self._pickle = pickle
            self._ext = "pkl"

    def make_callback(self, system, time, current_step: int):
        """

        Parameters
        ----------
        system :
            Each part of the system (i.e. rod, rigid body, etc)
        time :
            simulation time unit
        current_step : int
            simulation step
        """
        if current_step % self.step_skip == 0:
            position = system.position_collection.copy()
            velocity = system.velocity_collection.copy()
            director = system.director_collection.copy()

            self.buffer["time"].append(time)
            self.buffer["step"].append(current_step)
            self.buffer["position"].append(position)
            self.buffer["directors"].append(director)
            self.buffer["velocity"].append(velocity)

            self.buffer_size += (
                sys.getsizeof(position)
                + sys.getsizeof(velocity)
                + sys.getsizeof(director)
            )

        if (
            self.buffer_size > self.FILE_SIZE_CUTOFF
            or (current_step + 1) % self.file_save_interval == 0
        ):
            self._dump()

    def _dump(self, **kwargs):
        """
        Dump dictionary buffer (self.buffer) to a file and clear
        the buffer.
        """
        file_path = self.save_path.format(self.file_count, self._ext)
        data = {k: np.array(v) for k, v in self.buffer.items()}
        if self.method == ExportCallBack.AVAILABLE_METHOD[0]:
            # pickle
            with open(file_path, "wb") as file:
                self._pickle.dump(data, file)
        elif self.method == ExportCallBack.AVAILABLE_METHOD[1]:
            # npz
            self._savez(file_path, **data)
        elif self.method == ExportCallBack.AVAILABLE_METHOD[2]:
            # tempfile
            file = open(self._tempfile.name, "wb")
            self._pickle.dump(data, file)

        self.file_count += 1
        self.buffer_size = 0
        self.buffer.clear()

    def get_last_saved_path(self) -> str:
        """
        Return last saved file path. If no file has been saved,
        return None
        """
        if self.file_count == 0:
            return None
        else:
            return self.save_path.format(self.file_count - 1, self._ext)

    def close(self):
        """
        Save residual buffer
        """
        if self.buffer_size:
            self._dump()

    def clear(self):
        """
        Alias to `close`
        """
        self.close()
