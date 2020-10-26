__doc__ = """ Module contains callback classes to save simulation data for rod-like objects """
__all__ = ["CallBackBaseClass", "MyCallBack"]


class CallBackBaseClass:
    """
    This is the base class for callbacks for rod-like objects.

    Note
    ----
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

        Returns
        -------

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
