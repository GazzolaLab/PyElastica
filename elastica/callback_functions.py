__doc__ = """ Call back modules for rod-like objects """


class CallBackBaseClass:
    """
    This is the base class to do callback for rod-like objects.

    Note
    ----
    Every new callback class has to be derived from
    CallBackBaseClass.

    """

    def __init__(self):
        """
        CallBackBaseClass does not need any input parameters.
        """
        pass

    def make_callback(self, system, time, current_step: int):
        """
        This method will be called every time step. Users can define
        which parameters to be called back and recorded. Also users
        can define the sampling rate of these parameters inside this
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
    My call back class is derived from the base call back class.
    This is an example, user can use this class as an example to write
    new call back classes in his/her client file.

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
