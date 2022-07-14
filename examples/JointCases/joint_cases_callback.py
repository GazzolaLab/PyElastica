from elastica.callback_functions import CallBackBaseClass


class JointCasesCallback(CallBackBaseClass):
    """
    Call back function for testing joints
    """

    def __init__(self, step_skip: int, callback_params: dict):
        CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.callback_params = callback_params

    def make_callback(self, system, time, current_step: int):
        if current_step % self.every == 0:
            self.callback_params["time"].append(time)
            self.callback_params["step"].append(current_step)
            self.callback_params["position"].append(system.position_collection.copy())
            self.callback_params["director"].append(system.director_collection.copy())
            self.callback_params["velocity"].append(system.velocity_collection.copy())

        return
