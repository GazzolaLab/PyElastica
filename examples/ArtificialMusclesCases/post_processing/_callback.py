from elastica import *
import numpy as np

# Add callback functions for plotting position of the rod later on
class MuscleCallBack(CallBackBaseClass):
    """ """

    def __init__(self, step_skip: int, callback_params: dict):
        CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.callback_params = callback_params

    def make_callback(self, system, time, current_step: int):
        if current_step % self.every == 0:
            self.callback_params["time"].append(time)
            self.callback_params["step"].append(current_step)
            self.callback_params["position"].append(system.position_collection.copy())
            self.callback_params["velocity"].append(system.velocity_collection.copy())
            self.callback_params["acceleration"].append(
                system.acceleration_collection.copy()
            )
            self.callback_params["radius"].append(system.radius.copy())
            self.callback_params["com"].append(system.compute_position_center_of_mass())
            self.callback_params["com_velocity"].append(
                system.compute_velocity_center_of_mass()
            )

            total_energy = (
                system.compute_translational_energy()
                + system.compute_rotational_energy()
                + system.compute_bending_energy()
                + system.compute_shear_energy()
            )
            self.callback_params["total_energy"].append(total_energy)
            self.callback_params["directors"].append(system.director_collection.copy())
            self.callback_params["internal_force"].append(system.internal_forces.copy())
            self.callback_params["external_force"].append(system.external_forces.copy())

            return


class MeshRigidBodyCallBack(CallBackBaseClass):
    """ """

    def __init__(self, step_skip: int, callback_params: dict, **kwargs):
        CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.callback_params = callback_params
        self.kwargs = kwargs

    def make_callback(
        self,
        system,
        time,
        current_step: int,
    ):
        if current_step % self.every == 0:
            self.callback_params["time"].append(time)
            self.callback_params["step"].append(current_step)
            self.callback_params["position"].append(system.position_collection.copy())
            self.callback_params["radius"].append(system.radius.copy())
            self.callback_params["com"].append(system.compute_position_center_of_mass())
            self.callback_params["faces"].append(system.faces.copy())
            self.callback_params["face_centers"].append(system.face_centers.copy())
            self.callback_params["face_normal"].append(system.face_normals.copy())
            self.callback_params["directors"].append(system.director_collection.copy())
            system.update_faces()
            return
