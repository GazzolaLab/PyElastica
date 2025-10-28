import numpy as np
from tqdm import tqdm
import elastica as ea
from post_processing import plot_video_with_surface

start = np.zeros((3,))
direction = np.array([0.0, 1.0, 0.0])
normal = np.array([0.0, 0.0, 1.0])
base_length = 0.5
base_radius = 0.1

sphere_radius = 0.10
# overlap_perc = 1.0  # Should be no contact
# overlap_perc = 1.0 + 1e-2  # Should be no contact
overlap_perc = 1.0 - 1e-2  # Contact
sphere_center = np.array(
    [(base_radius + sphere_radius) * overlap_perc, base_length / 2, 0.0]
)


def rotate_random_axis_and_angle(R):
    """
    Randomly rotate the frame for testing purpose.
    """
    from scipy.spatial.transform import Rotation

    axis = np.random.rand(3)
    axis /= np.linalg.norm(axis)
    angle = np.random.rand() * 2 * np.pi
    return R @ Rotation.from_rotvec(angle * axis).as_matrix()


def main():
    class Simulator(
        ea.BaseSystemCollection,
        ea.Constraints,
        ea.Contact,
        ea.CallBacks,
        ea.Forcing,
        ea.Damping,
    ):
        pass

    simulator = Simulator()

    # time step etc
    final_time = 1.0
    time_step = 5e-4
    total_steps = int(final_time / time_step) + 1
    rendering_fps = 30  # 20 * 1e1
    step_skip = 100

    # Add rod
    density = 1000
    E = 3e5
    n_elem = 50

    rod = ea.CosseratRod.straight_rod(
        n_elem,
        start,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        youngs_modulus=E,
    )
    simulator.append(rod)

    simulator.constrain(rod).using(
        ea.FixedConstraint,
        constrained_position_idx=(0, -1),
        constrained_director_idx=(0, -1),
    )

    damping_constant = 1e-1
    simulator.dampen(rod).using(
        ea.AnalyticalLinearDamper,
        damping_constant=damping_constant,
        time_step=time_step,
    )

    # Add sphere
    density = 1000
    n_sphere = 1
    for _ in range(n_sphere):
        rr = rotate_random_axis_and_angle(np.eye(3))
        rigid_body = ea.Sphere(sphere_center, sphere_radius, density)
        rigid_body.director_collection[0] = rr[0][:, None]
        rigid_body.director_collection[1] = rr[1][:, None]
        rigid_body.director_collection[2] = rr[2][:, None]
        simulator.append(rigid_body)

        # Add contact between rigid body and rod
        simulator.detect_contact_between(rod, rigid_body).using(
            ea.RodSphereContact, k=3e4, nu=0.0
        )

    # Add callbacks
    post_processing_dict_list = []

    # For rod
    class StraightRodCallBack(ea.CallBackBaseClass):
        """
        Call back function for two arm octopus
        """

        def __init__(self, step_skip: int, callback_params: dict):
            ea.CallBackBaseClass.__init__(self)
            self.every = step_skip
            self.callback_params = callback_params

        def make_callback(self, system, time, current_step: int):
            if current_step % self.every == 0:
                self.callback_params["time"].append(time)
                self.callback_params["step"].append(current_step)
                self.callback_params["position"].append(
                    system.position_collection.copy()
                )
                self.callback_params["radius"].append(system.radius.copy())
                self.callback_params["com"].append(
                    system.compute_position_center_of_mass()
                )
                total_energy = (
                    system.compute_translational_energy()
                    + system.compute_rotational_energy()
                    + system.compute_bending_energy()
                    + system.compute_shear_energy()
                )
                self.callback_params["total_energy"].append(total_energy)
                return

    class RigidBodyCallback(ea.CallBackBaseClass):
        """
        Call back function for two arm octopus
        """

        def __init__(
            self,
            step_skip: int,
            callback_params: dict,
        ):
            ea.CallBackBaseClass.__init__(self)  # TODO: Is this necessary?
            self.every = step_skip
            self.callback_params = callback_params

        def make_callback(self, system, time, current_step: int):
            if current_step % self.every == 0:
                self.callback_params["time"].append(time)
                self.callback_params["step"].append(current_step)
                self.callback_params["position"].append(
                    system.position_collection.copy()
                )
                self.callback_params["director"].append(
                    system.director_collection.copy()
                )
                self.callback_params["radius"].append(np.array([system.radius.copy()]))
                self.callback_params["com"].append(
                    system.compute_position_center_of_mass()
                )

    post_processing_dict_list.append(ea.defaultdict(list))
    simulator.collect_diagnostics(rod).using(
        StraightRodCallBack,
        step_skip=step_skip,
        callback_params=post_processing_dict_list[0],
    )
    for _ in range(n_sphere):
        # For rigid body
        db = ea.defaultdict(list)
        post_processing_dict_list.append(db)
        simulator.collect_diagnostics(rigid_body).using(
            RigidBodyCallback,
            step_skip=step_skip,
            callback_params=db,
        )
    simulator.finalize()

    timestepper = ea.PositionVerlet()

    time = 0.0
    for i in tqdm(range(total_steps), disable=True):
        time = timestepper.step(simulator, time, time_step)

    # Plot the rods
    plot_video_with_surface(
        post_processing_dict_list,
        video_name="rod_sphere_contact.mp4",
        fps=rendering_fps,
        step=1,
        # The following parameters are optional
        x_limits=(-base_length * 5, base_length * 5),  # Set bounds on x-axis
        y_limits=(-base_length * 5, base_length * 5),  # Set bounds on y-axis
        z_limits=(-base_length * 5, base_length * 5),  # Set bounds on z-axis
        dpi=100,  # Set the quality of the image
        vis3D=True,  # Turn on 3D visualization
        vis2D=True,  # Turn on projected (2D) visualization
    )


if __name__ == "__main__":
    main()
