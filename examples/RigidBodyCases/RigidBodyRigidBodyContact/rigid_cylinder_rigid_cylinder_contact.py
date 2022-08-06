import numpy as np
from elastica import *
from post_processing import plot_video_with_surface, plot_velocity


def cylinder_cylinder_contact_case(inclination_angle=0.0):
    class ParallelCylinderCylinderContact(
        BaseSystemCollection, Constraints, Connections, Forcing, CallBacks
    ):
        pass

    cylinder_cylinder_contact_sim = ParallelCylinderCylinderContact()

    # Simulation parameters
    # time step etc
    final_time = 20.0
    time_step = 1e-4
    total_steps = int(final_time / time_step) + 1
    rendering_fps = 30  # 20 * 1e1
    step_skip = int(1.0 / (rendering_fps * time_step))

    cylinder_height = 0.5
    cylinder_radius = 0.1
    density = 1750
    start = np.zeros((3,))
    direction = np.array([np.sin(inclination_angle), 0.0, np.cos(inclination_angle)])
    normal = np.array([0.0, 1.0, 0.0])
    binormal = np.cross(direction, normal)

    rigid_body_one = Cylinder(
        start=start,
        direction=direction,
        normal=normal,
        base_length=cylinder_height,
        base_radius=cylinder_radius,
        density=density,
    )
    cylinder_cylinder_contact_sim.append(rigid_body_one)
    rigid_body_one.velocity_collection[0, :] -= 0.2

    cylinder_start = start + np.array(
        [-1.0, 0.0, 0.0]
    )  # start + binormal #- 2*cylinder_height/3 * direction
    cylinder_direction = np.array([0.0, 0.0, 1.0])  # direction
    cylinder_normal = np.array([0.0, 1.0, 0.0])  # normal

    rigid_body_two = Cylinder(
        start=cylinder_start,
        direction=cylinder_direction,
        normal=cylinder_normal,
        base_length=cylinder_height,
        base_radius=cylinder_radius,
        density=density,
    )
    cylinder_cylinder_contact_sim.append(rigid_body_two)
    # Contact between two rods
    cylinder_cylinder_contact_sim.connect(rigid_body_one, rigid_body_two).using(
        ExternalContactCylinderCylinder, k=1e3, nu=0.001
    )

    # Add call backs
    class RigidCylinderCallBack(CallBackBaseClass):
        """
        Call back function for two arm octopus
        """

        def __init__(
            self, step_skip: int, callback_params: dict, resize_cylinder_elems: int
        ):
            CallBackBaseClass.__init__(self)
            self.every = step_skip
            self.callback_params = callback_params
            self.n_elem_cylinder = resize_cylinder_elems
            self.n_node_cylinder = self.n_elem_cylinder + 1

        def make_callback(self, system, time, current_step: int):
            if current_step % self.every == 0:
                self.callback_params["time"].append(time)
                self.callback_params["step"].append(current_step)

                cylinder_center_position = system.position_collection
                cylinder_length = system.length
                cylinder_direction = system.director_collection[2, :, :].reshape(3, 1)
                cylinder_radius = system.radius

                # Expand cylinder data. Create multiple points on cylinder later to use for rendering.

                start_position = (
                    cylinder_center_position - cylinder_length / 2 * cylinder_direction
                )

                cylinder_position_collection = (
                    start_position
                    + np.linspace(0, cylinder_length[0], self.n_node_cylinder)
                    * cylinder_direction
                )
                cylinder_radius_collection = (
                    np.ones((self.n_elem_cylinder)) * cylinder_radius
                )
                cylinder_length_collection = (
                    np.ones((self.n_elem_cylinder)) * cylinder_length
                )
                cylinder_velocity_collection = (
                    np.ones((self.n_node_cylinder)) * system.velocity_collection
                )

                self.callback_params["position"].append(
                    cylinder_position_collection.copy()
                )
                self.callback_params["velocity"].append(
                    cylinder_velocity_collection.copy()
                )
                self.callback_params["radius"].append(cylinder_radius_collection.copy())
                self.callback_params["com"].append(
                    system.compute_position_center_of_mass()
                )

                self.callback_params["lengths"].append(
                    cylinder_length_collection.copy()
                )
                self.callback_params["com_velocity"].append(
                    system.velocity_collection[..., 0].copy()
                )

                total_energy = (
                    system.compute_translational_energy()
                    + system.compute_rotational_energy()
                )
                self.callback_params["total_energy"].append(total_energy[..., 0].copy())

                return

    post_processing_dict_list = []
    post_processing_dict_list.append(defaultdict(list))
    cylinder_cylinder_contact_sim.collect_diagnostics(rigid_body_one).using(
        RigidCylinderCallBack,
        step_skip=step_skip,
        callback_params=post_processing_dict_list[0],
        resize_cylinder_elems=50,
    )
    # For rigid body
    post_processing_dict_list.append(defaultdict(list))
    cylinder_cylinder_contact_sim.collect_diagnostics(rigid_body_two).using(
        RigidCylinderCallBack,
        step_skip=step_skip,
        callback_params=post_processing_dict_list[1],
        resize_cylinder_elems=50,
    )

    cylinder_cylinder_contact_sim.finalize()
    timestepper = PositionVerlet()

    integrate(timestepper, cylinder_cylinder_contact_sim, final_time, total_steps)

    base_length = cylinder_height
    # Plot the rods
    plot_video_with_surface(
        post_processing_dict_list,
        video_name="cylinder_cylinder_contact.mp4",
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

    filaname = "cylinder_cylinder_velocity.png"
    plot_velocity(
        post_processing_dict_list[0],
        post_processing_dict_list[1],
        filename=filaname,
        SAVE_FIGURE=True,
    )
