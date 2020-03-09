import numpy as np
import sys
from tqdm import tqdm

sys.path.append("../../../")
from examples.TapperedArmDecayingBasisFunctions.set_environment import Environment


def segment_activation_function(number_of_muscle_segments, time):
    """
    This function is an example activation function for users. Similar to
    this function users can write their own activation function.
    Note that it is important to set correctly activation array sizes, which
    is number of basis functions for that muscle segment. Also users has to
    pack activation arrays in correct order at the return step, thus correct
    activation array activates correct muscle segment.
    Note that, activation array values can take numbers between -1 and 1. If you
    put numbers different than these, Elastica clips the values. If activation value
    is -1, this basis function generates torque in opposite direction.
    Parameters
    ----------
    time

    Returns
    -------

    """

    def ramped_up(shifted_time, threshold=0.1):
        return (
            0.0
            if shifted_time < 0.0
            else (
                1.0
                if shifted_time > threshold
                else 0.5 * (1.0 - np.cos(np.pi * shifted_time / threshold))
            )
        )

    # Muscle segment acting in first bending direction or normal direction
    activation_arr_in_normal_direction = np.zeros((number_of_muscle_segments))
    activation_arr_in_normal_direction[:] = 1.0 * ramped_up(time - 1.0, 0.1)

    # Muscle segment acting in second bending direction or binormal direction
    activation_arr_in_binormal_direction = np.zeros((number_of_muscle_segments))
    activation_arr_in_binormal_direction[:] = 1.0 * ramped_up(time - 1.0, 0.1)

    # Muscle segment acting in twist direction or tangent direction
    activation_arr_in_tangent_direction = np.zeros((number_of_muscle_segments))
    activation_arr_in_tangent_direction[:] = 1.0 * ramped_up(time - 1.0, 0.1)

    return [
        [activation_arr_in_normal_direction],  # activation in normal direction
        [activation_arr_in_binormal_direction],  # activation in binormal direction
        [activation_arr_in_tangent_direction],  # activation in tangent direction
    ]


# User defined condition for exiting the simulation
def user_defined_condition_function(reward, systems, time):
    """
    This function will be defined by the user. Depending on
    the controller requirements and system states, with returning
    done=True boolean, simulation can be exited before reaching
    final simulation time. This function is thought for stopping the
    simulation if desired reward is reached.
    Parameters
    ----------
    reward: user defined reward
    systems: [shearable rod, rigid_rod] classes
    time: current simulation time

    Returns
    -------
    done: boolean
    """
    done = False
    # rod = systems[0]  # shearable rod or cyber-octopus
    # cylinder = systems[1]  # rigid body or target object
    if time > 20.0:
        done = True

    return done


def main():
    # Set simulation final time
    final_time = 20
    # Number of muscle segments
    number_of_muscle_segments = 10

    env = Environment(
        final_time,
        number_of_muscle_segments,
        alpha=10,
        COLLECT_DATA_FOR_POSTPROCESSING=True,
    )

    # Do multiple simulations for learning, or control
    for i_episodes in range(1):

        # Initialize the environment
        # Make RANDOM_TARGET_POSITION=True for randomly initialize cylinder for task 6
        RANDOM_TARGET_POSITION = False
        if RANDOM_TARGET_POSITION:
            alpha = 2 * np.pi * np.random.sample()
            target_position = np.array([1.0 * np.sin(alpha), 1.0 * np.cos(alpha), 0.0])
        else:
            target_position = np.array(
                [0.8, -0.6, 0.0]
            )  # target object initial position

        # Reset the environment before the new episode and get total number of simulation steps
        total_steps, systems = env.reset(target_position)

        # Simulation loop starts
        time = np.float64(0.0)
        user_defined_condition = False
        activation = segment_activation_function(number_of_muscle_segments, time)
        reward = 0.0

        for i_sim in tqdm(range(total_steps)):

            """ Learning loop """
            # Reward and activation does not have to be computed or updated every simulation step.
            # Simulation time step is chosen to satisfy numerical stability of Elastica simulation.
            # However, learning time step can be larger. For example in the below if loop,
            # we are updating activation every 200 step.
            if i_sim % 200:
                """ Use systems for observations """
                # Observations can be rod parameters and can be accessed after every time step.
                # shearable_rod.position_collection = position of the elements ( here octopus )
                # shearable_rod.velocity_collection = velocity of the elements ( here octopus )
                # rigid_body.position_collection = position of the rigid body (here target object)
                shearable_rod = systems[0]
                rigid_body = systems[1]

                """Reward function should be here"""
                # User has to define his/her own reward function
                reward = 0.0
                """Reward function should be here"""

                """ Compute the activation signal and pass to environment """
                # Based on the observations and reward function, have the learning algorithm
                # update the muscle activations. Make sure that the activation arrays are packaged
                # properly. See the segment_activation_function function defined above for an
                # example of manual activations.
                activation = segment_activation_function(
                    number_of_muscle_segments, time
                )

            # Do one simulation step. This function returns the current simulation time,
            # systems which are shearable rod (octopus) and rigid body, and done condition.
            time, systems, done = env.step(activation, time)

            """ User defined condition to exit simulation loop """
            # Below function has to be defined by the user. If user wants to exit the simulation
            # after some condition is reached before simulation completed, user
            # has to return a True boolean.
            user_defined_condition = user_defined_condition_function(
                reward, systems, time
            )
            if user_defined_condition == True:
                print(" User defined condition satisfied, exit simulation")
                print(" Episode finished after {} ".format(time))
                break

            # If done=True, NaN detected in simulation.
            # Exit the simulation loop before, reaching final time
            if done:
                print(" Episode finished after {} ".format(time))
                break

        print("Final time of simulation is : ", time)
        # Simulation loop ends

        # Post-processing
        # Make a video of octopus for current simulation episode. Note that
        # in order to make a video, COLLECT_DATA_FOR_POSTPROCESSING=True
        env.post_processing(
            filename_video="two_arm_simulation_3d_with_target.mp4",
            # The following parameters are optional
            x_limits=(-1.0, 1.0),  # Set bounds on x-axis
            y_limits=(-1.0, 1.0),  # Set bounds on y-axis
            z_limits=(-0.05, 1.00),  # Set bounds on z-axis
            dpi=100,  # Set the quality of the image
            vis3D=True,  # Turn on 3D visualization
            vis2D=True,  # Turn on projected (2D) visualization
        )

    return env


if __name__ == "__main__":
    env = main()

    positions = env.shearable_rod.position_collection
    avg_positions = 0.5 * (positions[..., :-1] + positions[..., 1:])
    radius = env.shearable_rod.radius

    from matplotlib import pyplot as plt
    from matplotlib.colors import to_rgb
    import matplotlib.animation as manimation
    from mpl_toolkits.mplot3d import proj3d, Axes3D

    fig = plt.figure(1, figsize=(10, 8), frameon=True, dpi=100)
    ax = plt.axes(projection="3d")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([-0.1, 1.1])
    ax.set_zlim([0, 0.5])

    # Rods next
    scatt = ax.scatter(
        avg_positions[0],
        avg_positions[1],
        avg_positions[2],
        s=np.pi * radius ** 2 * 1e4,
    )
    plt.show()
