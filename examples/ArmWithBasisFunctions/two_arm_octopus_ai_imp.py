import numpy as np
import sys
from tqdm import tqdm

sys.path.append("../../")
from examples.ArmWithBasisFunctions.set_environment import Environment


def segment_activation_function(time):
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

    # Muscle segment at the first arm, acting in first bending direction or normal direction
    activation_arr_1 = np.zeros((7))
    # Top level muscle segment
    activation_arr_1[0] = ramped_up(time - 1.0, 0.1)
    # Mid level muscle segment
    activation_arr_1[1:3] = ramped_up(time - 0.9, 0.1)
    # Bottom level muscle segment
    activation_arr_1[3:7] = ramped_up(time - 0.8, 0.1)

    # Muscle segment at the second arm, acting in first bending direction or normal direction
    activation_arr_2 = np.zeros((7))
    # Top level muscle segment
    activation_arr_2[0] = ramped_up(time - 1.0, 0.1)
    # Mid level muscle segment
    activation_arr_2[1:3] = ramped_up(time - 0.9, 0.1)
    # Bottom level muscle segment
    activation_arr_2[3:7] = ramped_up(time - 0.8, 0.1)

    # Muscle segment at the first arm, acting in second bending direction or binormal direction
    activation_arr_3 = np.zeros((7))
    # Top level muscle segment
    activation_arr_3[0] = ramped_up(time - 1.0, 0.1)
    # Mid level muscle segment
    activation_arr_3[1:3] = ramped_up(time - 0.9, 0.1)
    # Bottom level muscle segment
    activation_arr_3[3:7] = ramped_up(time - 0.8, 0.1)

    # Muscle segment at the second arm, acting in second bending direction or binormal direction
    activation_arr_4 = np.zeros((7))
    # Top level muscle segment
    activation_arr_4[0] = ramped_up(time - 1.0, 0.1)
    # Mid level muscle segment
    activation_arr_4[1:3] = ramped_up(time - 0.9, 0.1)
    # Bottom level muscle segment
    activation_arr_4[3:7] = ramped_up(time - 0.8, 0.1)

    # Muscle segment at the first arm, acting in twist direction or tangent direction
    activation_arr_5 = np.zeros((7))
    # Top level muscle segment
    activation_arr_5[0] = ramped_up(time - 1.0, 0.1)
    # Mid level muscle segment
    activation_arr_5[1:3] = ramped_up(time - 0.9, 0.1)
    # Bottom level muscle segment
    activation_arr_5[3:7] = ramped_up(time - 0.8, 0.1)

    # Muscle segment at the second arm, acting in twist direction or tangent direction
    activation_arr_6 = np.zeros((7))
    # Top level muscle segment
    activation_arr_6[0] = -1.0 * ramped_up(time - 1.0, 0.1)
    # Mid level muscle segment
    activation_arr_6[1:3] = -1.0 * ramped_up(time - 0.9, 0.1)
    # Bottom level muscle segment
    activation_arr_6[3:7] = -1.0 * ramped_up(time - 0.8, 0.1)

    return [
        [activation_arr_1, activation_arr_2],  # activation in normal direction
        [activation_arr_3, activation_arr_4],  # activation in binormal direction
        [activation_arr_5, activation_arr_6],  # activation in tangent direction
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
    rod = systems[0]  # shearable rod or cyber-octopus
    cylinder = systems[1]  # rigid body or target object
    if time > 20.0:
        done = True

    return done


def main():
    # Set simulation final time
    final_time = 10.0

    # Initialize the environment
    target_position = np.array([-0.4, 0.5, 0.0])
    # np.array([-0.4, 0.0, 0.4]) # target object initial position
    # For task 6 uncomment the below code and show that your algorithm can drive
    # octopus towards the random target.
    # alpha = np.random.sample()
    # target_position = np.array([-0.4*np.sin(alpha), 0.0, 0.5 + 0.4*np.cos(alpha)])
    env = Environment(final_time, target_position, COLLECT_DATA_FOR_POSTPROCESSING=True)
    total_steps, systems = env.reset()

    # Do multiple simulations for learning, or control
    for i_episodes in range(1):

        # Reset the environment before the new episode and get total number of simulation steps
        total_steps, systems = env.reset()

        # Simulation loop starts
        time = np.float64(0.0)
        user_defined_condition = False
        activation = segment_activation_function(time)
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
                activation = segment_activation_function(time)

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
        env.post_processing(filename_video="two_arm_simulation_3d_with_target.mp4")


if __name__ == "__main__":
    main()
