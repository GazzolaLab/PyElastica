# Hackathon Readme

> __NCSA-NVIDIA AI Hackathon__ held at the University of Illinois from March 7-8 2020.

## Problem Statement
The objective is to train a model to move a (cyber)-octopus with two soft arms and a head to reach a target location, and then grab an object. The octopus is modeled as an assembly of Cosserat rods and is activated by muscles surrounding its arms. Input to the mechanical model is the activation signals to the surrounding muscles, which causes it to contract, thus moving the arms. The output of the model comes from the octopus' environment. The mechanical model will be provided both for the octopus and its interaction with its environment. The goal is to find the correct muscle activation signals that make the octopus crawl to reach the target location and then make one arm to grab the object.

## Progression of specific goals
These goals build on each other, you need to successfully accomplish all prior goals to get credit for later goals.  

1) Make octopus crawl towards some direction. (5 points)
2) Make your octopus crawl to the target location. (7.5 points)  
3) Make octopus to move the object using its arms. (7.5 points)
4) Have your octopus grab the object by wrapping one arm around the object. (10 points) 
5) Make your octopus return to its starting location with the object. (20 points)
6) Generalize your policy to perform these tasks for an arbitrarily located object. (50 points)   

## Problem Context
Octopuses have flexible limbs made up of muscles with no internal bone structure. These limbs, know as muscular hydrostats, have an almost infinite number of degrees of freedom, allowing an octopus to perform complex actions with its arms, but also making them difficult to mathematically model. Attempts to model octopus arms are motivated not only by a desire to understand them biologically, but also to adapt their control ability and decision making processes to the rapidly developing field of soft robotics. We have developed a simulation package Elastica that models flexible 1-d rods, which can be used to represent octopus arms as a long, slender rod. We now want to learn methods for controlling these arms. 

You are being provided with a model of an octopus that consists of two arms connected by a head. Each arm can be controlled independently. These arms are actuated through the contraction of muscles in the arms. This muscle activation produces a torque profile along the arm, resulting in movement of the arm. The arms interact with the ground through friction. Your goal is to teach the octopus to crawl towards an object, grab it, and bring it back to where the octopus started. 

## Controlling octopus arms with hierarchical basis functions
For this problem, we abstract the activation of the octopus muscles to the generation of a torque profile defined by the activation of a set of hierarchical radial basis function. Here we are using Gaussian basis functions. 

<img src="https://github.com/GazzolaLab/PyElastica/blob/assets/archive/basis.png?raw=true" alt="image name" width="400"/>

<img src="https://github.com/GazzolaLab/PyElastica/blob/assets/archive/rotation.png?raw=true" alt="image name" width="500"/>

There are three levels of these basis functions, with 1 basis function in the first level, 2 in the second level and 4 in the third, leading to 7 basis functions in set. These levels have different maximum levels of activation. The lower levels have larger magnitudes than the higher levels, meaning they represent bulk motion of the rod while the higher levels allow finer control of the rod along the interval. In the code, the magnitude of each level will be fixed but you can choose the amount of activation at each level by setting the activation level between -1 and 1. 

There are two bending modes (in the normal and binormal directions) and a twisting mode (in the tangent direction), so we define torques in these three different directions and independently for each arm. This yields six different sets of basis functions that can be activated for a total of 42 inputs. 


## Overview of provided Elastica code
We are providing you the Elastica software package which is written in Python. Elastica simulates the dynamics and kinematics of 1-d slender rods. We have set up the model for you such that you do not need to worry about the details of the model, only the activation patterns of the muscle. 
In the provided `examples/ArmWithBasisFunctions/two_arm_octopus_ai_imp.py` file you will import the `Environment` class which will define and setup the simulation. 

`Environment` has three relevant functions:  
* `Environment.reset(self)`:  setups and initializes the simulation environment. Call this prior to running any simulations.  
* `Environment.step(self, activation_array_list, time)`: takes one timestep for muscle activations defined in `activation_array_list`. 
* `Environment.post_processing(self, filename_video)`: Makes 3D video based on saved data from simulation. Requires `ffmpeg`.  
We do not suggest changing `Environment` as it may cause unintended consequences to the simulation. 


You will want to work within `main()` to interface with the simulations and develop your learning model. In `main()`, the first thing you need to define is the length of your simulation and initialize the environment. `final_time` is the length of time that your simulation will run unless exited early. You want to give your octopus enough time to complete the task, but too much time will lead to excessively long simulation times.

```python 
    # Set simulation final time
    final_time = 10.0

    # Initialize the environment
    target_position = np.array([-0.4, 0.0, 0.5])
    env = Environment(final_time, target_position, COLLECT_DATA_FOR_POSTPROCESSING=True)
    total_steps, systems = env.reset()
```

With your system initialized, you are now ready to perform the simulation. To perform the simulation there are two steps:  
1) Evaluate the reward function and define the basis function activations
2) Perform time step  

There is also a user defined stopping condition. When met, this will immediately end the simulation. This can be useful to end the simulation if the octopus successfully complete the task early, or has a sufficiently low reward function that there is no point continuing the simulation. 

```python
    for i_sim in tqdm(range(total_steps)):
	""" Learning loop """
	if i_sim % 200:
           """ Add your learning algorithm here to define activation """
           # This will be based on your observations of the system and 
           # evaluation of your reward function.  
           shearable_rod = systems[0]
           rigid_body = systems[1]   
           reward = reward_function()   
           activation = segment_activation_function()

        """ Perform time step """
        time, systems, done = env.step(activation, time)

        """ User defined condition to exit simulation loop """
        done = user_defined_condition_function(reward, systems, time)
        if done:
            break
```

The state of the octopus is available in `shearable_rod`. The octopus consists of a series of 121 nodes. Nodes 0-49 relate to one arm, nodes 50-70 relate to the head, and nodes 71-120 relate to the second arm. `shearable_rod.position_collection` returns an array with entries relating to the position of each node.
The state of the target object is available in `rigid_body`.

It is important to properly define the activation function. It consists of a list of lists defining the activation of the two arms in each of the the three modes of deformation. The activation function should be a list with three entries for the three modes of deformation. Each of these entries is in turn a list with two entries, which are arrays of the basis function activations for the two arms. 

```python
    activation = [
        [arm_1_normal,   arm_2_normal],    # activation in normal direction
        [arm_1_binormal, arm_2_binormal],  # activation in binormal direction
        [arm_1_tangent,  arm_2_tangent],   # activation in tangent direction
        ]
```

Each activation array has 7 entries that relate to the activation of different basis functions. The ordering goes from the top level to the bottom level of the hierarchy. Each entry can vary from -1 to 1.

`activation_array[0]  ` -- One top level muscle segment  
`activation_array[1:3]` -- Two mid level muscle segment  
`activation_array[3:7]` -- Four bottom level muscle segment  

 

## A few practical notes
1) To save a video of the octopus with `Environment.post_processing()`, you need to install `ffmeg`. You can download and install it [here](https://www.ffmpeg.org/). 

2) The timestep size is set to 40 μs. This is necessary to keep the simulation stable, however, you may not need to update your muscle activations that often. Varying the learning time step will change how often your octopus updates its behaviour.

3) There is a 15-20 second startup delay while the simulation is initialized. This is a one time cost whenever the Python script is run and resetting the simulation using `.rest()` does not incur this delay for subsequent simulations. 

4) We suggest installing `requirements.txt` and `optional-requirements.txt`, to run Elastica without any problem.





