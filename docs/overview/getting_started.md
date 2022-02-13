# Getting Started

PyElastica is the Python implementation of Elastica and is the easiest version to get started with. This page contains useful information for getting PyElastica set up, using it to model single and multiple rod systems, and post-processing the results. 

## PyElastica Workflow
When using PyElastica, most users will want to setup a simulation in which they define a system of rods, define initial and boundary conditions on the rods, run the simulation, and then post-process the results. The typical outline for using PyElastica in such as case would be:

### Import necessary modules
There are several different modules from PyElastica that need to be imported. They can be broadly classified as:  
#### Wrappers
For the most general case, you would need to import the following:
```python
    from elastica.wrappers import (
        BaseSystemCollection,
        Connections,
        Constraints,
        Forcing,
        CallBacks)
```
#### System conditions (i.e. rods and boundary conditions
Some examples are:
```python
    from elastica.rod.cosserat_rod import CosseratRod
    from elastica.boundary_conditions import OneEndFixedRod
    from elastica.joint import FreeJoint, HingeJoint
    from elastica.external_forces import GravityForces, UniformForces
    from elastica.interaction import AnistropicFrictionalPlane
```
#### Call back functions
For saving state information during simulation.
```python
    from elastica.callback_functions import CallBackBaseClass
```
#### Time stepper functions
Currently `PositionVerlet` is the best default.
```python
    from elastica.timestepper.symplectic_steppers import PositionVerlet
    from elastica.timestepper import integrate
```

:::{note}
See the examples folder for a list to typical import statements. Future implementations will work to simplify this step.
:::

### Create simulator
We need to define an object that will contain the system we are about to create. To do this we define a simulator class that inherits the necessary attributes from the wrappers we previously imported. We will add rods and different boundary conditions to this class to create our system and pass it to the timestepper to be solved. The most generic simulator class is:  
```python
    class SystemSimulator(BaseSystemCollection, Constraints, Connections, Forcing, CallBacks): 
        pass 
```
This simply combines all the wrappers previously imported together. If a wrapper is not needed for the simulation, it does not need to be added here (i.e. if you only have one rod, you do not need to include the `Connections` class).

### Define parameters for each rod
Each rod has a number of physical parameters that need to be defined. These values then need to be assigned to the rod to create the object, and the rod needs to be added to the simulator. 
```python

    # Create rod
    rod1 = CosseratRod.straight_rod(
        n_elements = 50,                                # number of elements
        start = np.array([0.0, 0.0, 1.0]),              # Starting position of first node in rod
        direction = direction,                          # Direction the rod extends
        normal = np.cross(direction, binormal),         # normal vector of rod
        base_length=0.5,                                # original length of rod (m)
        base_radius=10e-2,                              # original radius of rod (m)
        density = 1e3,                                  # density of rod (kg/m^3)
        nu = 1e-3,                                      # Energy dissipation of rod
        youngs_modulus = 1e7,                           # Elastic Modulus (Pa)
        poisson_ratio = 0.5,                            # Poisson Ratio
    )

    # Add rod to SystemSimulator
    SystemSimulator.append(rod1)
```
This can be repeated to add multiple rods to the system. Be sure to remember to change the name of the rods (rod1 $\rightarrow$ rod2) and the starting location for each rod along with the rods physical properties as needed. 

:::{important}
**A note on notation:** Like other FEA packages such as Abaqus, PyElastica does not enforce units. This means that you are required to make sure that all units for your input variables are consistent. When in doubt, SI units are always safe, however, if you have a very small length scale ($\sim$ nm), then you may need to rescale your units to avoid needing prohibitively small time steps and/or roundoff errors. 
:::

### Define boundary conditions, forcings, and connections
Now that we have added all our rods to `SystemSimulator`, we need to apply the relevant boundary conditions. See the documentation and tutorials for in depth explanations of the different types of forcings available. 

As a simple example, to fix one end of a rod, we use the `OneEndFixedRod` boundary condition (which we imported in step 1 and apply it to the rod. Here we will be fixing the $0^{\text{th}}$ node as well as the $0^{\text{th}}$ element. 
```python 
    SystemSimulator.constrain(rod1).using(
        OneEndFixedRod,                 # Displacement BC being applied
        constrained_position_idx=(0,),  # Node number to apply BC
        constrained_director_idx=(0,)   # Element number to apply BC
    )
```
We have now fixed one end of the rod while leaving the other end free. We can also apply forces to free end using the `EndpointForces`
```python
    #Define 1x3 array of the applied forces
    origin_force = np.array([0.0, 0.0, 0.0])
    end_force = np.array([-15.0, 0.0, 0.0]) 
    SystemSimulator.add_forcing_to(rod1).using(
        EndpointForces,                 # Traction BC being applied
        origin_force,                   # Force vector applied at first node
        end_force,                      # Force vector applied at last node
        ramp_up_time=final_time / 2.0   # Ramp up time 
    )
```
We can also add more complex forcings, such as friction, gravity, or torque throughout the rod (see tutorials and documentation for details). One last condition we can define is the connections between rods. 
```python
    # Connect rod 1 and rod 2. '_connect_idx' specifies the node number that 
    # the connection should be applied to. You are specifying the index of a 
    # list so you can use -1 to access the last node. 
    SystemSimulator.connect(
        first_rod  = rod1, 
        second_rod = rod2, 
        first_connect_idx  = -1, # Connect to the last node of the first rod. 
        second_connect_idx =  0  # Connect to first node of the second rod. 
        ).using(
            FixedJoint,  # Type of connection between rods
            k  = 1e5,    # Spring constant of force holding rods together (F = k*x)
            nu = 0,      # Energy dissipation of joint
            kt = 5e3     # Rotational stiffness of rod to avoid rods twisting
            )
```

### Create callback functions (optional)
If you want to know what happens to the rod during the course of the simulation, you must create a callback function to output the data you need as the simulation runs. There is a base class `CallBackBaseClass` that can help with this. If you do not define a callback function, then at the end of the simulation, you will only have the final state of the system available.  
```python
    # MyCallBack class is derived from the base call back class.   
    class MyCallBack(CallBackBaseClass):
        def __init__(self, step_skip: int, callback_params):
            CallBackBaseClass.__init__(self)
            self.every = step_skip
            self.callback_params = callback_params
        
        # This function is called every time step
        def make_callback(self, system, time, current_step: int):         
            if current_step % self.every == 0:
                # Save time, step number, position, orientation and velocity
                self.callback_params["time"].append(time)
                self.callback_params["step"].append(current_step)
                self.callback_params["position" ].append(system.position_collection.copy())
                self.callback_params["directors"].append(system.director_collection.copy())
                self.callback_params["velocity" ].append(system.velocity_collection.copy())
                return

    # Create dictionary to hold data from callback function
    callback_data_rod1, callback_data_rod2 = defaultdict(list), defaultdict(list)

    # Add MyCallBack to SystemSimulator for each rod telling it how often to save data (step_skip)
    SystemSimulator.collect_diagnostics(rod1).using(
        MyCallBack, step_skip=1000, callback_params=callback_data_rod1)
    SystemSimulator.collect_diagnostics(rod2).using(
        MyCallBack, step_skip=1000, callback_params=callback_data_rod2)
```
You can define different callback functions for different rods and also have different data outputted at different time step intervals depending on your needs.

### Finalize system, define time stepper, and run simulation
Now that we have finished defining our rods, the different boundary conditions and connections between them, and how often we want to save data, we have finished setting up the simulation. We now need to finalize the simulator by calling `SystemSimulator.finalize()`. This goes through and collects all the rods and applied conditions, preparing the system for the simulation. 

With our system now ready to be run, we need to define which time stepping algorithm to use. Currently, we suggest using the position Verlet algorithm. We also need to define how much time we want to simulate as well as either the time step (dt) or the number of total time steps we want to take. Once we have defined these things, we can run the simulation by calling `integrate()`, which will start the simulation. 
```python
    timestepper = PositionVerlet()
    final_time = 10   # seconds
    dt = 1e-5         # seconds
    total_steps = int(final_time / dt) 
    integrate(timestepper, SystemSimulator, final_time, total_steps)
```

### Post-process the data
Once the simulation ends, it is time to analyze the data. If you defined a callback function, the data you outputted in available there (i.e. `callback_data_rod1`), otherwise you can access the final configuration of your system through your rod objects. For example, if you want the final position of one of your rods, you can get it from `rod1.position_collection[:]`. 

## Useful Information
To help get you started building initial intuition about PyElastica, here are some general rules of thumb to follow. 
:::{important}
These are based on general observations of how simulations tend to behave and are not guaranteed to always hold. Particularly for choosing dx and dt, it is important to perform a separate convergence study for your specific case.
:::

### Number of elements per rod
Generally, the more flexible your rod, the more elements you need. It is important to always perform a convergence test for your simulation, however, 30-50 elements per rod is a good starting point. 

### Choosing your dx and dt
Generally you will set your dx and then choose a stable dt. Your dx will be a combination of your problems length scale and the number of elements you want. Recall that units can be rescaled as long as they are consistent. If you have have a small rod, selecting a dx on the order of nm without scaling is 1e-9. This small value can cause numerical issues, so it is better to rescale your units so that nm $\sim O(1)$. 

When choosing your time step, there are a number of different conditions that can affect your choice. The most important consideration is that the time stepping algorithm remain stable. As a useful heuristic, we have found that dt = 0.01 dx $s/m$ tends to yield stable time steps, but depending on your problem this may not hold. If you wish to be able to resolve the propagation of different waves, then you need to make sure your dt is able to capture their propagation ($dt = dx \sqrt{\rho/G}$ for shear waves or $dt = dx \sqrt{\rho/E}$ for flexural waves).

### Run time scaling
PyElastica will scale linearly with the number of time steps, so if you halve your time step, your simulation will take twice as long to finish. 

The algorithms that PyElastica is based on scale linearly with the number of elements. However, due to overhead from calling functions in Python, PyElastica does not currently have a strong dependence on the number of nodes. Doubling the number of nodes may only lead to a 10-20% increase in run time. We are working on reducing this overhead, and future versions of PyElastica will be much faster due to implementation of C and C++ subroutines. While this means you can decrease your dx without a large run time penalty, remember that you also need to adjust your dt, which will affect the run time. 

Adding additional interactions with the environment, such as friction or gravity, will increase run time. Most of these interactions only have a small effect on run time except for rod collision and/or self-intersection. As implemented, these are expensive routines ($O(N^2)$) and should be avoided if possible as they will substantially lengthen your run time. We are working on developing more efficient methods of implementing these conditions. 

We are working to add parallel and HPC capabilities to PyElastica. If you are interested in helping us implement these changes, let us know.

### Call backs
The frequency at which you have your callback function save data will depend on what information you need from the simulation. Excessive call backs can cause performance penalties, however, it is rarely necessary to make call backs at a frequency that this becomes a problem. We have found that making a call back roughly every 100 iterations has a negligible performance penalty. 

Currently, all data saved from call back functions is saved in memory. If you have many rods or are running for a long time, you may want to consider editing the call back function to write the saved data to disk so you do not run out of memory during the simulation.


## Visualization
If you wish to visualize your system, make sure you define your callback function to output all necessary data. You can either plot your data using a python package such as `matplotlib`, or any rendering software that you choose. Note, many of the visualization scripts in the examples folders require [ffmpeg](https://www.ffmpeg.org/) (be sure to install with h264 libraries).

:::{note}
For high-quality visualization, we suggest [POVray](http://povray.com). See [this tutorial](https://github.com/GazzolaLab/PyElastica/tree/master/examples/Visualization) for examples of different ways of visualizing the system. 
:::

