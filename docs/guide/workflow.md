# Workflow

When using PyElastica, users will setup a simulation in which they define a system of rods, define initial and boundary conditions on the rods, run the simulation, and then post-process the results. Here, we outline a typical template of using PyElastica.

:::{important}
**A note on notation:** Like other FEA packages such as Abaqus, PyElastica does not enforce units. This means that you are required to make sure that all units for your input variables are consistent. When in doubt, SI units are always safe, however, if you have a very small length scale ($\sim$ nm), then you may need to rescale your units to avoid needing prohibitively small time steps and/or roundoff errors.
:::

<h2>1. Setup Simulation</h2>

```python
from elastica.modules import (
    BaseSystemCollection,
    Connections,
    Constraints,
    Forcing,
    CallBacks,
    Damping
)

class SystemSimulator(
    BaseSystemCollection,
    Constraints, # Enabled to use boundary conditions 'OneEndFixedBC'
    Forcing,     # Enabled to use forcing 'GravityForces'
    Connections, # Enabled to use FixedJoint
    CallBacks,   # Enabled to use callback
    Damping,     # Enabled to use damping models on systems.
):
    pass
```
This simply combines all the modules previously imported together. If a module is not needed for the simulation, it does not need to be added here.

Available components are:

|               Component               |               Note              |
|:-------------------------------------:|:-------------------------------:|
|         BaseSystemCollection          | **Required** for all simulator. |
| [Constraints](../api/constraints.rst) |                                 |
| [Forcing](../api/external_forces.rst) |                                 |
| [Connections](../api/connections.rst) |                                 |
| [CallBacks](../api/callback.rst)      |                                 |
| [Damping](../api/damping.rst)         |                                 |

:::{Note}
We adopted a composition and mixin design paradigm in building elastica. The detail of the implementation is not important in using the package, but we left some references to read [here](../advanced/PackageDesign.md).
:::


<h2>2. Create Rods</h2>
Each rod has a number of physical parameters that need to be defined. These values then need to be assigned to the rod to create the object, and the rod needs to be added to the simulator.

```python
from elastica.rod.cosserat_rod import CosseratRod

# Create rod
direction = np.array([0.0, 0.0, 1.0])
normal = np.array([0.0, 1.0, 0.0])
rod1 = CosseratRod.straight_rod(
    n_elements=50,                                # number of elements
    start=np.array([0.0, 0.0, 0.0]),              # Starting position of first node in rod
    direction=direction,                          # Direction the rod extends
    normal=normal,                                # normal vector of rod
    base_length=0.5,                              # original length of rod (m)
    base_radius=10e-2,                            # original radius of rod (m)
    density=1e3,                                  # density of rod (kg/m^3)
    youngs_modulus=1e7,                           # Elastic Modulus (Pa)
    shear_modulus=1e7/(2* (1+0.5)),               # Shear Modulus (Pa)
)
rod2 = CosseratRod.straight_rod(
    n_elements=50,                                # number of elements
    start=np.array([0.0, 0.0, 0.5]),              # Starting position of first node in rod
    direction=direction,                          # Direction the rod extends
    normal=normal,                                # normal vector of rod
    base_length=0.5,                              # original length of rod (m)
    base_radius=10e-2,                            # original radius of rod (m)
    density=1e3,                                  # density of rod (kg/m^3)
    youngs_modulus=1e7,                           # Elastic Modulus (Pa)
    shear_modulus=1e7/(2* (1+0.5)),               # Shear Modulus (Pa)
)

# create a SystemSimulator instance
simulator = SystemSimulator()

# Add rod to simulator
simulator.append(rod1)
simulator.append(rod2)
```

This can be repeated to create multiple rods. Supported geometries are listed in [API documentation](../api/rods.rst).

:::{note}
The number of element (`n_elements`) and `base_length` determines the spatial discretization `dx`. More detail discussion is included [here](discretization.md).
:::

<h2>3.a Define Boundary Conditions, Forcings, and Connections</h2>

Now that we have added all our rods to `simulator`, we
need to apply relevant boundary conditions.
See [this page](../api/constraints.rst) for in-depth explanations and documentation.

As a simple example, to fix one end of a rod, we use the `OneEndFixedBC` boundary condition (which we imported in step 1 and apply it to the rod. Here we will be fixing the $0^{\text{th}}$ node as well as the $0^{\text{th}}$ element.

```python
from elastica.boundary_conditions import OneEndFixedBC

simulator.constrain(rod1).using(
    OneEndFixedBC,                  # Displacement BC being applied
    constrained_position_idx=(0,),  # Node number to apply BC
    constrained_director_idx=(0,)   # Element number to apply BC
)
```

We have now fixed one end of the rod while leaving the other end free. We can also apply forces to free end using the `EndpointForces`. We can also add more complex forcings, such as friction, gravity, or torque throughout the rod. See [this page](../api/external_forces.rst) for in-depth explanations and documentation.

```python
from elastica.external_forces import EndpointForces

#Define 1x3 array of the applied forces
final_time = 10
origin_force = np.array([0.0, 0.0, 0.0])
end_force = np.array([-15.0, 0.0, 0.0])
simulator.add_forcing_to(rod1).using(
    EndpointForces,                 # Traction BC being applied
    origin_force,                   # Force vector applied at first node
    end_force,                      # Force vector applied at last node
    ramp_up_time=final_time / 2.0   # Ramp up time
)
```

One last condition we can define is the connections between rods. See [this page](../api/connections.rst) for in-depth explanations and documentation.

```python
from elastica.joint import FixedJoint

# Connect rod 1 and rod 2. '_connect_idx' specifies the node number that
# the connection should be applied to. You are specifying the index of a
# list so you can use -1 to access the last node.
simulator.connect(
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

:::{note}
Version 0.3.3: The order of the operation is defined by the order of the definition. For example, if you define the connection before the forcing condition, the connection will be applied first. This is less important for the boundary condition, forcing, and connection since they do not depend on each other. However, it is important for friction, contact, or any custom boundary conditions since they depend on other boundary conditions.
For example, friction should be defined after contact, since contact will define the normal force applied to the surface, which friction depends on. Contact should be defined before any other boundary conditions, since aggregated normal force is used to calculate the repelling force.
:::

<h2>3.b Define Damping </h2>

Next, if required, in order to numerically stabilize the simulation,
we can apply damping to the rods.
See [this page](../api/damping.rst) for in-depth explanations and documentation.

```python
from elastica.dissipation import AnalyticalLinearDamper

nu = 1e-3   # Damping constant of the rod
dt = 1e-5   # Time-step of simulation in seconds

simulator.dampen(rod1).using(
    AnalyticalLinearDamper,
    damping_constant = nu,
    time_step = dt,
)

simulator.dampen(rod2).using(
    AnalyticalLinearDamper,
    damping_constant = nu,
    time_step = dt,
)
```

<h2>4. Add Callback Functions (optional)</h2>

If you want to know what happens to the rod during the course of the simulation, you must collect data during the simulation. Here, we demonstrate how the callback function can be defined to export the data you need. There is a base class `CallBackBaseClass` that can help with this.

:::{note}
PyElastica __does not automatically saves__ the simulation result. If you do not define a callback function, you will only have the final state of the system at the end of the simulation.
:::

```python
from elastica.callback_functions import CallBackBaseClass

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
from collections import defaultdict
callback_data_rod1, callback_data_rod2 = defaultdict(list), defaultdict(list)

# Add MyCallBack to simulator for each rod telling it how often to save data (step_skip)
simulator.collect_diagnostics(rod1).using(
    MyCallBack, step_skip=1000, callback_params=callback_data_rod1)
simulator.collect_diagnostics(rod2).using(
    MyCallBack, step_skip=1000, callback_params=callback_data_rod2)
```

You can define different callback functions for different rods and also have different data outputted at different time step intervals depending on your needs. See [this page](../api/callback.rst) for more in-depth documentation.

:::{note}
During setting up callbacks, `.collect_diagnostics` can take python-collection (list, dict, tuple) of rods/systems:
```python
...
    def make_callback(self, system, time, current_step):
        rod1 = system[key1]
        rod2 = system[key2]
...
simulator.collect_diagnostics({key1: rod1, key2: rod2}).using(
    # custom callback class
)
```
to handle data from multiple rods at the same time.
Additionally, you can pass ellipsis (...) to collect data from all rods/systems in the simulator.
```python
...
    def make_callback(self, system, time, current_step):
        for rod in system:
            pass
...
simulator.collect_diagnostics(...).using(
    # custom callback class
)
```
:::

<h2>5. Finalize Simulator</h2>

Now that we have finished defining our rods, the different boundary conditions and connections between them, and how often we want to save data, we have finished setting up the simulation. We now need to finalize the simulator by calling

```python
simulator.finalize()
```

This goes through and collects all the rods and applied conditions, preparing the system for the simulation.

<h2>6. Set Timestepper</h2>

With our system now ready to be run, we need to define which time stepping algorithm to use. Currently, we suggest using the position Verlet algorithm. We also need to define how much time we want to simulate as well as either the time step (dt) or the number of total time steps we want to take. Once we have defined these things, we can run the simulation using an a timestepper loop.

>> We are still actively testing different integration and time-stepping techniques, `PositionVerlet` is the best default at this moment.

```python
from elastica.timestepper.symplectic_steppers import PositionVerlet

timestepper = PositionVerlet()
final_time = 10   # seconds
total_steps = int(final_time / dt)

# Explicit timestepper loop
dt = final_time / total_steps
time = 0.0
for i in range(total_steps):
    time = timestepper.step(simulator, time, dt)
```

More documentation on timestepper and integrator is included [here](../api/time_steppers.rst)

<h2>7. Post Process</h2>

Once the simulation ends, it is time to analyze the data. If you defined a callback function, the data you outputted in available there (i.e. `callback_data_rod1`), otherwise you can access the final configuration of your system through your rod objects. For example, if you want the final position of one of your rods, you can get it from `rod1.position_collection[:]`.
