"""
Timoshenko Beam
===============

Timoshenko beam validation case, for detailed explanation refer to
Gazzola et. al. R. Soc. 2018  section 3.4.3

This Elastica tutorial explains the basics of setting up and running a simple simulation of rods in Elastica. Elastica simulates Cosserat Rods, which are thin, 1-dimensional rods that undergo all possible modes of deformation. This example considers a Timoshenko beam, which is the deformation of a beam under a constant applied force while accounting for shear deforation and rotational bending. This is a good example of the capabilities of Elastica and Cosserat Rods as it requires accounting for the effects of shear deformation, something that the classical Euler-Bernoulli beam solution does not.

.. image:: ../../../assets/timoshenko_beam_figure.png

Getting Started
---------------

To set up the simulation, the first thing you need to do is import the necessary classes. Here we will only import the classes that we need. The `elastica.modules` classes make it easy to construct different simulation systems. Along with these modules, we need to import a rod class, classes for the boundary conditions, and time-stepping functions. As a note, this method of explicitly importing all classes can be a bit cumbersome. Future releases will simplify this step.
"""

import numpy as np
import elastica as ea
from elastica.version import VERSION

from timoshenko_postprocessing import plot_timoshenko

# %%
# Now that we have imported all the necessary classes, we want to create our beam system. We do this by combining all the modules we need to represent the physics that we to include in the simulation. In this case, that is the ``BaseSystemCollection``, ``Constraint``, ``Forcings`` and ``Damping`` because the simulation will consider a rod that is fixed in place on one end, and subject to an applied force on the other end.


class TimoshenkoBeamSimulator(
    ea.BaseSystemCollection, ea.Constraints, ea.Forcing, ea.CallBacks, ea.Damping
):
    pass


timoshenko_sim = TimoshenkoBeamSimulator()

# %%
# Creating Rods
# -------------
# With our simulator set up, we can now define the numerical, material, and geometric properties.
#
# First we define the number of elements in the rod. Next, the material properties are defined for every rod. These are the Young's modulus, the Poisson ratio, the density and the viscous damping coefficient. Finally, the geometry of the rod also needs to be defined by specifying the location of the rod and its orientation, length and radius.
#
# All of the values defined here are done in SI units, though this is not strictly necessary. You can rescale properties however you want, as long as you use consistent units throughout the simulation. See `here <https://info.simuleon.com/blog/units-in-abaqus>`_ for an example of consistent units.
#
# In order to make the difference between a shearable and unshearable rod more clear, we are using a Poisson ratio of 99. This is an unphysical value, as Poisson ratios can not exceed 0.5, however, it is used here for demonstration purposes.

# setting up test params
simulation_time = 500  # 5000.0  # (sec)

n_elem = 100
start = np.zeros((3,))
direction = np.array([0.0, 0.0, 1.0])
normal = np.array([0.0, 1.0, 0.0])
base_length = 3.0
base_radius = 0.25
base_area = np.pi * base_radius**2
density = 5000
nu = 0.1 / 7 / density / base_area
E = 1e6
# For shear modulus of 1e4, nu is 99!
poisson_ratio = 99
shear_modulus = E / (poisson_ratio + 1.0)

# %%
# With all of the rod's parameters set, we can now create a rod with the specificed properties and add the rod to the simulator system. **Important:** Make sure that any rods you create get added to the simulator system (``timoshenko_sim``), otherwise they will not be included in your simulation.

shearable_rod = ea.CosseratRod.straight_rod(
    n_elem,
    start,
    direction,
    normal,
    base_length,
    base_radius,
    density,
    youngs_modulus=E,
    shear_modulus=shear_modulus,
)
timoshenko_sim.append(shearable_rod)

# %%
# Adding Damping
# --------------
# With the rod added to the simulator, we can add damping to the rod. We do this using the ``.dampen()`` option and the ``AnalyticalLinearDamper``. We are modifying ``timoshenko_sim`` simulator to ``dampen`` the ``shearable_rod`` object using ``AnalyticalLinearDamper`` type of dissipation (damping) model.
#
# We also need to define ``damping_constant`` and simulation ``time_step`` and pass in ``.using()`` method.

dl = base_length / n_elem
dt = 0.07 * dl
timoshenko_sim.dampen(shearable_rod).using(
    ea.AnalyticalLinearDamper,
    damping_constant=nu,
    time_step=dt,
)

# %%
# Adding Boundary Conditions
# --------------------------
# With the rod added to the system, we need to apply boundary conditions. The first condition we will apply is fixing the location of one end of the rod. We do this using the ``.constrain()`` option and the ``OneEndFixedRod`` boundary condition. We are modifying the ``timoshenko_sim`` simulator to ``constrain`` the ``shearable_rod`` object using the ``OneEndFixedRod`` type of constraint.
#
# We also need to define which node of the rod is being constrained. We do this by passing the index of the nodes that we want to constrain to ``constrained_position_idx``. Here we are fixing the first node in the rod. In order to keep the rod from rotating around the fixed node, we also need to constrain an element between two nodes. This fixes the orientation of the rod. We do this by passing the index of the element that we want to fix to ``constrained_director_idx``. Like with the position, we are fixing the first element of the rod. Together, this constrains the position and orientation of the rod at the origin.

# One end of the rod is now fixed in place
timoshenko_sim.constrain(shearable_rod).using(
    ea.OneEndFixedBC, constrained_position_idx=(0,), constrained_director_idx=(0,)
)

# %%
# The next boundary condition that we want to apply is the endpoint force. Similarly to how we constrained one of the points, we want the ``timoshenko_sim`` simulator to ``add_forcing_to`` the ``shearable_rod`` object using the ``EndpointForces`` type of forcing. This ``EndpointForces`` applies forces to both ends of the rod. We want to apply a negative force in the :math:`d_1` direction, but only at the end of the rod. We do this by specifying the force vector to be applied at each end as ``origin_force`` and ``end_force``. We also want to ramp up the force over time, so we make the force take some ``ramp_up_time`` to reach its steady-state value. This helps avoid numerical errors due to discontinuities in the applied force.

# Forces added to the rod
end_force = np.array([-15.0, 0.0, 0.0])
timoshenko_sim.add_forcing_to(shearable_rod).using(
    ea.EndpointForces, 0.0 * end_force, end_force, ramp_up_time=simulation_time / 2.0
)

# %%
# Add Unshearable Rod
# -------------------
#
# Along with the shearable rod, we also want to add an unshearable rod to be able to compare the difference between the two. We do this the same way we did for the first rod, however, because this rod is unsherable, we need to change the Poisson ratio to make the rod unsherable. For a truely unsheraable rod, you would need a Poisson ratio of -1.0, however, this causes the system to be numerically unstable, so instead we make the system nearly unshearable by using a Poisson ratio of -0.85.

# Start into the plane
unshearable_start = np.array([0.0, -1.0, 0.0])
shear_modulus = E / (-0.7 + 1.0)
unshearable_rod = ea.CosseratRod.straight_rod(
    n_elem,
    unshearable_start,
    direction,
    normal,
    base_length,
    base_radius,
    density,
    youngs_modulus=E,
    # Unshearable rod needs G -> inf, which is achievable with -ve poisson ratio
    shear_modulus=shear_modulus,
)

timoshenko_sim.append(unshearable_rod)

# add damping
timoshenko_sim.dampen(unshearable_rod).using(
    ea.AnalyticalLinearDamper,
    damping_constant=nu,
    time_step=dt,
)
# add boundary conditions
timoshenko_sim.constrain(unshearable_rod).using(
    ea.OneEndFixedBC, constrained_position_idx=(0,), constrained_director_idx=(0,)
)
timoshenko_sim.add_forcing_to(unshearable_rod).using(
    ea.EndpointForces, 0.0 * end_force, end_force, ramp_up_time=simulation_time / 2.0
)

# %%
# Collect Data
# ------------


# Add call backs
class VelocityCallBack(ea.CallBackBaseClass):
    """
    Tracks the velocity norms of the rod
    """

    def __init__(self, step_skip: int, callback_params: dict):
        ea.CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.callback_params = callback_params

    def make_callback(self, system, time, current_step: int):

        if current_step % self.every == 0:

            self.callback_params["time"].append(time)
            # Collect x
            self.callback_params["velocity_norms"].append(
                np.linalg.norm(system.velocity_collection.copy())
            )
            return


recorded_history = ea.defaultdict(list)
timoshenko_sim.collect_diagnostics(shearable_rod).using(
    VelocityCallBack, step_skip=500, callback_params=recorded_history
)


# %%
# System Finalization
# -------------------
# We have now added all the necessary rods and boundary conditions to our system. The last thing we need to do is finalize the system. This goes through the system, rearranges things, and precomputes useful quantities to prepare the system for simulation.
#
# As a note, if you make any changes to the rod after calling finalize, you will need to re-setup the system. This requires rerunning all cells above this point.


timoshenko_sim.finalize()

# %%
# Define Simulation Time
# ----------------------
# The last thing we need to do deceide how long we want the simulation to run for and what timestepping method to use. Currently, the PositionVerlet algorithim is suggested default method.
#
# In this example, we are trying to match a steady-state solution by temporally evolving our system to reach equillibrium. As such, there is a tradeoff between letting the simulation run long enough to each the equillibrium and waiting around for the simulation to be done. Here we are running the simulation for 10 seconds, this produces reasonable agreement with the analytical solution without taking to long to finish. If you run the simulation for longer, you will get better agreement with the analytical solution.

timestepper = ea.PositionVerlet()
# timestepper = PEFRL()

total_steps = int(simulation_time / dt)
print("Total steps", total_steps)

# %%
# Run Simulation
# --------------
#
# We are now ready to perform the simulation. To run the simulation, we ``integrate`` the ``timoshenko_sim`` system using the ``timestepper`` method until ``final_time`` by taking ``total_steps``. As currently setup, the beam simulation takes about 1 minute to run.

time = 0.0
for i in range(total_steps):
    time = timestepper.step(timoshenko_sim, time, dt)

# %%
# Post Processing Results
# -----------------------
# Now that we have finished the simulation, we want to post-process the results. Post processing script is separately provided in ``timoshenko_postprocessing.py``.

plot_timoshenko(shearable_rod, end_force, False, True)
