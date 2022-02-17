# Useful Information
To help get you started building initial intuition about PyElastica, here are some general rules of thumb to follow. 
:::{important}
These are based on general observations of how simulations tend to behave and are not guaranteed to always hold. Particularly for choosing dx and dt, it is important to perform a separate convergence study for your specific case.
:::

## Number of elements per rod
Generally, the more flexible your rod, the more elements you need. It is important to always perform a convergence test for your simulation, however, 30-50 elements per rod is a good starting point. 

## Choosing your dx and dt
Generally you will set your dx and then choose a stable dt. Your dx will be a combination of your problems length scale and the number of elements you want. Recall that units can be rescaled as long as they are consistent. If you have have a small rod, selecting a dx on the order of nm without scaling is 1e-9. This small value can cause numerical issues, so it is better to rescale your units so that nm $\sim O(1)$. 

When choosing your time step, there are a number of different conditions that can affect your choice. The most important consideration is that the time stepping algorithm remain stable. As a useful heuristic, we have found that dt = 0.01 dx $s/m$ tends to yield stable time steps, but depending on your problem this may not hold. If you wish to be able to resolve the propagation of different waves, then you need to make sure your dt is able to capture their propagation ($dt = dx \sqrt{\rho/G}$ for shear waves or $dt = dx \sqrt{\rho/E}$ for flexural waves).

## Run time scaling
PyElastica will scale linearly with the number of time steps, so if you halve your time step, your simulation will take twice as long to finish. 

The algorithms that PyElastica is based on scale linearly with the number of elements. However, due to overhead from calling functions in Python, PyElastica does not currently have a strong dependence on the number of nodes. Doubling the number of nodes may only lead to a 10-20% increase in run time. We are working on reducing this overhead, and future versions of PyElastica will be much faster due to implementation of C and C++ subroutines. While this means you can decrease your dx without a large run time penalty, remember that you also need to adjust your dt, which will affect the run time. 

Adding additional interactions with the environment, such as friction or gravity, will increase run time. Most of these interactions only have a small effect on run time except for rod collision and/or self-intersection. As implemented, these are expensive routines ($O(N^2)$) and should be avoided if possible as they will substantially lengthen your run time. We are working on developing more efficient methods of implementing these conditions. 

We are working to add parallel and HPC capabilities to PyElastica. If you are interested in helping us implement these changes, let us know.

## Call backs
The frequency at which you have your callback function save data will depend on what information you need from the simulation. Excessive call backs can cause performance penalties, however, it is rarely necessary to make call backs at a frequency that this becomes a problem. We have found that making a call back roughly every 100 iterations has a negligible performance penalty. 

Currently, all data saved from call back functions is saved in memory. If you have many rods or are running for a long time, you may want to consider editing the call back function to write the saved data to disk so you do not run out of memory during the simulation.


