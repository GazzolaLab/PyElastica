Callback Functions
===================

The frequency at which you have your callback function save data will depend on what information you need from the simulation. Excessive call backs can cause performance penalties, however, it is rarely necessary to make call backs at a frequency that this becomes a problem. We have found that making a call back roughly every 100 iterations has a negligible performance penalty. 

Currently, all data saved from call back functions is saved in memory. If you have many rods or are running for a long time, you may want to consider editing the call back function to write the saved data to disk so you do not run out of memory during the simulation.

.. automodule:: elastica.callback_functions
   :members:
   :exclude-members: __weakref__

