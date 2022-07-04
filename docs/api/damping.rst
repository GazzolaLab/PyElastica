Damping
=======

.. automodule::  elastica.dissipation

Description
-----------

Damping is used to numerically stabilize the simulations.

.. rubric:: Available Damping

.. autosummary::
   :nosignatures:

   DamperBase
   ExponentialDamper
   FilterDamper

Compatibility
~~~~~~~~~~~~~

=============================== ==== =========== 
Damping/Numerical Dissipation   Rod  Rigid Body
=============================== ==== =========== 
ExponentialDamper               ✅       ✅
FilterDamper                    ✅       ❌
=============================== ==== ===========


Built-in Constraints
--------------------

.. autoclass:: DamperBase
   :inherited-members:
   :undoc-members:
   :exclude-members: 
   :show-inheritance:

.. autoclass:: ExponentialDamper
   :special-members: __init__

.. autoclass:: FilterDamper
   :special-members: __init__
