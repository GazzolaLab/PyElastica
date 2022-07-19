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
   LaplaceDissipationFilter

Compatibility
~~~~~~~~~~~~~

=============================== ==== =========== 
Damping/Numerical Dissipation   Rod  Rigid Body
=============================== ==== =========== 
ExponentialDamper               ✅       ✅
LaplaceDissipationFilter        ✅       ❌
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

.. autoclass:: LaplaceDissipationFilter
   :special-members: __init__
