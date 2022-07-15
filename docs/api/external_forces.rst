External Forces / Interactions
==============================

.. _external_forces:
.. _interactions:


Description
-----------

External force and environmental interaction are represented as force/torque boundary condition at different location.

.. rubric:: Available Forcing

.. automodule:: elastica.external_forces
.. autosummary::
   :nosignatures:

   NoForces
   EndpointForces
   GravityForces
   UniformForces
   UniformTorques
   MuscleTorques
   EndpointForcesSinusoidal

.. rubric:: Available Interaction

.. automodule:: elastica.interaction
.. autosummary::
   :nosignatures:

   AnisotropicFrictionalPlane
   InteractionPlane
   SlenderBodyTheory

Compatibility
~~~~~~~~~~~~~

========================== ======= ============
Forcing                     Rod     Rigid Body
========================== ======= ============
NoForces                    ✅      ✅
EndpointForces              ✅      ❌
GravityForces               ✅      ✅
UniformForces               ✅      ✅
UniformTorques              ✅      ✅
MuscleTorques               ✅      ❌
EndpointForcesSinusoidal    ✅      ❌
========================== ======= ============

========================== ======= ============
Interaction                 Rod     Rigid Body
========================== ======= ============
AnisotropicFrictionalPlane  ✅      ❌
InteractionPlane            ✅      ❌
SlenderBodyTheory           ✅      ❌
========================== ======= ============

Built-in External Forces
------------------------
.. automodule:: elastica.external_forces
   :noindex:

.. autoclass:: NoForces
   :special-members: __init__

.. autoclass:: EndpointForces
   :special-members: __init__

.. autoclass:: GravityForces
   :special-members: __init__

.. autoclass:: UniformForces
   :special-members: __init__

.. autoclass:: UniformTorques
   :special-members: __init__

.. autoclass:: MuscleTorques
   :special-members: __init__

.. autoclass:: EndpointForcesSinusoidal
   :special-members: __init__

Built-in Environment Interactions
---------------------------------
.. automodule:: elastica.interaction
   :noindex:

.. autoclass:: AnisotropicFrictionalPlane
   :special-members: __init__

.. autoclass:: InteractionPlane
   :special-members: __init__

.. autoclass:: SlenderBodyTheory
   :special-members: __init__
