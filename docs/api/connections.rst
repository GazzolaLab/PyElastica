Connections / Contact / Joints
==============================

.. _connections:

.. automodule:: elastica.joint

Description
-----------

.. rubric:: Available Connection/Contact/Joints

.. autosummary::
   :nosignatures:

   FreeJoint
   ExternalContact
   FixedJoint
   HingeJoint
   SelfContact

Compatibility
~~~~~~~~~~~~~

=============================== ==== =========== 
Connection / Contact / Joints   Rod  Rigid Body
=============================== ==== =========== 
FreeJoint                       ✅   ❌
ExternalContact                 ✅   ❌
FixedJoint                      ✅   ❌
HingeJoint                      ✅   ❌
SelfContact                     ✅   ❌
=============================== ==== =========== 

Built-in Connection / Contact / Joint
-------------------------------------

.. autoclass:: FreeJoint
   :special-members: __init__

.. autoclass:: ExternalContact
   :special-members: __init__

.. autoclass:: FixedJoint
   :special-members: __init__

.. autoclass:: HingeJoint
   :special-members: __init__

.. autoclass:: SelfContact
   :special-members: __init__
