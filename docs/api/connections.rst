Connections / Joints
==============================

.. _connections:

.. automodule:: elastica.joint

Description
-----------

.. rubric:: Available Connections/Joints

.. autosummary::
   :nosignatures:

   ConnectionBase
   FreeJoint
   FixedJoint
   HingeJoint

Compatibility
~~~~~~~~~~~~~

=============================== ==== ===========
Connection / Joints             Rod  Rigid Body
=============================== ==== ===========
ConnectionBase                  ✅   ✅
FreeJoint                       ✅   ❌
FixedJoint                      ✅   ❌
HingeJoint                      ✅   ❌
=============================== ==== ===========

Built-in Connection / Joint
-------------------------------------

.. autoclass:: ConnectionBase
   :special-members: __init__,apply_forces,apply_torques

.. autoclass:: FreeJoint
   :special-members: __init__,apply_forces,apply_torques

.. autoclass:: FixedJoint
   :special-members: __init__,apply_forces,apply_torques

.. autoclass:: HingeJoint
   :special-members: __init__,apply_forces,apply_torques
