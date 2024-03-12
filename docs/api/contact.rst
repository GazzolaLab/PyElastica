Contact
==============================

.. _contact:

.. automodule:: elastica.contact_forces

Description
-----------

.. rubric:: Available Contact Classes

.. autosummary::
   :nosignatures:

   NoContact
   RodRodContact
   RodCylinderContact
   RodSelfContact
   RodSphereContact
   RodPlaneContact
   RodPlaneContactWithAnisotropicFriction
   CylinderPlaneContact


Built-in Contact Classes
-------------------------------------

.. autoclass:: NoContact
   :special-members: __init__,apply_contact

.. autoclass:: RodRodContact
   :special-members: __init__,apply_contact

.. autoclass:: RodCylinderContact
   :special-members: __init__,apply_contact

.. autoclass:: RodSelfContact
   :special-members: __init__,apply_contact

.. autoclass:: RodSphereContact
   :special-members: __init__,apply_contact

.. autoclass:: RodPlaneContact
   :special-members: __init__,apply_contact

.. autoclass:: RodPlaneContactWithAnisotropicFriction
   :special-members: __init__,apply_contact

.. autoclass:: CylinderPlaneContact
   :special-members: __init__,apply_contact
