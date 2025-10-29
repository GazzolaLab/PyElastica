Contact
==============================

.. _contact:

.. automodule:: elastica.contact_forces

Description
-----------

.. note::
   (CAUTION)
   The contact is recommended to be added at last. This is because contact forces often includes
   friction that may depend on other normal forces and contraints to be calculated accurately.
   Be careful on the order of adding interactions.

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
