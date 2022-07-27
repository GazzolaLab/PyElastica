Controllers
==============================

.. _controllers:


Description
-----------

Interfaces for controllers applying forces and torques to systems as a function of the state of one or multiple systems.

.. rubric:: Available Controllers

.. automodule:: elastica.controllers
.. autosummary::
   :nosignatures:

   ControllerBase

Compatibility
~~~~~~~~~~~~~

========================== ======= ============
Controller                  Rod     Rigid Body
========================== ======= ============
ControllerBase              ✅      ✅
========================== ======= ============

Built-in Controllers
------------------------
.. automodule:: elastica.controllers
   :noindex:

.. autoclass:: ControllerBase
   :special-members: __init__
