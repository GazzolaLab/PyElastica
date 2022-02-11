Constraints
===========

.. _constraints:

.. automodule::  elastica.boundary_conditions

Description
-----------

Constraints are equivalent to displacement boundary condition.

.. rubric:: Available Constraint

.. autosummary::
   :nosignatures:

   ConstraintBase
   FreeBC
   OneEndFixedBC
   FixedConstraint
   HelicalBucklingBC
   FreeRod
   OneEndFixedRod

Examples
--------

.. note::
   PyElastica package provides basic built-in constraints, and we expect use to adapt their own boundary condition from our examples.


Built-in Constraints
--------------------

.. autoclass:: ConstraintBase
   :inherited-members:
   :undoc-members:
   :exclude-members: __weakref__
   :show-inheritance:

.. autoclass:: FreeBC
   :special-members: __init__

.. autoclass:: OneEndFixedBC
   :special-members: __init__

.. autoclass:: FixedConstraint
   :special-members: __init__

.. autoclass:: HelicalBucklingBC
   :special-members: __init__

.. autoclass:: FreeRod

.. autoclass:: OneEndFixedRod
