Constraints
===========

.. _constraints:

.. automodule::  elastica.boundary_conditions

Description
-----------

Constraints are equivalent to displacement boundary condition.

Built-in
--------

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

Compatibility
~~~~~~~~~~~~~

=============================== === ===========
Constraint / Boundary Condition Rod Rigid Body
=============================== === ===========
FreeBC                          ✅   ✅
OneEndFixedBC                   ✅   ✅
FixedConstraint                 ✅   ✅ 
HelicalBucklingBC               ✅   ❌
=============================== === ===========

Examples
--------

.. note::
   PyElastica package provides basic built-in constraints, and we expect use to adapt their own boundary condition from our examples.


Built-in Constraints
--------------------

.. autoclass:: ConstraintBase
   :inherited-members:
   :undoc-members:
   :exclude-members: 
   :show-inheritance:

.. autoclass:: FreeBC

.. autoclass:: OneEndFixedBC
   :special-members: __init__

.. autoclass:: FixedConstraint
   :special-members: __init__

.. autoclass:: HelicalBucklingBC
   :special-members: __init__

.. autoclass:: FreeRod

.. autoclass:: OneEndFixedRod
