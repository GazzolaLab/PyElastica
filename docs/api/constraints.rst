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
   GeneralConstraint
   FixedConstraint
   HelicalBucklingBC
   FreeRod
   OneEndFixedRod

Compatibility
~~~~~~~~~~~~~

=============================== ==== =========== 
Constraint / Boundary Condition Rod  Rigid Body  
=============================== ==== =========== 
FreeBC                           ✅   ✅
OneEndFixedBC                    ✅   ✅
GeneralConstraint                ✅   ✅
FixedConstraint                  ✅   ✅
HelicalBucklingBC                ✅   ❌
=============================== ==== =========== 

Examples
--------

.. note::
   PyElastica package provides basic built-in constraints, and we expect use to adapt their own boundary condition from our examples.

Customizing boundary condition examples:

- `Flexible Swinging Pendulum <https://github.com/GazzolaLab/PyElastica/tree/master/examples/FlexibleSwingingPendulumCase>`_
- `Plectonemes <https://github.com/GazzolaLab/PyElastica/tree/master/examples/RodContactCase/RodSelfContact/PlectonemesCase>`_
- `Solenoids <https://github.com/GazzolaLab/PyElastica/tree/master/examples/RodContactCase/RodSelfContact/SolenoidsCase>`_


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

.. autoclass:: GeneralConstraint
   :special-members: __init__

.. autoclass:: FixedConstraint
   :special-members: __init__

.. autoclass:: HelicalBucklingBC
   :special-members: __init__

.. autoclass:: FreeRod

.. autoclass:: OneEndFixedRod
