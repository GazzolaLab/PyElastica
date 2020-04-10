*************
PyElastica
*************

This the the documentation page for PyElastica. For more information about PyElastica, see the `project website`_. 

Rods
=====
.. automodule:: elastica.rod.__init__
   :members:
   :exclude-members: __weakref__

Cosserat Rod
~~~~~~~~~~~~
.. automodule:: elastica.rod.cosserat_rod
   :members:
   :exclude-members: __weakref__, _get_z_vector, __init__, _CosseratRodBase

.. Constitutive Models
.. ~~~~~~~~~~~~~~~~~~~
.. .. automodule:: elastica.rod.constitutive_model
..    :members:
..    :exclude-members: __weakref__


Boundary Conditions
====================

Endpoint Constraints
~~~~~~~~~~~~~~~~~~~~~
.. automodule:: elastica.boundary_conditions
   :members:
   :exclude-members: __weakref__

External Forces
~~~~~~~~~~~~~~~
.. automodule:: elastica.external_forces
   :members:
   :exclude-members: __weakref__

Environment Interactions
~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: elastica.interaction
   :members:
   :exclude-members: __weakref__


Multiple Rod Connections
=========================
.. automodule:: elastica.joint
   :members:
   :exclude-members: __weakref__


Callback Functions
===================
.. automodule:: elastica.callback_functions
   :members:
   :exclude-members: __weakref__


Time steppers
==============
.. automodule:: elastica.timestepper.symplectic_steppers
   :members:
   :exclude-members: __weakref__, __init__,  _SystemCollectionStepperMixin, SymplecticLinearExponentialIntegrator, SymplecticStepper

Wrappers
==========
.. automodule:: elastica.wrappers.__init__
   :members:
   :exclude-members: __weakref__

.. automodule:: elastica.wrappers.base_system
   :members:
   :exclude-members: __weakref__, __init__, __str__, insert

.. automodule:: elastica.wrappers.callbacks
   :members:
   :exclude-members: __weakref__, __init__, _callbacks, _CallBack

.. automodule:: elastica.wrappers.connections
   :members:
   :exclude-members: __weakref__, __init__, __call__, _Connect

.. automodule:: elastica.wrappers.constraints
   :members:
   :exclude-members: __weakref__, __init__, _Constraint

.. automodule:: elastica.wrappers.forcing
   :members:
   :exclude-members: __weakref__, __init__, __call__, _ExtForceTorque 



.. Utility Functions
.. ==================
.. 
.. Transformations
.. -----------------
.. .. automodule:: elastica.transformations
..    :members:
..    :exclude-members: __weakref__
.. 
.. Utils
.. ------
.. .. automodule:: elastica.utils
..    :members:
..    :exclude-members: __weakref__
.. 
.. Other Stuff
.. ------------
.. .. automodule:: elastica._calculus
..    :members:
..    :exclude-members: __weakref__
.. 
.. .. automodule:: elastica._linalg
..    :members:
..    :exclude-members: __weakref__
.. 
.. .. automodule:: elastica._rotations
..    :members:
..    :exclude-members: __weakref__
.. 
.. .. automodule:: elastica._spline
..    :members:
..    :exclude-members: __weakref__



.. Systems
.. ========
.. .. automodule:: elastica.systems.analytical
..    :members:
..    :exclude-members: __weakref__

.. _project website: https://cosseratrods.org/software/pyelastica
