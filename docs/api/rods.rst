Rods
====

.. automodule:: elastica.rod.rod_base
   :members:
   :exclude-members: __weakref__

Cosserat Rod
------------

+------------+-------------------+----------------------------------------+-----------------------------+
|            |   On Nodes (+1)   |        On Elements (n_elements)        |       On Voronoi (-1)       |
+============+===================+========================================+=============================+
| Geometry   | position          | | director, tangents                   | | rest voronoi length       |
|            |                   | | length, rest_length                  | | voronoi dilatation        |
|            |                   | | radius                               |                             |
|            |                   | | volume                               |                             |
|            |                   | | dilatation                           |                             |
+------------+-------------------+----------------------------------------+-----------------------------+
| Kinematics | | velocity        | | angular velocity (omega)             |                             |
|            | | acceleration    | | angular acceleration (alpha)         |                             |
|            | | external forces | | mass second moment of inertia        |                             |
|            | | damping forces  | |  +inverse                            |                             |
|            |                   | | dilatation rates                     |                             |
|            |                   | | external torques                     |                             |
|            |                   | | damping torques                      |                             |
+------------+-------------------+----------------------------------------+-----------------------------+
| Elasticity | internal forces   | | shear matrix (modulus)               | | bend matrix (modulus)     |
|            |                   | | shear/stretch strain (sigma)         | | bend/twist strain (kappa) |
|            |                   | | rest shear/stretch strain            | | rest bend/twist strain    |
|            |                   | | internal torques                     | | internal couple           |
|            |                   | | internal stress                      |                             |
+------------+-------------------+----------------------------------------+-----------------------------+
| Material   | mass              | | density                              |                             |
|            |                   | | dissipation constant (force, torque) |                             |
+------------+-------------------+----------------------------------------+-----------------------------+


.. automodule:: elastica.rod.cosserat_rod
   :exclude-members: __weakref__, __init__, update_accelerations, zeroed_out_external_forces_and_torques, compute_internal_forces_and_torques
   :members:
   :inherited-members:

.. Constitutive Models
.. ~~~~~~~~~~~~~~~~~~~
.. .. automodule:: elastica.rod.constitutive_model
..    :members:
..    :exclude-members: __weakref__


Knot Theory (Mixin)
~~~~~~~~~~~~~~~~~~~

.. .. autoclass:: elastica.rod.knot_theory.KnotTheory

.. .. autoclass:: elastica.rod.knot_theory.KnotTheoryCompatibleProtocol

.. automodule:: elastica.rod.knot_theory
   :exclude-members: __init__
   :members:
