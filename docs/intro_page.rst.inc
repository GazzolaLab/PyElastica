
**Elastica** is a *free* and *open-source* software project for the simulation of assemblies of slender, one-dimensional structures using Cosserat Rod theory.

More information about Elastica is available at the `project website`_

PyElastica
~~~~~~~~~~
.. image:: https://travis-ci.com/GazzolaLab/PyElastica.svg?branch=master
	:target: https://travis-ci.com/gazzolalab
	:alt: Build_status

.. image:: https://codecov.io/gh/gazzolalab/PyElastica/branch/master/graph/badge.svg
	:target: https://codecov.io/gh/gazzolalab/PyElastica
	:alt: codecov

.. image:: https://readthedocs.org/projects/pyelastica/badge/?version=latest
	:target: https://pyelastica.readthedocs.io/en/latest/?badge=latest
	:alt: Documentation Status

PyElastica is the python implementation of Elastica. The easiest way to install PyElastica is with PIP:

.. code-block:: bash

   $ pip install pyelastica

You can also download the source code for PyElastica directly from `GitHub <https://github.com/GazzolaLab/PyElastica>`__.

All options:

- ``examples``: installs dependencies to run example cases,
  found under the folder `examples <https://github.com/GazzolaLab/PyElastica/tree/master/examples>`__.
- ``docs``: packages to build documentation

Options can be combined, e.g.

.. code-block:: bash

    $ pip install "pyelastica[examples,docs]"

If you want to simulate magnetic Cosserat rods interacting with external magnetic environments you can install the derived package using

.. code-block:: bash

    $ pip install magneto_pyelastica

Details can be found `here <https://github.com/armantekinalp/MagnetoPyElastica>`__.


Elastica++
==========
Elastica++ is a C++ implementation of Elastica. The expected release date for the beta version is 2023 Q4.

.. _project website: https://cosseratrods.org
.. _GitHub repo: https://github.com/GazzolaLab/PyElastica
