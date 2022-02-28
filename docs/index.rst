.. pyelastica documentation master file, created by
   sphinx-quickstart on Sat Mar 21 19:25:06 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

************************
PyElastica Documentation
************************

.. include:: intro_page.rst


Community
~~~~~~~~~

.. image:: https://badges.gitter.im/PyElastica/community.svg
    :target: https://gitter.im/PyElastica/community
    :alt: on gitter

We mainly use `git-issue`_ to communicate the roadmap, updates, helps, and bug fixes.
If you have problem using PyElastica, check if similar issue is reported in `git-issue`_.

We also opened `gitter` channel for short and immediate feedbacks.


Contributing
~~~~~~~~~~~~

If you are interested to contribute, please read `contribution-guide`_ first.



.. toctree::
   :maxdepth: 2
   :caption: Elastica Overview

   overview/welcome_page
   overview/installation

.. toctree::
   :maxdepth: 1
   :caption: User Guide

   guide/workflow
   guide/discretization
   guide/pybind
   guide/example_cases
   guide/visualization

.. toctree::
   :maxdepth: 2
   :caption: API Documentation
   
   api/rods
   api/rigidbody
   api/constraints
   api/external_forces
   api/connections
   api/callback
   api/simulator
   api/time_steppers
   api/utility

..   api/elastica++

.. toctree::
   :maxdepth: 2
   :caption: Gallary

.. toctree::
   :maxdepth: 2
   :caption: Advanced Guide

   advanced/LocalizedForceTorque.md

---------

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. _git-issue: https://github.com/GazzolaLab/PyElastica/issues
.. _contribution-guide: https://github.com/GazzolaLab/PyElastica/blob/master/CONTRIBUTING.md
