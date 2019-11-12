"""
wrappers
--------

Wrappers are simple objects that you can subclass to provide extended
functionality to the simulation, such as adding an environment, joints
, controllers etc.
"""


from .base_system import BaseSystemCollection

try:
    from .connectionsss import Connections
except ImportError:
    pass
from .constraints import Constraints
