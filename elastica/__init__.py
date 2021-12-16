""" If Numba module present, use Numba functions"""
import os
import numpy as np

try:
    import numba

    if "IMPORT_TEST_NUMPY" in os.environ:
        # This is used to test if numpy version of classes can be imported.
        raise ImportError

    IMPORT_NUMBA = True

except ImportError:
    IMPORT_NUMBA = False

from elastica.wrappers import *
from elastica.timestepper import *
from elastica.rod.cosserat_rod import *
from elastica.rigidbody import *
from elastica.boundary_conditions import *
from elastica.external_forces import *
from elastica.callback_functions import *
from collections import defaultdict
from elastica.interaction import *
from elastica.joint import *
