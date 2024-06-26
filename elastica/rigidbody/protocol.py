from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from elastica.systems.protocol import SystemProtocol, SlenderBodyGeometryProtocol


class RigidBodyProtocol(SystemProtocol, SlenderBodyGeometryProtocol, Protocol):

    mass: np.float64
    volume: np.float64
    length: np.float64
    tangents: NDArray[np.float64]
    radius: np.float64

    mass_second_moment_of_inertia: NDArray[np.float64]
    inv_mass_second_moment_of_inertia: NDArray[np.float64]
