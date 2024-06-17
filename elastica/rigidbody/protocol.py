from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from elastica.systems.protocol import SystemProtocol, SlenderBodyGeometryProtocol


class RigidBodyProtocol(SystemProtocol, SlenderBodyGeometryProtocol, Protocol):

    mass: np.floating
    volume: np.floating
    length: np.floating
    tangents: NDArray[np.floating]
    radius: np.floating

    mass_second_moment_of_inertia: NDArray[np.floating]
    inv_mass_second_moment_of_inertia: NDArray[np.floating]
