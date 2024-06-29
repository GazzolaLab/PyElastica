from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class MeshProtocol(Protocol):
    faces: NDArray[np.float64]
    face_centers: NDArray[np.float64]
    face_normals: NDArray[np.float64]
