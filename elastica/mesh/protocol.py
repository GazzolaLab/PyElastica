from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class MeshProtocol(Protocol):
    faces: NDArray[np.floating]
    face_centers: NDArray[np.floating]
    face_normals: NDArray[np.floating]
