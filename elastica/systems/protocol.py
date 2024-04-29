__doc__ = """Base class for elastica system"""

from typing import Protocol

from numpy.typing import NDArray


class SystemProtocol(Protocol):
    """
    Protocol for all elastica system
    """

    @property
    def position_collection(self) -> NDArray: ...

    @property
    def omega_collection(self) -> NDArray: ...

    @property
    def acceleration_collection(self) -> NDArray: ...

    @property
    def alpha_collection(self) -> NDArray: ...

    @property
    def external_forces(self) -> NDArray: ...

    @property
    def external_torques(self) -> NDArray: ...
