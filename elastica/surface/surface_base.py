__doc__ = """Base class for surfaces"""
from typing import Type


class SurfaceBase:
    """
    Base class for all surfaces.

    Notes
    -----
    All new surface classes must be derived from this SurfaceBase class.

    """

    REQUISITE_MODULES: list[Type] = []

    def __init__(self) -> None:
        """
        SurfaceBase does not take any arguments.
        """
        pass
