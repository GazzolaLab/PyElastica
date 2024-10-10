__doc__ = """Abstract base class for all damper/dissipation modules"""
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar
import numpy as np


T = TypeVar("T")


class DamperBase(Generic[T], ABC):
    """Base class for damping module implementations.

    Notes
    -----
    All damper classes must inherit DamperBase class.


    Attributes
    ----------
    system : RodBase

    """

    _system: T

    # TODO typing can be made better
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize damping module"""
        try:
            self._system = kwargs["_system"]
        except KeyError:
            raise KeyError(
                "Please use simulator.dampen(...).using(...) syntax to establish "
                "damping."
            )

    @property
    def system(self) -> T:
        """
        get system (rod or rigid body) reference

        Returns
        -------
        SystemType

        """
        return self._system

    @abstractmethod
    def dampen_rates(self, system: T, time: np.float64) -> None:
        # TODO: In the future, we can remove rod and use self.system
        """
        Dampen rates (velocity and/or omega) of a rod object.

        Parameters
        ----------
        system : SystemType
            System (rod or rigid-body) object.
        time : float
            The time of simulation.

        """
        pass
