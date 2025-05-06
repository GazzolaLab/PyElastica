from typing import TYPE_CHECKING, TypeVar, Generic, Callable, Any
from collections.abc import Iterable, Iterator

import itertools

from .protocol import ModuleProtocol

T = TypeVar("T", bound=Callable)
F = TypeVar("F", bound=ModuleProtocol)


class OperatorGroupFIFO(Iterable, Generic[T, F]):
    """
    A class to store the features and their corresponding operators in a FIFO manner.
    Feature can be any user-defined object to label the operators, and operators are
    callable functions.

    This data structure is used to organize elastica operators, such as forcing,
    constraints, boundary condition, etc.

    Examples
    --------
    >>> class FeatureObj:
    ...     ADD: Callable
    ...     SUBTRACT: Callable
    ...     MULTIPLY: Callable

    >>> obj_1 = FeatureObj()
    >>> obj_2 = FeatureObj()
    >>>
    >>> operator_group = OperatorGroupFIFO()
    >>> operator_group.append_id(obj_1)
    >>> operator_group.append_id(obj_2)
    >>> operator_group.add_operators(obj_1, [obj_1.ADD, obj_1.SUBTRACT])
    >>> operator_group.add_operators(obj_2, [obj_2.SUBTRACT, obj_2.MULTIPLY])
    >>> list(iter(operator_group))
    [obj_1.ADD, obj_1.SUBTRACT, obj_2.SUBTRACT, obj_2.MULTIPLY]
    >>> for operator in operator_group:  # call operator in the group
    ...     operator()

    Attributes
    ----------
    _operator_collection : list[list[T]]
        A list of lists of operators. Each list of operators corresponds to a feature.
    _operator_ids : list[int]
        A list of ids of the features.

    Methods
    -------
    append_id(feature)
        Appends the id of the feature to the list of ids.
    add_operators(feature, operators)
        Adds the operators to the list of operators corresponding to the feature.
    is_last(feature)
        Checks if the feature is the last feature in the FIFO.
        Used to check if the specific feature is the last feature in the FIFO.
    """

    def __init__(self) -> None:
        self._operator_collection: list[list[T]] = []
        self._operator_ids: list[int] = []

    def __iter__(self) -> Iterator[T]:
        """Returns an operator iterator to satisfy the Iterable protocol."""
        operator_chain = itertools.chain.from_iterable(self._operator_collection)
        for operator in operator_chain:
            yield operator

    def append_id(self, feature: F) -> None:
        """Appends the id of the feature to the list of ids."""
        self._operator_ids.append(id(feature))
        self._operator_collection.append([])

    def add_operators(self, feature: F, operators: list[T]) -> None:
        """Adds the operators to the list of operators corresponding to the feature."""
        idx = self._operator_ids.index(id(feature))
        self._operator_collection[idx].extend(operators)

    def is_last(self, feature: F) -> bool:
        """Checks if the feature is the last feature in the FIFO."""
        return id(feature) == self._operator_ids[-1]


if TYPE_CHECKING:
    from elastica.typing import OperatorType

    _: Iterable[OperatorType] = OperatorGroupFIFO[OperatorType, Any]()
