from elastica.typing import OperatorType

from collections.abc import Iterable

import itertools


class OperatorGroupFIFO(Iterable):
    """
    A class to store the features and their corresponding operators in a FIFO manner.

    Examples
    --------
    >>> operator_group = OperatorGroupFIFO()
    >>> operator_group.append_id(obj_1)
    >>> operator_group.append_id(obj_2)
    >>> operator_group.add_operators(obj_1, [ADD, SUBTRACT])
    >>> operator_group.add_operators(obj_2, [SUBTRACT, MULTIPLY])
    >>> list(operator_group)
    [OperatorType.ADD, OperatorType.SUBTRACT, OperatorType.SUBTRACT, OperatorType.MULTIPLY]

    Attributes
    ----------
    _operator_collection : list[list[OperatorType]]
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

    def __init__(self):
        self._operator_collection: list[list[OperatorType]] = []
        self._operator_ids: list[int] = []

    def __iter__(self) -> OperatorType:
        """Returns an operator iterator to satisfy the Iterable protocol."""
        operator_chain = itertools.chain.from_iterable(self._operator_collection)
        for operator in operator_chain:
            yield operator

    def append_id(self, feature):
        """Appends the id of the feature to the list of ids."""
        self._operator_ids.append(id(feature))
        self._operator_collection.append([])

    def add_operators(self, feature, operators: list[OperatorType]):
        """Adds the operators to the list of operators corresponding to the feature."""
        idx = self._operator_ids.index(id(feature))
        self._operator_collection[idx].extend(operators)

    def is_last(self, feature) -> bool:
        """Checks if the feature is the last feature in the FIFO."""
        return id(feature) == self._operator_ids[-1]
