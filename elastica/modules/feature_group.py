from elastica.typing import OperatorType

from collections.abc import Iterable

import itertools


class FeatureGroupFIFO(Iterable):
    """
    A class to store the features and their corresponding operators in a FIFO manner.

    Examples
    --------
    >>> feature_group = FeatureGroupFIFO()
    >>> feature_group.append_id(obj_1)
    >>> feature_group.append_id(obj_2)
    >>> feature_group.add_operators(obj_1, [OperatorType.ADD, OperatorType.SUBTRACT])
    >>> feature_group.add_operators(obj_2, [OperatorType.SUBTRACT, OperatorType.MULTIPLY])
    >>> list(feature_group)
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
