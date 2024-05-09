from elastica.typing import OperatorType

from collections.abc import Iterable

import itertools


class FeatureGroupFIFO(Iterable):
    def __init__(self):
        self._operator_collection: list[list[OperatorType]] = []
        self._operator_ids: list[int] = []

    def __iter__(self) -> OperatorType:
        if not self._operator_collection:
            raise RuntimeError("Feature group is not instantiated.")
        operator_chain = itertools.chain.from_iterable(self._operator_collection)
        for operator in operator_chain:
            yield operator

    def append_id(self, feature):
        self._operator_ids.append(id(feature))
        self._operator_collection.append([])

    def add_operators(self, feature, operators: list[OperatorType]):
        idx = self._operator_ids.index(id(feature))
        self._operator_collection[idx].extend(operators)
