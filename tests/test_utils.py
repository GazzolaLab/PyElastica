#!/usr/bin/env python3

__doc__ = """Test utility scripts"""

import pytest

from elastica.utils import isqrt, perm_parity


def test_isqrt_with_zero():
    assert isqrt(0) == 0


# Splitting tests for small and big
# integers as this should be the common use case
@pytest.mark.parametrize("inp, out", [(1, 1), (4, 2), (9, 3)])
def test_isqrt_small_numbers(inp, out):
    assert isqrt(inp) == out


@pytest.mark.parametrize("inp, out", [(56 ** 2, 56), (98 ** 2, 98)])
def test_isqrt_large_numbers(inp, out):
    assert isqrt(inp) == out


@pytest.mark.parametrize("seq", [list(range(3)), [2, 0, 1], [1, 2, 3]])
def test_perm_parity_correctness_on_even_sequences(seq):
    assert perm_parity(seq) == 1


@pytest.mark.parametrize("seq", [[1, 0, 2], [2, 1, 0], [4, 3, 2, 1]])
def test_perm_parity_correctness_on_odd_sequences(seq):
    assert perm_parity(seq) == -1
