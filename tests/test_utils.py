#!/usr/bin/env python3

__doc__ = """Test utility scripts"""

import pytest

from elastica.utils import isqrt


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
