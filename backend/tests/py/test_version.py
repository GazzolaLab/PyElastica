"""Test for elasticapp.version function."""

import pytest
import elasticapp
import importlib.metadata

def test_version():
    """Test that elasticapp.version returns the current version."""
    version = elasticapp.version()
    assert isinstance(version, str)
    assert version == importlib.metadata.version("elasticapp")
