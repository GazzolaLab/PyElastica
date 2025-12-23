"""Elasticapp: A CPP accelerated backend for PyElastica kernels."""

# Import version function from version module
from elasticapp.version import version

# Import BlockRodSystem from C++ module
from elasticapp._memory_block import BlockRodSystem, BlockRodSystemView

# Import MemoryBlockCosseratRod after BlockRodSystem is defined to avoid circular import
from .memory_block_rod import MemoryBlockCosseratRod

__all__ = [
    "BlockRodSystem",
    "BlockRodSystemView",
    "MemoryBlockCosseratRod",
    "version",
]

# import submodules so that they can be easily accessed
# import elasticapp._linalg
# import elasticapp._rotations

# Note: These imports are commented out as they depend on blaze,
# which is being removed. They will be refactored in later tasks.
# import elasticapp._PyTags
# import elasticapp._PyArrays
# import elasticapp._PyCosseratRods
# import elasticapp._PyExamples
