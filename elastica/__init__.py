""" If Numba module present, use Numba functions"""

try:
    import numba

    # raise ImportError
    IMPORT_NUMBA = True

except ImportError:
    IMPORT_NUMBA = False
