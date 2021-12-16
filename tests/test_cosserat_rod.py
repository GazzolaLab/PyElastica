__doc__ = """ Cosserat rod module import test"""

# System imports
import pytest
import importlib
import elastica


def test_import_numpy_version_of_cosserat_rod_modules(monkeypatch):
    """
    Testing import of the Numpy Cosserat rod module. In case there is ImportError and Numba cannot be found,
    then automatically Numpy code has to be imported.  In order to generate an ImportError we create an environment
    variable called IMPORT_TEST_NUMPY and it is only used for raising ImportError. This test case imports Numpy code
    and compares the manually imported Numpy module.

    Returns
    -------

    """

    # First change the environment variable to import Numpy
    monkeypatch.setenv("IMPORT_TEST_NUMPY", "True", prepend=False)
    # After changing the import flag reload the modules.
    importlib.reload(elastica)
    importlib.reload(elastica.rod.cosserat_rod)

    # Test importing Cosserat rod class
    from elastica._elastica_numpy._rod._cosserat_rod import (
        CosseratRod as CosseratRod_numpy,
    )
    from elastica.rod.cosserat_rod import CosseratRod

    assert CosseratRod == CosseratRod_numpy, str(
        " Imported modules are not matching "
        + str(CosseratRod)
        + " and "
        + str(CosseratRod_numpy)
    )

    # Remove the import flag
    monkeypatch.delenv("IMPORT_TEST_NUMPY")
    # Reload the elastica after changing flag
    importlib.reload(elastica)
    importlib.reload(elastica.rod.cosserat_rod)


def test_import_numba_version_of_cosserat_rod_modules(monkeypatch):
    """
    Testing import of the Numba Cosserat rod module. This test case imports Numba code
    and compares the manually imported Numba module.

    Returns
    -------

    """

    # Test importing FreeRod class
    from elastica._elastica_numba._rod._cosserat_rod import (
        CosseratRod as CosseratRod_numba,
    )
    from elastica.rod.cosserat_rod import CosseratRod

    assert CosseratRod == CosseratRod_numba, str(
        " Imported modules are not matching "
        + str(CosseratRod)
        + " and "
        + str(CosseratRod_numba)
    )


if __name__ == "__main__":
    from pytest import main

    main([__file__])
