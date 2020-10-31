__doc__ = """ Boundary conditions module import test"""

# System imports
import pytest
import importlib
import elastica


def test_import_numpy_version_of_boundary_conditions_modules(monkeypatch):
    """
    Testing import of the Numpy boundary conditions module. In case there is ImportError and Numba cannot be found,
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
    importlib.reload(elastica.boundary_conditions)

    # Test importing FreeRod class
    from elastica._elastica_numpy._boundary_conditions import FreeRod as FreeRod_numpy
    from elastica.boundary_conditions import FreeRod

    assert FreeRod == FreeRod_numpy, str(
        " Imported modules are not matching "
        + str(FreeRod)
        + " and "
        + str(FreeRod_numpy)
    )

    # Test importing OneEndFixedRod class
    from elastica._elastica_numpy._boundary_conditions import (
        OneEndFixedRod as OneEndFixedRod_numpy,
    )
    from elastica.boundary_conditions import OneEndFixedRod

    assert OneEndFixedRod == OneEndFixedRod_numpy, str(
        " Imported modules are not matching "
        + str(OneEndFixedRod)
        + " and "
        + str(OneEndFixedRod_numpy)
    )

    # Test importing HelicalBucklingBC_numpy
    from elastica._elastica_numpy._boundary_conditions import (
        HelicalBucklingBC as HelicalBucklingBC_numpy,
    )
    from elastica.boundary_conditions import HelicalBucklingBC

    assert HelicalBucklingBC == HelicalBucklingBC_numpy, str(
        " Imported modules are not matching "
        + str(HelicalBucklingBC)
        + " and "
        + str(HelicalBucklingBC_numpy)
    )

    # Remove the import flag
    monkeypatch.delenv("IMPORT_TEST_NUMPY")
    # Reload the elastica after changing flag
    importlib.reload(elastica)
    importlib.reload(elastica.boundary_conditions)


def test_import_numba_version_of_boundary_conditions_modules(monkeypatch):
    """
    Testing import of the Numba boundary conditions module. This test case imports Numba code
    and compares the manually imported Numba module.

    Returns
    -------

    """

    # Test importing FreeRod class
    from elastica._elastica_numba._boundary_conditions import FreeRod as FreeRod_numba
    from elastica.boundary_conditions import FreeRod

    assert FreeRod == FreeRod_numba, str(
        " Imported modules are not matching "
        + str(FreeRod)
        + " and "
        + str(FreeRod_numba)
    )

    # Test importing OneEndFixedRod class
    from elastica._elastica_numba._boundary_conditions import (
        OneEndFixedRod as OneEndFixedRod_numba,
    )
    from elastica.boundary_conditions import OneEndFixedRod

    assert OneEndFixedRod == OneEndFixedRod_numba, str(
        " Imported modules are not matching "
        + str(OneEndFixedRod)
        + " and "
        + str(OneEndFixedRod_numba)
    )

    # Test importing HelicalBucklingBC_numpy
    from elastica._elastica_numba._boundary_conditions import (
        HelicalBucklingBC as HelicalBucklingBC_numba,
    )
    from elastica.boundary_conditions import HelicalBucklingBC

    assert HelicalBucklingBC == HelicalBucklingBC_numba, str(
        " Imported modules are not matching "
        + str(HelicalBucklingBC)
        + " and "
        + str(HelicalBucklingBC_numba)
    )


if __name__ == "__main__":
    from pytest import main

    main([__file__])
