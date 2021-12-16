__doc__ = """ External forcing module import test """

# System imports
import pytest
import importlib
import elastica


def test_import_numpy_version_of_external_forces_modules(monkeypatch):
    """
    Testing import of the Numpy forcing module. In case there is ImportError and Numba cannot be found,
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
    importlib.reload(elastica.external_forces)

    # Test importing NoForces class
    from elastica._elastica_numpy._external_forces import NoForces as NoForces_numpy
    from elastica.external_forces import NoForces

    assert NoForces == NoForces_numpy, str(
        " Imported modules are not matching "
        + str(NoForces)
        + " and "
        + str(NoForces_numpy)
    )

    # Test importing GravityForces class
    from elastica._elastica_numpy._external_forces import (
        GravityForces as GravityForces_numpy,
    )
    from elastica.external_forces import GravityForces

    assert GravityForces == GravityForces_numpy, str(
        " Imported modules are not matching "
        + str(GravityForces)
        + " and "
        + str(GravityForces_numpy)
    )

    # Test importing EndpointForces
    from elastica._elastica_numpy._external_forces import (
        EndpointForces as EndpointForces_numpy,
    )
    from elastica.external_forces import EndpointForces

    assert EndpointForces == EndpointForces_numpy, str(
        " Imported modules are not matching "
        + str(EndpointForces)
        + " and "
        + str(EndpointForces_numpy)
    )

    # Test importing UniformTorques
    from elastica._elastica_numpy._external_forces import (
        UniformTorques as UniformTorques_numpy,
    )
    from elastica.external_forces import UniformTorques

    assert UniformTorques == UniformTorques_numpy, str(
        " Imported modules are not matching "
        + str(UniformTorques)
        + " and "
        + str(UniformTorques_numpy)
    )

    # Test importing UniformForces
    from elastica._elastica_numpy._external_forces import (
        UniformForces as UniformForces_numpy,
    )
    from elastica.external_forces import UniformForces

    assert UniformForces == UniformForces_numpy, str(
        " Imported modules are not matching "
        + str(UniformForces)
        + " and "
        + str(UniformForces_numpy)
    )

    # Test importing MuscleTorques
    from elastica._elastica_numpy._external_forces import (
        MuscleTorques as MuscleTorques_numpy,
    )
    from elastica.external_forces import MuscleTorques

    assert MuscleTorques == MuscleTorques_numpy, str(
        " Imported modules are not matching "
        + str(MuscleTorques)
        + " and "
        + str(MuscleTorques_numpy)
    )

    # Remove the import flag
    monkeypatch.delenv("IMPORT_TEST_NUMPY")
    # Reload the elastica after changing flag
    importlib.reload(elastica)
    importlib.reload(elastica.external_forces)


def test_import_numba_version_of_external_forces_modules(monkeypatch):
    """
    Testing import of the Numba forcing module. This test case imports Numba code
    and compares the manually imported Numba module.

    Returns
    -------

    """

    # Test importing NoForces class
    from elastica._elastica_numba._external_forces import NoForces as NoForces_numba
    from elastica.external_forces import NoForces

    assert NoForces == NoForces_numba, str(
        " Imported modules are not matching "
        + str(NoForces)
        + " and "
        + str(NoForces_numba)
    )

    # Test importing GravityForces class
    from elastica._elastica_numba._external_forces import (
        GravityForces as GravityForces_numba,
    )
    from elastica.external_forces import GravityForces

    assert GravityForces == GravityForces_numba, str(
        " Imported modules are not matching "
        + str(GravityForces)
        + " and "
        + str(GravityForces_numba)
    )

    # Test importing EndpointForces
    from elastica._elastica_numba._external_forces import (
        EndpointForces as EndpointForces_numba,
    )
    from elastica.external_forces import EndpointForces

    assert EndpointForces == EndpointForces_numba, str(
        " Imported modules are not matching "
        + str(EndpointForces)
        + " and "
        + str(EndpointForces_numba)
    )

    # Test importing UniformTorques
    from elastica._elastica_numba._external_forces import (
        UniformTorques as UniformTorques_numba,
    )
    from elastica.external_forces import UniformTorques

    assert UniformTorques == UniformTorques_numba, str(
        " Imported modules are not matching "
        + str(UniformTorques)
        + " and "
        + str(UniformTorques_numba)
    )

    # Test importing UniformForces
    from elastica._elastica_numba._external_forces import (
        UniformForces as UniformForces_numba,
    )
    from elastica.external_forces import UniformForces

    assert UniformForces == UniformForces_numba, str(
        " Imported modules are not matching "
        + str(UniformForces)
        + " and "
        + str(UniformForces_numba)
    )

    # Test importing MuscleTorques
    from elastica._elastica_numba._external_forces import (
        MuscleTorques as MuscleTorques_numba,
    )
    from elastica.external_forces import MuscleTorques

    assert MuscleTorques == MuscleTorques_numba, str(
        " Imported modules are not matching "
        + str(MuscleTorques)
        + " and "
        + str(MuscleTorques_numba)
    )


if __name__ == "__main__":
    from pytest import main

    main([__file__])
