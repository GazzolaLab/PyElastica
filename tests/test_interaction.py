__doc__ = """ Interaction module import test"""

# System imports
import pytest
import importlib
import elastica


def test_import_numpy_version_of_interaction_modules(monkeypatch):
    """
    Testing import of the Numpy interaction module. In case there is ImportError and Numba cannot be found,
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
    importlib.reload(elastica.interaction)

    # Test importing AnisotropicFrictionalPlane class
    from elastica._elastica_numpy._interaction import (
        AnisotropicFrictionalPlane as AnisotropicFrictionalPlane_numpy,
    )
    from elastica.interaction import AnisotropicFrictionalPlane

    assert AnisotropicFrictionalPlane == AnisotropicFrictionalPlane_numpy, str(
        " Imported modules are not matching "
        + str(AnisotropicFrictionalPlane)
        + " and "
        + str(AnisotropicFrictionalPlane_numpy)
    )

    # Test importing InteractionPlane class
    from elastica._elastica_numpy._interaction import (
        InteractionPlane as InteractionPlane_numpy,
    )
    from elastica.interaction import InteractionPlane

    assert InteractionPlane == InteractionPlane_numpy, str(
        " Imported modules are not matching "
        + str(InteractionPlane)
        + " and "
        + str(InteractionPlane_numpy)
    )

    # # Test importing AnisotropicFrictionalPlaneRigidBody
    # from elastica._elastica_numpy._interaction import (
    #     AnisotropicFrictionalPlaneRigidBody as AnisotropicFrictionalPlaneRigidBody_numpy,
    # )
    # from elastica.interaction import AnisotropicFrictionalPlaneRigidBody
    #
    # assert (
    #     AnisotropicFrictionalPlaneRigidBody == AnisotropicFrictionalPlaneRigidBody_numpy
    # ), str(
    #     " Imported modules are not matching "
    #     + str(AnisotropicFrictionalPlaneRigidBody)
    #     + " and "
    #     + str(AnisotropicFrictionalPlaneRigidBody_numpy)
    # )
    #
    # # Test importing InteractionPlaneRigidBody
    # from elastica._elastica_numpy._interaction import (
    #     InteractionPlaneRigidBody as InteractionPlaneRigidBody_numpy,
    # )
    # from elastica.interaction import InteractionPlaneRigidBody
    #
    # assert InteractionPlaneRigidBody == InteractionPlaneRigidBody_numpy, str(
    #     " Imported modules are not matching "
    #     + str(InteractionPlaneRigidBody)
    #     + " and "
    #     + str(InteractionPlaneRigidBody_numpy)
    # )

    # Test importing SlenderBodyTheory
    from elastica._elastica_numpy._interaction import (
        SlenderBodyTheory as SlenderBodyTheory_numpy,
    )
    from elastica.interaction import SlenderBodyTheory

    assert SlenderBodyTheory == SlenderBodyTheory_numpy, str(
        " Imported modules are not matching "
        + str(SlenderBodyTheory)
        + " and "
        + str(SlenderBodyTheory_numpy)
    )

    # Remove the import flag
    monkeypatch.delenv("IMPORT_TEST_NUMPY")
    # Reload the elastica after changing flag
    importlib.reload(elastica)
    importlib.reload(elastica.interaction)


def test_import_numba_version_of_interaction_modules(monkeypatch):
    """
    Testing import of the Numba interaction module. This test case imports Numba code
    and compares the manually imported Numba module.

    Returns
    -------

    """

    # Test importing AnisotropicFrictionalPlane class
    from elastica._elastica_numba._interaction import (
        AnisotropicFrictionalPlane as AnisotropicFrictionalPlane_numba,
    )
    from elastica.interaction import AnisotropicFrictionalPlane

    assert AnisotropicFrictionalPlane == AnisotropicFrictionalPlane_numba, str(
        " Imported modules are not matching "
        + str(AnisotropicFrictionalPlane)
        + " and "
        + str(AnisotropicFrictionalPlane_numba)
    )

    # Test importing InteractionPlane class
    from elastica._elastica_numba._interaction import (
        InteractionPlane as InteractionPlane_numba,
    )
    from elastica.interaction import InteractionPlane

    assert InteractionPlane == InteractionPlane_numba, str(
        " Imported modules are not matching "
        + str(InteractionPlane)
        + " and "
        + str(InteractionPlane_numba)
    )

    # # Test importing AnisotropicFrictionalPlaneRigidBody
    # from elastica._elastica_numba._interaction import (
    #     AnisotropicFrictionalPlaneRigidBody as AnisotropicFrictionalPlaneRigidBody_numba,
    # )
    # from elastica.interaction import AnisotropicFrictionalPlaneRigidBody
    #
    # assert (
    #     AnisotropicFrictionalPlaneRigidBody == AnisotropicFrictionalPlaneRigidBody_numba
    # ), str(
    #     " Imported modules are not matching "
    #     + str(AnisotropicFrictionalPlaneRigidBody)
    #     + " and "
    #     + str(AnisotropicFrictionalPlaneRigidBody_numba)
    # )
    #
    # # Test importing InteractionPlaneRigidBody
    # from elastica._elastica_numba._interaction import (
    #     InteractionPlaneRigidBody as InteractionPlaneRigidBody_numba,
    # )
    # from elastica.interaction import InteractionPlaneRigidBody
    #
    # assert InteractionPlaneRigidBody == InteractionPlaneRigidBody_numba, str(
    #     " Imported modules are not matching "
    #     + str(InteractionPlaneRigidBody)
    #     + " and "
    #     + str(InteractionPlaneRigidBody_numba)
    # )

    # Test importing SlenderBodyTheory
    from elastica._elastica_numba._interaction import (
        SlenderBodyTheory as SlenderBodyTheory_numba,
    )
    from elastica.interaction import SlenderBodyTheory

    assert SlenderBodyTheory == SlenderBodyTheory_numba, str(
        " Imported modules are not matching "
        + str(SlenderBodyTheory)
        + " and "
        + str(SlenderBodyTheory_numba)
    )


if __name__ == "__main__":
    from pytest import main

    main([__file__])
