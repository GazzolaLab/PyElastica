__doc__ = """ Time steppers module import test"""

# System imports
import pytest
import importlib
import elastica


def test_import_numpy_version_of_common_time_steppers_modules(monkeypatch):
    """
    Testing import of the Numpy time steppers __init__ module. In case there is ImportError and Numba cannot be found,
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
    importlib.reload(elastica.timestepper)

    # Test importing extend_stepper_interface class
    from elastica._elastica_numpy._timestepper import (
        extend_stepper_interface as extend_stepper_interface_numpy,
    )
    from elastica.timestepper import extend_stepper_interface

    assert extend_stepper_interface == extend_stepper_interface_numpy, str(
        " Imported modules are not matching "
        + str(extend_stepper_interface)
        + " and "
        + str(extend_stepper_interface_numpy)
    )

    # Test importing integrate class
    from elastica._elastica_numpy._timestepper import integrate as integrate_numpy
    from elastica.timestepper import integrate

    assert integrate == integrate_numpy, str(
        " Imported modules are not matching "
        + str(integrate)
        + " and "
        + str(integrate_numpy)
    )

    # Remove the import flag
    monkeypatch.delenv("IMPORT_TEST_NUMPY")
    # Reload the elastica after changing flag
    importlib.reload(elastica)
    importlib.reload(elastica.timestepper)


def test_import_numba_version_of_common_time_steppers_modules(monkeypatch):
    """
    Testing import of the Numba time steppers __init__ module. This test case imports Numba code
    and compares the manually imported Numba module.

    Returns
    -------

    """

    # Test importing extend_stepper_interface class
    from elastica._elastica_numba._timestepper import (
        extend_stepper_interface as extend_stepper_interface_numba,
    )
    from elastica.timestepper import extend_stepper_interface

    assert extend_stepper_interface == extend_stepper_interface_numba, str(
        " Imported modules are not matching "
        + str(extend_stepper_interface)
        + " and "
        + str(extend_stepper_interface_numba)
    )

    # Test importing integrate class
    from elastica._elastica_numba._timestepper import integrate as integrate_numba
    from elastica.timestepper import integrate

    assert integrate == integrate_numba, str(
        " Imported modules are not matching "
        + str(integrate)
        + " and "
        + str(integrate_numba)
    )


def test_import_numpy_version_of_symplectic_steppers_modules(monkeypatch):
    """
    Testing import of the Numpy symplectic steppers module. In case there is ImportError and Numba cannot be found,
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
    importlib.reload(elastica.timestepper.symplectic_steppers)

    # Test importing SymplecticStepperTag class
    from elastica._elastica_numpy._timestepper._symplectic_steppers import (
        SymplecticStepperTag as SymplecticStepperTag_numpy,
    )
    from elastica.timestepper.symplectic_steppers import SymplecticStepperTag

    assert SymplecticStepperTag == SymplecticStepperTag_numpy, str(
        " Imported modules are not matching "
        + str(SymplecticStepperTag)
        + " and "
        + str(SymplecticStepperTag_numpy)
    )

    # Test importing PositionVerlet class
    from elastica._elastica_numpy._timestepper._symplectic_steppers import (
        PositionVerlet as PositionVerlet_numpy,
    )
    from elastica.timestepper.symplectic_steppers import PositionVerlet

    assert PositionVerlet == PositionVerlet_numpy, str(
        " Imported modules are not matching "
        + str(PositionVerlet)
        + " and "
        + str(PositionVerlet_numpy)
    )

    # Test importing PEFRL
    from elastica._elastica_numpy._timestepper._symplectic_steppers import (
        PEFRL as PEFRL_numpy,
    )
    from elastica.timestepper.symplectic_steppers import PEFRL

    assert PEFRL == PEFRL_numpy, str(
        " Imported modules are not matching " + str(PEFRL) + " and " + str(PEFRL_numpy)
    )

    # Remove the import flag
    monkeypatch.delenv("IMPORT_TEST_NUMPY")
    # Reload the elastica after changing flag
    importlib.reload(elastica)
    importlib.reload(elastica.timestepper.symplectic_steppers)


def test_import_numba_version_of_interaction_modules(monkeypatch):
    """
    Testing import of the Numba symplectic steppers module. This test case imports Numba code
    and compares the manually imported Numba module.

    Returns
    -------

    """

    # Test importing SymplecticStepperTag class
    from elastica._elastica_numba._timestepper._symplectic_steppers import (
        SymplecticStepperTag as SymplecticStepperTag_numba,
    )
    from elastica.timestepper.symplectic_steppers import SymplecticStepperTag

    assert SymplecticStepperTag == SymplecticStepperTag_numba, str(
        " Imported modules are not matching "
        + str(SymplecticStepperTag)
        + " and "
        + str(SymplecticStepperTag_numba)
    )

    # Test importing PositionVerlet class
    from elastica._elastica_numba._timestepper._symplectic_steppers import (
        PositionVerlet as PositionVerlet_numba,
    )
    from elastica.timestepper.symplectic_steppers import PositionVerlet

    assert PositionVerlet == PositionVerlet_numba, str(
        " Imported modules are not matching "
        + str(PositionVerlet)
        + " and "
        + str(PositionVerlet_numba)
    )

    # Test importing PEFRL
    from elastica._elastica_numba._timestepper._symplectic_steppers import (
        PEFRL as PEFRL_numba,
    )
    from elastica.timestepper.symplectic_steppers import PEFRL

    assert PEFRL == PEFRL_numba, str(
        " Imported modules are not matching " + str(PEFRL) + " and " + str(PEFRL_numba)
    )


def test_import_numpy_version_of_explicit_steppers_modules(monkeypatch):
    """
    Testing import of the Numpy explicit steppers module. In case there is ImportError and Numba cannot be found,
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
    importlib.reload(elastica.timestepper.explicit_steppers)

    # Test importing ExplicitStepperTag class
    from elastica._elastica_numpy._timestepper._explicit_steppers import (
        ExplicitStepperTag as ExplicitStepperTag_numpy,
    )
    from elastica.timestepper.explicit_steppers import ExplicitStepperTag

    assert ExplicitStepperTag == ExplicitStepperTag_numpy, str(
        " Imported modules are not matching "
        + str(ExplicitStepperTag)
        + " and "
        + str(ExplicitStepperTag_numpy)
    )

    # Test importing RungeKutta4 class
    from elastica._elastica_numpy._timestepper._explicit_steppers import (
        RungeKutta4 as RungeKutta4_numpy,
    )
    from elastica.timestepper.explicit_steppers import RungeKutta4

    assert RungeKutta4 == RungeKutta4_numpy, str(
        " Imported modules are not matching "
        + str(RungeKutta4)
        + " and "
        + str(RungeKutta4_numpy)
    )

    # Test importing PEFRL
    from elastica._elastica_numpy._timestepper._explicit_steppers import (
        EulerForward as EulerForward_numpy,
    )
    from elastica.timestepper.explicit_steppers import EulerForward

    assert EulerForward == EulerForward_numpy, str(
        " Imported modules are not matching "
        + str(EulerForward)
        + " and "
        + str(EulerForward_numpy)
    )

    # Remove the import flag
    monkeypatch.delenv("IMPORT_TEST_NUMPY")
    # Reload the elastica after changing flag
    importlib.reload(elastica)
    importlib.reload(elastica.timestepper.explicit_steppers)


def test_import_numba_version_of_explicit_steppers_modules(monkeypatch):
    """
    Testing import of the Numba explicit steppers module. This test case imports Numba code
    and compares the manually imported Numba module.

    Returns
    -------

    """

    # Test importing ExplicitStepperTag class
    from elastica._elastica_numba._timestepper._explicit_steppers import (
        ExplicitStepperTag as ExplicitStepperTag_numba,
    )
    from elastica.timestepper.explicit_steppers import ExplicitStepperTag

    assert ExplicitStepperTag == ExplicitStepperTag_numba, str(
        " Imported modules are not matching "
        + str(ExplicitStepperTag)
        + " and "
        + str(ExplicitStepperTag_numba)
    )

    # Test importing RungeKutta4 class
    from elastica._elastica_numba._timestepper._explicit_steppers import (
        RungeKutta4 as RungeKutta4_numba,
    )
    from elastica.timestepper.explicit_steppers import RungeKutta4

    assert RungeKutta4 == RungeKutta4_numba, str(
        " Imported modules are not matching "
        + str(RungeKutta4)
        + " and "
        + str(RungeKutta4_numba)
    )

    # Test importing PEFRL
    from elastica._elastica_numba._timestepper._explicit_steppers import (
        EulerForward as EulerForward_numba,
    )
    from elastica.timestepper.explicit_steppers import EulerForward

    assert EulerForward == EulerForward_numba, str(
        " Imported modules are not matching "
        + str(EulerForward)
        + " and "
        + str(EulerForward_numba)
    )


if __name__ == "__main__":
    from pytest import main

    main([__file__])
