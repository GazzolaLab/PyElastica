__doc__ = """ Data structures module import test"""

# System imports
import pytest
import importlib
import elastica


def test_import_numpy_version_of_data_structures_modules(monkeypatch):
    """
    Testing import of the Numpy data structures module. In case there is ImportError and Numba cannot be found,
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
    importlib.reload(elastica.rod.data_structures)

    # Test importing _RodSymplecticStepperMixin class
    from elastica._elastica_numpy._rod._data_structures import (
        _RodSymplecticStepperMixin as _RodSymplecticStepperMixin_numpy,
    )
    from elastica.rod.data_structures import _RodSymplecticStepperMixin

    assert _RodSymplecticStepperMixin == _RodSymplecticStepperMixin_numpy, str(
        " Imported modules are not matching "
        + str(_RodSymplecticStepperMixin)
        + " and "
        + str(_RodSymplecticStepperMixin_numpy)
    )

    # Test importing _bootstrap_from_data class
    from elastica._elastica_numpy._rod._data_structures import (
        _bootstrap_from_data as _bootstrap_from_data_numpy,
    )
    from elastica.rod.data_structures import _bootstrap_from_data

    assert _bootstrap_from_data == _bootstrap_from_data_numpy, str(
        " Imported modules are not matching "
        + str(_bootstrap_from_data)
        + " and "
        + str(_bootstrap_from_data_numpy)
    )

    # Test importing _State
    from elastica._elastica_numpy._rod._data_structures import (
        _State as _State_numpy,
    )
    from elastica.rod.data_structures import _State

    assert _State == _State_numpy, str(
        " Imported modules are not matching "
        + str(_State)
        + " and "
        + str(_State_numpy)
    )

    # Test importing _DerivativeState
    from elastica._elastica_numpy._rod._data_structures import (
        _DerivativeState as _DerivativeState_numpy,
    )
    from elastica.rod.data_structures import _DerivativeState

    assert _DerivativeState == _DerivativeState_numpy, str(
        " Imported modules are not matching "
        + str(_DerivativeState)
        + " and "
        + str(_DerivativeState_numpy)
    )

    # Test importing _KinematicState
    from elastica._elastica_numpy._rod._data_structures import (
        _KinematicState as _KinematicState_numpy,
    )
    from elastica.rod.data_structures import _KinematicState

    assert _KinematicState == _KinematicState_numpy, str(
        " Imported modules are not matching "
        + str(_KinematicState)
        + " and "
        + str(_KinematicState_numpy)
    )

    # Test importing _KinematicState
    from elastica._elastica_numpy._rod._data_structures import (
        _DynamicState as _DynamicState_numpy,
    )
    from elastica.rod.data_structures import _DynamicState

    assert _DynamicState == _DynamicState_numpy, str(
        " Imported modules are not matching "
        + str(_DynamicState)
        + " and "
        + str(_DynamicState_numpy)
    )

    # Remove the import flag
    monkeypatch.delenv("IMPORT_TEST_NUMPY")
    # Reload the elastica after changing flag
    importlib.reload(elastica)
    importlib.reload(elastica.rod.data_structures)


def test_import_numba_version_of_data_structures_modules(monkeypatch):
    """
    Testing import of the Numba data structures module. This test case imports Numba code
    and compares the manually imported Numba module.

    Returns
    -------

    """

    importlib.reload(elastica)
    importlib.reload(elastica.rod.data_structures)

    # Test importing _RodSymplecticStepperMixin class
    from elastica._elastica_numba._rod._data_structures import (
        _RodSymplecticStepperMixin as _RodSymplecticStepperMixin_numba,
    )
    from elastica.rod.data_structures import _RodSymplecticStepperMixin

    assert _RodSymplecticStepperMixin == _RodSymplecticStepperMixin_numba, str(
        " Imported modules are not matching "
        + str(_RodSymplecticStepperMixin)
        + " and "
        + str(_RodSymplecticStepperMixin_numba)
    )

    # Test importing _bootstrap_from_data class
    from elastica._elastica_numba._rod._data_structures import (
        _bootstrap_from_data as _bootstrap_from_data_numpy,
    )
    from elastica.rod.data_structures import _bootstrap_from_data

    assert _bootstrap_from_data == _bootstrap_from_data_numpy, str(
        " Imported modules are not matching "
        + str(_bootstrap_from_data)
        + " and "
        + str(_bootstrap_from_data_numpy)
    )

    # Test importing _State
    from elastica._elastica_numba._rod._data_structures import (
        _State as _State_numpy,
    )
    from elastica.rod.data_structures import _State

    assert _State == _State_numpy, str(
        " Imported modules are not matching "
        + str(_State)
        + " and "
        + str(_State_numpy)
    )

    # Test importing _DerivativeState
    from elastica._elastica_numba._rod._data_structures import (
        _DerivativeState as _DerivativeState_numpy,
    )
    from elastica.rod.data_structures import _DerivativeState

    assert _DerivativeState == _DerivativeState_numpy, str(
        " Imported modules are not matching "
        + str(_DerivativeState)
        + " and "
        + str(_DerivativeState_numpy)
    )

    # Test importing _KinematicState
    from elastica._elastica_numba._rod._data_structures import (
        _KinematicState as _KinematicState_numpy,
    )
    from elastica.rod.data_structures import _KinematicState

    assert _KinematicState == _KinematicState_numpy, str(
        " Imported modules are not matching "
        + str(_KinematicState)
        + " and "
        + str(_KinematicState_numpy)
    )

    # Test importing _KinematicState
    from elastica._elastica_numba._rod._data_structures import (
        _DynamicState as _DynamicState_numpy,
    )
    from elastica.rod.data_structures import _DynamicState

    assert _DynamicState == _DynamicState_numpy, str(
        " Imported modules are not matching "
        + str(_DynamicState)
        + " and "
        + str(_DynamicState_numpy)
    )


if __name__ == "__main__":
    from pytest import main

    main([__file__])
