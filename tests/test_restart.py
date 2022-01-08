__doc__ = """Test restart functionality """

import pytest
import numpy as np
from numpy.testing import assert_allclose
from elastica.utils import Tolerance
from elastica.wrappers import (
    BaseSystemCollection,
    Constraints,
    Forcing,
    Connections,
    CallBacks,
)
from elastica.restart import save_state, load_state


class GenericSimulatorClass(
    BaseSystemCollection, Constraints, Forcing, Connections, CallBacks
):
    pass


class TestRestartFunctionsWithFeaturesUsingCosseratRod:
    @pytest.fixture(scope="function")
    def load_collection(self):
        sc = GenericSimulatorClass()
        from elastica.rod.cosserat_rod import CosseratRod

        # rod = RodBase()
        rod_list = []
        for _ in range(5):
            rod = CosseratRod.straight_rod(
                n_elements=10,
                start=np.zeros((3)),
                direction=np.array([0, 1, 0.0]),
                normal=np.array([1, 0, 0.0]),
                base_length=1,
                base_radius=1,
                density=1,
                nu=1,
                youngs_modulus=1,
            )
            # Bypass check, but its fine for testing
            sc._systems.append(rod)

            # Also add rods to a separate list
            rod_list.append(rod)

        return sc, rod_list

    def test_restart_save_load(self, load_collection):
        simulator_class, rod_list = load_collection

        # Finalize simulator
        simulator_class.finalize()

        directory = "restart_test_data/"
        time = np.random.rand()

        # save state
        save_state(simulator_class, directory, time=time)

        # load state
        restart_time = load_state(simulator_class, directory)

        # check if restart time loaded correctly
        assert_allclose(restart_time, time, atol=Tolerance.atol())

        # check if rods are loaded correctly
        for idx, correct_rod in enumerate(rod_list):
            test_rod = simulator_class[idx]
            for key, value in correct_rod.__dict__.items():

                # get correct values
                correct_value = getattr(correct_rod, key)

                # get test values
                test_value = getattr(test_rod, key)

                assert_allclose(test_value, correct_value)


class TestRestartFunctionsWithFeaturesUsingRigidBodies:
    @pytest.fixture(scope="function")
    def load_collection(self):
        sc = GenericSimulatorClass()
        from elastica.rigidbody import Cylinder

        # rod = RodBase()
        cylinder_list = []
        for _ in range(5):
            cylinder = Cylinder(
                start=np.zeros((3)),
                direction=np.array([0, 1, 0.0]),
                normal=np.array([1, 0, 0.0]),
                base_length=1,
                base_radius=1,
                density=1,
            )
            # Bypass check, but its fine for testing
            sc._systems.append(cylinder)

            # Also add rods to a separate list
            cylinder_list.append(cylinder)

        return sc, cylinder_list

    def test_restart_save_load(self, load_collection):
        simulator_class, cylinder_list = load_collection

        # Finalize simulator
        simulator_class.finalize()

        directory = "restart_test_data/"
        time = np.random.rand()

        # save state
        save_state(simulator_class, directory, time=time)

        # load state
        restart_time = load_state(simulator_class, directory)

        # check if restart time loaded correctly
        assert_allclose(restart_time, time, atol=Tolerance.atol())

        # check if rods are loaded correctly
        for idx, correct_cylinder in enumerate(cylinder_list):
            test_cylinder = simulator_class[idx]
            for key, value in correct_cylinder.__dict__.items():

                # get correct values
                correct_value = getattr(correct_cylinder, key)

                # get test values
                test_value = getattr(test_cylinder, key)

                assert_allclose(test_value, correct_value)


if __name__ == "__main__":
    from pytest import main

    main([__file__])
