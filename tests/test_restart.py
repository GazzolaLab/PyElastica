__doc__ = """Test restart functionality """

import pytest
import numpy as np
from numpy.testing import assert_allclose
from elastica.utils import Tolerance
from elastica.modules import (
    BaseSystemCollection,
    Constraints,
    Forcing,
    Connections,
    CallBacks,
)
from elastica.restart import save_state, load_state
import elastica as ea


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

    def run_sim(self, final_time, load_from_restart, save_data_restart):
        class BaseSimulatorClass(
            BaseSystemCollection, Constraints, Forcing, Connections, CallBacks
        ):
            pass

        simulator_class = BaseSimulatorClass()

        rod_list = []
        for _ in range(5):
            rod = ea.CosseratRod.straight_rod(
                n_elements=10,
                start=np.zeros((3)),
                direction=np.array([0, 1, 0.0]),
                normal=np.array([1, 0, 0.0]),
                base_length=1,
                base_radius=1,
                density=1,
                youngs_modulus=1,
            )
            # Bypass check, but its fine for testing
            simulator_class._systems.append(rod)

            # Also add rods to a separate list
            rod_list.append(rod)

        for rod in rod_list:
            simulator_class.add_forcing_to(rod).using(
                ea.EndpointForces,
                start_force=np.zeros(
                    3,
                ),
                end_force=np.array([0, 0.1, 0]),
                ramp_up_time=0.1,
            )

        # Finalize simulator
        simulator_class.finalize()

        directory = "restart_test_data/"

        time_step = 1e-4
        total_steps = int(final_time / time_step)

        if load_from_restart:
            restart_time = ea.load_state(simulator_class, directory, True)

        else:
            restart_time = np.float64(0.0)

        timestepper = ea.PositionVerlet()
        time = ea.integrate(
            timestepper,
            simulator_class,
            final_time,
            total_steps,
            restart_time=restart_time,
        )

        if save_data_restart:
            ea.save_state(simulator_class, directory, time, True)

        # Compute final time accelerations
        recorded_list = np.zeros((len(rod_list), rod_list[0].n_elems + 1))
        for i, rod in enumerate(rod_list):
            recorded_list[i, :] = rod.acceleration_collection[1, :]

        return recorded_list

    @pytest.mark.parametrize("final_time", [0.2, 1.0])
    def test_save_restart_run_sim(self, final_time):

        # First half of simulation
        _ = self.run_sim(
            final_time / 2, load_from_restart=False, save_data_restart=True
        )

        # Second half of simulation
        recorded_list = self.run_sim(
            final_time / 2, load_from_restart=True, save_data_restart=False
        )
        recorded_list_second_half = recorded_list.copy()

        # Full simulation
        recorded_list = self.run_sim(
            final_time, load_from_restart=False, save_data_restart=False
        )
        recorded_list_full_sim = recorded_list.copy()

        # Compare final accelerations of rods
        assert_allclose(recorded_list_second_half, recorded_list_full_sim)


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
