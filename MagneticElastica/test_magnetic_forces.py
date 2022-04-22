import numpy as np
import pytest
from MagneticElastica.magnetic_forces import (
    compute_ramp_factor,
    BaseMagneticField,
    ConstantMagneticField,
    SingleModeOscillatingMagneticField,
    ExternalMagneticFieldForces,
)
from elastica.utils import Tolerance


@pytest.mark.parametrize("time", [0.0, 1.0, 2.0, 4.0, 8.0])
def test_compute_ramp_factor(time):
    ramp_interval = 1.0
    start_time = 2.0
    end_time = 5.0

    factor = compute_ramp_factor(
        time=time, ramp_interval=ramp_interval, start_time=start_time, end_time=end_time
    )

    correct_factor = 0.0
    if time > start_time:
        correct_factor = (time > start_time) * min(
            1.0, (time - start_time) / ramp_interval
        )
    if time > end_time:
        correct_factor = max(0.0, -1 / ramp_interval * (time - end_time) + 1.0)

    np.testing.assert_allclose(factor, correct_factor, atol=Tolerance.atol())


@pytest.mark.parametrize("time", [0.0, 1.0, 2.0, 4.0, 8.0])
def test_base_magnetic_field(time):
    magnetic_field_object = BaseMagneticField()
    magnetic_field_value = magnetic_field_object.value(time=time)
    # base class does nothing!
    assert magnetic_field_value == None


@pytest.mark.parametrize("time", [4.0, 8.0, 16.0])
@pytest.mark.parametrize("ramp_interval", [1.0, 2.0])
@pytest.mark.parametrize("start_time", [0.0, 1.0, 2.0])
@pytest.mark.parametrize("end_time", [4.0, 8.0])
def test_constant_magnetic_field(time, ramp_interval, start_time, end_time):
    dim = 3
    magnetic_field_amplitude = np.random.rand(dim)
    magnetic_field_object = ConstantMagneticField(
        magnetic_field_amplitude=magnetic_field_amplitude,
        ramp_interval=ramp_interval,
        start_time=start_time,
        end_time=end_time,
    )
    magnetic_field_value = magnetic_field_object.value(time=time)
    correct_factor = 0.0
    if time > start_time:
        correct_factor = (time > start_time) * min(
            1.0, (time - start_time) / ramp_interval
        )
    if time > end_time:
        correct_factor = max(0.0, -1 / ramp_interval * (time - end_time) + 1.0)
    correct_magnetic_field_value = correct_factor * magnetic_field_amplitude

    np.testing.assert_allclose(
        magnetic_field_value, correct_magnetic_field_value, atol=Tolerance.atol()
    )


@pytest.mark.parametrize("time", [4.0, 8.0, 16.0])
@pytest.mark.parametrize("ramp_interval", [1.0, 2.0])
@pytest.mark.parametrize("start_time", [0.0, 1.0, 2.0])
@pytest.mark.parametrize("end_time", [4.0, 8.0])
def test_single_mode_oscillating_magnetic_field(
    time, ramp_interval, start_time, end_time
):
    dim = 3
    magnetic_field_amplitude = np.random.rand(dim)
    magnetic_field_angular_frequency = np.random.rand(dim)
    magnetic_field_phase_difference = np.random.rand(dim)
    magnetic_field_object = SingleModeOscillatingMagneticField(
        magnetic_field_amplitude=magnetic_field_amplitude,
        magnetic_field_angular_frequency=magnetic_field_angular_frequency,
        magnetic_field_phase_difference=magnetic_field_phase_difference,
        ramp_interval=ramp_interval,
        start_time=start_time,
        end_time=end_time,
    )
    magnetic_field_value = magnetic_field_object.value(time=time)
    correct_factor = 0.0
    if time > start_time:
        correct_factor = (time > start_time) * min(
            1.0, (time - start_time) / ramp_interval
        )
    if time > end_time:
        correct_factor = max(0.0, -1 / ramp_interval * (time - end_time) + 1.0)

    correct_magnetic_field_value = (
        correct_factor
        * magnetic_field_amplitude
        * np.sin(
            magnetic_field_angular_frequency * time + magnetic_field_phase_difference
        )
    )

    np.testing.assert_allclose(
        magnetic_field_value, correct_magnetic_field_value, atol=Tolerance.atol()
    )


def mock_magnetic_rod_init(self):
    self.n_elems = 0.0
    self.external_forces = 0.0
    self.external_torques = 0.0
    self.director_collection = 0.0
    self.magnetization_collection = 0.0


MockMagneticRod = type(
    "MockMagneticRod", (object,), {"__init__": mock_magnetic_rod_init}
)


@pytest.mark.parametrize("n_elems", [2, 4, 16])
@pytest.mark.parametrize("time", [4.0, 8.0, 16.0])
@pytest.mark.parametrize("ramp_interval", [1.0, 2.0])
@pytest.mark.parametrize("start_time", [0.0, 1.0, 2.0])
@pytest.mark.parametrize("end_time", [4.0, 8.0])
def test_external_magnetic_field_forces(
    n_elems, time, ramp_interval, start_time, end_time
):
    dim = 3
    mock_rod = MockMagneticRod()
    mock_rod.external_torques = np.zeros((dim, n_elems))
    mock_rod.n_elems = n_elems
    mock_rod.director_collection = np.repeat(
        np.identity(dim)[:, :, np.newaxis], n_elems, axis=2
    )
    magnetization_vector = np.random.rand(dim)
    mock_rod.magnetization_collection = magnetization_vector.reshape(dim, 1) * np.ones(
        (n_elems,)
    )
    magnetic_field_amplitude = np.random.rand(dim)
    magnetic_field_object = ConstantMagneticField(
        magnetic_field_amplitude=magnetic_field_amplitude,
        ramp_interval=ramp_interval,
        start_time=start_time,
        end_time=end_time,
    )
    external_magnetic_field_forcing = ExternalMagneticFieldForces(
        external_magnetic_field=magnetic_field_object
    )

    external_magnetic_field_forcing.apply_torques(system=mock_rod, time=time)

    correct_factor = 0.0
    if time > start_time:
        correct_factor = (time > start_time) * min(
            1.0, (time - start_time) / ramp_interval
        )
    if time > end_time:
        correct_factor = max(0.0, -1 / ramp_interval * (time - end_time) + 1.0)
    correct_magnetic_field_value = correct_factor * magnetic_field_amplitude
    correct_magnetic_field_torques = np.cross(
        magnetization_vector, correct_magnetic_field_value
    ).reshape(dim, 1) * np.ones((n_elems,))

    # no effect on forces
    np.testing.assert_allclose(mock_rod.external_forces, 0.0, atol=Tolerance.atol())
    np.testing.assert_allclose(
        mock_rod.external_torques, correct_magnetic_field_torques, atol=Tolerance.atol()
    )
