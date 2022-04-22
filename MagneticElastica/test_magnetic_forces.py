import numpy as np
import pytest
from MagneticElastica.magnetic_forces import compute_ramp_factor
from elastica.utils import Tolerance


@pytest.mark.parametrize("time", [0.0, 1.0, 2.0, 4.0, 8.0])
def test_compute_ramp_factor(time):
    # tests computation of ramp factor
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
