import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
import elastica as ea
from analytical_dynamic_cantilever import AnalyticalDynamicCantilever


def simulate_dynamic_cantilever_with(
    density=2000.0,
    n_elem=100,
    final_time=300.0,
    mode=0,
    rendering_fps=30.0,  # For visualization
):
    """
    This function completes a dynamic cantilever simulation with the given parameters.

    Parameters
    ----------
    density: float
        Density of the rod
    n_elem: int
        The number of elements of the rod
    final_time: float
        Total simulation time. The timestep is determined by final_time / n_steps.
    mode: int
        Index of the first 'mode' th natural frequency.
        Up to the first four modes are supported.
    rendering_fps: float
        Frames per second for video plotting.
        The call back step-skip is also determined by rendering_fps.

    Returns
    -------
    dict of {str : int}
        A collection of parameters for post-processing.

    """

    class DynamicCantileverSimulator(
        ea.BaseSystemCollection, ea.Constraints, ea.CallBacks
    ):
        pass

    cantilever_sim = DynamicCantileverSimulator()

    # Add test parameters
    start = np.zeros((3,))
    direction = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 1.0, 0.0])
    base_length = 1
    base_radius = 0.02
    base_area = np.pi * base_radius ** 2
    youngs_modulus = 1e5

    moment_of_inertia = np.pi / 4 * base_radius ** 4

    dl = base_length / n_elem
    dt = dl * 0.05
    step_skips = int(1.0 / (rendering_fps * dt))

    # Add Cosserat rod
    cantilever_rod = ea.CosseratRod.straight_rod(
        n_elem,
        start,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        youngs_modulus=youngs_modulus,
    )

    # Add constraints
    cantilever_sim.append(cantilever_rod)
    cantilever_sim.constrain(cantilever_rod).using(
        ea.OneEndFixedBC, constrained_position_idx=(0,), constrained_director_idx=(0,)
    )

    end_velocity = 0.005
    analytical_cantilever_soln = AnalyticalDynamicCantilever(
        base_length,
        base_area,
        moment_of_inertia,
        youngs_modulus,
        density,
        mode=mode,
        end_velocity=end_velocity,
    )

    initial_velocity = analytical_cantilever_soln.get_initial_velocity_profile(
        cantilever_rod.position_collection[0, :]
    )
    cantilever_rod.velocity_collection[2, :] = initial_velocity

    # Add call backs
    class CantileverCallBack(ea.CallBackBaseClass):
        def __init__(self, step_skip: int, callback_params: dict):
            ea.CallBackBaseClass.__init__(self)
            self.every = step_skip
            self.callback_params = callback_params

        def make_callback(self, system, time, current_step: int):

            if current_step % self.every == 0:

                self.callback_params["time"].append(time)
                self.callback_params["position"].append(
                    system.position_collection.copy()
                )
                self.callback_params["deflection"].append(
                    system.position_collection[2, -1].copy()
                )
                return

    recorded_history = ea.defaultdict(list)
    cantilever_sim.collect_diagnostics(cantilever_rod).using(
        CantileverCallBack, step_skip=step_skips, callback_params=recorded_history
    )
    cantilever_sim.finalize()

    total_steps = int(final_time / dt)
    print(f"Total steps: {total_steps}")

    timestepper = ea.PositionVerlet()

    ea.integrate(
        timestepper,
        cantilever_sim,
        final_time,
        total_steps,
    )

    # FFT
    amplitudes = np.abs(fft(recorded_history["deflection"]))
    fft_length = len(amplitudes)
    amplitudes = amplitudes * 2 / fft_length
    omegas = fftfreq(fft_length, dt * step_skips) * 2 * np.pi  # [rad/s]

    try:
        peaks, _ = find_peaks(amplitudes)
        peak = peaks[np.argmax(amplitudes[peaks])]

        simulated_frequency = omegas[peak]
        theoretical_frequency = analytical_cantilever_soln.get_omega()

        simulated_amplitude = max(recorded_history["deflection"])
        theoretical_amplitude = analytical_cantilever_soln.get_amplitude()

        print(
            f"Theoretical frequency: {theoretical_frequency} rad/s \n"
            f"Simulated frequency: {simulated_frequency} rad/s \n"
            f"Theoretical amplitude: {theoretical_amplitude} m \n"
            f"Simulated amplitude: {simulated_amplitude} m"
        )

        return {
            "rod": cantilever_rod,
            "recorded_history": recorded_history,
            "fft_frequencies": omegas,
            "fft_amplitudes": amplitudes,
            "analytical_cantilever_soln": analytical_cantilever_soln,
            "peak": peak,
            "simulated_frequency": simulated_frequency,
            "theoretical_frequency": theoretical_frequency,
            "simulated_amplitude": simulated_amplitude,
            "theoretical_amplitude": theoretical_amplitude,
        }

    except RuntimeError:
        print("No peaks detected: change input parameters.")
