import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from elastica import *
from dynamic_cantilever_helper import DynamicCantileverVibration


def simulate_dynamic_cantilever_with(
    density=2000,
    n_elem=100,
    final_time=300,
    mode=0,
    rendering_fps=30,  # For visualization
):
    class DynamicCantileverSimulator(BaseSystemCollection, Constraints, CallBacks):
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

    moi = np.pi / 4 * base_radius ** 4

    dl = base_length / n_elem
    dt = dl * 0.05
    step_skips = int(1.0 / (rendering_fps * dt))

    # Add Cosserat rod
    cantilever = CosseratRod.straight_rod(
        n_elem,
        start,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        0.0,
        youngs_modulus,
    )

    # Add constraints
    cantilever_sim.append(cantilever)
    cantilever_sim.constrain(cantilever).using(
        OneEndFixedBC, constrained_position_idx=(0,), constrained_director_idx=(0,)
    )

    end_velocity = 0.005
    vibration = DynamicCantileverVibration(
        base_length,
        base_area,
        moi,
        youngs_modulus,
        density,
        mode=mode,
        end_velocity_z=end_velocity,
    )

    input_velocity = vibration.get_initial_velocity_profile(
        cantilever.position_collection[0, :]
    )
    cantilever.velocity_collection[2, :] = input_velocity

    # Add call backs
    class CantileverCallBack(CallBackBaseClass):
        def __init__(self, step_skip: int, callback_params: dict):
            CallBackBaseClass.__init__(self)
            self.every = step_skip
            self.callback_params = callback_params

        def make_callback(self, system, time, current_step: int):

            if current_step % self.every == 0:

                self.callback_params["time"].append(time)
                self.callback_params["position"].append(
                    system.position_collection.copy()
                )
                self.callback_params["z_position"].append(
                    system.position_collection[2, -1].copy()
                )
                return

    recorded_history = defaultdict(list)
    cantilever_sim.collect_diagnostics(cantilever).using(
        CantileverCallBack, step_skip=step_skips, callback_params=recorded_history
    )
    cantilever_sim.finalize()

    total_steps = int(final_time / dt)
    print(f"Total steps: {total_steps}")

    timestepper = PositionVerlet()

    integrate(
        timestepper,
        cantilever_sim,
        final_time,
        total_steps,
    )

    # FFT
    amps = np.abs(fft(recorded_history["z_position"]))
    f_length = len(amps)
    amps = amps * 2 / f_length
    omegas = fftfreq(f_length, dt * step_skips) * 2 * np.pi  # [rad/s]

    try:
        peaks, _ = find_peaks(amps)
        peak = peaks[np.argmax(amps[peaks])]

        simulated_frequency = omegas[peak]
        theoretical_frequency = vibration.get_omega()

        simulated_amplitude = max(recorded_history["z_position"])
        theoretical_amplitude = vibration.get_amplitude()

        print(
            f"Theoretical frequency: {theoretical_frequency} rad/s \n"
            f"Simulated frequency: {simulated_frequency} rad/s \n"
            f"Theoretical amplitude: {theoretical_amplitude} m \n"
            f"Simulated amplitude: {simulated_amplitude} m"
        )

        # error_frequency = abs(simulated_frequency - resonant_frequency) / resonant_frequency
        # error_amplitude = abs(simulated_amplitude - resonant_amplitude) / resonant_amplitude

        return {
            "rod": cantilever,
            "recorded_history": recorded_history,
            "fft_frequencies": omegas,
            "fft_amplitudes": amps,
            "vibration": vibration,
            "peak": peak,
            "simulated_frequency": simulated_frequency,
            "theoretical_frequency": theoretical_frequency,
            "simulated_amplitude": simulated_amplitude,
            "theoretical_amplitude": theoretical_amplitude,
        }

    except IndexError:
        print("No peaks detected: change input parameters.")
