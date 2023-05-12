__doc__ = """Visualization of simulated dynamic cantilever beam"""

import elastica as ea
from dynamic_cantilever import simulate_dynamic_cantilever_with
from dynamic_cantilever_post_processing import (
    plot_end_position_with,
    plot_dynamic_cantilever_video_with,
)


class DynamicCantileverSimulator(ea.BaseSystemCollection, ea.Constraints, ea.CallBacks):
    pass


cantilever_sim = DynamicCantileverSimulator()

# Options
PLOT_FIGURE = True
SAVE_FIGURE = True
PLOT_VIDEO = True

# Add test parameters
n_elem = 200
final_time = 100
rendering_fps = 30
mode = 0

sim_result = simulate_dynamic_cantilever_with(
    density=500,
    n_elem=n_elem,
    final_time=final_time,
    mode=mode,
    rendering_fps=rendering_fps,
)

cantilever = sim_result["rod"]
recorded_history = sim_result["recorded_history"]
omegas = sim_result["fft_frequencies"]
amplitudes = sim_result["fft_amplitudes"]
analytical_cantilever_soln = sim_result["analytical_cantilever_soln"]
peak = sim_result["peak"]

# Plotting end-point position over time
# and frequency-domain equivalent
if PLOT_FIGURE or SAVE_FIGURE:
    plot_end_position_with(
        recorded_history,
        analytical_cantilever_soln,
        omegas,
        amplitudes,
        peak,
        PLOT_FIGURE,
        SAVE_FIGURE,
    )

# Plotting video
if PLOT_VIDEO:
    plot_dynamic_cantilever_video_with(mode, recorded_history, rendering_fps)
