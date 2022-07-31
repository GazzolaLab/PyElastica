__doc__ = """Timoshenko beam validation case, for detailed explanation refer to
Gazzola et. al. R. Soc. 2018  section 3.4.3 """

import numpy as np
from elastica import *
from examples.TimoshenkoBeamCase.timoshenko_postprocessing import plot_timoshenko


class TimoshenkoBeamSimulator(
    BaseSystemCollection, Constraints, Forcing, CallBacks, Damping
):
    pass


timoshenko_sim = TimoshenkoBeamSimulator()
final_time = 5000.0

# Options
PLOT_FIGURE = True
SAVE_FIGURE = True
SAVE_RESULTS = False
ADD_UNSHEARABLE_ROD = False

# setting up test params
n_elem = 100
start = np.zeros((3,))
direction = np.array([0.0, 0.0, 1.0])
normal = np.array([0.0, 1.0, 0.0])
base_length = 3.0
base_radius = 0.25
base_area = np.pi * base_radius ** 2
density = 5000
nu = 0.1 / 7 / density / base_area
E = 1e6
# For shear modulus of 1e4, nu is 99!
poisson_ratio = 99
shear_modulus = E / (poisson_ratio + 1.0)

shearable_rod = CosseratRod.straight_rod(
    n_elem,
    start,
    direction,
    normal,
    base_length,
    base_radius,
    density,
    0.0,  # internal damping constant, deprecated in v0.3.0
    E,
    shear_modulus=shear_modulus,
)

timoshenko_sim.append(shearable_rod)
# add damping
dl = base_length / n_elem
dt = 0.07 * dl
timoshenko_sim.dampen(shearable_rod).using(
    AnalyticalLinearDamper,
    damping_constant=nu,
    time_step=dt,
)

timoshenko_sim.constrain(shearable_rod).using(
    OneEndFixedBC, constrained_position_idx=(0,), constrained_director_idx=(0,)
)

end_force = np.array([-15.0, 0.0, 0.0])
timoshenko_sim.add_forcing_to(shearable_rod).using(
    EndpointForces, 0.0 * end_force, end_force, ramp_up_time=final_time / 2.0
)


if ADD_UNSHEARABLE_ROD:
    # Start into the plane
    unshearable_start = np.array([0.0, -1.0, 0.0])
    shear_modulus = E / (-0.7 + 1.0)
    unshearable_rod = CosseratRod.straight_rod(
        n_elem,
        unshearable_start,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        0.0,  # internal damping constant, deprecated in v0.3.0
        E,
        # Unshearable rod needs G -> inf, which is achievable with -ve poisson ratio
        shear_modulus=shear_modulus,
    )

    timoshenko_sim.append(unshearable_rod)

    # add damping
    timoshenko_sim.dampen(unshearable_rod).using(
        AnalyticalLinearDamper,
        damping_constant=nu,
        time_step=dt,
    )
    timoshenko_sim.constrain(unshearable_rod).using(
        OneEndFixedBC, constrained_position_idx=(0,), constrained_director_idx=(0,)
    )
    timoshenko_sim.add_forcing_to(unshearable_rod).using(
        EndpointForces, 0.0 * end_force, end_force, ramp_up_time=final_time / 2.0
    )

# Add call backs
class VelocityCallBack(CallBackBaseClass):
    """
    Tracks the velocity norms of the rod
    """

    def __init__(self, step_skip: int, callback_params: dict):
        CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.callback_params = callback_params

    def make_callback(self, system, time, current_step: int):

        if current_step % self.every == 0:

            self.callback_params["time"].append(time)
            # Collect x
            self.callback_params["velocity_norms"].append(
                np.linalg.norm(system.velocity_collection.copy())
            )
            return


recorded_history = defaultdict(list)
timoshenko_sim.collect_diagnostics(shearable_rod).using(
    VelocityCallBack, step_skip=500, callback_params=recorded_history
)

timoshenko_sim.finalize()
timestepper = PositionVerlet()
# timestepper = PEFRL()

total_steps = int(final_time / dt)
print("Total steps", total_steps)
integrate(timestepper, timoshenko_sim, final_time, total_steps)

if PLOT_FIGURE:
    plot_timoshenko(shearable_rod, end_force, SAVE_FIGURE, ADD_UNSHEARABLE_ROD)

if SAVE_RESULTS:
    import pickle

    filename = "Timoshenko_beam_data.dat"
    file = open(filename, "wb")
    pickle.dump(shearable_rod, file)
    file.close()

    tv = (
        np.asarray(recorded_history["time"]),
        np.asarray(recorded_history["velocity_norms"]),
    )

    def as_time_series(v):
        return v.T

    np.savetxt(
        "velocity_norms.csv",
        as_time_series(np.stack(tv)),
        delimiter=",",
    )
