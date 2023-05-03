import numpy as np
from collections import defaultdict
import elastica as ea


class SnakeSimulator(
    ea.BaseSystemCollection, ea.Constraints, ea.Forcing, ea.Damping, ea.CallBacks
):
    pass


def run_snake(b_coeff, SAVE_RESULTS=False):

    snake_sim = SnakeSimulator()

    # setting up test params
    n_elem = 20
    start = np.zeros((3,))
    direction = np.array([0.0, 0.0, 1.0])
    normal = np.array([0.0, 1.0, 0.0])
    base_length = 1.0
    base_radius = 0.025
    density = 1000
    E = 1e7
    poisson_ratio = 0.5
    shear_modulus = E / (poisson_ratio + 1.0)

    shearable_rod = ea.CosseratRod.straight_rod(
        n_elem,
        start,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        youngs_modulus=E,
        shear_modulus=shear_modulus,
    )

    snake_sim.append(shearable_rod)

    # Add gravitational forces
    gravitational_acc = -9.80665
    snake_sim.add_forcing_to(shearable_rod).using(
        ea.GravityForces, acc_gravity=np.array([0.0, gravitational_acc, 0.0])
    )

    period = 1.0
    wave_length = b_coeff[-1]
    snake_sim.add_forcing_to(shearable_rod).using(
        ea.MuscleTorques,
        base_length=base_length,
        b_coeff=b_coeff[:-1],
        period=period,
        wave_number=2.0 * np.pi / (wave_length),
        phase_shift=0.0,
        direction=normal,
        rest_lengths=shearable_rod.rest_lengths,
        ramp_up_time=period,
        with_spline=True,
    )

    # Add friction forces
    origin_plane = np.array([0.0, -base_radius, 0.0])
    normal_plane = normal
    slip_velocity_tol = 1e-8
    froude = 0.1
    mu = base_length / (period * period * np.abs(gravitational_acc) * froude)
    kinetic_mu_array = np.array(
        [mu, 1.5 * mu, 2.0 * mu]
    )  # [forward, backward, sideways]
    static_mu_array = 2 * kinetic_mu_array
    snake_sim.add_forcing_to(shearable_rod).using(
        ea.AnisotropicFrictionalPlane,
        k=1.0,
        nu=1e-6,
        plane_origin=origin_plane,
        plane_normal=normal_plane,
        slip_velocity_tol=slip_velocity_tol,
        static_mu_array=static_mu_array,
        kinetic_mu_array=kinetic_mu_array,
    )

    # add damping
    damping_constant = 5.0
    dt = 5.0e-5 * period
    snake_sim.dampen(shearable_rod).using(
        ea.AnalyticalLinearDamper,
        damping_constant=damping_constant,
        time_step=dt,
    )

    # Add call backs
    class ContinuumSnakeCallBack(ea.CallBackBaseClass):
        """
        Call back function for continuum snake
        """

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

                return

    pp_list = defaultdict(list)
    snake_sim.collect_diagnostics(shearable_rod).using(
        ContinuumSnakeCallBack, step_skip=200, callback_params=pp_list
    )

    snake_sim.finalize()
    timestepper = ea.PositionVerlet()
    # timestepper = PEFRL()

    final_time = (11.0 + 0.01) * period
    total_steps = int(final_time / dt)
    print("Total steps", total_steps)
    ea.integrate(timestepper, snake_sim, final_time, total_steps)

    if SAVE_RESULTS:
        import pickle

        filename = "continuum_snake.dat"
        file = open(filename, "wb")
        pickle.dump(pp_list, file)
        file.close()

    return pp_list


if __name__ == "__main__":

    # Options
    SAVE_RESULTS = True

    # Add muscle forces on the rod
    t_coeff_optimized = np.array([17.4, 48.5, 5.4, 14.7, 0.97])

    # run the simulation
    pp_list = run_snake(t_coeff_optimized, SAVE_RESULTS)

    print("Datafile Created")
