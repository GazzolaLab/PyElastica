import numpy as np
from elastica.boundary_conditions import OneEndFixedBC
from elastica.external_forces import EndpointForces
from elastica.timestepper.symplectic_steppers import PositionVerlet
import elastica as ea
from examples.convergence_functions import calculate_error_norm
from cantilever_transversal_load_postprocessing import adjust_square_cross_section
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb
import json


def analytical_results(index):
    with open("cantilever_transversal_load_data.json", "r") as file:
        analytical_results = json.load(file)

    return analytical_results[index]


def cantilever_subjected_to_a_transversal_load(n_elem=19):
    start = np.zeros((3,))
    direction = np.array([0.0, 1.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    radius = 1
    base_length = 0.25 * radius * np.pi
    base_radius = 0.01 / (
        np.pi ** (1 / 2)
    )  # The Cross-sectional area is 1e-4(we assume its equivalent to a square cross-sectional surface with same area)
    base_area = 1e-4
    density = 1000
    youngs_modulus = 1e9
    poisson_ratio = 0
    shear_modulus = youngs_modulus / (poisson_ratio + 1.0)

    class SquareRodSimulator(
        ea.BaseSystemCollection, ea.Constraints, ea.Forcing, ea.Damping, ea.CallBacks
    ):
        pass

    squarerod_sim = SquareRodSimulator()

    density = 1000
    t = np.linspace(0, 0.25 * np.pi, n_elem + 1)
    tmp = np.zeros((3, n_elem + 1), dtype=np.float64)
    tmp[0, :] = -radius * np.cos(t) + 1
    tmp[1, :] = radius * np.sin(t)
    tmp[2, :] *= 0.0
    director = np.zeros((3, 3, n_elem), dtype=np.float64)
    tan = tmp[:, 1:] - tmp[:, :-1]
    tan = tan / np.linalg.norm(tan, axis=0)
    side_length = 0.01

    d1 = np.array([0.0, 0.0, 1.0]).reshape((3, 1))
    d2 = np.cross(tan, d1, axis=0)

    director[0, :, :] = d1
    director[1, :, :] = d2
    director[2, :, :] = tan

    rod = ea.CosseratRod.straight_rod(
        n_elem,
        start,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        youngs_modulus=youngs_modulus,
        shear_modulus=shear_modulus,
        position=tmp,
        directors=director,
    )

    # Adjust the Cross Section
    adjust_square_cross_section(rod, youngs_modulus, side_length)

    squarerod_sim.append(rod)

    # squarerod_sim.finalize()
    rod.rest_kappa[...] = rod.kappa

    dl = base_length / n_elem
    dt = 0.01 * dl / 100

    squarerod_sim.constrain(rod).using(
        OneEndFixedBC, constrained_position_idx=(0,), constrained_director_idx=(0,)
    )

    print("One end of the rod is now fixed in place")

    squarerod_sim.dampen(rod).using(
        ea.AnalyticalLinearDamper,
        damping_constant=0.3,
        time_step=dt,
    )

    ramp_up_time = 1

    origin_force = np.array([0.0, 0.0, 0.0])
    end_force = np.array([0.0, 0.0, 6.0])

    squarerod_sim.add_forcing_to(rod).using(
        EndpointForces, origin_force, end_force, ramp_up_time=ramp_up_time
    )
    print("Forces added to the rod")

    # Finalization and Run the Project
    final_time = 5
    total_steps = int(final_time / dt)
    print("Total steps to take", total_steps)

    squarerod_sim.finalize()
    print("System finalized")

    # The simulation result from Project3.3.2 with 400 elements/ Tip position Z

    # generate analytical solution array from [400]

    analytical_results_sub = np.zeros(n_elem + 1)

    for i in range(n_elem + 1):
        analytical_results_converge_index = round((i * dl) / (base_length / 400))
        position_left = analytical_results(analytical_results_converge_index)
        analytical_results_sub[i] = position_left

    timestepper = PositionVerlet()

    dt = final_time / total_steps
    time = 0.0
    for i in range(total_steps):
        time = timestepper.step(squarerod_sim, time, dt)
    print(rod.position_collection[2, ...])

    error, l1, l2, linf = calculate_error_norm(
        analytical_results_sub,
        rod.position_collection[2, ...],
        n_elem,
    )

    return {"rod": rod, "error": error, "l1": l1, "l2": l2, "linf": linf}


if __name__ == "__main__":

    results = []

    convergence_elements = [
        25,
        26,
        27,
        28,
        29,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        100,
        200,
        420,
    ]
    for i in convergence_elements:
        results.append(cantilever_subjected_to_a_transversal_load(i))

    l1 = []
    l2 = []
    linf = []

    for result in results:
        l1.append(result["l1"])
        l2.append(result["l2"])
        linf.append(result["linf"])

    fig = plt.figure(figsize=(10, 8), frameon=True, dpi=150)
    ax = fig.add_subplot(111)
    ax.grid(which="minor", color="k", linestyle="--")
    ax.grid(which="major", color="k", linestyle="-")
    ax.set_xlabel("N_element")  # X-axis label
    ax.set_ylabel("Error")  # Y-axis label
    ax.set_title("Error Convergence Analysis")

    ax.loglog(
        convergence_elements,
        l1,
        marker="o",
        ms=10,
        c=to_rgb("xkcd:bluish"),
        lw=2,
        label="l1",
    )
    ax.loglog(
        convergence_elements,
        l2,
        marker="o",
        ms=10,
        c=to_rgb("xkcd:reddish"),
        lw=2,
        label="l2",
    )
    ax.loglog(convergence_elements, linf, marker="o", ms=10, c="k", lw=2, label="linf")
    fig.legend(prop={"size": 20})

    fig.show()
