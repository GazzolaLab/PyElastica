import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb


def analytical_shearable(arg_rod, arg_end_force, n_elem=500):
    base_length = np.sum(arg_rod.rest_lengths)
    arg_s = np.linspace(0.0, base_length, n_elem)
    if type(arg_end_force) is np.ndarray:
        acting_force = arg_end_force[np.nonzero(arg_end_force)]
    else:
        acting_force = arg_end_force
    acting_force = np.abs(acting_force)

    linear_prefactor = -acting_force / arg_rod.shear_matrix[0, 0, 0]
    quadratic_prefactor = (
        -acting_force
        * np.sum(arg_rod.rest_lengths)
        / 2.0
        / arg_rod.bend_matrix[0, 0, 0]
    )
    cubic_prefactor = acting_force / 6.0 / arg_rod.bend_matrix[0, 0, 0]
    return (
        arg_s,
        arg_s
        * (linear_prefactor + arg_s * (quadratic_prefactor + arg_s * cubic_prefactor)),
    )


def analytical_unshearable(arg_rod, arg_end_force, n_elem=500):
    base_length = np.sum(arg_rod.rest_lengths)
    arg_s = np.linspace(0.0, base_length, n_elem)
    if type(arg_end_force) is np.ndarray:
        acting_force = arg_end_force[np.nonzero(arg_end_force)]
    else:
        acting_force = arg_end_force
    acting_force = np.abs(acting_force)

    quadratic_prefactor = (
        -acting_force
        * np.sum(arg_rod.rest_lengths)
        / 2.0
        / arg_rod.bend_matrix[0, 0, 0]
    )
    cubic_prefactor = acting_force / 6.0 / arg_rod.bend_matrix[0, 0, 0]
    return arg_s, arg_s ** 2 * (quadratic_prefactor + arg_s * cubic_prefactor)


def plot_timoshenko(rod, end_force, SAVE_FIGURE, ADD_UNSHEARABLE_ROD=False):
    fig = plt.figure(figsize=(10, 8), frameon=True, dpi=150)
    ax = fig.add_subplot(111)
    ax.grid(which="minor", color="k", linestyle="--")
    ax.grid(which="major", color="k", linestyle="-")
    analytical_shearable_positon = analytical_shearable(rod, end_force)
    ax.plot(
        analytical_shearable_positon[0],
        analytical_shearable_positon[1],
        "k--",
        label="Timoshenko",
    )
    ax.plot(
        rod.position_collection[2, ...],
        rod.position_collection[0, ...],
        c=to_rgb("xkcd:bluish"),
        label="n=" + str(rod.n_elems),
    )
    if ADD_UNSHEARABLE_ROD:
        analytical_unshearable_positon = analytical_unshearable(rod, end_force)
        ax.plot(
            analytical_unshearable_positon[0],
            analytical_unshearable_positon[1],
            "r-.",
            label="Euler-Bernoulli",
        )
    fig.legend(prop={"size": 20})
    plt.show()
    if SAVE_FIGURE:
        fig.savefig("Timoshenko_beam_test" + str(rod.n_elems) + ".png")
