import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.colors import to_rgb
from scipy.linalg import norm


def calculate_error_norm(true_solution, computed_solution, n_elem):
    assert (
        true_solution.shape == computed_solution.shape
    ), "Shape of computed and true solution does not match"
    error = true_solution - computed_solution
    l1 = norm(error, 1) / n_elem
    l2 = norm(error, 2) / n_elem
    linf = norm(error, np.inf)

    return error, l1, l2, linf


def plot_convergence(results, SAVE_FIGURE, filename):
    convergence_elements = []
    l1 = []
    l2 = []
    linf = []

    for result in results:
        convergence_elements.append(result["rod"].n_elems)
        l1.append(result["l1"])
        l2.append(result["l2"])
        linf.append(result["linf"])

    fig = plt.figure(figsize=(10, 8), frameon=True, dpi=150)
    ax = fig.add_subplot(111)
    ax.grid(b=True, which="minor", color="k", linestyle="--")
    ax.grid(b=True, which="major", color="k", linestyle="-")
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
    if SAVE_FIGURE:
        assert filename != "", "provide a file name for figure"
        fig.savefig(filename)
    fig.show()
