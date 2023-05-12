from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb


def plot_friction_validation(results, SAVE_FIGURE, filename):

    sweep = []
    translational_energy = []
    rotational_energy = []
    analytical_translational_energy = []
    analytical_rotational_energy = []

    for result in results:
        sweep.append(result["sweep"])
        translational_energy.append(result["translational_energy"])
        rotational_energy.append(result["rotational_energy"])
        analytical_translational_energy.append(
            result["analytical_translational_energy"]
        )
        analytical_rotational_energy.append(result["analytical_rotational_energy"])

    fig = plt.figure(figsize=(10, 8), frameon=True, dpi=150)
    ax = fig.add_subplot(111)
    ax.plot(
        sweep,
        translational_energy,
        c=to_rgb("xkcd:bluish"),
        lw=4,
        label="Translational",
    )
    ax.plot(
        sweep, rotational_energy, c=to_rgb("xkcd:reddish"), lw=4, label="Rotational"
    )
    ax.plot(
        sweep,
        analytical_translational_energy,
        "-.k",
        lw=2,
        label="Translational analytical",
    )
    ax.plot(
        sweep, analytical_rotational_energy, "--k", lw=2, label="Rotational analytical"
    )

    # fig.legend(prop={"size": 20})
    ax.legend(loc="upper left", fontsize="xx-large", prop={"size": 20})
    if SAVE_FIGURE:
        assert filename != "", "provide a file name for figure"
        fig.savefig(filename)
    fig.show()


def plot_axial_friction_validation(results, SAVE_FIGURE, filename):

    sweep = []
    translational_energy = []
    rotational_energy = []
    analytical_translational_energy = []
    analytical_rotational_energy = []

    for result in results:
        sweep.append(result["sweep"])
        translational_energy.append(result["translational_energy"])
        rotational_energy.append(result["rotational_energy"])
        analytical_translational_energy.append(
            result["analytical_translational_energy"]
        )
        analytical_rotational_energy.append(result["analytical_rotational_energy"])

    fig = plt.figure(figsize=(10, 8), frameon=True, dpi=150)
    ax = fig.add_subplot(111)
    ax.grid(which="major", color="k", linestyle="-")
    ax.plot(
        sweep,
        translational_energy,
        c=to_rgb("xkcd:bluish"),
        lw=4,
        label="Translational",
    )
    ax.plot(
        sweep,
        analytical_translational_energy,
        "-.k",
        lw=2,
        label="Translational analytical",
    )

    # fig.legend(prop={"size": 20})
    ax.legend(loc="upper center", fontsize="xx-large", prop={"size": 20})
    if SAVE_FIGURE:
        assert filename != "", "provide a file name for figure"
        fig.savefig(filename)
    fig.show()
