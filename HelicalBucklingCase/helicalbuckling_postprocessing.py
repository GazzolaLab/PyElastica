import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.colors import to_rgb
from scipy.linalg import norm


def envelope(arg_pos):
    """
    Given points, computes the arc length and envelope of the curve
    """
    n_points = arg_pos.shape[1]

    # Computes the direction in which the rod points
    # in our cases it should be the z-axis
    rod_direction = arg_pos[:, -1] - arg_pos[:, 0]
    rod_direction /= norm(rod_direction, ord=2, axis=0)

    # Compute local tangent directions
    tangent_s = np.diff(arg_pos, n=1, axis=-1)  # x_(i+1)-x(i)
    length_s = norm(tangent_s, ord=2, axis=0)
    tangent_s /= length_s

    # Dot product with direction is cos_phi, see RSOS
    cos_phi_s = np.einsum("ij,i->j", tangent_s, rod_direction)

    # Compute phi-max now
    phi = np.arccos(cos_phi_s)
    cos_phi_max = np.cos(np.max(phi))

    # Return envelope and arclength
    envelope = (cos_phi_s - cos_phi_max) / (1.0 - cos_phi_max)
    # -0.5 * length accounts for the element/node business
    arclength = np.cumsum(length_s) - 0.5 * length_s[0]

    return arclength, envelope


def analytical_solution(L, n_elem=10000):
    """ Gives the analytical solution of the helicalbuckling case
    """
    # Physical parameters, set from the simulation
    B = 1.345
    C = 0.789
    gamma = C / B
    R = 27.0 * 2.0 * np.pi
    d = 0.03
    D = d * L
    nu = 1.0 / gamma - 1.0

    # These are magic constants, but you can obtain them by solving
    # this equation (accoring to matlab syntax)
    # syms x y
    # S = vpasolve([d == sqrt(16/y*(1-x*x/(4*y))), R == x/gamma+4*acos(x/(2*sqrt(y)))], [x, y]);
    # moment = double(S.x); # dimensionless end moment
    # tension = double(S.y); # dimensionless end torque
    # This comes from  Eqs. 14-15 of "Writhing instabilities of twisted rods: from
    # infinite to finite length", 2001
    # We did not want to introduce sympy dependency here, so we decided to hardcode
    # the solutions instead
    moment = 98.541496171190744
    tension = 2.900993205792131e3

    # Compute maximum envelope angle according to Eq. 13 of "Writhing
    # instabilities of twisted rods: from infinite to finite length", 2001
    thetaMax = np.arccos(moment * moment / (2.0 * tension) - 1.0)

    # Compute actual end torque and tension according to "Writhing
    # instabilities of twisted rods: from infinite to finite length", 2001
    M = moment * B / L
    T = tension * B / (L * L)

    # Compute dimensionless load according to Eq. 30 of "Helical and localised
    # buckling in twisted rods: a unified analysis of the symmetric case", 2000
    m = M / np.sqrt(B * T)

    # Setup for analytical curve calculation
    s = np.linspace(-0.5, 0.5, n_elem)
    t = T * L * L / (4 * np.pi * np.pi * B)
    mz = M * L / (2 * np.pi * B)
    root = np.sqrt(4 * t - mz * mz)

    # This is the analytical curve computed
    # according to Eqs. 27 and 52 of
    # "Instability and self-contact phenomena in the writhing of clamped rods",
    # 2003
    xs = (
        1.0
        / (2.0 * np.pi * t)
        * root
        * np.sin(mz * np.pi * s)
        / np.cosh(np.pi * s * root)
    )
    ys = (
        -1.0
        / (2.0 * np.pi * t)
        * root
        * np.cos(mz * np.pi * s)
        / np.cosh(np.pi * s * root)
    )
    zs = s - 1.0 / (2.0 * np.pi * t) * root * np.tanh(np.pi * s * root)
    pos = np.vstack((xs, ys, zs)) * L
    return envelope(pos)


def plot_helicalbuckling(rod, SAVE_FIGURE):

    plt.figure()
    plt.axes(projection="3d")
    plt.plot(
        rod.position_collection[0, ...],
        rod.position_collection[1, ...],
        rod.position_collection[2, ...],
    )
    plt.xlim([-10, 10])
    plt.ylim([-10, 10])
    if SAVE_FIGURE:
        plt.savefig("HelicalBuckling_3d.png")
    plt.show()

    base_length = np.sum(rod.rest_lengths)
    phi_analytical_envelope = analytical_solution(base_length)
    phi_computed_envelope = envelope(rod.position_collection)

    plt.figure()
    plt.plot(phi_analytical_envelope[0], phi_analytical_envelope[1], label="analytical")
    plt.plot(
        phi_computed_envelope[0],
        phi_computed_envelope[1],
        label="n=" + str(rod.n_elems),
    )
    plt.legend()
    if SAVE_FIGURE:
        plt.savefig("HelicalBuckling_Envelope.png")
    plt.show()


def calculate_error_norm(rod):
    base_length = np.sum(rod.rest_lengths)
    n_elems = rod.n_elems
    # Since we need to evaluate analytical solution only on nodes, n_nodes = n_elems+1
    phi_analytical_envelope = analytical_solution(base_length, n_elems + 1)
    phi_computed_envelope = envelope(rod.position_collection)

    # calculate errors
    # use the second index of envelope, first is position on the rod
    # and second index is envelope
    error = phi_computed_envelope[1] - phi_analytical_envelope[1]
    l1 = norm(error, 1) / n_elems
    l2 = norm(error, 2) / n_elems
    linf = norm(error, np.inf) / n_elems

    return error, l1, l2, linf


def plot_convergence_helicalbuckling(results, SAVE_FIGURE):
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
        fig.savefig("HelicalBuckling_convergence_test.png")
    fig.show()
