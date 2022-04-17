import sys
from scipy.optimize import fsolve, minimize_scalar, minimize
from scipy.integrate import odeint
import multiprocessing as mp
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb
# FIXME without appending sys.path make it more generic
sys.path.append("../")
sys.path.append("../../")
from elastica import *
from MagneticElastica.magnetic_forces import MagneticTorquesForUniformMagneticField


class MagneticBeamAnalytical:
    """

    """
    def __init__(self, F, EI, phi, L, ksi=np.linspace(0,1,100), theta_dot_initial_guess=0.0):
        self.F = F
        self.EI = EI
        self.phi = phi

        self.ksi = ksi
        self.q = F * L**2 / (EI)
        self.theta_dot_initial_guess = theta_dot_initial_guess

    def compute_governing_equations(self, x,  ksi, *args, **kwargs):

        x1, x2 = x

        x1_dot = x2

        x2_dot = -self.q * np.sin(self.phi - x1 )

        return [x1_dot, x2_dot]

    def solve_governing_equations(self, x2_0):
        x0 = [0, x2_0]  # x2(0) not known it will be computed using shooting method.

        sol = odeint(self.compute_governing_equations, x0, self.ksi)

        return sol

    def shooting_method(self, x2_0):
        sol = self.solve_governing_equations(x2_0)

        # return sol[-1,1]-0
        return (sol[-1,1]-0)**2


    def find_boundary_conditions(self):
        """
        This is for doing shooting method, guess the initial theta_dot at ksi=0 and integrate the beam equations, and
        check the if bc at the free end theta_dot satisfied which is 0.
        Returns
        -------

        """

        # x2_0 = fsolve(self.shooting_method, x0=self.theta_dot_initial_guess, xtol=1E-4)
        x2_0 = minimize(self.shooting_method, x0=self.theta_dot_initial_guess).x
        return x2_0

    def __call__(self, *args, **kwargs):

        # Find new boundary conditions
        x2_0 = self.find_boundary_conditions()

        sol = self.solve_governing_equations(x2_0)

        theta = sol[:,0]

        current_deflection = self._compute_deflection(theta)


        return theta[-1], current_deflection, x2_0


    def _compute_deflection(self, theta):

        return np.trapz(np.sin(theta), x=self.ksi)




class MagneticBeamSimulator(BaseSystemCollection, Constraints, Forcing):
    pass


def run_magnetic_beam_sim(magnetization_density, magnetic_field_angle, magnetic_field):
    magnetic_beam_sim = MagneticBeamSimulator()
    # setting up test params
    n_elem = 100
    start = np.zeros((3,))
    direction = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 1.0, 0.0])
    base_length = 6.0
    base_radius = 0.15
    base_area = np.pi * base_radius ** 2
    density = 5000
    nu = 10
    E = 1E6
    poisson_ratio  = 0.5
    shear_modulus = E / (2*poisson_ratio + 1.0)

    shearable_rod = CosseratRod.straight_rod(
        n_elem,
        start,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        nu,
        E,
        shear_modulus=shear_modulus,
    )
    magnetic_beam_sim.append(shearable_rod)

    # Add boundary conditions, one end of rod is clamped
    magnetic_beam_sim.constrain(shearable_rod).using(
        OneEndFixedBC, constrained_position_idx=(0,), constrained_director_idx=(0,)
    )

    # Apply magnetic forces
    magnetic_field_vector =  magnetic_field * np.array([np.cos(magnetic_field_angle), np.sin(magnetic_field_angle), 0])
    magnetic_beam_sim.add_forcing_to(shearable_rod).using(
        MagneticTorquesForUniformMagneticField,
        ramp_interval = 500.0,
        start_time = 0.0,
        end_time = 100000,
        magnetization_density=magnetization_density*np.ones((n_elem)),
        magnetic_field_vector=magnetic_field_vector
    )

    magnetic_beam_sim.finalize()
    timestepper = PositionVerlet()
    final_time = 1000
    dl = base_length / n_elem
    dt = 0.01 * dl
    total_steps = int(final_time / dt)
    integrate(timestepper, magnetic_beam_sim, final_time, total_steps)

    # Compute MBAL2/EI
    moment_of_inertia = np.pi/4 * base_radius**4
    MBAL2_EI = magnetization_density * magnetic_field * base_area * base_length**2 / (E * moment_of_inertia)

    # Get the final tip deflection
    deflection = shearable_rod.position_collection[...,-1][1]/base_length

    # Get the tip angle
    tip_angle = np.arccos(np.dot(shearable_rod.tangents[...,-1], np.array([1., 0., 0.])))

    return MBAL2_EI, deflection, tip_angle


def compute_analytical_solution(magnetization_density, magnetic_field_angle, magnetic_field):
    base_length = 6
    base_radius = 0.15
    base_area = np.pi * base_radius**2
    I = np.pi/4 * base_radius**4
    E = 1E6

    f = magnetization_density * base_area * magnetic_field

    deflection = np.zeros((f.shape[0]))
    theta = np.zeros((f.shape[0]))
    theta_dot = np.zeros((f.shape[0]))

    for i in range(f.shape[0]):
        # For computing theta dot at x=0 we are using shooting method. Shooting method has to satify the boundary
        # condition at x=L theta_dot = 0. However, optimization methods used can converge to different optima, so
        # we start optimization using the theta_dot computed in previous iteration.
        if i == 0:
            theta_dot_initial_guess = theta_dot[i]
        else:
            theta_dot_initial_guess = theta_dot[i-1]

        magneto = MagneticBeamAnalytical(F=f[i], EI=E*I, phi=magnetic_field_angle, L=base_length, ksi=np.linspace(0,1,100), theta_dot_initial_guess=theta_dot_initial_guess)
        theta[i], deflection[i], theta_dot[i] = magneto()

    return f*base_length**2/(E*I), deflection, theta


if __name__ == "__main__":

    magnetization_density = 144E3
    # magnetic_field_angle = np.deg2rad(180-0.5)#np.pi/3
    magnetic_field = np.linspace(0, 42, 10) * 1E-3
    magnetic_field_analytical = np.linspace(0, 42, 400) * 1E-3

    # MBAL2_EI_analytical, deflection_analytical, theta_analytical = compute_analytical_solution(magnetization_density,magnetic_field_angle , magnetic_field_analytical)
    #
    # MBAL2_EI, deflection, theta = run_magnetic_beam_sim(magnetization_density, magnetic_field_angle, magnetic_field[5])

    magnetic_field_angle = np.array([30, 60, 90, 120, 150, 180-0.5])
    # Run elastica simulations as a batch job
    simulation_list = []
    for i in range(magnetic_field_angle.shape[0]):
        for j in range(magnetic_field.shape[0]):
            simulation_list.append((magnetization_density, magnetic_field_angle[i], magnetic_field[j]))


    with mp.Pool(mp.cpu_count()) as pool:
        result = pool.starmap(run_magnetic_beam_sim, simulation_list)

    MBAL2_EI = np.zeros((magnetic_field_angle.shape[0], magnetic_field.shape[0]))
    deflection = np.zeros((magnetic_field_angle.shape[0], magnetic_field.shape[0]))
    theta  = np.zeros((magnetic_field_angle.shape[0], magnetic_field.shape[0]))

    counter = 0
    for i in range(magnetic_field_angle.shape[0]):
        for j in range(magnetic_field.shape[0]):
            simulation_result = result[counter]
            MBAL2_EI[i,j] = simulation_result[0]
            deflection[i,j] = simulation_result[1]
            theta[i,j] = simulation_result[2]
            counter += 1


    # Run analytical solutions
    analytical_list = []
    for i in range(magnetic_field_angle.shape[0]):
        for j in range(magnetic_field_analytical.shape[0]):
            analytical_list.append((magnetization_density, magnetic_field_angle[i], magnetic_field_analytical[j]))

    with mp.Pool(mp.cpu_count()) as pool:
        result_analytical = pool.starmap(compute_analytical_solution, analytical_list)

    MBAL2_EI_analytical = np.zeros((magnetic_field_angle.shape[0], magnetic_field_analytical.shape[0]))
    deflection_analytical = np.zeros((magnetic_field_angle.shape[0], magnetic_field_analytical.shape[0]))
    theta_analytical = np.zeros((magnetic_field_angle.shape[0], magnetic_field_analytical.shape[0]))

    counter = 0
    for i in range(magnetic_field_angle.shape[0]):
        for j in range(magnetic_field_analytical.shape[0]):
            analytical_result = result_analytical[counter]
            MBAL2_EI_analytical[i,j] = analytical_result[0]
            deflection_analytical[i,j] = analytical_result[1]
            theta_analytical[i,j] = analytical_result[2]
            counter += 1




    plt.rcParams.update({"font.size": 22})
    fig = plt.figure(figsize=(10, 10), frameon=True, dpi=150)

    axs = []
    axs.append(plt.subplot2grid((1, 1), (0, 0)))
    for i in range(magnetic_field_angle.shape[0]):
        axs[0].plot(
            MBAL2_EI_analytical[i,:],
            deflection_analytical[i,:],
            label="phi=" + str(np.ceil(np.rad2deg(magnetic_field_angle[i]))),
            )
        axs[0].plot(
            MBAL2_EI[i,:],
            deflection[i,:],
            '*',
            markersize=16,
            color='green',
            # label="phi=" + str(np.ceil(np.rad2deg(magnetic_field_angle))),
        )
    axs[0].set_xlabel("MBAL2/EI", fontsize=20)
    axs[0].set_ylabel("delta_y/L", fontsize=20)
    plt.tight_layout()
    fig.align_ylabels()
    fig.legend(prop={"size": 20})
    fig.savefig("magnetic_beam_analytical_deflection.png")
    plt.close(plt.gcf())

    plt.rcParams.update({"font.size": 22})
    fig = plt.figure(figsize=(10, 10), frameon=True, dpi=150)

    axs = []
    axs.append(plt.subplot2grid((1, 1), (0, 0)))
    for i in range(magnetic_field_angle.shape[0]):
        axs[0].plot(
            MBAL2_EI_analytical[i,:],
            theta_analytical[i,:],
            label="phi=" + str(np.ceil(np.rad2deg(magnetic_field_angle[i]))),
        )
        axs[0].plot(
            MBAL2_EI[i,:],
            theta[i,:],
            '*',
            markersize=16,
            color='green',
            # label="phi=" + str(np.ceil(np.rad2deg(magnetic_field_angle))),
        )
    axs[0].set_xlabel("MBAL2/EI", fontsize=20)
    axs[0].set_ylabel("theta (L)", fontsize=20)
    plt.tight_layout()
    fig.align_ylabels()
    fig.legend(prop={"size": 20})
    fig.savefig("magnetic_beam_analytical_theta.png")
    plt.close(plt.gcf())
