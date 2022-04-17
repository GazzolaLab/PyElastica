import numpy as np
import multiprocessing as mp
import sys
sys.path.append("../../")
from scipy.optimize import fsolve, minimize_scalar, minimize
from scipy.integrate import odeint


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



phi = np.array([30, 60, 90, 120, 150, 180-0.5])
phi = np.deg2rad(phi)

base_length = 6
base_radius = 0.15
base_area = np.pi * base_radius**2
I = np.pi/4 * base_radius**4
E = 1E6
M = 144E3
B = np.linspace(0, 42, 400) * 1E-3
f = M * B * base_area


deflection =  np.zeros((phi.shape[0], f.shape[0]))
theta =  np.zeros((phi.shape[0], f.shape[0]))
theta_dot = np.zeros((phi.shape[0], f.shape[0]))

for i in range(phi.shape[0]):
    for j in range(f.shape[0]):
        # For computing theta dot at x=0 we are using shooting method. Shooting method has to satify the boundary
        # condition at x=L theta_dot = 0. However, optimization methods used can converge to different optima, so
        # we start optimization using the theta_dot computed in previous iteration.
        if j == 0:
            theta_dot_initial_guess = theta_dot[i,j]
        else:
            theta_dot_initial_guess = theta_dot[i,j-1]


        magneto = MagneticBeamAnalytical(F=f[j], EI=E*I, phi=phi[i], L=base_length, ksi=np.linspace(0,1,100), theta_dot_initial_guess=theta_dot_initial_guess)
        temp_theta, temp_deflection, temp_theta_dot = magneto()
        theta[i,j] = temp_theta
        deflection[i,j]  = temp_deflection
        theta_dot[i,j]=temp_theta_dot


theta = np.rad2deg(theta)


import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb

plt.rcParams.update({"font.size": 22})
fig = plt.figure(figsize=(10, 10), frameon=True, dpi=150)

axs = []
axs.append(plt.subplot2grid((1, 1), (0, 0)))
for i in range(phi.shape[0]):
    axs[0].plot(
    f * base_length**2 / (E*I),
    deflection[i, :],
    label="phi=" + str(np.ceil(np.rad2deg(phi[i]))),
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
for i in range(phi.shape[0]):
    axs[0].plot(
        f * base_length**2 / (E*I),
        theta[i, :],
        label="phi=" + str(np.ceil(np.rad2deg(phi[i]))),
        )
axs[0].set_xlabel("MBAL2/EI", fontsize=20)
axs[0].set_ylabel("theta(L)", fontsize=20)
plt.tight_layout()
fig.align_ylabels()
fig.legend(prop={"size": 20})
fig.savefig("magnetic_beam_analytical_tip_angle.png")
plt.close(plt.gcf())
