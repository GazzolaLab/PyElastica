import numpy as np
from matplotlib import pyplot as plt

from elastica.wrappers import BaseSystemCollection, Connections, Constraints, Forcing
from elastica.rod.cosserat_rod import CosseratRod
from elastica.boundary_conditions import OneEndFixedRod, FreeRod
from elastica.external_forces import EndpointForces
from elastica.timestepper.symplectic_steppers import PositionVerlet, PEFRL
from elastica.timestepper import integrate


class TimoshenkoBeamSimulator(BaseSystemCollection, Constraints, Forcing):
    pass


timoshenko_sim = TimoshenkoBeamSimulator()

# setting up test params
n_elem = 100
start = np.zeros((3,))
direction = np.array([0., 0., 1.])
normal = np.array([0., 1., 0.])
base_length = 3.0
base_radius = 0.25
base_area = np.pi * base_radius ** 2
density = 5000
nu = 0.1#2.9
E = 1e6
# For shear modulus of 1e4, nu is 99!
poisson_ratio = 99

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
    poisson_ratio,
)

final_time = 5000

timoshenko_sim.append(shearable_rod)
timoshenko_sim.constrain(shearable_rod).using(OneEndFixedRod, positions=(0,), directors=(0,))
# end_force = np.array([-15.0, 0.0, 0.0])
end_force = np.array([-15.0, 0.0, 0.0])
timoshenko_sim.add_forcing_to(shearable_rod).using(EndpointForces, 0.0 * end_force, end_force, rampupTime=5000/2)

ADD_UNSHEARABLE_ROD = False

if ADD_UNSHEARABLE_ROD:
    # Start into the plane
    unshearable_start = np.array([0.0, -1.0, 0.0])
    unshearable_rod = CosseratRod.straight_rod(
        n_elem,
        unshearable_start,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        nu,
        E,
        # Unshearable rod needs G -> inf, which is achievable with -ve poisson ratio
        poisson_ratio=-0.7,
    )

    timoshenko_sim.append(unshearable_rod)
    timoshenko_sim.constrain(unshearable_rod).using(OneEndFixedRod, positions=(0,), directors=(0,))
    timoshenko_sim.add_forcing_to(unshearable_rod).using(EndpointForces, 0.0 * end_force, end_force, rampupTime=final_time/2.0)

timoshenko_sim.finalize()
timestepper = PositionVerlet()
# timestepper = PEFRL()

positions_over_time = []
directors_over_time = []
velocities_over_time = []
# final_time = 2000.0
dl = base_length/n_elem
dt = 0.01*dl
# dt = 3.0/40 * 0.01
total_steps = int(final_time/dt)
print("Total steps", total_steps)
positions_over_time, directors_over_time, velocities_over_time = integrate(timestepper, timoshenko_sim, final_time, total_steps)
# positions_over_time, directors_over_time, velocities_over_time = integrate(timestepper, timoshenko_sim, 500.0, int(8e5))
# positions_over_time, directors_over_time, velocities_over_time = integrate(timestepper, timoshenko_sim, 5.0, int(5e3))

def analytical_shearable(arg_s, arg_end_force, arg_rod):
    if type(arg_end_force) is np.ndarray:
        acting_force = arg_end_force[np.nonzero(arg_end_force)]
    else:
        acting_force = arg_end_force
    acting_force = np.abs(acting_force)

    linear_prefactor = -acting_force / arg_rod.shear_matrix[0, 0, 0]
    quadratic_prefactor = -acting_force * np.sum(arg_rod.rest_lengths) / 2.0 / arg_rod.bend_matrix[0, 0, 0]
    cubic_prefactor = acting_force / 6.0 / arg_rod.bend_matrix[0, 0, 0]
    return arg_s * (linear_prefactor + arg_s * (quadratic_prefactor + arg_s * cubic_prefactor))


def analytical_unshearable(arg_s, arg_end_force, arg_rod):
    if type(arg_end_force) is np.ndarray:
        acting_force = arg_end_force[np.nonzero(arg_end_force)]
    else:
        acting_force = arg_end_force
    acting_force = np.abs(acting_force)

    quadratic_prefactor = -acting_force * np.sum(arg_rod.rest_lengths) / 2.0 / arg_rod.bend_matrix[0, 0, 0]
    cubic_prefactor = acting_force / 6.0 / arg_rod.bend_matrix[0, 0, 0]
    return arg_s ** 2 * (quadratic_prefactor + arg_s * cubic_prefactor)


fig = plt.figure(figsize=(10, 8), frameon=True, dpi=150)
fig.clear()
ax = fig.add_subplot(111)
n_pos = len(positions_over_time)
# for i_pos, position in enumerate(positions_over_time):
#     ax.plot(position[2, ...], position[0, ...], c='b', alpha=10**((i_pos/n_pos)-1))
centerline = np.linspace(0.0, base_length, 500)
ax.plot(centerline, analytical_shearable(centerline, end_force, shearable_rod), 'k--', lw=2)
ax.plot(shearable_rod.position_collection[2, ...], shearable_rod.position_collection[0, ...], c='b', lw=2)
if ADD_UNSHEARABLE_ROD:
    ax.plot(centerline, analytical_unshearable(centerline, end_force, unshearable_rod), 'k--', lw=2)
    ax.plot(unshearable_rod.position_collection[2, ...], unshearable_rod.position_collection[0, ...], c='r', lw=2)
# ax.set_aspect('equal')

# vel_fig = plt.figure(figsize=(10, 8), frameon=True, dpi=150)
# vel_fig.clear()
# vel_ax = vel_fig.add_subplot(111)
# velocity_norms = [np.linalg.norm(v, 2) for v in velocities_over_time]
# vel_ax.plot(velocity_norms)
#
plt.show()

com = []
fig = plt.figure(figsize=(10, 8), frameon=True, dpi=150)
bx = fig.add_subplot(111)
for position in positions_over_time[-10:]:
    bx.plot(position[2, ...], position[0, ...])
    com.append(np.sum(position * shearable_rod.mass, axis=1) / np.sum(shearable_rod.mass))
bx.plot(centerline, analytical_shearable(centerline, end_force, shearable_rod), 'k--', lw=2)
plt.show()