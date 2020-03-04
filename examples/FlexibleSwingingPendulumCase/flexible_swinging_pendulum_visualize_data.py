import numpy as np
from matplotlib import pyplot as plt
import os

# def main():
data_file_name = "flexible_swinging_pendulum.dat"
if os.path.exists(data_file_name):
    import pickle

    with open(data_file_name, "rb") as file_handle:
        recorded_history = pickle.load(file_handle)

# Generate data in six separate figures and not in one subplot
NODAL_SELECTION = np.arange(0, 10 + 2, 2)
ELEMENT_SELECTION = list(range(0, 8 + 2, 2)) + [9]
VORONOI_SELECTION = range(0, 9)
FORCE_SELECTION = range(0, 10, 3)

# 1. Centroid positions in vertical plane
time = np.array(recorded_history["time"])
positions = np.array(recorded_history["position"])

if False:
    fig = plt.figure(1, figsize=(8, 5))
    ax = fig.add_subplot(111)
    for node in NODAL_SELECTION:
        ax.plot(time, positions[:, 2, node])

    fig = plt.figure(2, figsize=(8, 5))
    ax = fig.add_subplot(111)
    for node in NODAL_SELECTION:
        ax.plot(time, positions[:, 0, node])

    fig = plt.figure(3, figsize=(8, 5))
    ax = fig.add_subplot(111)
    # (time, 3, 3, n_elem) array
    directors = np.array(recorded_history["directors"])
    # Plot d1 . e1
    projected_director = np.einsum(
        "ijk,j->ik", directors[:, 0, :, :], np.array([1.0, 0.0, 0.0])
    )
    for elem in ELEMENT_SELECTION:
        ax.plot(time, projected_director[:, elem])

    fig = plt.figure(4, figsize=(8, 5))
    ax = fig.add_subplot(111)
    # (n_time, 3, n_elem)
    internal_couple = np.array(recorded_history["internal_couple"])
    for voronoi in VORONOI_SELECTION:
        ax.plot(time[1:], internal_couple[:, 1, voronoi])


fig = plt.figure(5, figsize=(8, 5))
ax = fig.add_subplot(111)
internal_stress = np.array(recorded_history["internal_stress"])
for elem in FORCE_SELECTION:
    ax.plot(time[1:], internal_stress[:, 0, elem])


fig = plt.figure(6, figsize=(8, 5))
ax = fig.add_subplot(111)
for elem in FORCE_SELECTION:
    ax.plot(time[1:], internal_stress[:, 2, elem])

plt.show()
# if __name__ == "__main__":
# main()
