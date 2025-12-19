import numpy as np
from matplotlib import pyplot as plt
import os
import pickle

data_file_name = "flexible_swinging_pendulum.dat"
if os.path.exists(data_file_name):
    with open(data_file_name, "rb") as file_handle:
        recorded_history = pickle.load(file_handle)

# Generate data in separate figures
FORCE_SELECTION = range(0, 10, 3)

# Extract time and positions
time = np.array(recorded_history["time"])
internal_stress = np.array(recorded_history["internal_stress"])

# Plot internal stress in x-direction
fig = plt.figure(5, figsize=(8, 5))
ax = fig.add_subplot(111)
for elem in FORCE_SELECTION:
    ax.plot(time[1:], internal_stress[:, 0, elem])

# Plot internal stress in z-direction
fig = plt.figure(6, figsize=(8, 5))
ax = fig.add_subplot(111)
for elem in FORCE_SELECTION:
    ax.plot(time[1:], internal_stress[:, 2, elem])

plt.show()
