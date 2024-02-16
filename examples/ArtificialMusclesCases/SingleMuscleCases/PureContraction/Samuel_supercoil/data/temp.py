import numpy as np
import os

DATA_PATH = "/Users/ali-7800/Desktop/Research/dev_artificial_muscle/PyElastica/examples/ArtificialMusclesCases/SingleMuscleCases/PureContraction/Samuel_supercoil/data/PureContractionSamuel_supercoil.dat"
SAVE_PICKLE = True
assert os.path.exists(DATA_PATH), "File does not exists"
try:
    if SAVE_PICKLE:
        import pickle as pk

        with open(DATA_PATH, "rb") as fptr:
            data = pk.load(fptr)
    else:
        # (TODO) add importing npz file format
        raise NotImplementedError("Only pickled data is supported")
except OSError as err:
    print("Cannot open the datafile {}".format(DATA_PATH))
    print(str(err))
    raise


time = np.array(data[0]["time"])
n_elem = np.array((data[0]["position"])).shape[2] - 1

n_muscle_rod = 3
muscle_rods_position_history = np.zeros((n_muscle_rod, time.shape[0], 3, n_elem + 1))
muscle_rods_radius_history = np.zeros((n_muscle_rod, time.shape[0], n_elem))
for i in range(3):
    muscle_rods_position_history[i, :, :, :] = np.array(data[i]["position"])
    muscle_rods_radius_history[i, :, :] = np.array(data[i]["radius"])
np.savez(
    os.path.join("PureContractionSamuel_supercoil.npz"),
    time=time,
    muscle_rods_position_history=muscle_rods_position_history,
    muscle_rods_radius_history=muscle_rods_radius_history,
)
