"""
Example script to benchmark C++ operations with timing done entirely in C++.

This script demonstrates how to use the benchmark_cpp function to measure
performance without Python loop or pybind11 call overhead.
"""

import numpy as np
import matplotlib.pyplot as plt
import elastica as epy
import elasticapp as epp

# %%
# Create test rods
n_rods = 50
n_elems_per_rod = 200

print(f"Creating {n_rods} rods with {n_elems_per_rod} elements each...")
keys = list(epp.memory_block_rod.PY2CPP_VARNAMES.keys())
rods_cpp = [
    epy.CosseratRod.straight_rod(
        n_elements=n_elems_per_rod,
        start=np.zeros(3),
        direction=np.array([0.0, 0.0, 1.0]),
        normal=np.array([1.0, 0.0, 0.0]),
        base_length=1.0,
        base_radius=0.01,
        density=3000,
        youngs_modulus=1e6,
    )
    for _ in range(n_rods)
]
rods_py = [
    epy.CosseratRod.straight_rod(
        n_elements=n_elems_per_rod,
        start=np.zeros(3),
        direction=np.array([0.0, 0.0, 1.0]),
        normal=np.array([1.0, 0.0, 0.0]),
        base_length=1.0,
        base_radius=0.01,
        density=3000,
        youngs_modulus=1e6,
    )
    for _ in range(n_rods)
]
rng = np.random.default_rng(43)
for i in range(n_rods):
    for key in ["rest_kappa", "rest_sigma"]:
        shape = getattr(rods_py[i], key).shape
        values = rng.random(size=shape)
        getattr(rods_py[i], key)[...] = values.copy()
        getattr(rods_cpp[i], key)[...] = values.copy()

# Create block rod system
block_cpp = epp.MemoryBlockCosseratRod(rods_cpp, range(n_rods))
block_py = epy.MemoryBlockCosseratRod(rods_py, range(n_rods))

# %%
# Cross-check ghost indices
# -------------------------
np.testing.assert_array_equal(block_cpp.ghost_nodes_idx, block_py.ghost_nodes_idx)
np.testing.assert_array_equal(block_cpp.ghost_elems_idx[:-1], block_py.ghost_elems_idx)
np.testing.assert_array_equal(
    block_cpp.ghost_voronoi_idx[:-2], block_py.ghost_voronoi_idx
)


# %%
# Cross-check block memory
# ------------------------
def cross_check_block_memory():
    for key in keys:
        cpp_value = getattr(block_cpp, key)
        py_value = getattr(block_py, key)
        assert cpp_value.shape == py_value.shape
        assert np.allclose(cpp_value, py_value), f"{key} is not equal"
        print(f"{key} |  values and shapes checked")


cross_check_block_memory()

# %%
# Cross-check computing internal forces and torques
# -------------------------------------------------
block_cpp.compute_internal_forces_and_torques(0.0)
block_py.compute_internal_forces_and_torques(0.0)

cross_check_block_memory()

# %%
# Cross-check updating accelerations
# ----------------------------------
block_cpp.update_accelerations(0.0)
block_py.update_accelerations(0.0)

cross_check_block_memory()
# %%
# Cross-check updating kinematics
# -----------------------------
block_cpp.update_kinematics(0.0, 1.4)
block_py.update_kinematics(0.0, 1.4)

cross_check_block_memory()
# %%
# Cross-check updating dynamics
# -----------------------------
block_cpp.update_dynamics(0.0, 1.6)
block_py.update_dynamics(0.0, 1.6)

cross_check_block_memory()
# %%
