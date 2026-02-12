# C++ for PyElastica

This directory includes basic C++ backend for experimental purposes.
The full functionality is not yet moved to this directory.
The purpose of this update is to provide a way to implement functionality in C++, such as multithreading and more controlled vectorization.
For large-number of rods and complex environment, computation with contact and self-interactions requires higher control over parallelization.

> Note: manual tuning of vectorization and threading is required for optimal performance, which may vary depending on the hardware.

> Note: expected performance improvement depends on the vector size. It is expected to be faster for large rod systems, mostly due to the overhead created by the python-binding and interpreter. In most of the case with less than 1e5 elements, the performance of `numba` could be better.

## Usage

```python
import elastica as ea
import elasticapp as epp

class SystemSimulator(
    ea.BaseSystemCollection,
    ...
):
    pass

simulator = SystemSimulator()
# Replace C++ block module for CosseratRod
simulator.enable_block_supports(ea.CosseratRod, epp.MemoryBlockCosseratRod)
```

## Installation

From the `backend` directory, run:

```bash
cd backend
make install  # or run pip install "."
```
> Make sure you install the package from _PyElastica source tree_.

## Testing

Test includes both `cpp` and `python` testing.

```bash
make test
```

## File structure

- All cpp files are in `src` directory. `src/py` directory includes python bindings for C++ classes.
- All test files are in `tests` directory. `cpp` tests are in `tests/cpp` directory, and `python` tests are in `tests/py` directory.
- All benchmark files are in `benchmarking` directory.


## Contributed By

- Tejaswin Parthasarathy (Teja)
- [Seung Hyun Kim](https://github.com/skim0119)
- Ankith Pai
- [Yashraj Bhosale](https://github.com/bhosale2)
- Arman Tekinalp
- Songyuan Cui
