# Elasticapp backend for PyElastica

This file serves as the documentation for the `elasticapp` backend.

## Installation

In the root of the PyElastica source tree, run the following command

```
pip install ./backend
```

> Make sure you install the package from _PyElastica source tree_.

This command will take care of installation of all build-time dependencies, compilation of C++ source files and install the a python package called `elasticapp`.

## Testing

Make sure you have `pytest` installed. In the root of the PyElastica source tree, run the following command

```
pytest backend/tests
```

## Benchmarking

Standalone scripts for benchmarking purposes are available in `backend/benchmarking` folder.

### Benchmarking `matmul`

For benchmarking various `matmul` implementations, run

```
python3 backend/benchmarking/matmul.py
```

## Contributed By

- Tejaswin Parthasarathy (Teja)
- [Seung Hyun Kim](https://github.com/skim0119)
- Ankith Pai
- [Yashraj Bhosale](https://github.com/bhosale2)
- Arman Tekinalp
- Songyuan Cui
