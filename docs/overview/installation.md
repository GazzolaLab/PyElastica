# Installation

## Instruction

PyElastica requires Python 3.8 - 3.10, which needs to be installed prior to using PyElastica. For information on installing Python, see [here](https://realpython.com/installing-python/). If you are interested in using a package manager like Conda, see [here](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html).

:::{note}
Python version above 3.8 is tested only in Ubuntu and Mac OS. For Windows 10, some of the dependencies were not yet compatible.
:::

The easiest way to install PyElastica is with `pip`:

```bash
$ pip install pyelastica
```
You can also download the source code for PyElastica directly from [GitHub](https://github.com/GazzolaLab/PyElastica).

All options:
- `examples`: installs dependencies to run example cases,
found under the folder [`examples`](https://github.com/GazzolaLab/PyElastica/tree/master/examples).
- `docs`: packages to build documentation

Options can be combined e.g.
```bash
$ pip install "pyelastica[examples,docs]"
```

If you want to simulate magnetic Cosserat rods interacting with external magnetic environments you can install the derived package using

```bash
$ pip install magneto_pyelastica
```

Details can be found [here](https://github.com/armantekinalp/MagnetoPyElastica).

## Dependencies

The core of PyElastica is developed using:

- numpy
- numba
- scipy
- tqdm
- matplotlib (visualization)

Above packages will be installed along with PyElastica if you used `pip` to install.
If you have directly downloaded the source code, you must install these packages separately.
