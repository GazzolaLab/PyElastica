# Installation

## Instruction

PyElastica requires Python 3.5 - 3.8, which needs to be installed prior to using PyElastica. For information on installing Python, see [here](https://realpython.com/installing-python/). If you are interested in using a package manager like Conda, see [here](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html).

:::{note}
Python version above 3.8 is tested only in Ubuntu and Mac OS. For Windows 10, some of the dependencies were not yet compatible.
:::

The easiest way to install PyElastica is with `pip`:

```sh
$ pip install pyelastica
```
You can also download the source code for PyElastica directly from [GitHub](https://github.com/GazzolaLab/PyElastica). 

## Dependencies

The core of PyElastica is developed using:

- numpy
- numba
- scipy
- tqdm
- matplotlib (visualization)

Above packages will be installed along with PyElastica if you used `pip` to install.
If you have directly downloaded the source code, you must install these packages separately. 
