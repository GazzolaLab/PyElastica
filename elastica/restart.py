__doc__ = """Generate or load restart file implementations."""
__all__ = ["save_state", "load_state"]
import numpy as np
import os
from itertools import groupby


def all_equal(iterable):
    """
    Checks if all elements of list are equal.
    Parameters
    ----------
    iterable : list
        Iterable list
    Returns
    -------
        Boolean
    References
    ----------
        https://stackoverflow.com/questions/3844801/check-if-all-elements-in-a-list-are-identical
    """
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


def save_state(simulator, directory: str = "", time=0.0, verbose: bool = False):
    """
    Save state parameters of each rod.
    TODO : environment list variable is not uniform at the current stage of development.
    It would be nice if we have set list (like env.system) that iterates all the rods.
    Parameters
    ----------
    simulator : object
        Simulator object.
    directory : str
        Directory path name. The path must exist.
    time : float
        Simulation time.
    verbose : boolean

    """
    os.makedirs(directory, exist_ok=True)
    for idx, rod in enumerate(simulator):
        path = os.path.join(directory, "system_{}.npz".format(idx))
        np.savez(path, time=time, **rod.__dict__)

    if verbose:
        print("Save complete: {}".format(directory))


def load_state(simulator, directory: str = "", verbose: bool = False):
    """
    Load the rod-state. Compatibale with 'save_state' method.
    If the save-file does not exist, it returns error.
    Call this function after finalize method.

    Parameters
    ----------
    simulator : object
        Simulator object.
    directory : str
        Directory path name.
    verbose : boolean

    Returns
    ------
    time : float
        Simulation time of systems when they are saved.
    """
    time_list = []  # Simulation time of rods when they are saved.
    for idx, rod in enumerate(simulator):
        path = os.path.join(directory, "system_{}.npz".format(idx))
        data = np.load(path, allow_pickle=True)
        for key, value in data.items():
            if key == "time":
                time_list.append(value.item())
                continue

            if value.shape != ():
                # Copy data into placeholders
                getattr(rod, key)[:] = value
            else:
                # For single-value data
                setattr(rod, key, value)

    if not all_equal(time_list):
        raise ValueError(
            "Restart time of loaded rods are different, check your inputs!"
        )

    # Apply boundary conditions, after loading the systems.
    simulator.constrain_values(0.0)
    simulator.constrain_rates(0.0)

    if verbose:
        print("Load complete: {}".format(directory))

    return time_list[0]
