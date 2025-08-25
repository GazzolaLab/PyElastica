__doc__ = """Generate or load restart file implementations."""
from typing import Iterable, Iterator, Any

import numpy as np
import os
import json
from itertools import groupby

from .memory_block import MemoryBlockCosseratRod, MemoryBlockRigidBody

from .typing import SystemType, SystemCollectionType


def all_equal(iterable: Iterable[Any]) -> bool:
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
    g: Iterator[Any] = groupby(iterable)
    return next(g, True) and not next(g, False)


def save_state(
    simulator: SystemCollectionType,
    directory: str = "saved",
    time: np.float64 = np.float64(0.0),
    verbose: bool = False,
) -> None:
    """
    Save state parameters of each systems.
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

    # Save system state
    for idx, system in enumerate(simulator):
        name = system.__class__.__name__
        path = os.path.join(directory, f"{name}_{idx}.npz")
        np.savez(path, **system.__dict__)  # type: ignore

    # Save meta-data
    with open(os.path.join(directory, "meta.json"), "w") as f:
        json.dump({"time": time}, f)

    if verbose:
        print(f"Save complete: {directory}")
        print(f"  Saved time: {time}")


def load_state(
    simulator: SystemCollectionType, directory: str = "saved", verbose: bool = False
) -> float:
    """
    Load the simulator state. Compatible with 'save_state' method.
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
    # Load meta-data
    with open(os.path.join(directory, "meta.json"), "r") as f:
        meta = json.load(f)
    time = meta["time"]

    # Load system state
    for idx, system in enumerate(simulator):
        # TODO: Not exactly sure why this condition is necessary.
        if isinstance(system, (MemoryBlockCosseratRod, MemoryBlockRigidBody)):
            continue
        name = system.__class__.__name__  # type: ignore
        path = os.path.join(directory, f"{name}_{idx}.npz")
        data = np.load(path, allow_pickle=True)
        for key, value in data.items():
            if value.shape != ():
                # Copy data into placeholders
                getattr(system, key)[:] = value
            else:
                # For single-value data
                setattr(system, key, value[()])

    # Apply boundary conditions. Ring rods have periodic BC, so we need to update periodic elements in memory block
    simulator.constrain_values(time)
    simulator.constrain_rates(time)

    if verbose:
        print(f"Load complete: {directory}")
        print(f"  Loaded time: {time}")

    return time
