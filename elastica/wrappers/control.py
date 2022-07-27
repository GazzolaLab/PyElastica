__doc__ = """
Control
-------

Provides the control interface to apply forces and torques to systems as a function of the state of the simulation.
"""
from elastica.typing import SystemType
from typing import Dict, List


class Control:
    """
    The Control class is a wrapper for applying boundary conditions that
    consist of applied external forces. To apply forcing on system objects,
    the simulator class must be derived from the Forcing class.

        Attributes
        ----------
        _controllers: list
            List of forcing class defined for rod-like objects.
    """

    def __init__(self):
        self._controllers = []
        super(Control, self).__init__()
        self._feature_group_synchronize.append(self._call_controller)
        self._feature_group_finalize.append(self._finalize_control)

    def control(self, systems: Dict[str, SystemType]):
        """
        This method applies external forces and torques on the relevant
        user-defined system or rod-like object. You must input the system
        or rod-like object that you want to apply external forces and torques on.

        Parameters
        ----------
        systems: Dict[SystemType]
            Dict of system objects, whose state gets passed to the Controller class

        Returns
        -------

        """
        sys_indices = {}
        for key, system in systems.items():
            sys_idx = self._get_sys_idx_if_valid(system)
            sys_indices[key] = sys_idx

        # Create _Constraint object, cache it and return to user
        controller = _Controller(sys_indices)
        self._controllers.append(controller)

        return controller

    def _finalize_control(self):
        # From stored _Controller objects, and instantiate a Controller object
        # inplace : https://stackoverflow.com/a/1208792

        # dev : the first index stores the rod index to apply the boundary condition
        # to. Technically we can use another array but it its one more book-keeping
        # step. Being lazy, I put them both in the same array
        self._controllers[:] = [
            (controller.ids().keys(), controller.ids().values(), controller())
            for controller in self._controllers
        ]

    def _call_controller(self, time, *args, **kwargs):
        for keys, sys_indices, controller in self._controllers:
            systems = {key: self._systems[sys_idx] for key, sys_idx in zip(keys, sys_indices)}
            controller.apply_forces(systems=systems, time=time, *args, **kwargs)
            controller.apply_torques(systems=systems, time=time, *args, **kwargs)


class _Controller:
    """
    Controller wrapper private class

    Attributes
    ----------
    _sys_indices: Dict[str, int]
    _control_cls: List[int]
    *args
        Variable length argument list.
    **kwargs
        Arbitrary keyword arguments.
    """

    def __init__(self, sys_indices: Dict[str, int]):
        """

        Parameters
        ----------
        sys_idx: int
        """
        self._sys_indices = sys_indices
        self._controller_cls = None
        self._args = ()
        self._kwargs = {}

    def using(self, controller_cls, *args, **kwargs):
        """
        This method is a wrapper to set which forcing class is used to apply forcing
        to user defined rod-like objects.

        Parameters
        ----------
        controller_cls: object
            User defined controller class.
        *args
            Variable length argument list
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------

        """
        from elastica.controllers import ControllerBase

        assert issubclass(
            controller_cls, ControllerBase
        ), "{} is not a valid controller. Did you forget to derive from ControllerBase?".format(
            controller_cls
        )
        self._controller_cls = controller_cls
        self._args = args
        self._kwargs = kwargs
        return self

    def ids(self):
        return self._sys_indices

    def __call__(self, *args, **kwargs):
        """Constructs a controller after checks

        Parameters
        ----------
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------

        """
        if not self._controller_cls:
            raise RuntimeError(
                "No controller provided to act on system ids {0}"
                "but a force was registered. Did you forget to call"
                "the `using` method".format(self.ids().values())
            )

        try:
            return self._controller_cls(*self._args, **self._kwargs)
        except (TypeError, IndexError):
            raise TypeError(
                r"Unable to construct forcing class.\n"
                r"Did you provide all necessary controller properties?"
            )
