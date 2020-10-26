__doc__ = """
Forcing
-------

Provides the forcing interface to apply forces and torques to rod-like objects
(external point force, muscle torques, etc).
"""


class Forcing:
    """
    The Forcing class is a wrapper for applying boundary conditions that
    consist of applied external forces. To apply forcing on rod-like objects,
    the simulator class must be derived from the Forcing class.

        Attributes
        ----------
        _ext_forces_torques: list
            List of forcing class defined for rod-like objects.
    """

    def __init__(self):
        self._ext_forces_torques = []
        super(Forcing, self).__init__()

    def add_forcing_to(self, system):
        """
        This method applies external forces and torques on the relevant
        user-defined system or rod-like object. You must input the system
        or rod-like object that you want to apply external forces and torques on.

        Parameters
        ----------
        system: object
            System is a rod-like object.

        Returns
        -------

        """
        sys_idx = self._get_sys_idx_if_valid(system)

        # Create _Constraint object, cache it and return to user
        _ext_force_torque = _ExtForceTorque(sys_idx)
        self._ext_forces_torques.append(_ext_force_torque)

        return _ext_force_torque

    def _finalize(self):
        # From stored _ExtForceTorque objects, and instantiate a Force
        # inplace : https://stackoverflow.com/a/1208792

        # dev : the first index stores the rod index to apply the boundary condition
        # to. Technically we can use another array but it its one more book-keeping
        # step. Being lazy, I put them both in the same array
        self._ext_forces_torques[:] = [
            (ext_force_torque.id(), ext_force_torque())
            for ext_force_torque in self._ext_forces_torques
        ]

        # Sort from lowest id to highest id for potentially better memory access
        # _ext_forces_torques contains list of tuples. First element of tuple is
        # rod number and following elements are the type of boundary condition such as
        # [(0, NoForces, GravityForces), (1, UniformTorques), ... ]
        # Thus using lambda we iterate over the list of tuples and use rod number (x[0])
        # to sort _ext_forces_torques.
        self._ext_forces_torques.sort(key=lambda x: x[0])

    def __call__(self, time, *args, **kwargs):
        for sys_id, ext_force_torque in self._ext_forces_torques:
            ext_force_torque.apply_forces(self._systems[sys_id], time, *args, **kwargs)
            ext_force_torque.apply_torques(self._systems[sys_id], time, *args, **kwargs)
            # TODO Apply torque, see if necessary


class _ExtForceTorque:
    """
    Forcing wrapper private class

    Attributes
    ----------
    _sys_idx: int
    _forcing_cls: list
    *args
        Variable length argument list.
    **kwargs
        Arbitrary keyword arguments.
    """

    def __init__(self, sys_idx: int):
        """

        Parameters
        ----------
        sys_idx: int
        """
        self._sys_idx = sys_idx
        self._forcing_cls = None
        self._args = ()
        self._kwargs = {}

    def using(self, forcing_cls, *args, **kwargs):
        """
        This method is a wrapper to set which forcing class is used to apply forcing
        to user defined rod-like objects.

        Parameters
        ----------
        forcing_cls: object
            User defined forcing class.
        *args
            Variable length argument list
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------

        """
        from elastica.external_forces import NoForces

        assert issubclass(
            forcing_cls, NoForces
        ), "{} is not a valid forcing. Did you forget to derive from NoForces?".format(
            forcing_cls
        )
        self._forcing_cls = forcing_cls
        self._args = args
        self._kwargs = kwargs
        return self

    def id(self):
        return self._sys_idx

    def __call__(self, *args, **kwargs):
        """Constructs a constraint after checks

        Parameters
        ----------
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------

        """
        if not self._forcing_cls:
            raise RuntimeError(
                "No forcing provided to act on rod id {0}"
                "but a force was registered. Did you forget to call"
                "the `using` method".format(self.id())
            )

        try:
            return self._forcing_cls(*self._args, **self._kwargs)
        except (TypeError, IndexError):
            raise TypeError(
                r"Unable to construct forcing class.\n"
                r"Did you provide all necessary force properties?"
            )
