"""
forcing
-------

Add forces and torques to rod (external point force, b-spline torques etc).
"""


class Forcing:
    """
    Forcing wrapper class for calling external force functions

    Attributes
    ----------
    _ext_forces_torques: list
    """

    def __init__(self):
        self._ext_forces_torques = []
        super(Forcing, self).__init__()

    def add_forcing_to(self, system):
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
    def __init__(self, sys_idx: int):
        self._sys_idx = sys_idx
        self._forcing_cls = None
        self._args = ()
        self._kwargs = {}

    def using(self, forcing_cls, *args, **kwargs):
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
        """ Constructs a constraint after checks

        Parameters
        ----------
        args
        kwargs

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
