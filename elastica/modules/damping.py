__doc__ = """
Damping
-------

(added in version 0.3.0)

Provides the damper interface to apply damping
on the rods. (see `dissipation.py`).

"""

from elastica.dissipation import DamperBase


class Damping:
    """
    The Damping class is a module for applying damping
    on rod-like objects, the simulator class must be derived from
    Damping class.

        Attributes
        ----------
        _dampers: list
            List of damper classes defined for rod-like objects.
    """

    def __init__(self):
        self._dampers = []
        super(Damping, self).__init__()
        self._feature_group_constrain_rates.append(self._dampen_rates)
        self._feature_group_finalize.append(self._finalize_dampers)

    def dampen(self, system):
        """
        This method applies damping on relevant user-defined
        system or rod-like object. You must input the system or rod-like
        object that you want to apply damping on.

        Parameters
        ----------
        system: object
            System is a rod-like object.

        Returns
        -------

        """
        sys_idx = self._get_sys_idx_if_valid(system)

        # Create _Damper object, cache it and return to user
        _damper = _Damper(sys_idx)
        self._dampers.append(_damper)

        return _damper

    def _finalize_dampers(self):
        # From stored _Damping objects, instantiate the dissipation/damping
        # inplace : https://stackoverflow.com/a/1208792

        self._dampers[:] = [
            (damper.id(), damper(self._systems[damper.id()]))
            for damper in self._dampers
        ]

        # Sort from lowest id to highest id for potentially better memory access
        # _dampers contains list of tuples. First element of tuple is rod number and
        # following elements are the type of damping.
        # Thus using lambda we iterate over the list of tuples and use rod number (x[0])
        # to sort dampers.
        self._dampers.sort(key=lambda x: x[0])

    def _dampen_rates(self, time, *args, **kwargs):
        for sys_id, damper in self._dampers:
            damper.dampen_rates(self._systems[sys_id], time, *args, **kwargs)


class _Damper:
    """
    Damper module private class

    Attributes
    ----------
    _sys_idx: int
    _damper_cls: list
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
        self._damper_cls = None
        self._args = ()
        self._kwargs = {}

    def using(self, damper_cls, *args, **kwargs):
        """
        This method is a module to set which damper class is used to
        enforce damping from user defined rod-like objects.

        Parameters
        ----------
        damper_cls : object
            User defined damper class.
        *args
            Variable length argument list
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------

        """
        assert issubclass(
            damper_cls, DamperBase
        ), "{} is not a valid damper. Damper must be driven from DamperBase.".format(
            damper_cls
        )
        self._damper_cls = damper_cls
        self._args = args
        self._kwargs = kwargs
        return self

    def id(self):
        return self._sys_idx

    def __call__(self, rod, *args, **kwargs):
        """Constructs a Damper class object after checks

        Parameters
        ----------
        args
        kwargs

        Returns
        -------

        """
        if not self._damper_cls:
            raise RuntimeError(
                "No damper provided to dampen rod id {0} at {1},"
                "but damping was intended. Did you"
                "forget to call the `using` method?".format(self.id(), rod)
            )

        try:
            damper = self._damper_cls(*self._args, _system=rod, **self._kwargs)
            return damper
        except (TypeError, IndexError):
            raise TypeError("Unable to construct damping class.\n")
