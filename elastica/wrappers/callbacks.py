__doc__ = """
CallBacks
-----------

Provides the callBack interface to collect data over time (see `callback_functions.py`).
"""

from elastica.callback_functions import CallBackBaseClass


class CallBacks:
    """
    CallBacks class is a wrapper for calling callback functions, set by the user. If the user
    wants to collect data from the simulation, the simulator class has to be derived
    from the CallBacks class.

        Attributes
        ----------
        _callbacks: list
            List of call back classes defined for rod-like objects.
    """

    def __init__(self):
        self._callbacks = []
        super(CallBacks, self).__init__()

    def collect_diagnostics(self, system):
        """
        This method calls user-defined call-back classes for a
        user-defined system or rod-like object. You need to input the
        system or rod-like object that you want to collect data from.

        Parameters
        ----------
        system: object
            System is a rod-like object.

        Returns
        -------

        """
        sys_idx = self._get_sys_idx_if_valid(system)

        # Create _Constraint object, cache it and return to user
        _callbacks = _CallBack(sys_idx)
        self._callbacks.append(_callbacks)

        return _callbacks

    def _finalize(self):
        # From stored _CallBack objects, instantiate the boundary conditions
        # inplace : https://stackoverflow.com/a/1208792

        # dev : the first index stores the rod index to collect data.
        # Technically we can use another array but it its one more book-keeping
        # step. Being lazy, I put them both in the same array
        self._callbacks[:] = [
            (callback.id(), callback(self._systems[callback.id()]))
            for callback in self._callbacks
        ]

        # Sort from lowest id to highest id for potentially better memory access
        # _callbacks contains list of tuples. First element of tuple is rod number and
        # following elements are the type of boundary condition such as
        # [(0, MyCallBack), (1, MyVelocityCallBack), ... ]
        # Thus using lambda we iterate over the list of tuples and use rod number (x[0])
        # to sort callbacks.
        self._callbacks.sort(key=lambda x: x[0])

        self._callBack(time=0.0, current_step=0)

    # TODO: same as above naming of _callBack function
    def _callBack(self, time, current_step: int, *args, **kwargs):
        for sys_id, callback in self._callbacks:
            callback.make_callback(
                self._systems[sys_id], time, current_step, *args, **kwargs
            )


class _CallBack:
    """
    CallBack wrapper private class

        Attributes
        ----------
        _sys_idx: rod object index
        _callback_cls: list
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
        self._callback_cls = None
        self._args = ()
        self._kwargs = {}

    def using(self, callback_cls, *args, **kwargs):
        """
        This method is a wrapper to set which callback class is used to collect data
        from user defined rod-like object.

        Parameters
        ----------
        callback_cls: object
            User defined callback class.
        *args
            Variable length argument list
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------

        """
        assert issubclass(
            callback_cls, CallBackBaseClass
        ), "{} is not a valid call back. Did you forget to derive from CallBackClass?".format(
            callback_cls
        )
        self._callback_cls = callback_cls
        self._args = args
        self._kwargs = kwargs
        return self

    def id(self):
        return self._sys_idx

    def __call__(self, *args, **kwargs):
        """Constructs a callback functions after checks

        Parameters
        ----------
        args
        kwargs

        Returns
        -------

        """
        if not self._callback_cls:
            raise RuntimeError(
                "No callback provided to act on rod id {0}"
                "but a callback was registered. Did you forget to call"
                "the `using` method".format(self.id())
            )

        try:
            return self._callback_cls(*self._args, **self._kwargs)
        except (TypeError, IndexError):
            raise TypeError(
                r"Unable to construct callback class.\n"
                r"Did you provide all necessary callback properties?"
            )
