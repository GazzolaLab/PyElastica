"""
callback
-----------

Provides the CallBack interface to collect data in time (see `callback_functions.py`).
"""

from elastica.callback_functions import CallBackBaseClass


class CallBacks:
    def __init__(self):
        self._callbacks = []
        super(CallBacks, self).__init__()

    def callback_of(self, system):
        sys_idx = self._get_sys_idx_if_valid(system)

        # Create _Constraint object, cache it and return to user
        _callbacks = _CallBack(sys_idx)
        self._callbacks.append(_callbacks)

        return _callbacks

    def _finalize(self):
        # From stored _CallBack objects, instantiate the boundary conditions
        # inplace : https://stackoverflow.com/a/1208792

        # dev : the first index stores the rod index to apply the boundary condition
        # to. Technically we can use another array but it its one more book-keeping
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
    def __init__(self, sys_idx: int):
        self._sys_idx = sys_idx
        self._callback_cls = None
        self._args = ()
        self._kwargs = {}

    def using(self, callback_cls, *args, **kwargs):
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
        """ Constructs a callback functions after checks

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
