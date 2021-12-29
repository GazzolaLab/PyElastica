__doc__ = """
Connect
-------

Provides the connections interface to connect entities (rods,
rigid bodies) using joints (see `joints.py`).
"""
import numpy as np
from elastica.joint import FreeJoint


class Connections:
    """
    The Connections class is a wrapper for connecting rod-like objects using joints selected
    by the user. To connect two rod-like objects, the simulator class must be derived from
    the Connections class.

        Attributes
        ----------
        _connections: list
            List of joint classes defined for rod-like objects.
    """

    def __init__(self):
        self._connections = []
        super(Connections, self).__init__()

    def connect(
        self, first_rod, second_rod, first_connect_idx=None, second_connect_idx=None
    ):
        """
        This method connects two rod-like objects using the selected joint class.
        You need to input the two rod-like objects that are to be connected as well
        as set the element indexes of these rods where the connection occurs.

        Parameters
        ----------
        first_rod : object
            Rod-like object
        second_rod : object
            Rod-like object
        first_connect_idx : int
            Index of first rod for joint.
        second_connect_idx : int
            Index of second rod for joint.

        Returns
        -------

        """
        sys_idx = [None] * 2
        for i_sys, sys in enumerate((first_rod, second_rod)):
            sys_idx[i_sys] = self._get_sys_idx_if_valid(sys)

        # For each system identified, get max dofs
        # FIXME: Revert back to len, it should be able to take, systems without elements!
        # sys_dofs = [len(self._systems[idx]) for idx in sys_idx]
        sys_dofs = [self._systems[idx].n_elems for idx in sys_idx]

        # Create _Connect object, cache it and return to user
        _connector = _Connect(*sys_idx, *sys_dofs)
        _connector.set_index(first_connect_idx, second_connect_idx)
        self._connections.append(_connector)

        return _connector

    def _finalize(self):
        # From stored _Connect objects, instantiate the joints and store it

        # dev : the first indices stores the
        # (first rod index, second_rod_idx, connection_idx_on_first_rod, connection_idx_on_second_rod)
        # to apply the connections to
        # Technically we can use another array but it its one more book-keeping
        # step. Being lazy, I put them both in the same array
        self._connections[:] = [
            (*connection.id(), connection()) for connection in self._connections
        ]

        # Need to finally solve CPP here, if we are doing things properly
        # This is to optimize the call tree for better memory accesses
        # https://brooksandrew.github.io/simpleblog/articles/intro-to-graph-optimization-solving-cpp/

    def __call__(self, *args, **kwargs):
        for (
            first_sys_idx,
            second_sys_idx,
            first_connect_idx,
            second_connect_idx,
            connection,
        ) in self._connections:
            connection.apply_forces(
                self._systems[first_sys_idx],
                first_connect_idx,
                self._systems[second_sys_idx],
                second_connect_idx,
            )
            connection.apply_torques(
                self._systems[first_sys_idx],
                first_connect_idx,
                self._systems[second_sys_idx],
                second_connect_idx,
            )


class _Connect:
    """
    Connect wrapper private class

    Attributes
    ----------
    _first_sys_idx: int
    _second_sys_idx: int
    _first_sys_n_lim: int
    _second_sys_n_lim: int
    _connect_class: list
    first_sys_connection_idx: int
    second_sys_connection_idx: int
    *args
        Variable length argument list.
    **kwargs
        Arbitrary keyword arguments.
    """

    def __init__(
        self,
        first_sys_idx: int,
        second_sys_idx: int,
        first_sys_nlim: int,
        second_sys_nlim: int,
    ):
        """

        Parameters
        ----------
        first_sys_idx: int
        second_sys_idx: int
        first_sys_nlim: int
        second_sys_nlim: int
        """
        self._first_sys_idx = first_sys_idx
        self._second_sys_idx = second_sys_idx
        self._first_sys_n_lim = first_sys_nlim
        self._second_sys_n_lim = second_sys_nlim
        self._connect_cls = None
        self._args = ()
        self._kwargs = {}
        self.first_sys_connection_idx = None
        self.second_sys_connection_idx = None

    def set_index(self, first_idx, second_idx):
        # TODO assert range
        # First check if the types of first rod idx and second rod idx variable are same.
        assert type(first_idx) == type(
            second_idx
        ), "Type of first_connect_idx :{}".format(
            type(first_idx)
        ) + " is different than second_connect_idx :{}".format(
            type(second_idx)
        )

        # Check if the type of idx variables are correct.
        assert isinstance(
            first_idx, (int, np.int_, list, tuple, np.ndarray, type(None))
        ), "Connection index type is not supported :{}".format(
            type(first_idx)
        ) + ", please try one of the following :{}".format(
            (int, np.int_, list, tuple, np.ndarray)
        )

        # If type of idx variables are tuple or list or np.ndarray, check validity of each entry.
        if (
            isinstance(first_idx, tuple)
            or isinstance(first_idx, list)
            or isinstance(first_idx, np.ndarray)
        ):

            for i in range(len(first_idx)):
                assert isinstance(first_idx[i], (int, np.int_)), (
                    "Connection index of first rod is not integer :{}".format(
                        first_idx[i]
                    )
                    + " It should be :{}".format((int, np.int_))
                    + " Check your input!"
                )
                assert isinstance(second_idx[i], (int, np.int_)), (
                    "Connection index of second rod is not integer :{}".format(
                        second_idx[i]
                    )
                    + " It should be :{}".format((int, np.int_))
                    + " Check your input!"
                )

                # The addition of +1 and and <= check on the RHS is because
                # connections can be made to the node indices as well
                assert (
                    -(self._first_sys_n_lim + 1)
                    <= first_idx[i]
                    <= self._first_sys_n_lim
                ), "Connection index of first rod exceeds its dof : {}".format(
                    self._first_sys_n_lim
                )
                assert (
                    -(self._second_sys_n_lim + 1)
                    <= second_idx[i]
                    <= self._second_sys_n_lim
                ), "Connection index of second rod exceeds its dof : {}".format(
                    self._second_sys_n_lim
                )
        elif first_idx is None:
            # Do nothing if idx are None
            pass
        else:

            # The addition of +1 and and <= check on the RHS is because
            # connections can be made to the node indices as well
            assert (
                -(self._first_sys_n_lim + 1) <= first_idx <= self._first_sys_n_lim
            ), "Connection index of first rod exceeds its dof : {}".format(
                self._first_sys_n_lim
            )
            assert (
                -(self._second_sys_n_lim + 1) <= second_idx <= self._second_sys_n_lim
            ), "Connection index of second rod exceeds its dof : {}".format(
                self._second_sys_n_lim
            )

        self.first_sys_connection_idx = first_idx
        self.second_sys_connection_idx = second_idx

    def using(self, connect_cls, *args, **kwargs):
        """
        This method is a wrapper to set which joint class is used to connect
        user defined rod-like objects.

        Parameters
        ----------
        connect_cls: object
            User defined callback class.
        *args
            Variable length argument list
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------

        """
        assert issubclass(
            connect_cls, FreeJoint
        ), "{} is not a valid joint class. Did you forget to derive from FreeJoint?".format(
            connect_cls
        )
        self._connect_cls = connect_cls
        self._args = args
        self._kwargs = kwargs
        return self

    def id(self):
        return (
            self._first_sys_idx,
            self._second_sys_idx,
            self.first_sys_connection_idx,
            self.second_sys_connection_idx,
        )

    def __call__(self, *args, **kwargs):
        if not self._connect_cls:
            raise RuntimeError(
                "No connections provided to link rod id {0}"
                "(at {2}) and {1} (at {3}), but a Connection"
                "was intended as per code. Did you forget to"
                "call the `using` method?".format(*self.id())
            )

        try:
            return self._connect_cls(*self._args, **self._kwargs)
        except (TypeError, IndexError):
            raise TypeError(
                r"Unable to construct connection class.\n"
                r"Did you provide all necessary joint properties?"
            )
