"""
connect
-------

Provides the Connections interface to connect entities (rods,
rigid bodies) using Joints (see `joints.py`).
"""

from elastica.joint import FreeJoint


class Connections:
    def __init__(self):
        self._connections = []

    def connect(
        self, first_rod, second_rod, first_connect_idx=0, second_connect_idx=-1
    ):
        # START: Should be in its own function
        from elastica._rod import RodBase
        from numpy import int_ as npint

        n_systems = len(self._systems)  # Total number of systems from mixed-in class
        sys_idx = [None] * 2
        for i_sys, sys in enumerate((first_rod, second_rod)):
            if isinstance(sys, (int, npint)):
                # 1. If they are indices themselves, check range
                assert (
                    -n_systems <= sys < n_systems
                ), "Rod index {} exceeds number of registered systems".format(sys)
                sys_idx[i_sys] = sys
            elif issubclass(sys.__class__, RodBase):
                # 2. If they are sys objects (most likely), lookup indices
                # index might have some problems : https://stackoverflow.com/a/176921
                try:
                    sys_idx[i_sys] = self._systems.index(sys)
                except ValueError:
                    raise ValueError(
                        "Rod {} was not found, did you append it to the system?".format(
                            sys
                        )
                    )
            else:
                raise TypeError("argument {} is not a sys index/object".format(sys))
        # END

        # For each system identified, get max dofs
        sys_dofs = [len(self._systems[idx]) for idx in sys_idx]

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
            connection.apply_force(
                self._systems[first_sys_idx],
                self._systems[second_sys_idx],
                first_connect_idx,
                second_connect_idx,
            )
            connection.apply_torue(
                self._systems[first_sys_idx],
                self._systems[second_sys_idx],
                first_connect_idx,
                second_connect_idx,
            )


class _Connect:
    def __init__(
        self,
        first_sys_idx: int,
        second_sys_idx: int,
        first_sys_nlim: int,
        second_sys_nlim: int,
    ):
        self._first_sys_idx = first_sys_idx
        self._second_sys_idx = second_sys_idx
        self._first_sys_n_lim = first_sys_nlim
        self._second_sys_n_lim = second_sys_nlim
        self._connect_cls = None
        self._args = ()
        self._kwargs = {}
        self.first_sys_connection_idx = None
        self.second_sys_connection_idx = None

    def set_index(self, first_idx: int, second_idx: int):
        # TODO assert range
        assert (
            -self._first_sys_n_lim <= first_idx < self._first_sys_n_lim
        ), "Connection index of first rod exceeds its dof : {}".format(
            self._first_sys_n_lim
        )
        assert (
            -self._second_sys_n_lim <= second_idx < self._second_sys_n_lim
        ), "Connection index of second rod exceeds its dof : {}".format(
            self._second_sys_n_lim
        )

        self.first_sys_connection_idx = first_idx
        self.second_sys_connection_idx = second_idx

    def using(self, connect_cls, *args, **kwargs):
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
                "No connections provided to link rod id {0} (at {1}) and {2} (at {3})".format(
                    self._first_sys_idx,
                    self.first_sys_connection_idx,
                    self._second_sys_idx,
                    self.second_sys_connection_idx,
                )
            )

        try:
            return self._connect_cls(*self._args, **self._kwargs)
        except:
            raise TypeError(
                r"Unable to construct connnection class.\n"
                r"Did you provide all necessary joint properties?"
            )
