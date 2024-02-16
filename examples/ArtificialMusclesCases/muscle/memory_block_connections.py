__doc__ = """
MemoryBlockConnect
-------
Provides the connections interface to connect rods, rigid bodies. Different than the Connections class this class
groups same type of connections together and turns it into a single connection. It is faster than Connections class
when there are many connections. You should be careful while using this class because it is more complicated and
requires extra care for indexes and other arguments.
"""

from elastica.modules.connections import Connections, _Connect
import numpy as np
from typing import Sequence
from collections import defaultdict
from elastica.joint import FreeJoint


class MemoryBlockConnections(Connections):
    def __init__(self):
        # super(Connections, self).__init__()
        # self._connections = []
        self._connections = []
        super(Connections, self).__init__()
        self._feature_group_synchronize.append(self._call_connections)
        self._feature_group_finalize.append(self._finalize_connections)

    # TODO: first and second connect idx can be a list and take multiple idx?
    def connect(self, first_rod, second_rod, first_connect_idx=0, second_connect_idx=0):
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

    def _finalize_connections(self):

        # TODO: Allow passing more than one index for first idx and second idx for a connection.
        # TODO: Sort the connections for better memory access.

        # First find the types of connections among all the connections. We will group connections depending on their
        # type and memory blocks they belong.
        _connection_cls_type_list = list(
            set([connection._connect_cls for connection in self._connections])
        )
        _connection_cls_idx_list = [[] for _ in _connection_cls_type_list]
        number_of_connection_cls_type = len(_connection_cls_type_list)
        number_of_memory_block = len(self._memory_blocks)
        # TODO: improve memory_block_connections for number of memory_blocks>2.
        if number_of_memory_block > 2:
            raise NotImplemented(
                "We haven't implemented memory block connections in case there are three types of"
                " memory block or systems."
            )

        for k in range(number_of_connection_cls_type):
            for _ in range(2 ** number_of_memory_block - 1):
                _connection_cls_idx_list[k].append([])

        for k in range(len(_connection_cls_type_list)):
            for i in range(len(self._connections)):
                if _connection_cls_type_list[k] == self._connections[i]._connect_cls:
                    first_sys_idx = self._connections[i]._first_sys_idx
                    second_sys_idx = self._connections[i]._second_sys_idx

                    # In case having more than one memory block i.e. CosseratRod and MuscularRod, then we need to
                    # distinguish which blocks does the rods belong and group them. There are three possibilities.
                    # 1) first system and second system belongs to memory block one => my_block_idx=0
                    # 2) first system belongs to memory block one and second system belongs to memory block two =>
                    #   my_block_idx = 0+1 = 1
                    # 3) first system and second system belongs to memory block two => my_block_idx=1+1=2
                    my_block_idx = 0
                    for block_idx in range(number_of_memory_block):
                        if (
                            np.where(
                                first_sys_idx
                                == self._memory_blocks[block_idx].system_idx_list
                            )[0].size
                            == 1
                        ):
                            my_block_idx += block_idx

                        if (
                            np.where(
                                second_sys_idx
                                == self._memory_blocks[block_idx].system_idx_list
                            )[0].size
                            == 1
                        ):
                            my_block_idx += block_idx

                    _connection_cls_idx_list[k][my_block_idx].append(i)

        if not len(self._connections) == 0:
            _connection_cls_idx_list, _connection_cls_type_list = (
                list(t)
                for t in zip(
                    *sorted(zip(_connection_cls_idx_list, _connection_cls_type_list))
                )
            )

        _memory_block_connector = []
        _memory_block_connector_list_idx = 0

        for i, same_type_connection_list in enumerate(_connection_cls_idx_list):

            for k, same_memory_block_group_list in enumerate(same_type_connection_list):
                # These connections have the same type. In addition to that we grouped them together because
                # system one and two are in the same memory block, or system one is in one memory block and system two
                # is in the other memory block.

                if len(same_memory_block_group_list) == 0:
                    # if there is no connection, then don't do anything.
                    continue

                # Find which memory block does the systems connected belong. Check the first connection on the list.
                # Since all the remaining connections on same_memory_block_group_list are belong to same memory blocks.
                first_sys_idx = self._connections[
                    same_memory_block_group_list[0]
                ]._first_sys_idx
                second_sys_idx = self._connections[
                    same_memory_block_group_list[0]
                ]._second_sys_idx
                for block_idx in range(number_of_memory_block):
                    if (
                        np.where(
                            first_sys_idx
                            == self._memory_blocks[block_idx].system_idx_list
                        )[0].size
                        == 1
                    ):
                        # Index on memory_blocks list
                        first_block_idx = block_idx
                    if (
                        np.where(
                            second_sys_idx
                            == self._memory_blocks[block_idx].system_idx_list
                        )[0].size
                        == 1
                    ):
                        # Index on memory_blocks list
                        second_block_idx = block_idx
                # Get the memory block index on system list.
                first_block_sys_idx = self._get_sys_idx_if_valid(
                    self._memory_blocks[first_block_idx]
                )
                second_block_sys_idx = self._get_sys_idx_if_valid(
                    self._memory_blocks[second_block_idx]
                )

                # Create and append MemoryBlockConnection class. This is a wrapper we use.
                _memory_block_connector.append(
                    _MemoryBlockConnect(first_block_sys_idx, second_block_sys_idx)
                )
                # Update the connection type.
                _memory_block_connector[
                    _memory_block_connector_list_idx
                ]._connect_cls = _connection_cls_type_list[i]

                for _, idx in enumerate(same_memory_block_group_list):
                    # These connections are same type and belong to the same memory block group. So they should be
                    # packed together.
                    first_sys_idx = self._connections[idx]._first_sys_idx
                    second_sys_idx = self._connections[idx]._second_sys_idx
                    first_sys_connection_idx = self._connections[
                        idx
                    ].first_sys_connection_idx
                    second_sys_connection_idx = self._connections[
                        idx
                    ].second_sys_connection_idx

                    first_sys_idx_on_block = np.where(
                        self._memory_blocks[first_block_idx].system_idx_list
                        == first_sys_idx
                    )[0]
                    second_sys_idx_on_block = np.where(
                        self._memory_blocks[second_block_idx].system_idx_list
                        == second_sys_idx
                    )[0]

                    # Find the node or element or voronoi offset we need, to get the corresponding
                    # index on memory block.
                    first_sys_offset = self._memory_blocks[
                        first_block_idx
                    ].start_idx_in_rod_nodes[first_sys_idx_on_block][0]
                    second_sys_offset = self._memory_blocks[
                        second_block_idx
                    ].start_idx_in_rod_nodes[second_sys_idx_on_block][0]

                    _memory_block_connector[_memory_block_connector_list_idx].set_index(
                        first_sys_connection_idx,
                        second_sys_connection_idx,
                        first_sys_offset,
                        second_sys_offset,
                    )

                    _memory_block_connector[
                        _memory_block_connector_list_idx
                    ]._args += self._connections[idx]._args[:]
                    for key in self._connections[idx]._kwargs.keys():
                        _memory_block_connector[
                            _memory_block_connector_list_idx
                        ]._kwargs[key].append(self._connections[idx]._kwargs[key])

                    _memory_block_connector[_memory_block_connector_list_idx]._kwargs[
                        "first_sys_idx"
                    ].append(first_sys_idx)
                    _memory_block_connector[_memory_block_connector_list_idx]._kwargs[
                        "second_sys_idx"
                    ].append(second_sys_idx)
                    _memory_block_connector[_memory_block_connector_list_idx]._kwargs[
                        "first_sys_idx_offset"
                    ].append(first_sys_offset)
                    _memory_block_connector[_memory_block_connector_list_idx]._kwargs[
                        "second_sys_idx_offset"
                    ].append(second_sys_offset)
                    _memory_block_connector[_memory_block_connector_list_idx]._kwargs[
                        "first_sys_idx_on_block"
                    ].append(first_sys_idx_on_block)
                    _memory_block_connector[_memory_block_connector_list_idx]._kwargs[
                        "second_sys_idx_on_block"
                    ].append(second_sys_idx_on_block)

                _memory_block_connector_list_idx += 1

        self._connections = _memory_block_connector

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


class _MemoryBlockConnect:
    """
    Connect wrapper private class
    Attributes
    ----------
    _first_sys_idx: int
    _second_sys_idx: int
    _connect_cls: class
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
    ):
        """
        Parameters
        ----------
        first_sys_idx: int
        second_sys_idx: int
        """
        self._first_sys_idx = first_sys_idx
        self._second_sys_idx = second_sys_idx
        self._connect_cls = None
        self._args = ()
        self._kwargs = defaultdict(list)
        self.first_sys_connection_idx = []
        self.second_sys_connection_idx = []

    def set_index(
        self,
        first_idx: int,
        second_idx: int,
        first_sys_offset: int,
        second_sys_offset: int,
    ):
        self.first_sys_connection_idx.append(first_idx + first_sys_offset)
        self.second_sys_connection_idx.append(second_idx + second_sys_offset)

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
            # first element of the list corresponds to memory block
            self._first_sys_idx,
            # second element of the list corresponds to memory block
            self._second_sys_idx,
            np.array(self.first_sys_connection_idx, dtype=np.int),
            np.array(self.second_sys_connection_idx, dtype=np.int),
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
