from elastica.modules.operator_group import OperatorGroupFIFO


def test_add_ids():
    group = OperatorGroupFIFO()
    group.append_id(1)
    group.append_id(2)
    group.append_id(3)

    assert group._operator_ids == [id(1), id(2), id(3)]


def test_add_operators():
    group = OperatorGroupFIFO()
    group.append_id(1)
    group.add_operators(1, [1, 2, 3])
    group.append_id(2)
    group.add_operators(2, [4, 5, 6])
    group.append_id(3)
    group.add_operators(3, [7, 8, 9])

    assert group._operator_collection == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert group._operator_ids == [id(1), id(2), id(3)]

    group.append_id(4)
    group.add_operators(4, [10, 11, 12])

    assert group._operator_collection == [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12],
    ]
    assert group._operator_ids == [id(1), id(2), id(3), id(4)]


def test_grouping():
    group = OperatorGroupFIFO()
    group.append_id(1)
    group.add_operators(1, [1, 2, 3])
    group.append_id(2)
    group.add_operators(2, [4, 5, 6])
    group.append_id(3)
    group.add_operators(3, [7, 8, 9])

    assert list(group) == [1, 2, 3, 4, 5, 6, 7, 8, 9]

    group.append_id(4)
    group.add_operators(4, [10, 11, 12])

    assert list(group) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    group.append_id(1)
    group.add_operators(1, [13, 14, 15])

    assert list(group) == [1, 2, 3, 13, 14, 15, 4, 5, 6, 7, 8, 9, 10, 11, 12]


def test_is_last():
    group = OperatorGroupFIFO()
    group.append_id(1)
    group.add_operators(1, [1, 2, 3])
    group.append_id(2)
    group.add_operators(2, [4, 5, 6])

    assert group.is_last(1) == False
    assert group.is_last(2) == True
