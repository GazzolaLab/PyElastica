from elastica.modules.feature_group import FeatureGroupFIFO


def test_add_ids():
    feature_group = FeatureGroupFIFO()
    feature_group.append_id(1)
    feature_group.append_id(2)
    feature_group.append_id(3)

    assert feature_group._operator_ids == [id(1), id(2), id(3)]


def test_add_operators():
    feature_group = FeatureGroupFIFO()
    feature_group.append_id(1)
    feature_group.add_operators(1, [1, 2, 3])
    feature_group.append_id(2)
    feature_group.add_operators(2, [4, 5, 6])
    feature_group.append_id(3)
    feature_group.add_operators(3, [7, 8, 9])

    assert feature_group._operator_collection == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert feature_group._operator_ids == [id(1), id(2), id(3)]

    feature_group.append_id(4)
    feature_group.add_operators(4, [10, 11, 12])

    assert feature_group._operator_collection == [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12],
    ]
    assert feature_group._operator_ids == [id(1), id(2), id(3), id(4)]


def test_grouping():
    feature_group = FeatureGroupFIFO()
    feature_group.append_id(1)
    feature_group.add_operators(1, [1, 2, 3])
    feature_group.append_id(2)
    feature_group.add_operators(2, [4, 5, 6])
    feature_group.append_id(3)
    feature_group.add_operators(3, [7, 8, 9])

    assert list(feature_group) == [1, 2, 3, 4, 5, 6, 7, 8, 9]

    feature_group.append_id(4)
    feature_group.add_operators(4, [10, 11, 12])

    assert list(feature_group) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    feature_group.append_id(1)
    feature_group.add_operators(1, [13, 14, 15])

    assert list(feature_group) == [1, 2, 3, 13, 14, 15, 4, 5, 6, 7, 8, 9, 10, 11, 12]
