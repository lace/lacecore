import numpy as np
import pytest
from ._group_map import GroupMap


def create_group_map():
    # Test with a cube with triangular faces.
    return GroupMap.from_dict(
        {
            "bottom": [0, 1],
            "left_side": [2, 3],
            "front_side": [4, 5],
            "right_side": [6, 7],
            "back_side": [8, 9],
            "top": [10, 11],
            "sides": [2, 3, 4, 5, 6, 7, 8, 9],
            "top_and_bottom": [0, 1, 10, 11],
            "empty": [],
        },
        12,
    )


def test_group_map_num_elements():
    groups = create_group_map()
    assert groups.num_elements == 12


def test_group_map_len():
    groups = create_group_map()
    assert len(groups) == 9


def test_group_map_get_item():
    groups = create_group_map()
    np.testing.assert_array_equal(groups["bottom"], np.array([True] * 2 + [False] * 10))
    np.testing.assert_array_equal(groups["bottom"].nonzero()[0], np.array([0, 1]))
    np.testing.assert_array_equal(groups["empty"], np.zeros(12))
    np.testing.assert_array_equal(groups["empty"].nonzero()[0], np.array([]))
    with pytest.raises(ValueError, match="assignment destination is read-only"):
        groups["empty"][3] = True
    with pytest.raises(KeyError, match="Unknown group: nope"):
        groups["nope"]


def test_group_map_iteration():
    groups = create_group_map()
    names = [name for name in groups]
    assert names == [
        "bottom",
        "left_side",
        "front_side",
        "right_side",
        "back_side",
        "top",
        "sides",
        "top_and_bottom",
        "empty",
    ]


def test_group_contains():
    groups = create_group_map()
    assert "front_side" in groups
    assert "nope" not in groups


def test_group_map_keys():
    groups = create_group_map()
    assert groups.keys() == [
        "bottom",
        "left_side",
        "front_side",
        "right_side",
        "back_side",
        "top",
        "sides",
        "top_and_bottom",
        "empty",
    ]


def test_group_map_union():
    groups = create_group_map()
    np.testing.assert_array_equal(
        groups.union("bottom"), np.array([True] * 2 + [False] * 10)
    )
    np.testing.assert_array_equal(
        groups.union("bottom", "sides"), np.array([True] * 10 + [False] * 2)
    )
    np.testing.assert_array_equal(
        groups.union("top_and_bottom", "top"),
        np.array([True] * 2 + [False] * 8 + [True] * 2),
    )
    np.testing.assert_array_equal(
        groups.union("left_side", "bottom"), np.array([True] * 4 + [False] * 8)
    )

    # Verify is writable.
    mask = groups.union("left_side", "bottom")
    mask[5] = True
    with pytest.raises(KeyError, match="Unknown groups: a, b, c"):
        groups.union("a", "left_side", "b", "c")

    with pytest.raises(ValueError, match="Group names must be strings"):
        groups.union(["a", "left_side", "b", "c"])


def test_group_map_indices_out_of_range():
    with pytest.raises(ValueError, match="Element indices should be less than 12"):
        GroupMap.from_dict({"too_big": range(20)}, 12)


def test_group_map_with_copy_true_makes_readonly_copy():
    masks = np.zeros((3, 12), dtype=np.bool)
    groups = GroupMap(
        num_elements=12, group_names=["a", "b", "c"], masks=masks, copy_masks=True
    )

    # Verify original is writable.
    masks[0][0] = True

    with pytest.raises(ValueError, match="assignment destination is read-only"):
        groups["a"][0] = True


def test_group_map_with_copy_false_makes_read_only():
    masks = np.zeros((3, 12), dtype=np.bool)
    GroupMap(
        num_elements=12, group_names=["a", "b", "c"], masks=masks, copy_masks=False
    )
    with pytest.raises(ValueError, match="assignment destination is read-only"):
        masks[0][0] = True


def test_invalid_num_elements_throws_error():
    with pytest.raises(
        ValueError, match="num_elements should be a non-negative integer"
    ):
        GroupMap(
            num_elements=-1,
            group_names=["a", "b", "c"],
            masks=np.zeros((3, 12), dtype=np.bool),
        )
    with pytest.raises(
        ValueError, match="num_elements should be a non-negative integer"
    ):
        GroupMap(
            num_elements="foo",
            group_names=["a", "b", "c"],
            masks=np.zeros((3, 12), dtype=np.bool),
        )


def test_invalid_group_name_throws_error():
    with pytest.raises(ValueError, match="group_names should be a list of strings"):
        GroupMap(
            num_elements=12,
            group_names=[1, 2, 3],
            masks=np.zeros((3, 12), dtype=np.bool),
        )


def test_invalid_mask_throws_error():
    with pytest.raises(ValueError, match="Expected masks to be a bool array"):
        GroupMap(
            num_elements=12,
            group_names=["a", "b", "c"],
            masks=np.zeros((3, 12), dtype=np.int),
        )
