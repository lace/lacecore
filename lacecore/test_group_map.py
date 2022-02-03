from lacecore import GroupMap
import numpy as np
import pytest
from ._selection.test_selection_mixin import (
    create_group_map,
    cube_at_origin,
    groups as group_data,
)
from ._common.reindexing import reindex_faces
from ._mesh import Mesh

# Remove overlapping groups.
non_overlapping_group_data = dict(group_data)
del non_overlapping_group_data["sides"]
del non_overlapping_group_data["top_and_bottom"]


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
        "top",
        "back_side",
        "right_side",
        "front_side",
        "left_side",
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
        "top",
        "back_side",
        "right_side",
        "front_side",
        "left_side",
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
        groups.union("bottom", "sides"), np.array([True] * 2 + [False] * 2 + [True] * 8)
    )
    np.testing.assert_array_equal(
        groups.union("top_and_bottom", "top"),
        np.array([True] * 4 + [False] * 8),
    )
    np.testing.assert_array_equal(
        groups.union("left_side", "bottom"),
        np.array([True] * 2 + [False] * 8 + [True] * 2),
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


def test_to_dict():
    assert GroupMap.from_dict(group_data, 12).to_dict() == group_data


def test_mask_for_element():
    groups = create_group_map()
    np.testing.assert_array_equal(
        groups.mask_for_element(0), np.array([True] + [False] * 6 + [True, False])
    )
    np.testing.assert_array_equal(
        groups.mask_for_element(10),
        np.array([False] * 5 + [True] * 2 + [False] * 2),
    )


def test_group_names_for_element_mask():
    groups = create_group_map()
    assert groups.group_names_for_element_mask(groups.mask_for_element(0)) == [
        "bottom",
        "top_and_bottom",
    ]
    assert groups.group_names_for_element_mask(groups.mask_for_element(10)) == [
        "left_side",
        "sides",
    ]


def test_reindexed():
    groups = create_group_map()

    # Split each face in two.
    f_new_to_old = np.repeat(np.arange(groups.num_elements), 2)
    reindexed = groups.reindexed(f_new_to_old)

    expected_groups = {
        "bottom": [0, 1, 2, 3],
        "top": [4, 5, 6, 7],
        "back_side": [8, 9, 10, 11],
        "right_side": [12, 13, 14, 15],
        "front_side": [16, 17, 18, 19],
        "left_side": [20, 21, 22, 23],
        "sides": [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        "top_and_bottom": [0, 1, 2, 3, 4, 5, 6, 7],
        "empty": [],
    }

    for group_name in reindexed:
        np.testing.assert_array_equal(
            reindexed[group_name].nonzero()[0], expected_groups[group_name]
        )


def test_defragment():
    groups = GroupMap.from_dict(non_overlapping_group_data, 12)

    # Since these groups aren't fragmented, this preserves the original group
    # order and is a no-op.
    ordering = groups.defragment()
    np.testing.assert_array_equal(ordering, np.arange(12))

    # Specifying a different order should put the groups in that order.
    group_order = [
        "left_side",
        "right_side",
        "front_side",
        "back_side",
        "bottom",
        "top",
    ]
    ordering = groups.defragment(group_order=group_order)
    np.testing.assert_array_equal(
        ordering, np.array([8, 9, 10, 11, 6, 7, 2, 3, 4, 5, 0, 1])
    )


def test_defragment_and_reindex():
    # Prepare.
    original = Mesh(
        v=cube_at_origin.v,
        f=cube_at_origin.f,
        face_groups=GroupMap.from_dict(
            non_overlapping_group_data, cube_at_origin.num_f
        ),
    )

    # Act.
    defragmented = reindex_faces(
        mesh=original,
        ordering=original.face_groups.defragment(
            group_order=[
                "left_side",
                "right_side",
                "front_side",
                "back_side",
                "bottom",
                "top",
            ]
        ),
    )

    # Confidence check.
    np.testing.assert_array_equal(defragmented.v, original.v)
    assert np.all(defragmented.f == original.f) == False

    # Assert.
    np.testing.assert_array_equal(
        defragmented.v[defragmented.f[defragmented.face_groups["front_side"]]],
        original.v[original.f[original.face_groups["front_side"]]],
    )
    np.testing.assert_array_equal(
        defragmented.v[defragmented.f[defragmented.face_groups["top"]]],
        original.v[original.f[original.face_groups["top"]]],
    )


def test_defragment_errors():
    groups = create_group_map()

    with pytest.raises(
        ValueError,
        match=r"group_order is missing groups: back_side, bottom, front_side, right_side, sides, top, top_and_bottom",
    ):
        groups.defragment(group_order=["left_side"])

    with pytest.raises(
        ValueError,
        match=r"group_order contains unknown groups: foo",
    ):
        groups.defragment(
            group_order=[
                "bottom",
                "top",
                "back_side",
                "right_side",
                "front_side",
                "left_side",
                "sides",
                "top_and_bottom",
                "empty",
                "foo",
            ]
        )

    with pytest.raises(
        ValueError, match=r"Group \"sides\" overlaps with previous groups"
    ):
        groups.defragment()
