import numpy as np
import pytest
from .reindexing import (
    indices_of_original_elements_after_applying_mask,
    reindex_faces,
    reindex_vertices,
)
from .. import shapes
from .._group_map import GroupMap
from .._mesh import Mesh


def create_cube_with_face_group():
    cube = shapes.cube(np.zeros(3), 3.0)
    selection = np.array([0, 1, 5, 9])
    return Mesh(
        v=cube.v,
        f=cube.f,
        face_groups=GroupMap.from_dict(
            {"selection": selection}, num_elements=cube.num_f
        ),
    )


def assert_has_same_rows(first, second):
    assert first.shape == second.shape
    assert np.all([row in first for row in second])
    assert np.all([row in second for row in first])


def test_reindexing_mask():
    mask = np.array(
        [False, False, True, True, False, True, False, False, True, False, False, False]
    )
    np.testing.assert_array_equal(
        indices_of_original_elements_after_applying_mask(mask),
        np.array([-1, -1, 0, 1, -1, 2, -1, -1, 3, -1, -1, -1]),
    )


def test_reindexing_mask_noop():
    np.testing.assert_array_equal(
        indices_of_original_elements_after_applying_mask(np.repeat(True, 12)),
        np.arange(12),
    )


def test_reindexing_mask_remove_all():
    np.testing.assert_array_equal(
        indices_of_original_elements_after_applying_mask(np.repeat(False, 12)),
        np.repeat(-1, 12),
    )


def test_reindex_vertices():
    cube = shapes.cube(np.zeros(3), 3.0)

    ordering = np.random.permutation(8)
    reindexed_cube = reindex_vertices(cube, ordering)

    np.testing.assert_array_equal(reindexed_cube.v, cube.v[ordering])
    np.testing.assert_array_equal(reindexed_cube.v[reindexed_cube.f], cube.v[cube.f])


def test_reindex_vertices_with_face_groups():
    cube = create_cube_with_face_group()

    reindexed_cube = reindex_vertices(cube, np.random.permutation(8))

    assert_has_same_rows(
        reindexed_cube.v[reindexed_cube.f[reindexed_cube.face_groups["selection"]]],
        cube.v[cube.f[cube.face_groups["selection"]]],
    )


def test_reindex_vertices_error():
    with pytest.raises(
        ValueError,
        match="Expected new vertex indices to be unique, and range from 0 to 7",
    ):
        reindex_vertices(
            shapes.cube(np.zeros(3), 3.0), np.array([0, 0, 1, 2, 3, 4, 5, 6])
        )


def test_reindex_faces():
    cube = shapes.cube(np.zeros(3), 3.0)

    ordering = np.random.permutation(12)
    reindexed_cube = reindex_faces(cube, ordering)

    np.testing.assert_array_equal(reindexed_cube.f, cube.f[ordering])
    np.testing.assert_array_equal(reindexed_cube.v, cube.v)


def test_reindex_faces_with_face_groups():
    cube = create_cube_with_face_group()

    reindexed_cube = reindex_faces(cube, np.random.permutation(12))

    assert_has_same_rows(
        reindexed_cube.f[reindexed_cube.face_groups["selection"]],
        cube.f[cube.face_groups["selection"]],
    )


def test_reindex_faces_error():
    with pytest.raises(
        ValueError,
        match="Expected new face indices to be unique, and range from 0 to 11",
    ):
        reindex_faces(
            shapes.cube(np.zeros(3), 3.0),
            np.array([0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        )
