import numpy as np
import pytest
from .reindexing import (
    indices_of_original_elements_after_applying_mask,
    reindex_vertices,
)
from .. import shapes


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


def test_reindex_vertices_error():
    with pytest.raises(ValueError, match="Expected new vertex indices to be unique, and range from 0 to 8"):
        reindex_vertices(
            shapes.cube(np.zeros(3), 3.0),
            np.array([0, 0, 1, 2, 3, 4, 5, 6])
        )
