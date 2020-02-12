import numpy as np
from .reindexing import indices_of_original_elements_after_applying_mask


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
