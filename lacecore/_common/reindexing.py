import numpy as np


def indices_of_original_elements_after_applying_mask(mask):
    """
    Given a mask that represents which of the original elements should be kept,
    produce an array containing the new indices of the original elements. Returns
    -1 as the index of the removed elements.
    """
    result = np.repeat(np.int(-1), len(mask))
    result[mask] = np.arange(np.count_nonzero(mask))
    return result
