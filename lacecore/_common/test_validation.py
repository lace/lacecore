import numpy as np
import pytest
from .validation import check_arity, check_indices


def test_check_arity_valid():
    check_arity(np.arange(24).reshape(-1, 3))
    check_arity(np.arange(24).reshape(-1, 4))


def test_check_arity_invalid():
    with pytest.raises(ValueError, match="Expected 3 or 4 vertices per face"):
        check_arity(np.arange(24).reshape(-1, 6))


def test_check_indices_valid():
    check_indices(np.repeat(np.arange(8), 3).reshape(-1, 3), 8, "f")
    check_indices(np.zeros((0, 3)), 8, "f")


def test_check_indices_invalid():
    with pytest.raises(ValueError, match="Expected indices in f to be less than 8"):
        check_indices(np.arange(24).reshape(-1, 3), 8, "f")
