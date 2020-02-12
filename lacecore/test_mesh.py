import numpy as np
from . import shapes


def test_repr():
    assert repr(shapes.cube(np.zeros(3), 3.0)) == "lacecore.Mesh(num_v=8, num_f=12)"
