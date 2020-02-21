from ._obj import load, LoadException, ArityException
from ._group_map import GroupMap
import numpy as np
import pytest

def test_loads_from_local_path_using_serializer_success_1():
    m = load("./examples/tinyobjloader/models/cube.obj")
    assert(m.num_v == 8)
    np.testing.assert_array_equal(m.v[0], np.array([0.0, 2.0, 2.0]))
    np.testing.assert_array_equal(m.f[0], np.array([0,1,2,3]))
    np.testing.assert_array_equal(m.f[-1], np.array([1,5,6,2]))
    assert(m.num_f == 6)
    assert(isinstance(m.face_groups, GroupMap))
    assert m.face_groups.keys() == [
        "front",
        "cube",
        "back",
        "right",
        "top",
        "left",
        "bottom",
    ]

def test_loads_from_local_path_using_serializer_failure_1():
    #test for failure2
    with pytest.raises(LoadException):
        m = load("./thispathdoesnotexist")

def test_loads_from_local_path_using_serializer_failure_2():
    #test for failure
    with pytest.raises(ArityException):
        m = load("./examples/tinyobjloader/models/smoothing-group-two-squares.obj")

