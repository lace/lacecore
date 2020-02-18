from ._obj import load, LoadException, ArityException
import pytest

def test_loads_from_local_path_using_serializer_success_1():
    m = load("./examples/tinyobjloader/models/cube.obj")
    assert(m.num_v == 8)
    assert(m.num_f == 6)

def test_loads_from_local_path_using_serializer_failure_1():
    #test for failure2
    with pytest.raises(LoadException):
        m = load("./thispathdoesnotexist")

def test_loads_from_local_path_using_serializer_failure_2():
    #test for failure
    with pytest.raises(ArityException):
        m = load("./examples/tinyobjloader/models/smoothing-group-two-squares.obj")

