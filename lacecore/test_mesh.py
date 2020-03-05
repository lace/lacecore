import numpy as np
from . import shapes


def test_repr():
    assert repr(shapes.cube(np.zeros(3), 3.0)) == "lacecore.Mesh(num_v=8, num_f=12)"


def test_is_tri():
    assert shapes.cube(np.zeros(3), 3.0).is_tri is True


def test_is_quad():
    assert shapes.cube(np.zeros(3), 3.0).is_quad is False


def test_write_obj(tmp_path):
    obj_path = str(tmp_path / "cube.obj")
    shapes.cube(np.zeros(3), 3.0).write_obj(obj_path)

    with open(obj_path, "r") as f:
        obj_contents = f.read()

    assert obj_contents.count("v ") == 8
    assert obj_contents.count("f ") == 12
