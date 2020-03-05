import numpy as np
import pytest
from .._group_map import GroupMap
from .loader import ArityException, LoadException, load


@pytest.fixture
def write_tmp_mesh(tmp_path):
    def _write_tmp_mesh(mesh_contents, basename="example.obj"):
        test_mesh_path = str(tmp_path / basename)
        with open(test_mesh_path, "w") as f:
            f.write(mesh_contents)
        return test_mesh_path

    return _write_tmp_mesh


def test_loads_from_local_path_using_serializer_success_1():
    m = load("./examples/tinyobjloader/models/cube.obj")
    assert m.num_v == 8
    np.testing.assert_array_equal(m.v[0], np.array([0.0, 2.0, 2.0]))
    np.testing.assert_array_equal(m.f[0], np.array([0, 1, 2, 3]))
    np.testing.assert_array_equal(m.f[-1], np.array([1, 5, 6, 2]))
    assert m.num_f == 6
    assert isinstance(m.face_groups, GroupMap)
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
    # test for failure2
    with pytest.raises(LoadException):
        load("./thispathdoesnotexist")


def test_loads_from_local_path_using_serializer_failure_2():
    # test for failure
    with pytest.raises(ArityException):
        load("./examples/tinyobjloader/models/smoothing-group-two-squares.obj")


def test_triangulation_is_abc_acd(write_tmp_mesh):
    """
    There is some complex code in tinyobjloader which occasionally switches
    the axes of triangulation based on the vertex positions. This is
    undesirable in lacecore as it scrambles correspondence.
    """
    mesh_path = write_tmp_mesh(
        """
v 0 0 0
v 0 0 0
v 0 0 0
v 0 0 0
f 1 2 3 4
v 46.367584 82.676086 8.867414
v 46.524185 82.81955 8.825487
v 46.59864 83.086678 8.88121
v 46.461926 82.834091 8.953863
f 5 6 7 8
    """
    )

    # ABC + ACD
    expected_triangle_faces = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]])
    triangulated_mesh = load(mesh_path, triangulate=True)
    np.testing.assert_array_equal(triangulated_mesh.f, expected_triangle_faces)


def test_mesh_with_no_faces_has_empty_triangle_f(write_tmp_mesh):
    mesh_path = write_tmp_mesh(
        """
v 0.0 0.0 0.0
    """
    )

    mesh = load(mesh_path)
    np.testing.assert_array_equal(mesh.v, np.zeros((1, 3)))
    np.testing.assert_array_equal(mesh.f, np.zeros((0, 3)))
