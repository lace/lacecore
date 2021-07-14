from lacecore import ArityException, GroupMap, LoadException, load_obj, load_obj_string
import numpy as np
import pytest


@pytest.fixture
def write_tmp_mesh(tmp_path):
    def _write_tmp_mesh(mesh_contents, basename="example.obj"):
        test_mesh_path = str(tmp_path / basename)
        with open(test_mesh_path, "w") as f:
            f.write(mesh_contents)
        return test_mesh_path

    return _write_tmp_mesh


def assert_is_cube_mesh(mesh):
    assert mesh.num_v == 8
    np.testing.assert_array_equal(mesh.v[0], np.array([0.0, 2.0, 2.0]))
    np.testing.assert_array_equal(mesh.f[0], np.array([0, 1, 2, 3]))
    np.testing.assert_array_equal(mesh.f[-1], np.array([1, 5, 6, 2]))
    assert mesh.num_f == 6
    assert isinstance(mesh.face_groups, GroupMap)
    assert mesh.face_groups.keys() == [
        "front",
        "cube",
        "back",
        "right",
        "top",
        "left",
        "bottom",
    ]
    assert np.issubdtype(mesh.f.dtype, np.integer)


def test_loads_from_local_path():
    mesh = load_obj("./examples/tinyobjloader/models/cube.obj")
    assert_is_cube_mesh(mesh)


def test_loads_from_string():
    with open("./examples/tinyobjloader/models/cube.obj", "r") as f:
        contents = f.read()
    mesh = load_obj_string(contents)
    assert_is_cube_mesh(mesh)


def test_loads_from_string_with_error():
    contents = """
    f 0 0 0
    """
    with pytest.raises(LoadException, match="^Failed parse `f' line"):
        load_obj_string(contents)


def test_loads_from_local_path_with_nonexistent_file():
    with pytest.raises(
        LoadException, match=r"^Cannot open file \[./thispathdoesnotexist\]"
    ):
        load_obj("./thispathdoesnotexist")


def test_loads_from_local_path_with_mixed_arities():
    with pytest.raises(ArityException):
        load_obj("./examples/tinyobjloader/models/smoothing-group-two-squares.obj")


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
    mesh = load_obj(mesh_path, triangulate=True)
    np.testing.assert_array_equal(mesh.f, expected_triangle_faces)
    assert np.issubdtype(mesh.f.dtype, np.integer)


def test_mesh_with_mixed_tris_and_quads_returns_expected(write_tmp_mesh):
    mesh_path = write_tmp_mesh(
        """
v 0 1 1
v 0 2 2
v 0 3 3
v 0 4 4
v 0 5 5
f 1 2 3 4
f 1 4 5
    """
    )

    expected_triangle_faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4]])
    mesh = load_obj(mesh_path, triangulate=True)
    np.testing.assert_array_equal(mesh.f, expected_triangle_faces)
    assert np.issubdtype(mesh.f.dtype, np.integer)


def test_mesh_with_no_faces_has_empty_triangle_f(write_tmp_mesh):
    mesh_path = write_tmp_mesh(
        """
v 0.0 0.0 0.0
    """
    )

    mesh = load_obj(mesh_path)
    np.testing.assert_array_equal(mesh.v, np.zeros((1, 3)))
    np.testing.assert_array_equal(mesh.f, np.zeros((0, 3)))
    assert np.issubdtype(mesh.f.dtype, np.integer)


def test_mesh_with_ngons_raises_expected_error(write_tmp_mesh):
    mesh_path = write_tmp_mesh(
        """
v 0 0 0
v 0 0 0
v 0 0 0
v 0 0 0
v 0 0 0
f 1 2 3 4 5
    """
    )

    with pytest.raises(
        ArityException,
        match="OBJ Loader does not support arities greater than 4 or less than 3",
    ):
        load_obj(mesh_path)
