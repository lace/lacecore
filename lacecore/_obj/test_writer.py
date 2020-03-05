import io
import numpy as np
from .writer import write
from .. import shapes
from .._mesh import Mesh
from ..test_group_map import create_group_map


def test_write_empty_mesh():
    f = io.StringIO()

    mesh = Mesh(v=np.zeros((0, 3)), f=np.zeros((0, 4)))
    write(f, mesh)

    assert f.getvalue() == ""


def test_write_pointcloud():
    f = io.StringIO()

    mesh = Mesh(v=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), f=np.zeros((0, 4)))
    write(f, mesh)

    assert (
        f.getvalue()
        == """\
v 1.0 2.0 3.0
v 4.0 5.0 6.0
"""
    )


def test_write_mesh_without_face_groups():
    f = io.StringIO()

    mesh = Mesh(
        v=np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
        ),
        f=np.array([[0, 1, 2], [0, 1, 3]]),
    )
    write(f, mesh)

    assert (
        f.getvalue()
        == """\
v 1.0 2.0 3.0
v 4.0 5.0 6.0
v 7.0 8.0 9.0
v 10.0 11.0 12.0
f 1 2 3
f 1 2 4
"""
    )


def test_write_quads():
    f = io.StringIO()

    mesh = Mesh(
        v=np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
        ),
        f=np.array([[0, 1, 2, 3]]),
    )
    write(f, mesh)

    assert (
        f.getvalue()
        == """\
v 1.0 2.0 3.0
v 4.0 5.0 6.0
v 7.0 8.0 9.0
v 10.0 11.0 12.0
f 1 2 3 4
"""
    )

def test_write_mesh_with_face_groups():
    f = io.StringIO()

    mesh = shapes.cube(np.zeros(3), 3.0)
    mesh = Mesh(v=mesh.v, f=mesh.f, face_groups=create_group_map())
    write(f, mesh)

    assert (
        f.getvalue()
        == """\
v 0.0 0.0 0.0
v 3.0 0.0 0.0
v 3.0 0.0 3.0
v 0.0 0.0 3.0
v 0.0 3.0 0.0
v 3.0 3.0 0.0
v 3.0 3.0 3.0
v 0.0 3.0 3.0
g bottom top_and_bottom
f 1 2 3
f 1 3 4
g left_side sides
f 8 7 6
f 8 6 5
g front_side sides
f 5 6 2
f 5 2 1
g right_side sides
f 6 7 3
f 6 3 2
g back_side sides
f 7 8 4
f 7 4 3
g top top_and_bottom
f 4 8 5
f 4 5 1
"""
    )
