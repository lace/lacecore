import numpy as np
from . import _shapes as shapes


def test_cube():
    mesh = shapes.cube(origin=np.array([3.0, 4.0, 5.0]), size=2.0)
    assert mesh.num_v == 8
    assert mesh.num_f == 12


def test_rectangular_prism():
    mesh = shapes.rectangular_prism(
        origin=np.array([3.0, 4.0, 5.0]), size=np.array([2.0, 10.0, 20.0])
    )
    assert mesh.num_v == 8
    assert mesh.num_f == 12


def test_triangular_prism():
    mesh = shapes.triangular_prism(
        p1=np.array([3.0, 0.0, 0.0]),
        p2=np.array([0.0, 3.0, 0.0]),
        p3=np.array([0.0, 0.0, 3.0]),
        height=1.0,
    )
    assert mesh.num_v == 6
    assert mesh.num_f == 8


def test_rectangle():
    mesh = shapes.rectangle()
    assert mesh.num_v == 4
    assert mesh.num_f == 2
