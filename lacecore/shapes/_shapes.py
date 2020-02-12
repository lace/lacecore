from polliwog import shapes
from .._mesh import Mesh


__all__ = [
    "rectangular_prism",
    "cube",
    "triangular_prism",
    "rectangle",
]


def _mesh_from_shape_fn(shape_factory_fn, *args, **kwargs):
    vertices, faces = shape_factory_fn(
        *args, ret_unique_vertices_and_faces=True, **kwargs
    )
    return Mesh(v=vertices, f=faces)


def rectangular_prism(origin, size):
    return _mesh_from_shape_fn(
        shapes.create_rectangular_prism, origin=origin, size=size
    )


def cube(origin, size):
    return _mesh_from_shape_fn(shapes.create_cube, origin=origin, size=size)


def triangular_prism(p1, p2, p3, height):
    return _mesh_from_shape_fn(
        shapes.create_triangular_prism, p1=p1, p2=p2, p3=p3, height=height
    )


def rectangle():
    return _mesh_from_shape_fn(shapes.create_rectangle)
