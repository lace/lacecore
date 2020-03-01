from collections import OrderedDict
import numpy as np

from ._group_map import GroupMap
from ._mesh import Mesh

ERROR_MESSAGE = "tinyobjloader library has not been installed. You will not be able to load OBJ files"

try:
    from tinyobjloader import ObjReader, ObjReaderConfig
except Exception:  # pragma: no cover
    ObjReader = None
    ObjReaderConfig = None
    import sys

    print("WARNING: %s" % ERROR_MESSAGE, file=sys.stderr)


class LoadException(Exception):
    pass


class ArityException(Exception):
    pass


def load(mesh_path, triangulate=False):
    """
    Load a `Mesh` from a path to an OBJ file.

    Args:
        mesh_path (str): A path to an OBJ file
        triangulate (bool): A flag that indicates whether to triangulate the mesh on load.

    Returns:
        lacecore.Mesh: A `Mesh` instance
    """
    if ObjReader is None:  # pragma: no cover
        raise Exception(ERROR_MESSAGE)
    reader = ObjReader()
    config = ObjReaderConfig()
    config.triangulate = triangulate
    success = reader.ParseFromFile(mesh_path, config)
    if success is False:
        raise LoadException(reader.Warning() or reader.Error())
    attrib = reader.GetAttrib()
    shapes = reader.GetShapes()
    tinyobj_vertices = attrib.numpy_vertices().reshape(-1, 3)
    all_vertices_per_face = np.concatenate(
        [shape.mesh.numpy_num_face_vertices() for shape in shapes]
    )
    first_arity = all_vertices_per_face[0]
    if (
        np.any(all_vertices_per_face != first_arity)
        or np.any(all_vertices_per_face > 4)
        or np.any(all_vertices_per_face < 3)
    ):
        raise ArityException(
            "OBJ Loader does not support mixed arities, or arities greater than 4 or less than 3"
        )

    segm = OrderedDict()
    all_faces = None

    for shape in shapes:
        tinyobj_all_indices = shape.mesh.numpy_indices().reshape(-1, 3)[:, 0]
        faces = tinyobj_all_indices.reshape(-1, first_arity)
        start = len(all_faces) if all_faces is not None else 0
        end = start + len(faces)
        all_faces = faces if all_faces is None else np.concatenate((all_faces, faces))

        group_names = shape.name.split()
        for name in group_names:
            if name not in segm:
                segm[name] = []
            segm[name] = segm[name] + list(range(start, end))

    group_map = GroupMap.from_dict(segm, len(all_faces))
    return Mesh(v=tinyobj_vertices, f=all_faces, face_groups=group_map)
