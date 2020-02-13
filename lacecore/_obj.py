import numpy as np
from ._mesh import Mesh

from tinyobjloader import ObjReader, ObjReaderConfig

class LoadException(Exception):
    pass

class ArityException(Exception):
    pass

def load(mesh_path):
    reader = ObjReader()
    config = ObjReaderConfig()
    config.triangulate = False 
    success = reader.ParseFromFile(mesh_path, config)
    if success == False:
        raise LoadException(reader.Warning() or reader.Error())
    attrib = reader.GetAttrib()
    shapes = reader.GetShapes()
    tinyobj_vertices = attrib.numpy_vertices().reshape(-1, 3)
    aggregate_tinyobj_faces = []
    arities = set()
    for shape in shapes:
        vertices_per_face = shape.mesh.numpy_num_face_vertices()
        tinyobj_all_indices = shape.mesh.numpy_indices().reshape(-1, 3).transpose()[0]
        start = 0
        for v in vertices_per_face:
            arities.add(v)
            end = start + v
            aggregate_tinyobj_faces.append(np.array(tinyobj_all_indices[start:end]))
            start = end
    if all([a == 3 for a in arities]) or all([a == 4 for a in arities]):
        f = np.array(aggregate_tinyobj_faces)
        return Mesh(v=tinyobj_vertices, f=f)
    else:
        raise ArityException('OBJ Loader does not support mixed arities, or arities greater than 4 or less than 3')

