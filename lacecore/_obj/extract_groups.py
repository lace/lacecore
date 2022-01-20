from lacecore._obj.loader import create_reader_and_config, _get_arity, LoadException


def read_groups(reader):
    shapes = reader.GetShapes()

    if _get_arity(shapes) != 3:
        raise ValueError("Only supported for triangulated meshes")

    groups = []

    for shape in shapes:
        print(shape.name)
        these_face_indices = shape.mesh.numpy_indices().reshape(-1, 3)[:, 0]
        groups.append({"name": shape.name, "num_faces": len(these_face_indices)})

    return groups


def extract_group_slices(mesh_path):
    reader, config = create_reader_and_config()
    success = reader.ParseFromFile(mesh_path, config)
    if not success:
        raise LoadException(reader.Warning() or reader.Error())
    return read_groups(reader)


def main():
    import json

    print(json.dumps(extract_group_slices("examples/tinyobjloader/models/cube.obj")))

if __name__ == "__main__":
    main()
