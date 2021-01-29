# Changelog

## 0.6.0

- Add `faces_triangulated()` function.
- Correctly preserve groups when reindexing faces.
- Upgrade to tinyobjloader 2.0.0rc8.

## 0.5.0

- Add `load_obj_string()` function.
- Upgrade to tinyobjloader 2.0.0rc7.

## 0.4.1

- Ensure faces have integral dtype.

## 0.4.0

- obj: Add support for triangulating mixed arities.
- Upgrade to tinyobjloader 2.0.0rc6.
- Upgrade to polliwog 1.0.0b10.

## 0.3.0

- Upgrade to polliwog 1.0.0b8.
- Remove `lacecore.shapes.rectangle()`.

## 0.2.0

### New features

- Optional OBJ loading with `pip install lacecore[obj]`.
- Add transform methods.
- Add group selection.
- Add `apex()` method.
- Add `vertex_centroid` and `bounding_box` properties.

### Bug fixes

- Pass through `face_groups` when returning a new mesh.
- Reject invalid `point` from selection methods.
- Avoid scrambling correspondence with inconsistent triangulation.


## 0.1.0

Initial release.
