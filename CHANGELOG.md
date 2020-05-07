# Changelog

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
