from click.testing import CliRunner
from lacecore import load_obj, shapes
import numpy as np

from .convert_units import convert_units


def test_convert_units_cli(tmp_path, tmpdir) -> None:
    obj_path_m = str(tmp_path / "cube.obj")
    shapes.cube(np.zeros(3), 3.0).write_obj(obj_path_m)

    runner = CliRunner()

    with tmpdir.as_cwd():
        result = runner.invoke(convert_units, ["m", "cm", obj_path_m])

    assert result.exit_code == 0
    assert (
        result.stderr == f"Converting {obj_path_m} from m to cm\n  Wrote cube_cm.obj\n"
    )

    mesh = load_obj(str(tmpdir / "cube_cm.obj"))
    assert mesh.bounding_box.width == 300.0


def test_convert_units_cli_with_outdir(tmp_path, tmpdir) -> None:
    obj_path_m = str(tmp_path / "cube.obj")
    shapes.cube(np.zeros(3), 3.0).write_obj(obj_path_m)

    runner = CliRunner()
    result = runner.invoke(
        convert_units, ["--outdir", str(tmpdir), "m", "cm", obj_path_m]
    )
    assert result.exit_code == 0

    mesh = load_obj(str(tmp_path / "cube.obj"))
    assert mesh.bounding_box.width == 300.0
