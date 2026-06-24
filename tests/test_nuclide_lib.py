import json
import pytest
from awf.io.nuclide_lib import (
    _cfloat, _cint, GammaLine, Nuclide,
    parse_lsrm_lib, to_json_obj, load_nuclides_json, default_library,
)

def _by_name(nuclides):
    """Загрузчики возвращают список Nuclide; для удобства индексируем по имени."""
    return {n.name: n for n in nuclides}

def _write_lib(tmp_path):
    text = '''<?xml version="1.0" encoding="windows-1251"?>
<Library library_type="gamma" library_version="2.0" database_version="26">
  <Comment></Comment>
  <Nuclide name="Am-241" half_life_value="432,6" half_life_unit="year" gamma_constant="0,118348" atomic_mass="241">
    <Line energy="59,5409" d_energy="0,0001" intensity="35,9" d_intensity="0,4"/>
  </Nuclide>
  <Nuclide name="Test-2" half_life_value="10" half_life_unit="day">
    <Line energy="100,0" intensity="50,0" line_type="X" used="false"/>
    <Line energy="200,0" intensity="90,0"/>
    <Line energy="300,0" intensity="5,0"/>
    <Line d_energy="0,1" intensity="1,0"/>
  </Nuclide>
</Library>'''
    path = tmp_path / "lib.lib"
    path.write_bytes(text.encode("windows-1251"))
    return str(path)

def test_cfloat():
    assert _cfloat("432,6") == 432.6
    assert _cfloat("1248000000") == 1248000000.0
    assert _cfloat(None) is None
    assert _cfloat("") is None

def test_cint():
    assert _cint("241") == 241
    assert _cint(None) is None
    assert _cint("") is None

def test_parse_counts(tmp_path):
    lib = _by_name(parse_lsrm_lib(_write_lib(tmp_path)))
    assert len(lib) == 2
    assert len(lib["Am-241"].lines) == 1
    assert len(lib["Test-2"].lines) == 3

def test_parse_values(tmp_path):
    lib = _by_name(parse_lsrm_lib(_write_lib(tmp_path)))
    nuc = lib["Am-241"]
    assert nuc.half_life_value == 432.6
    assert nuc.half_life_unit == "year"
    assert nuc.gamma_constant == 0.118348
    assert nuc.atomic_mass == 241
    assert nuc.lines[0].energy == 59.5409
    assert nuc.lines[0].intensity == 35.9
    assert nuc.lines[0].d_energy == 0.0001
    assert nuc.lines[0].d_intensity == 0.4

def test_used_flag(tmp_path):
    lib = _by_name(parse_lsrm_lib(_write_lib(tmp_path)))
    line1 = lib["Test-2"].lines[0]
    line2 = lib["Test-2"].lines[1]
    assert line1.used is False
    assert line2.used is True

def test_line_type(tmp_path):
    lib = _by_name(parse_lsrm_lib(_write_lib(tmp_path)))
    line1 = lib["Test-2"].lines[0]
    line2 = lib["Test-2"].lines[1]
    assert line1.line_type == "X"
    assert line2.line_type is None

def test_missing_optional(tmp_path):
    lib = _by_name(parse_lsrm_lib(_write_lib(tmp_path)))
    nuc = lib["Test-2"]
    assert nuc.gamma_constant is None
    assert nuc.atomic_mass is None

def test_half_life_seconds():
    assert Nuclide("X", 1.0, "year").half_life_seconds() == 365.25 * 86400
    assert Nuclide("X", 2.0, "day").half_life_seconds() == 2 * 86400
    assert Nuclide("X", None, "year").half_life_seconds() is None
    assert Nuclide("X", 1.0, None).half_life_seconds() is None

def test_major_lines():
    n = Nuclide("X", lines=(
        GammaLine(100.0, 50.0, used=False),
        GammaLine(200.0, 90.0),
        GammaLine(300.0, 5.0),
    ))
    ml = n.major_lines(only_used=True)
    assert [l.energy for l in ml] == [200.0, 300.0]
    ml2 = n.major_lines(min_intensity=10.0, only_used=False)
    assert [l.energy for l in ml2] == [200.0, 100.0]

def test_json_roundtrip(tmp_path):
    src = parse_lsrm_lib(_write_lib(tmp_path))
    obj = to_json_obj(src, provenance={"k": "v"})
    jsonpath = tmp_path / "test.json"
    jsonpath.write_text(json.dumps(obj), encoding="utf-8")
    back = _by_name(load_nuclides_json(jsonpath))
    assert len(back) == 2
    assert back["Am-241"].name == "Am-241"
    assert len(back["Am-241"].lines) == 1
    assert back["Am-241"].lines[0].energy == 59.5409
    assert back["Am-241"].lines[0].intensity == 35.9
    assert back["Am-241"].lines[0].used is True
    assert back["Test-2"].name == "Test-2"
    assert len(back["Test-2"].lines) == 3
    assert obj["_provenance"] == {"k": "v"}
    assert "nuclides" in obj
    assert len(obj["nuclides"]) == 2

def test_default_library_present():
    lib = _by_name(default_library())
    assert len(lib) >= 1
    assert "Cs-137" in lib
    cs = lib["Cs-137"]
    assert abs(cs.lines[0].energy - 661.657) < 0.01
