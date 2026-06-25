import pytest
from awf.io.iaea_fetcher import (
    IaeaGammaLine, fetch_iaea_gamma_lines, load_iaea_gamma_lines_from_cache,
    merge_iaea_into_internal,
    _normalize_nuclide_name, _denormalize_nuclide_name, _cache_path, _parse_iaea_csv,
)
from pathlib import Path
import urllib.request
import tempfile
import os

def _csv():
    return ("energy,unc_e,intensity,unc_i,decay,p_energy,parent\n"
            "63.29,0.02,4.8,0.5,B-,0,234TH\n"
            "92.38,0.01,2.81,0.2,B-,0,234TH\n"
            "1001.03,0.03,0.837,0.05,B-,0,234TH\n")

class _FakeResp:
    def __init__(self, data): self._data = data
    def read(self): return self._data
    def __enter__(self): return self
    def __exit__(self, *a): return False

def test_parse_csv():
    lines = _parse_iaea_csv(_csv())
    assert len(lines) == 3
    assert lines[0].energy_keV == 63.29
    assert lines[0].intensity_pct == 4.8
    assert lines[0].energy_uncertainty_keV == 0.02
    assert lines[0].parent_nuclide == "234TH"
    assert lines[2].energy_keV == 1001.03

def test_parse_skips_empty_energy():
    csv_data = "energy,intensity,parent\n,5.0,234TH\n100.0,3.0,234TH\n"
    lines = _parse_iaea_csv(csv_data)
    assert len(lines) == 1
    assert lines[0].energy_keV == 100.0

def test_normalize_ok():
    assert _normalize_nuclide_name("Th-234") == "234th"
    assert _normalize_nuclide_name("234Th") == "234th"
    assert _normalize_nuclide_name("234TH") == "234th"
    assert _normalize_nuclide_name("th234") == "234th"

@pytest.mark.parametrize("bad", ["?evil=1", "../etc/passwd", "x"*33, "", "Th--/"])
def test_normalize_fail_loud(bad):
    with pytest.raises(ValueError):
        _normalize_nuclide_name(bad)

def test_denormalize():
    assert _denormalize_nuclide_name("234TH") == "Th-234"
    assert _denormalize_nuclide_name("234th") == "Th-234"

def test_cache_path():
    p = _cache_path(Path("/tmp/c"), "Th-234", "g")
    assert p.name == "234th_g.csv"

def test_cache_roundtrip(tmp_path):
    cache_file = tmp_path / "234th_g.csv"
    cache_file.write_text(_csv(), encoding="utf-8")
    lines = load_iaea_gamma_lines_from_cache("Th-234", cache_dir=tmp_path)
    assert lines is not None
    assert len(lines) == 3
    assert load_iaea_gamma_lines_from_cache("Cs-137", cache_dir=tmp_path) is None

def test_fetch_uses_cache_no_network(tmp_path, monkeypatch):
    cache_file = tmp_path / "234th_g.csv"
    cache_file.write_text(_csv(), encoding="utf-8")
    
    def _boom(*a, **k):
        raise AssertionError("network called")
    monkeypatch.setattr("urllib.request.urlopen", _boom)
    
    lines = fetch_iaea_gamma_lines("Th-234", cache_dir=tmp_path, force_refresh=False)
    assert len(lines) == 3

def test_fetch_writes_cache_atomically(tmp_path, monkeypatch):
    def fake_urlopen(req, timeout=None):
        return _FakeResp(b"energy,intensity,parent\n100.0,5.0,234TH\n")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    
    lines = fetch_iaea_gamma_lines("Th-234", cache_dir=tmp_path, force_refresh=True)
    assert len(lines) == 1
    assert lines[0].energy_keV == 100.0
    
    cache_file = tmp_path / "234th_g.csv"
    assert cache_file.exists()
    assert cache_file.read_text().startswith("energy")
    
    temp_file = tmp_path / "234th_g.csv.tmp"
    assert not temp_file.exists()

def test_merge_basic():
    lines = _parse_iaea_csv(_csv())
    d = merge_iaea_into_internal(lines, "Th-234")
    assert "lines" in d
    assert len(d["lines"]) == 3
    assert d["lines"][0][0] == 63.29
    assert d["lines"][2][0] == 1001.03
    assert all(len(line) == 3 for line in d["lines"])

def test_merge_min_intensity():
    lines = _parse_iaea_csv(_csv())
    d = merge_iaea_into_internal(lines, "Th-234", min_intensity_pct=1.0)
    assert len(d["lines"]) == 2
    assert all(line[1] >= 1.0 for line in d["lines"])

def test_merge_excited_parent_excluded():
    lines = [
        IaeaGammaLine(50.0, None, 10.0, None, "234TH", "B-", 0.0),
        IaeaGammaLine(60.0, None, 20.0, None, "234TH", "B-", 74.0)
    ]
    d = merge_iaea_into_internal(lines, "Th-234", only_ground_state_parent=True)
    assert len(d["lines"]) == 1
    assert d["lines"][0][0] == 50.0
