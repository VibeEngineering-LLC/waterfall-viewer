"""
Fetcher for IAEA LiveChart Decay Radiation API.

The IAEA Nuclear Data Section maintains the authoritative ENSDF
(Evaluated Nuclear Structure Data File) database. Their public API
exposes ENSDF data as CSV downloads:

    https://www-nds.iaea.org/relnsd/v1/data?fields=decay_rads&nuclides=<n>&rad_types=g

  • `fields=decay_rads` — emitted radiations (γ, X-ray, α, β±, e⁻)
  • `nuclides=234th`     — parent nuclide name (lowercase mass-symbol)
  • `rad_types=g`        — gamma rays specifically

This module fetches ENSDF γ-ray data for nuclides missing from the
bundled internal nuclide library. Users invoke it explicitly when
adding nuclides to their working library; it is NOT called
automatically by identification logic (deterministic, reproducible
behaviour requires user awareness of which library entries come
from which source).

Citation requirement: if data fetched via this module is used in
published results, cite ENSDF and the IAEA LiveChart of Nuclides
per their guidance: https://www-nds.iaea.org/

Local caching: every fetch writes the raw CSV response into a local
cache directory keyed by nuclide name + rad_type, so subsequent
runs are network-free and reproducible.

Reference: https://www-nds.iaea.org/relnsd/vcharthtml/api_v0_guide.html
(API v0 and v1 endpoints are equivalent at the time of writing.)
"""

from __future__ import annotations

import csv
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

# Standard library only — no network here yet. Network access is
# explicit per-call via urllib.request.


# IAEA LiveChart service URL
IAEA_API_URL = "https://www-nds.iaea.org/relnsd/v1/data"

# Default local cache directory
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "iaea_cache"


@dataclass(frozen=True)
class IaeaGammaLine:
    """One γ-line from IAEA decay_rads CSV."""
    energy_keV: float
    energy_uncertainty_keV: Optional[float]
    intensity_pct: float
    intensity_uncertainty_pct: Optional[float]
    parent_nuclide: str            # "234TH" → "Th-234"
    decay_mode: Optional[str]      # B-, EC, A, etc.
    parent_energy_keV: Optional[float]  # parent level energy (0 = g.s.)


#: SEC-02 hardening. Longest real nuclide name (e.g. `Md-258m`) is ~7
#: chars; anything beyond 32 chars before normalisation is either a typo
#: or adversarial input. A bounded check guards against URL-injection
#: payloads where the attacker embeds query parameters or path-traversal
#: into the nuclide field interpolated at `_build_url:103`.
_MAX_NUCLIDE_NAME_LEN = 32


def _normalize_nuclide_name(name: str) -> str:
    """Convert 'Th-234' / 'th234' / '234TH' to IAEA's canonical form '234th'.

    SEC-02 (security): raises ``ValueError`` on non-whitelisted input
    instead of silently returning the unsanitised lowercase string.
    Previously a fall-through ``return s`` allowed adversary-supplied
    characters (``?``, ``/``, ``%``, ``..``) to reach the URL builder
    and cache-path builder unmodified, enabling URL parameter injection
    and cache-dir escape. Now both callers fail-loud on invalid input.
    """
    # Defensive length cap. Fail-loud per SEC-02 directive: no silent
    # truncation, no fallback. 32 is generous (real names ≤ 7 chars).
    if not isinstance(name, str) or len(name) > _MAX_NUCLIDE_NAME_LEN:
        raise ValueError(f"Invalid nuclide name: {name!r}")
    s = name.strip().replace("-", "").lower()
    if not s:
        raise ValueError(f"Invalid nuclide name: {name!r}")
    # Split letters and digits
    import re
    m = re.match(r"^([a-z]+)(\d+)([a-z]*)$", s)
    if m:
        sym, num, meta = m.groups()
        return f"{num}{sym}{meta}"
    m = re.match(r"^(\d+)([a-z]+)([a-z]*)$", s)
    if m:
        num, sym, meta = m.groups()
        return f"{num}{sym}{meta}"
    # SEC-02: fail-loud instead of `return s`. The silent fall-through
    # previously allowed URL-injection / path-traversal characters
    # (`?evil=`, `../etc/passwd`) to reach `_build_url` and `_cache_path`
    # unsanitised. Callers now propagate this ValueError as a clear
    # operator-facing error.
    raise ValueError(f"Invalid nuclide name: {name!r}")


def _denormalize_nuclide_name(api_name: str) -> str:
    """Convert IAEA's '234TH' or '234th' back to 'Th-234' display form."""
    import re
    s = api_name.strip()
    m = re.match(r"^(\d+)([A-Za-z]+)([a-z]*)$", s)
    if m:
        num, sym, meta = m.groups()
        sym = sym.capitalize()
        suffix = meta if meta else ""
        return f"{sym}-{num}{suffix}"
    return s


def _cache_path(cache_dir: Path, nuclide: str, rad_type: str) -> Path:
    """Path to cached CSV for one nuclide/rad_type pair."""
    nuc_norm = _normalize_nuclide_name(nuclide)
    return cache_dir / f"{nuc_norm}_{rad_type}.csv"


def _build_url(nuclide: str, rad_type: str = "g") -> str:
    """Build the IAEA API URL for one decay_rads query."""
    nuc_norm = _normalize_nuclide_name(nuclide)
    return (f"{IAEA_API_URL}?fields=decay_rads&nuclides={nuc_norm}"
            f"&rad_types={rad_type}")


def _parse_iaea_csv(csv_text: str) -> List[IaeaGammaLine]:
    """
    Parse the IAEA decay_rads CSV response.

    Real CSV columns (as of 2025) for fields=decay_rads&rad_types=g
    typically include: energy, unc_e, intensity, unc_i, type, p_energy,
    daughter, decay, parent, parent_z, parent_n, etc. Column names
    have changed in minor revisions, so we look them up by name rather
    than positional index.

    Returns:
        list of IaeaGammaLine (only successfully parsed γ rows).
    """
    reader = csv.DictReader(io.StringIO(csv_text))
    lines = []
    for row in reader:
        # Energy field — has been called "energy" or "energy_kev"
        E_str = (row.get("energy") or row.get("energy_kev")
                 or row.get("e_g") or row.get("e_lvl_g") or "")
        if not E_str.strip():
            continue
        try:
            E = float(E_str)
        except (ValueError, TypeError):
            continue

        dE_str = (row.get("unc_e") or row.get("d_energy") or "").strip()
        try:
            dE = float(dE_str) if dE_str else None
        except ValueError:
            dE = None

        I_str = (row.get("intensity") or row.get("intensity_g") or
                 row.get("i_g") or "").strip()
        try:
            I = float(I_str) if I_str else 0.0
        except ValueError:
            I = 0.0

        dI_str = (row.get("unc_i") or row.get("d_intensity") or "").strip()
        try:
            dI = float(dI_str) if dI_str else None
        except ValueError:
            dI = None

        parent_field = (row.get("parent") or row.get("p_symbol") or "").strip()
        decay_mode = (row.get("decay") or row.get("decay_mode")
                      or "").strip() or None

        p_E_str = (row.get("p_energy") or row.get("parent_energy")
                   or "").strip()
        try:
            p_E = float(p_E_str) if p_E_str else None
        except ValueError:
            p_E = None

        lines.append(IaeaGammaLine(
            energy_keV=E,
            energy_uncertainty_keV=dE,
            intensity_pct=I,
            intensity_uncertainty_pct=dI,
            parent_nuclide=parent_field,
            decay_mode=decay_mode,
            parent_energy_keV=p_E,
        ))
    return lines


def load_iaea_gamma_lines_from_cache(
    nuclide: str,
    *,
    cache_dir = DEFAULT_CACHE_DIR,
    rad_type: str = "g",
) -> Optional[List[IaeaGammaLine]]:
    """
    Read previously-fetched IAEA data from local cache.

    Args:
        nuclide: nuclide name (any common form: "Th-234", "234Th", "th234")
        cache_dir: where cached CSV files live
        rad_type: "g" for γ (default), "x" for X-ray, "bm"/"bp" for β

    Returns:
        list of IaeaGammaLine, or None if not cached.
    """
    cache_dir = Path(cache_dir)
    path = _cache_path(cache_dir, nuclide, rad_type)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        csv_text = f.read()
    if not csv_text.strip():
        return None
    return _parse_iaea_csv(csv_text)


def fetch_iaea_gamma_lines(
    nuclide: str,
    *,
    cache_dir = DEFAULT_CACHE_DIR,
    rad_type: str = "g",
    force_refresh: bool = False,
    timeout_seconds: float = 30.0,
) -> List[IaeaGammaLine]:
    """
    Fetch γ-radiation data for a nuclide from IAEA LiveChart API.

    First checks local cache. If absent (or force_refresh=True),
    downloads from IAEA, writes to cache, and parses.

    Args:
        nuclide: nuclide name (any common form)
        cache_dir: local cache directory (created if missing)
        rad_type: "g" for γ-rays (default)
        force_refresh: bypass cache and re-fetch from IAEA
        timeout_seconds: HTTP timeout

    Returns:
        list of IaeaGammaLine.

    Raises:
        urllib.error.URLError, urllib.error.HTTPError on network failure
        (caller should handle these and decide whether to fall back to
        cache or skip the nuclide).

    Notes:
        IAEA API sometimes returns HTTP 403 to default user-agents;
        we set a permissive User-Agent header to avoid this.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, nuclide, rad_type)

    if not force_refresh and path.exists():
        with open(path, "r", encoding="utf-8") as f:
            csv_text = f.read()
        if csv_text.strip():
            return _parse_iaea_csv(csv_text)

    import urllib.request, urllib.error
    url = _build_url(nuclide, rad_type=rad_type)
    req = urllib.request.Request(
        url, headers={"User-Agent": "atomspectra-waterfall-viewer/0.1 (scientific use)"},
    )
    with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
        csv_text = resp.read().decode("utf-8", errors="replace")

    # REL-03: atomic cache write. Naive `open(path, 'w')` truncates the
    # destination at open() — a Ctrl-C / OOM / power-loss between truncate
    # and the final flush leaves a zero-length or torn cache that breaks
    # the next reproducible load. Write to a sibling tempfile and commit
    # with `os.replace`, which is atomic on POSIX and NTFS. Pattern mirrors
    # the standard write-temp-then-rename idiom.
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(csv_text)
        os.replace(tmp_path, path)
    finally:
        # Best-effort cleanup if os.replace failed mid-write. Never raise
        # from the finally; the original exception (if any) propagates.
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass

    return _parse_iaea_csv(csv_text)


def merge_iaea_into_internal(
    iaea_lines: List[IaeaGammaLine],
    target_nuclide_name: str,
    *,
    min_intensity_pct: float = 0.1,
    only_ground_state_parent: bool = True,
) -> dict:
    """
    Convert IAEA γ-lines into one entry suitable for the internal
    nuclide_library format.

    Args:
        iaea_lines: lines from fetch_iaea_gamma_lines() or
            load_iaea_gamma_lines_from_cache()
        target_nuclide_name: name to assign in the internal library
            (typically the parent nuclide, e.g. "Th-234")
        min_intensity_pct: drop lines below this intensity (default
            0.1%, keeps low-I diagnostics but filters out tiny ones)
        only_ground_state_parent: when True (default), only include
            lines from the ground-state parent decay. Metastable
            parents and excited parents are excluded.

    Returns:
        dict with keys: "lines" — [[E_keV, I_pct, dI_pct], ...] sorted
        by energy.
    """
    filtered = []
    for ln in iaea_lines:
        if ln.intensity_pct < min_intensity_pct:
            continue
        if only_ground_state_parent and ln.parent_energy_keV is not None \
                and ln.parent_energy_keV > 0.001:
            continue
        dI = ln.intensity_uncertainty_pct if ln.intensity_uncertainty_pct \
            else max(0.01, 0.01 * ln.intensity_pct)
        filtered.append([ln.energy_keV, ln.intensity_pct, dI])
    filtered.sort(key=lambda x: x[0])
    return {"lines": filtered}


__all__ = [
    "IAEA_API_URL", "DEFAULT_CACHE_DIR",
    "IaeaGammaLine",
    "fetch_iaea_gamma_lines",
    "load_iaea_gamma_lines_from_cache",
    "merge_iaea_into_internal",
]
