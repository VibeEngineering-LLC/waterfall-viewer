# Waterfall Viewer

A viewer and analyzer for **waterfall spectrograms** from gamma-ray spectrometers.
3D view with rotation and zoom, 2D Time×Energy map, time slices / cross-sections / rectangular ROI selections.

**Supported formats:** AtomSpectra (`.aswf`, native binary waterfall), RadiaCode
(`.rcspg`, JSON — tested on RadiaCode-110), and ANSI N42.42-2011 (`.n42`/`.xml`).
Energy calibration is read directly from the file (polynomial from the `.aswf` header / `coefficients` / `CoefficientValues`).

![Python](https://img.shields.io/badge/Python-3.12%20%7C%203.14-blue) ![GUI](https://img.shields.io/badge/GUI-PySide6%20(Qt)-green) ![3D](https://img.shields.io/badge/3D-pyqtgraph%20%2B%20OpenGL-orange) ![Format](https://img.shields.io/badge/format-ANSI%20N42.42--2011-yellowgreen) ![License](https://img.shields.io/badge/license-MIT-lightgrey)

Data source device: [atomspectra-waterfall-esp32](https://github.com/VibeEngineering-LLC/atomspectra-waterfall-esp32).
Architectural inspiration — [InterSpec (Sandia)](https://github.com/sandialabs/interspec).

## Features v1

- **3D surface** Time × Energy × Counts (OpenGL, rotate with LMB, zoom with scroll wheel, pan with MMB).
- **2D map** Time×Energy with rectangular ROI selection.
- **Z-scale** — contrast switch (linear / √ / log10) for both the 2D map and 3D relief.
- **Time slices** — spectrum of the selected time slice.
- **Cross-sections (by channel/band)** — time series of intensity in a selected energy band.
- **ROI selections** — spectrum of the time window + time series of the band + total counts in rectangle.
- **Nuclide library** — select a nuclide/family (21 nuclides, LSRM SpectraLine data) to highlight gamma-line energies with vertical markers on the spectrum; intensity filter included.
- **Unlimited scale** — day-long recordings (tens of thousands of slices) are loaded streaming (`lxml.iterparse`, bounded memory); 3D and 2D are rendered via LOD downsampling (`Spectrogram.downsample`).

## Download Ready-to-Run Distribution

**Pre-built Windows executable** — available on the [Releases](https://github.com/VibeEngineering-LLC/waterfall-viewer/releases/latest) page.  
Download `waterfall-viewer-vX.Y.Z-win64.zip`, extract, and run `waterfall-viewer.exe`.  
No Python required.

## Installation from Source

> Step-by-step installation from scratch (no Python experience required) — see [`INSTALL.en.md`](INSTALL.en.md).
> Known issues and compatibility — see [`KNOWN_ISSUES.md`](docs/dev/KNOWN_ISSUES.md).

**Option A — venv (isolated, recommended):**

```bat
cd "path\to\waterfall-viewer"
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**Option B — global (no venv):**

```bat
py -3.14 -m pip install -r requirements.txt
```

**Easiest — just run `run.bat`:** on first launch it automatically installs missing dependencies
into the user `site-packages` (`pip install --user`, no admin rights needed) and then opens the
app. First run takes ~1 minute due to installation (`Installing dependencies on first run...`
appears in the console). Manual installation (Options A/B above) is only needed if you want an
isolated `.venv` or auto-install is unavailable (no network).

`run.bat` picks the interpreter automatically: if `.venv` exists it uses it; otherwise falls back
to the global Python 3.14.

## Running

```bat
run.bat                                 REM empty window, open a file via menu
run.bat "C:\path\to\waterfall.n42"     REM open a file immediately
```

or directly:

```bat
.venv\Scripts\python.exe -m awf "C:\path\to\file.n42"
```

## Controls

| Action | How |
|---|---|
| Open file | Menu **File → Open…** (Ctrl+O) |
| Rotate 3D | Left mouse button + drag |
| Zoom 3D | Mouse scroll wheel |
| Pan 3D | Middle mouse button |
| ROI selection | Tab **2D Map** → drag the yellow rectangle; slice panels on the right update live |
| Z-scale | Toolbar **View → Z-scale** (linear / √ / log10) — changes 2D and 3D contrast |
| Nuclides | Dock **Nuclide library** (left) → check nuclides; line energies are highlighted on the spectrum plot |

## Architecture

```
awf/
  model/spectrogram.py   — numerical core: Calibration (polynomial), Spectrogram
                           (slices, sections, ROI sums, LOD downsample). No Qt dependency.
  io/n42_loader.py       — streaming N42 loader (CountedZeroes decoder: vectorised + scalar
                           reference; iterparse with memory release for day-long files).
  io/aswf_loader.py      — AtomSpectra loader (.aswf): ASWF magic + uint32 len + JSON header +
                           uint16 LE matrix (channels per row); calib/interval/t0 from header.
  io/rcspg_loader.py     — RadiaCode loader (.rcspg, JSON): pulses→counts, calib from coefficients.
  io/nuclide_lib.py      — nuclide library: LSRM .lib parser (win-1251 XML) + our JSON loader.
  data/nuclides.json     — built-in library (21 nuclides, gamma-line energies/intensities).
  ui/view3d.py           — Waterfall3DView (GLViewWidget + GLSurfacePlotItem).
  ui/panels.py           — HeatmapPanel (2D + ROI), SlicePanel (spectrum + time series + nuclide markers).
  ui/nuclide_panel.py    — NuclidePanel (nuclide/family selection, intensity filter).
  ui/zscale.py           — shared apply_z_scale (linear/sqrt/log10) for 2D and 3D.
  ui/main_window.py      — MainWindow + background loading (QThread).
  __main__.py            — entry point (python -m awf).
```

The numerical core is decoupled from the UI: it can be tested and used without a display.

## Tests

The data layer is covered by automated tests (reference values from an independent oracle on
real file bytes):

```bat
.venv\Scripts\python.exe -m pytest -q
```

(Tests cover ISO durations, CountedZeroes hand-cases + fuzz vec≡scalar, polynomial calibration,
inverse energy→channel mapping, real sample loading, slice limit, analytical primitives;
plus AtomSpectra `.aswf` loader: shape/dtype, count values, writable copy, calibration, time
axes, t0, slice limit, invalid signature, saved_rows cap and auto-inference from size, default
calibration; plus RadiaCode `.rcspg` loader: shape/dtype, channel zero-padding, calibration,
time axes, t0, slice limit, empty file, auto-channel count inference; plus nuclide library:
LSRM `.lib` parser (comma separator, used flag, line_type), major_lines, half_life,
round-trip JSON↔parser bit-for-bit, built-in library.)

Graphical modules are smoke-tested by scripts in `scripts/` (build widgets on a real sample
without GUI interaction).

## Stack

Python 3.12 / 3.14 · numpy · lxml · PySide6 (Qt) · pyqtgraph + PyOpenGL.

## Roadmap (v2)

Spectroscopy: peak search, isotope identification, calibration by reference peaks, ROI net/background, export.
Possible packaging as `.exe` (PyInstaller).

## License

[MIT](LICENSE) © 2026 Verter73.
