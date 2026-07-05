# Installing Waterfall Viewer from Scratch

Step-by-step instructions: what to install and how to run the waterfall spectrogram viewer
on Windows. No Python experience required.

> Quick reference for experienced users — see [`README.en.md`](README.en.md), section "Installation".
> This file is for users who have nothing installed yet.

---

## Table of Contents

0. [Option 0: Pre-built exe (no Python required)](#option-0-pre-built-exe)
1. [What you'll end up with](#what-youll-end-up-with)
2. [System requirements](#system-requirements)
3. [Step 1: Install Python](#step-1-install-python)
4. [Step 2: Get the program files](#step-2-get-the-program-files)
5. [Step 3: Install dependencies — Path A (venv) or B (global)](#step-3-install-dependencies)
6. [Step 4: Launch](#step-4-launch)
7. [Step 5: Open a spectrogram and verify](#step-5-open-a-spectrogram-and-verify)
8. [Tests](#tests)
9. [Troubleshooting](#troubleshooting)

---

## Option 0: Pre-built exe

**The easiest way.** No Python installation required.

1. Open the [Releases](https://github.com/VibeEngineering-LLC/waterfall-viewer/releases/latest) page.
2. Download `waterfall-viewer-vX.Y.Z-win64.zip`.
3. Extract the archive to any folder.
4. Run `waterfall-viewer.exe` inside the extracted folder.

**Requirements:** Windows 10/11 x64, GPU with OpenGL 3.x support (for the 3D waterfall tab).

> If Windows SmartScreen blocks the launch — click "More info" → "Run anyway".
> The executable is not signed with a publisher certificate.

Steps 1–5 below are only needed if you want to run **from source** (with Python).

---

## What You'll End Up With

A desktop window with two tabs:

- **3D Waterfall** — Time × Energy × Counts surface (OpenGL): rotate with the mouse,
  zoom with the scroll wheel, pan with the middle button.
- **2D Map (Time×Energy)** — a 2D map with log contrast and rectangular ROI selection.
  On the right — a slice panel: spectrum of the selected time window, time series of
  intensity in an energy band, and total counts in the rectangle.

Day-long recordings (tens of thousands of time slices) are loaded streaming and rendered
via LOD downsampling — no upper limit on recording length.

---

## System Requirements

- **OS:** Windows 10/11 (developed and tested on Windows 11).
- **Python:** 3.12 or 3.14 (64-bit). Tested on 3.14.5.
- **GPU:** driver with **OpenGL 2.1+** support (for the 3D tab). Nearly any integrated or
  discrete GPU from the last 10 years qualifies.
- **RAM:** 4 GB minimum; more for day-long recordings (the loader is streaming, but the
  count matrix is kept in RAM).
- **Disk space:** ~600 MB for dependencies (mostly PySide6/Qt).

---

## Step 1: Install Python

If Python is already installed — skip this step (verify: `py -3.14 --version` or
`python --version` in PowerShell).

1. Download the installer from [python.org/downloads](https://www.python.org/downloads/)
   (version 3.12 or 3.14, 64-bit).
2. During installation **check the box** **"Add python.exe to PATH"**.
3. Verify in PowerShell:
   ```powershell
   py -3.14 --version
   ```
   You should see `Python 3.14.x`.

---

## Step 2: Get the Program Files

Via git:
```powershell
git clone <repository-URL> waterfall-viewer
cd waterfall-viewer
```

Or download the ZIP archive of the repository and extract it to any folder.

---

## Step 3: Install Dependencies

Dependencies are listed in [`requirements.txt`](requirements.txt):
`numpy`, `lxml`, `PySide6` (Qt), `pyqtgraph` + `PyOpenGL`, `pytest`.

### Path A — venv (isolated, recommended)

```powershell
cd "path\to\waterfall-viewer"
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

> **Important.** Create the venv with the same Python you'll use to run the app. **Do not**
> recreate a venv over an existing one with a different interpreter (e.g., you had 3.12 and
> create a new one with 3.14): `python -m venv` will replace `python.exe` but **will not**
> reinstall packages — you'll get mixed binaries (`.pyd` from 3.12 under a 3.14 interpreter)
> and a `ModuleNotFoundError: numpy._core._multiarray_umath` error. Fix: delete `.venv` and
> reinstall. See [`KNOWN_ISSUES.md`](docs/dev/KNOWN_ISSUES.md).

### Path B — global (no venv)

```powershell
py -3.14 -m pip install -r requirements.txt
```

`run.bat` will choose automatically: if `.venv` exists it uses it; otherwise falls back to
global `py -3.14`.

---

## Step 4: Launch

```powershell
.\run.bat                                     # empty window, open a file via menu
.\run.bat "C:\path\to\waterfall.n42"          # open a file immediately
```

or directly as a module:
```powershell
py -3.14 -m awf "C:\path\to\file.n42"
```

---

## Step 5: Open a Spectrogram and Verify

1. **File → Open…** (Ctrl+O), select a file: AtomSpectra `.aswf`, RadiaCode `.rcspg`,
   or N42 `.n42`/`.xml`.
2. The status bar will show "total counts …" — the file is loaded.
3. **3D Waterfall tab:** hold the **left button** and drag — the surface rotates;
   **scroll wheel** — zoom; **middle button** — pan.
4. **2D Map tab:** drag the yellow rectangle — the slice panel on the right updates
   live (window spectrum + band time series + count total).

---

## Tests

The data layer is covered by automated tests (reference values from an independent oracle on
real file bytes):

```powershell
.venv\Scripts\python.exe -m pytest -q      # from venv
py -3.14 -m pytest -q                       # global
```

Tests cover ISO durations, CountedZeroes (hand-cases + fuzz vec≡scalar), polynomial
calibration and inverse mapping, sample loading, slice limit, analytical primitives;
AtomSpectra `.aswf` and RadiaCode `.rcspg` loaders (shape/dtype, values, calibration,
time axes, t0, edge cases — on synthetic files); nuclide library (LSRM `.lib` parser,
round-trip JSON).

> The N42 sample loading test is marked `skipif` and **skipped** if no `waterfall_sample.n42`
> is present in `sample_data/`. The sample itself **is not included in the repository**
> (N42 may contain GPS/device metadata, see `.gitignore`) — place your own `.n42` there
> under that name to run that test too. The `.aswf`/`.rcspg` loader tests build synthetic
> files in a temp folder and do not require a real sample.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: numpy._core._multiarray_umath` | venv recreated with a different Python over the old one — mixed binaries | Delete `.venv`, recreate, and `pip install -r requirements.txt` |
| `run.bat` shows "is not recognized" / garbled text | `.bat` not run from its own folder, or edited in an editor that added non-ASCII chars to comments | Run from the project folder; keep comments in `run.bat` ASCII-only |
| **Two** windows open | Old `run.bat` version with fallback `\|\| python` | Update `run.bat` from the repository (single `py -3.14 -m awf %*` run) |
| Black 3D tab / OpenGL error | GPU driver without OpenGL 2.1 | Update GPU driver; on VM enable 3D acceleration |
| Window opened but empty | No file opened | **File → Open…**, select `.aswf` / `.rcspg` / `.n42` |

Full breakdown of known issues and compatibility — in [`KNOWN_ISSUES.md`](docs/dev/KNOWN_ISSUES.md).
