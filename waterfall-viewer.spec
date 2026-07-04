# PyInstaller spec — Задача #184. Сборка onedir Windows-exe.
# Запуск: py -3.14 -m PyInstaller waterfall-viewer.spec --noconfirm
import os
import PySide6
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

hiddenimports = (
    collect_submodules("pyqtgraph")
    + collect_submodules("pyqtgraph.opengl")
    + collect_submodules("OpenGL")
    + ["awf.ui.main_window"]
)

# Задача #185 F7: iaea_cache регенерируемый и в дистрибутив не входит (pyproject.toml),
# ядро идентификации — nuclides.json.
datas = collect_data_files("awf", includes=["data/*.json"])

_pyside_root = os.path.dirname(PySide6.__file__)
# Задача #185 F4: imageformats исключён — awf не читает файлы изображений
# (QPixmap/QImage только in-memory из numpy, Format_RGB888).
for _sub in ("platforms", "styles", "iconengines"):
    _src = os.path.join(_pyside_root, "plugins", _sub)
    if os.path.isdir(_src):
        datas.append((_src, os.path.join("PySide6", "plugins", _sub)))

a = Analysis(
    ["awf/__main__.py"],
    pathex=["."],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "unittest", "matplotlib", "PIL", "pandas",
              "PySide6.QtTest", "PySide6.QtNetwork"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="waterfall-viewer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe, a.binaries, a.datas,
    strip=False, upx=False, upx_exclude=[], name="waterfall-viewer",
)

# Задача #185 F1: удалить software-fallback OpenGL (20.6 МБ) — hardware GL уже требование awf/ui/view3d.py.
_opengl_sw = os.path.join(DISTPATH, "waterfall-viewer", "_internal", "PySide6", "opengl32sw.dll")
if os.path.exists(_opengl_sw):
    os.remove(_opengl_sw)
    print(f"[spec] Задача #185 F1: removed {_opengl_sw}")
