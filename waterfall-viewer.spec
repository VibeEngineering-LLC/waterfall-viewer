# PyInstaller spec — waterfall-viewer, PySide6 (стек после отката #228..#230)
# Запуск: py -3.14 -m PyInstaller waterfall-viewer.spec --noconfirm
import os

from PyInstaller.utils.hooks import collect_submodules, collect_data_files
import PySide6

hiddenimports = (
    collect_submodules("pyqtgraph")
    + collect_submodules("pyqtgraph.opengl")
    + collect_submodules("OpenGL")
    + ["awf.ui.main_window",
       "PySide6", "PySide6.QtCore", "PySide6.QtWidgets", "PySide6.QtGui",
       "PySide6.QtOpenGL", "PySide6.QtOpenGLWidgets"]
)

# Данные awf (nuclides.json и т.п.)
datas = collect_data_files("awf", includes=["data/*.json"])

# Плагины PySide6 (платформы, стили, иконки)
_pyside_root = os.path.dirname(PySide6.__file__)
_qt6_plugins = os.path.join(_pyside_root, "plugins")
for _sub in ("platforms", "styles", "iconengines"):
    _src = os.path.join(_qt6_plugins, _sub)
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
              "PyQt5", "PyQt5.QtCore", "PySide2", "PyQtAds"],
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
