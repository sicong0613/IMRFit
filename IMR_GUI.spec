# -*- mode: python ; coding: utf-8 -*-
#
# PyInstaller spec for IMR_GUI (onedir / folder distribution)
#
# Build command:
#   cd <project root>
#   pyinstaller IMR_GUI.spec
#
# Output: dist/IMR_GUI/
#   IMR_GUI.exe          <- launcher
#   settings.json        <- written here at first run (user-editable)
#   _internal/           <- all bundled Python + Qt files

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# --- Data files -----------------------------------------------------------
# Explicitly copy the constitutive JSON files into the bundle at the correct
# relative path so importlib.resources can find them at runtime.
datas = [
    ("imr_gui/constitutive/*.json", "imr_gui/constitutive"),
    ("imr_gui/view_colors.json", "imr_gui"),
    ("imr_gui/icon.ico", "imr_gui"),   # app icon (also used by settings_path logic)
]

# mat73 ships its own data files (if any); collect defensively
datas += collect_data_files("mat73", include_py_files=False)

# --- Hidden imports -------------------------------------------------------
# scipy's BDF/Radau integrators and sparse-matrix back-ends are loaded
# dynamically and not detected by PyInstaller's static analysis.
hiddenimports = [
    # scipy integrators
    "scipy.integrate._ivp.bdf",
    "scipy.integrate._ivp.radau",
    "scipy.integrate._ivp.rk",
    "scipy.integrate._ivp.lsoda",
    "scipy.integrate._ivp.common",
    "scipy.integrate._ivp.base",
    # scipy sparse
    "scipy.sparse.csgraph._shortest_path",
    "scipy.sparse.csgraph._traversal",
    "scipy.sparse.linalg._dsolve.umfpack",
    # mat73 / h5py internals
    "h5py",
    "h5py._hl",
    "h5py.defs",
    "h5py.utils",
    "h5py.h5ac",
    "h5py.h5z",
    # matplotlib backends
    "matplotlib.backends.backend_agg",
    "matplotlib.backends.backend_svg",
]
hiddenimports += collect_submodules("scipy.integrate._ivp")

# --------------------------------------------------------------------------
a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # keep the bundle lean
        "tkinter",
        "wx",
        "PyQt5",
        "PyQt6",
        "IPython",
        "jupyter",
        "notebook",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,   # onedir: binaries stay in the folder
    name="IMR_GUI",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,               # UPX can corrupt Qt DLLs; keep off
    console=False,           # no console window (GUI app)
    icon="imr_gui/icon.ico",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="IMR_GUI",
)
