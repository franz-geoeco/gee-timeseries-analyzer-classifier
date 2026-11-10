"""
Check of the Python environment before anything imports numpy / pandas /
matplotlib (standard library only).

The classic breakage: a NumPy 2.x in the user's personal site-packages
(%APPDATA%\\Python\\Python312\\site-packages on Windows -- typically
installed by another plugin or a `pip install`) shadows the NumPy 1.x that
QGIS ships. QGIS's GDAL (osgeo.gdal_array: every pixel read into numpy) is
built for NumPy 1.x and then fails -- inside QGIS (sys.stderr is None) with the cryptic
"AttributeError: 'NoneType' object has no attribute 'write'".
"""
import glob
import importlib.util
import os
import site
import sys


def _version_in(folder, package="numpy"):
    hits = glob.glob(os.path.join(folder, f"{package}-*.dist-info"))
    if not hits:
        return None
    return os.path.basename(sorted(hits)[-1])[len(package) + 1:-len(".dist-info")]


def _bundled_dirs():
    try:
        dirs = site.getsitepackages()
    except AttributeError:                     # old virtualenvs
        dirs = []
    return [d for d in dirs if os.path.isdir(d)]


def numpy_conflict():
    """-> dict(loaded, loaded_dir, bundled, bundled_dir) if NumPy is loaded
    from outside QGIS's own site-packages with a different major version
    than the one QGIS ships; else None."""
    mod = sys.modules.get("numpy")
    path = getattr(mod, "__file__", None) if mod is not None else None
    if path is None:
        try:
            spec = importlib.util.find_spec("numpy")          # does not import numpy
        except (ImportError, ValueError):
            spec = None
        path = spec.origin if spec is not None else None
    if not path:
        return None
    loaded_dir = os.path.dirname(os.path.dirname(os.path.abspath(path)))   # .../site-packages
    bundled = [(d, _version_in(d)) for d in _bundled_dirs()]
    bundled = [(d, v) for d, v in bundled if v]
    if not bundled:
        return None
    norm = os.path.normcase
    if any(norm(os.path.abspath(d)) == norm(loaded_dir) for d, _v in bundled):
        return None
    loaded = _version_in(loaded_dir)
    bundled_dir, bundled_v = bundled[0]
    if not loaded or loaded.split(".")[0] == bundled_v.split(".")[0]:
        return None
    return {"loaded": loaded, "loaded_dir": loaded_dir, "bundled": bundled_v, "bundled_dir": bundled_dir}


def numpy_conflict_text(c):
    """Explanation and fix for numpy_conflict()."""
    major = c["bundled"].split(".")[0]
    return (
        f"NumPy {c['loaded']} is loaded from\n    {c['loaded_dir']}\n"
        f"instead of the NumPy {c['bundled']} that QGIS ships\n    ({c['bundled_dir']}).\n\n"
        f"QGIS's own GDAL (and possibly other bundled libraries) is built for NumPy {major}.x and fails with "
        "this copy -- typically with \"AttributeError: 'NoneType' object has no attribute 'write'\" or "
        "\"numpy.core.multiarray failed to import\". Everything that reads raster pixels is affected, in this "
        "plugin (local scenes, Indices, offline classification, exports) and in QGIS itself. The extra NumPy "
        "was usually installed by another plugin or a 'pip install'.\n\n"
        "Fix (removes only that extra copy):\n"
        "  1. Close QGIS.\n"
        "  2. Start the 'OSGeo4W Shell' (Windows start menu, QGIS folder; NOT as administrator) and run\n"
        "         python -c \"import numpy; print(numpy.__version__, numpy.__file__)\"\n"
        f"     It should show {c['loaded']} and the folder above. Only then run\n"
        "         python -m pip uninstall numpy\n"
        f"     and the first command again: it must now show {c['bundled']} from the QGIS folder.\n"
        "  3. Start QGIS again.\n\n"
        "Packages in that folder that need NumPy 2 (installed by other plugins) may then need older versions. "
        "Alternatively, QGIS can ignore the personal folder completely: Settings > Options > System > "
        "Environment, add PYTHONNOUSERSITE = 1 (Overwrite) and restart -- packages installed there (e.g. "
        "scikit-learn for the offline classification) are then not available either.")
