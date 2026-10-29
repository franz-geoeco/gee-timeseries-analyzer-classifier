"""Builds dist/gee_timeseries_analyzer.zip (installable via QGIS → Plugins →
Install from ZIP). Run from anywhere: python scripts/build_plugin_zip.py

Only the plugin sources are packed -- no __pycache__, no local files."""
import os
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLUGIN = "gee_timeseries_analyzer"
SRC = os.path.join(ROOT, "qgis_plugin", PLUGIN)
DIST = os.path.join(ROOT, "dist")
SKIP_DIRS = {"__pycache__", ".git"}
SKIP_EXT = {".pyc", ".pyo", ".zip"}


def main():
    os.makedirs(DIST, exist_ok=True)
    out = os.path.join(DIST, f"{PLUGIN}.zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for root, dirs, files in os.walk(SRC):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for f in sorted(files):
                if os.path.splitext(f)[1] in SKIP_EXT:
                    continue
                full = os.path.join(root, f)
                z.write(full, os.path.join(PLUGIN, os.path.relpath(full, SRC)).replace(os.sep, "/"))
    print(f"{out} ({os.path.getsize(out) / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
