"""
Spectral indices from local Sentinel-2 / Landsat rasters (Indices section).

* LIBRARY: ~25 indices written in terms of band ROLES (blue, red, nir,
  swir1, red edge ...), so one formula serves Sentinel-2 and Landsat.
* detect(): finds the band of every role from the raster's metadata --
  wavelength metadata (GDAL IMAGERY domain, WAVELENGTH_NM, "(842 nm)" in
  the description), else band names (B8A, SR_B5, ...) with the sensor's
  wavelength table, else the sensor's usual band order -- and how to
  convert the values to surface reflectance (scale / offset metadata,
  Sentinel-2 digital numbers incl. the -1000 offset of baseline >= 04.00,
  Landsat Collection 2 Level-2).
* compute(): block-wise (any raster size), Float32 GeoTIFF(s) with band
  names and formula metadata, NoData = NaN.

GDAL + numpy only, no Qt / QGIS imports.
"""
import math
import os
import re

import numpy as np

# ---------------------------------------------------------------------------
# Band roles: target wavelength and accepted window (nm)
# ---------------------------------------------------------------------------
ROLES = {
    "coastal": ("Coastal aerosol", 443, 425, 455),
    "blue": ("Blue", 490, 455, 515),
    "green": ("Green", 560, 525, 600),
    "red": ("Red", 665, 625, 690),
    "re1": ("Red edge 1", 705, 695, 715),
    "re2": ("Red edge 2", 740, 730, 750),
    "re3": ("Red edge 3", 783, 770, 795),
    "nir": ("NIR", 842, 800, 900),
    "nir2": ("NIR narrow", 865, 850, 880),
    "swir1": ("SWIR 1", 1610, 1550, 1700),
    "swir2": ("SWIR 2", 2190, 2080, 2350),
}

# wavelength tables (nm) per sensor, by band id
SENSORS = {
    "s2": {"label": "Sentinel-2 MSI",
           "bands": {"B1": 443, "B2": 490, "B3": 560, "B4": 665, "B5": 705, "B6": 740, "B7": 783, "B8": 842,
                     "B8A": 865, "B9": 945, "B10": 1375, "B11": 1610, "B12": 2190},
           # usual band orders of stacks without names, by band count
           "orders": {12: ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"],
                      13: ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12", "SCL"],
                      10: ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"],
                      4: ["B2", "B3", "B4", "B8"]}},
    "l89": {"label": "Landsat 8/9 OLI",
            "bands": {"B1": 443, "B2": 482, "B3": 561, "B4": 655, "B5": 865, "B6": 1609, "B7": 2201},
            "orders": {7: ["B1", "B2", "B3", "B4", "B5", "B6", "B7"]}},
    "l457": {"label": "Landsat 4-7 TM / ETM+",
             "bands": {"B1": 485, "B2": 560, "B3": 660, "B4": 835, "B5": 1650, "B7": 2215},
             "orders": {6: ["B1", "B2", "B3", "B4", "B5", "B7"]}},
}

# ---------------------------------------------------------------------------
# Index library
# ---------------------------------------------------------------------------
VEG, REDEDGE, WATER, SOIL = "Vegetation", "Red edge (Sentinel-2)", "Water and moisture", "Soil, burn and built-up"


def _nd(a, b):
    return (a - b) / (a + b)


def _ndvi(b):
    return _nd(b["nir"], b["red"])


# key: (category, full name, formula text, roles, function, (vmin, vmax) for display, colour ramp, reference)
LIBRARY = {
    "NDVI": (VEG, "Normalized Difference Vegetation Index", "(NIR − Red) / (NIR + Red)", ("nir", "red"),
             _ndvi, (-0.2, 0.9), "RdYlGn", "Rouse et al. 1974"),
    "EVI": (VEG, "Enhanced Vegetation Index", "2.5 (NIR − Red) / (NIR + 6 Red − 7.5 Blue + 1)",
            ("nir", "red", "blue"),
            lambda b: 2.5 * (b["nir"] - b["red"]) / (b["nir"] + 6 * b["red"] - 7.5 * b["blue"] + 1),
            (-0.2, 0.9), "RdYlGn", "Huete et al. 2002"),
    "EVI2": (VEG, "Two-band Enhanced Vegetation Index", "2.5 (NIR − Red) / (NIR + 2.4 Red + 1)", ("nir", "red"),
             lambda b: 2.5 * (b["nir"] - b["red"]) / (b["nir"] + 2.4 * b["red"] + 1),
             (-0.2, 0.9), "RdYlGn", "Jiang et al. 2008"),
    "SAVI": (VEG, "Soil Adjusted Vegetation Index", "1.5 (NIR − Red) / (NIR + Red + 0.5)", ("nir", "red"),
             lambda b: 1.5 * (b["nir"] - b["red"]) / (b["nir"] + b["red"] + 0.5),
             (-0.2, 0.8), "RdYlGn", "Huete 1988"),
    "MSAVI": (VEG, "Modified Soil Adjusted Vegetation Index",
              "(2 NIR + 1 − √((2 NIR + 1)² − 8 (NIR − Red))) / 2", ("nir", "red"),
              lambda b: (2 * b["nir"] + 1 - np.sqrt((2 * b["nir"] + 1) ** 2 - 8 * (b["nir"] - b["red"]))) / 2,
              (-0.2, 0.8), "RdYlGn", "Qi et al. 1994"),
    "OSAVI": (VEG, "Optimized Soil Adjusted Vegetation Index", "(NIR − Red) / (NIR + Red + 0.16)", ("nir", "red"),
              lambda b: (b["nir"] - b["red"]) / (b["nir"] + b["red"] + 0.16),
              (-0.2, 0.9), "RdYlGn", "Rondeaux et al. 1996"),
    "GNDVI": (VEG, "Green Normalized Difference Vegetation Index", "(NIR − Green) / (NIR + Green)",
              ("nir", "green"), lambda b: _nd(b["nir"], b["green"]), (-0.2, 0.9), "RdYlGn", "Gitelson et al. 1996"),
    "CIgreen": (VEG, "Green Chlorophyll Index", "NIR / Green − 1", ("nir", "green"),
                lambda b: b["nir"] / b["green"] - 1, (0, 8), "YlGn", "Gitelson et al. 2003"),
    "ARVI": (VEG, "Atmospherically Resistant Vegetation Index",
             "(NIR − (2 Red − Blue)) / (NIR + (2 Red − Blue))", ("nir", "red", "blue"),
             lambda b: _nd(b["nir"], 2 * b["red"] - b["blue"]), (-0.2, 0.9), "RdYlGn", "Kaufman & Tanré 1992"),
    "VARI": (VEG, "Visible Atmospherically Resistant Index", "(Green − Red) / (Green + Red − Blue)",
             ("green", "red", "blue"), lambda b: (b["green"] - b["red"]) / (b["green"] + b["red"] - b["blue"]),
             (-0.5, 0.5), "RdYlGn", "Gitelson et al. 2002"),
    "NIRv": (VEG, "Near-Infrared Reflectance of Vegetation", "NDVI × NIR", ("nir", "red"),
             lambda b: _ndvi(b) * b["nir"], (0, 0.4), "YlGn", "Badgley et al. 2017"),
    "kNDVI": (VEG, "Kernel NDVI", "tanh(NDVI²)", ("nir", "red"),
              lambda b: np.tanh(_ndvi(b) ** 2), (0, 0.7), "YlGn", "Camps-Valls et al. 2021"),
    "NDRE": (REDEDGE, "Normalized Difference Red Edge", "(NIR narrow − Red edge 1) / (NIR narrow + Red edge 1)",
             ("nir2", "re1"), lambda b: _nd(b["nir2"], b["re1"]), (-0.1, 0.7), "RdYlGn", "Barnes et al. 2000"),
    "CIre": (REDEDGE, "Red-edge Chlorophyll Index", "Red edge 3 / Red edge 1 − 1", ("re3", "re1"),
             lambda b: b["re3"] / b["re1"] - 1, (0, 6), "YlGn", "Gitelson et al. 2003"),
    "MCARI": (REDEDGE, "Modified Chlorophyll Absorption in Reflectance Index",
              "((Red edge 1 − Red) − 0.2 (Red edge 1 − Green)) × (Red edge 1 / Red)", ("re1", "red", "green"),
              lambda b: ((b["re1"] - b["red"]) - 0.2 * (b["re1"] - b["green"])) * (b["re1"] / b["red"]),
              (0, 0.3), "YlGn", "Daughtry et al. 2000"),
    "IRECI": (REDEDGE, "Inverted Red-Edge Chlorophyll Index", "(Red edge 3 − Red) / (Red edge 1 / Red edge 2)",
              ("re3", "red", "re1", "re2"), lambda b: (b["re3"] - b["red"]) / (b["re1"] / b["re2"]),
              (0, 1.5), "YlGn", "Frampton et al. 2013"),
    "S2REP": (REDEDGE, "Sentinel-2 Red-Edge Position (nm)",
              "705 + 35 × ((Red edge 3 + Red) / 2 − Red edge 1) / (Red edge 2 − Red edge 1)",
              ("re3", "red", "re1", "re2"),
              lambda b: 705 + 35 * ((b["re3"] + b["red"]) / 2 - b["re1"]) / (b["re2"] - b["re1"]),
              (700, 735), "viridis", "Frampton et al. 2013"),
    "NDWI": (WATER, "Normalized Difference Water Index (open water)", "(Green − NIR) / (Green + NIR)",
             ("green", "nir"), lambda b: _nd(b["green"], b["nir"]), (-0.8, 0.6), "RdBu", "McFeeters 1996"),
    "MNDWI": (WATER, "Modified Normalized Difference Water Index", "(Green − SWIR 1) / (Green + SWIR 1)",
              ("green", "swir1"), lambda b: _nd(b["green"], b["swir1"]), (-0.8, 0.8), "RdBu", "Xu 2006"),
    "NDMI": (WATER, "Normalized Difference Moisture Index", "(NIR − SWIR 1) / (NIR + SWIR 1)",
             ("nir", "swir1"), lambda b: _nd(b["nir"], b["swir1"]), (-0.4, 0.6), "RdBu", "Gao 1996"),
    "AWEInsh": (WATER, "Automated Water Extraction Index (no shadow)",
                "4 (Green − SWIR 1) − (0.25 NIR + 2.75 SWIR 2)", ("green", "swir1", "nir", "swir2"),
                lambda b: 4 * (b["green"] - b["swir1"]) - (0.25 * b["nir"] + 2.75 * b["swir2"]),
                (-1.5, 0.5), "RdBu", "Feyisa et al. 2014"),
    "NBR": (SOIL, "Normalized Burn Ratio", "(NIR − SWIR 2) / (NIR + SWIR 2)", ("nir", "swir2"),
            lambda b: _nd(b["nir"], b["swir2"]), (-0.5, 0.9), "RdYlGn", "Key & Benson 2006"),
    "NBR2": (SOIL, "Normalized Burn Ratio 2 (also tillage index NDTI)", "(SWIR 1 − SWIR 2) / (SWIR 1 + SWIR 2)",
             ("swir1", "swir2"), lambda b: _nd(b["swir1"], b["swir2"]), (-0.1, 0.4), "RdYlGn",
             "Key & Benson 2006; van Deventer et al. 1997"),
    "NDBI": (SOIL, "Normalized Difference Built-up Index", "(SWIR 1 − NIR) / (SWIR 1 + NIR)", ("swir1", "nir"),
             lambda b: _nd(b["swir1"], b["nir"]), (-0.6, 0.4), "YlOrBr", "Zha et al. 2003"),
    "BSI": (SOIL, "Bare Soil Index", "((SWIR 1 + Red) − (NIR + Blue)) / ((SWIR 1 + Red) + (NIR + Blue))",
            ("swir1", "red", "nir", "blue"),
            lambda b: _nd(b["swir1"] + b["red"], b["nir"] + b["blue"]), (-0.5, 0.4), "YlOrBr",
            "Rikimaru et al. 2002"),
}
CATEGORIES = (VEG, REDEDGE, WATER, SOIL)
COMMON = ("NDVI", "EVI", "NDWI", "NDMI", "NBR")


def info(key):
    cat, name, formula, roles, _fn, rng, ramp, ref = LIBRARY[key]
    return {"key": key, "category": cat, "name": name, "formula": formula, "roles": roles, "range": rng,
            "ramp": ramp, "reference": ref}


def available(key, roles):
    """True if every band role the index needs is assigned."""
    return all(roles.get(r) for r in LIBRARY[key][3])


def missing_roles(key, roles):
    return [ROLES[r][0] for r in LIBRARY[key][3] if not roles.get(r)]


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
_NM = re.compile(r"(\d{3,4}(?:\.\d+)?)\s*nm", re.I)
# "B8A", "B08", "SR_B5" -- not "Band 4" (that is a band number, not a band id)
_ID = re.compile(r"(?:^|[^A-Za-z0-9])(?:SR_)?B0?(\d{1,2}A?)(?![0-9A-Za-z])", re.I)


def detect_sensor(path, ds):
    """'s2', 'l89', 'l457' or None, from dataset metadata, band names and the file name."""
    md = {k.upper(): str(v) for k, v in (ds.GetMetadata() or {}).items()}
    text = " ".join([os.path.basename(path)] + [md.get(k, "") for k in
                    ("SPACECRAFT", "SPACECRAFT_NAME", "PRODUCT", "SENTINEL2_PRODUCT", "PRODUCT_URI", "SENSOR",
                     "SATELLITE", "PRODUCT_TYPE", "DATATAKE_1_SPACECRAFT_NAME")]).upper()
    if re.search(r"SENTINEL-?2|S2[ABC]_|MSIL[12][AC]|\bS2[ABC]\b|_T\d{2}[A-Z]{3}_", text):
        return "s2"
    if re.search(r"LC0?[89]|LANDSAT[-_ ]?[89]|OLI", text):
        return "l89"
    if re.search(r"L[TE]0?[457]|LANDSAT[-_ ]?[457]|\bTM\b|ETM", text):
        return "l457"
    names = " ".join((ds.GetRasterBand(i + 1).GetDescription() or "") for i in range(ds.RasterCount)).upper()
    if "B8A" in names or "B11" in names or "B12" in names:
        return "s2"
    return None


def _band_wavelength(band):
    """Wavelength (nm) from band metadata / description, or None."""
    um = band.GetMetadataItem("CENTRAL_WAVELENGTH_UM", "IMAGERY")
    for v, factor in ((um, 1000.0), (band.GetMetadataItem("WAVELENGTH_NM"), 1.0),
                      (band.GetMetadataItem("wavelength"), None), (band.GetMetadataItem("WAVELENGTH"), None)):
        try:
            if v is not None:
                x = float(str(v).split()[0])
                return x * (factor if factor else (1000.0 if x < 30 else 1.0))
        except ValueError:
            pass
    m = _NM.search(band.GetDescription() or "")
    return float(m.group(1)) if m else None


def _band_id(text):
    m = _ID.search(text or "")
    return f"B{m.group(1).upper().lstrip('0') or '0'}" if m else None


def describe_bands(path, sensor=None):
    """-> (sensor, [{index, label, wavelength, how}]) for every band."""
    from osgeo import gdal
    ds = gdal.Open(path)
    if ds is None:
        raise RuntimeError(f"Could not open {path}")
    sensor = sensor or detect_sensor(path, ds)
    table = SENSORS.get(sensor, {}).get("bands", {})
    order = SENSORS.get(sensor, {}).get("orders", {}).get(ds.RasterCount)
    out = []
    for i in range(ds.RasterCount):
        b = ds.GetRasterBand(i + 1)
        desc = (b.GetDescription() or "").strip()
        label = desc or f"Band {i + 1}"
        wl, how = _band_wavelength(b), "wavelength metadata"
        if wl is None:
            bid = _band_id(b.GetMetadataItem("BANDNAME") or desc)
            if bid in table:
                wl, how = table[bid], "band name"
        if wl is None and order and i < len(order) and order[i] in table:
            wl, how = table[order[i]], "band order (assumed)"
            label = f"{label} (= {order[i]}?)"
        out.append({"index": i + 1, "label": label, "wavelength": wl, "how": how if wl is not None else ""})
    return sensor, out


def assign_roles(bands):
    """Role -> band index (closest wavelength inside the role's window)."""
    roles = {}
    for role, (_label, target, lo, hi) in ROLES.items():
        cands = [(abs(b["wavelength"] - target), b["index"]) for b in bands
                 if b["wavelength"] is not None and lo <= b["wavelength"] <= hi]
        roles[role] = min(cands)[1] if cands else None
    return roles


def _baseline(path, ds):
    md = ds.GetMetadata() or {}
    v = md.get("PROCESSING_BASELINE")
    if not v:
        m = re.search(r"_N(\d{2})(\d{2})_", os.path.basename(path) + " " + md.get("PRODUCT", "") +
                      " " + md.get("SENTINEL2_PRODUCT", ""))
        v = f"{m.group(1)}.{m.group(2)}" if m else None
    try:
        return float(v) if v else None
    except ValueError:
        return None


SCALINGS = {
    "auto": "Automatic (metadata / sensor)",
    "none": "Values are reflectance already (0-1)",
    "s2_offset": "Sentinel-2 L2A digital numbers, baseline ≥ 04.00: (DN − 1000) / 10000",
    "div10000": "Digital numbers / 10000 (Sentinel-2 before 2022, many stacks)",
    "landsat_c2": "Landsat Collection 2 Level-2: DN × 0.0000275 − 0.2",
}


def _sample(band, size=256):
    a = band.ReadAsArray(buf_xsize=min(size, band.XSize), buf_ysize=min(size, band.YSize))
    return None if a is None else a.astype(np.float64)


def resolve_scaling(path, roles, sensor, mode="auto"):
    """-> ({band index: (scale, offset)}, text). Values become surface
    reflectance as value * scale + offset."""
    from osgeo import gdal
    ds = gdal.Open(path)
    used = sorted({i for i in roles.values() if i})
    if mode != "auto":
        so = {"none": (1.0, 0.0), "s2_offset": (1e-4, -0.1), "div10000": (1e-4, 0.0),
              "landsat_c2": (2.75e-5, -0.2)}[mode]
        return {i: so for i in used}, SCALINGS[mode]
    out, how = {}, set()
    for i in used:
        b = ds.GetRasterBand(i)
        s, o = b.GetScale(), b.GetOffset()
        if (s not in (None, 1.0)) or (o not in (None, 0.0)):
            out[i] = (s if s is not None else 1.0, o if o is not None else 0.0)
            how.add("scale / offset from the band metadata")
            continue
        smp = _sample(b)
        valid = smp[np.isfinite(smp) & (smp != 0)] if smp is not None else np.array([])
        med = float(np.median(valid)) if valid.size else 0.0
        if b.DataType in (gdal.GDT_Float32, gdal.GDT_Float64) and med <= 2.0:
            out[i] = (1.0, 0.0)
            how.add("values are reflectance already")
        elif sensor in ("l89", "l457") and med > 5000:
            out[i] = (2.75e-5, -0.2)
            how.add("Landsat Collection 2 Level-2 (× 0.0000275 − 0.2)")
        elif sensor == "s2" and (_baseline(path, ds) or 0) >= 4.0:
            out[i] = (1e-4, -0.1)
            how.add("Sentinel-2 baseline ≥ 04.00: (DN − 1000) / 10000")
        elif med > 2.0:
            out[i] = (1e-4, 0.0)
            how.add("digital numbers / 10000 (assumed)")
        else:
            out[i] = (1.0, 0.0)
            how.add("values used as they are")
    return out, "; ".join(sorted(how)) or "–"


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------
def pixel_window(path, bbox):
    """bbox (xmin, ymin, xmax, ymax) in the raster CRS -> (xoff, yoff, w, h); None = whole raster."""
    from osgeo import gdal
    ds = gdal.Open(path)
    if bbox is None:
        return 0, 0, ds.RasterXSize, ds.RasterYSize
    gt = ds.GetGeoTransform()
    if gt[2] or gt[4]:
        return 0, 0, ds.RasterXSize, ds.RasterYSize          # rotated grid: whole raster
    xs = sorted(((bbox[0] - gt[0]) / gt[1], (bbox[2] - gt[0]) / gt[1]))
    ys = sorted(((bbox[3] - gt[3]) / gt[5], (bbox[1] - gt[3]) / gt[5]))
    x0, x1 = max(0, math.floor(xs[0])), min(ds.RasterXSize, math.ceil(xs[1]))
    y0, y1 = max(0, math.floor(ys[0])), min(ds.RasterYSize, math.ceil(ys[1]))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("The extent does not overlap the raster.")
    return x0, y0, x1 - x0, y1 - y0


def _create(path, w, h, n, gt, wkt):
    from osgeo import gdal
    out = gdal.GetDriverByName("GTiff").Create(
        path, w, h, n, gdal.GDT_Float32,
        ["COMPRESS=DEFLATE", "PREDICTOR=3", "TILED=YES", "BIGTIFF=IF_SAFER", "NUM_THREADS=ALL_CPUS"])
    if out is None:
        raise RuntimeError(f"Could not create {path}")
    out.SetGeoTransform(gt)
    out.SetProjection(wkt)
    return out


def _tag(band, key, roles, labels):
    i = info(key)
    band.SetDescription(key)
    band.SetNoDataValue(float("nan"))
    band.SetMetadata({"INDEX": key, "NAME": i["name"], "FORMULA": i["formula"], "REFERENCE": i["reference"],
                      "BANDS": "; ".join(f"{ROLES[r][0]} = {labels.get(roles[r], roles[r])}" for r in i["roles"])})


def compute(path, keys, roles, scaling, out, mode="stack", window=None, band_labels=None, overviews=True,
            progress_cb=None, cancel_cb=None):
    """Writes the indices `keys` of raster `path`.
    roles: {role: band index}; scaling: {band index: (scale, offset)} (see
    resolve_scaling); mode "stack" -> one GeoTIFF `out` (one band per index,
    named), "single" -> one GeoTIFF per index in folder `out`
    (<prefix><KEY>.tif with out = (folder, prefix)); window: (xoff, yoff,
    w, h) or None. DN 0 of integer rasters, the band NoData value and
    non-finite results are NoData (NaN). -> list of written paths."""
    from osgeo import gdal
    keys = [k for k in keys if k in LIBRARY]
    if not keys:
        raise ValueError("Choose at least one index.")
    bad = [k for k in keys if not available(k, roles)]
    if bad:
        raise ValueError(f"Bands missing for {', '.join(bad)}.")
    ds = gdal.Open(path)
    if ds is None:
        raise RuntimeError(f"Could not open {path}")
    xoff, yoff, w, h = window or (0, 0, ds.RasterXSize, ds.RasterYSize)
    gt0 = ds.GetGeoTransform()
    gt = (gt0[0] + xoff * gt0[1] + yoff * gt0[2], gt0[1], gt0[2], gt0[3] + xoff * gt0[4] + yoff * gt0[5], gt0[4],
          gt0[5])
    wkt = ds.GetProjection()
    labels = band_labels or {}
    need = sorted({roles[r] for k in keys for r in LIBRARY[k][3]})
    src = {i: ds.GetRasterBand(i) for i in need}
    nodata = {i: src[i].GetNoDataValue() for i in need}
    is_int = {i: src[i].DataType not in (gdal.GDT_Float32, gdal.GDT_Float64) for i in need}

    if mode == "stack":
        paths = [out]
        dst = _create(out, w, h, len(keys), gt, wkt)
        targets = {k: dst.GetRasterBand(j + 1) for j, k in enumerate(keys)}
        dsts = [dst]
    else:
        folder, prefix = out
        os.makedirs(folder, exist_ok=True)
        paths, targets, dsts = [], {}, []
        for k in keys:
            p = os.path.join(folder, f"{prefix}{k}.tif")
            d = _create(p, w, h, 1, gt, wkt)
            paths.append(p)
            dsts.append(d)
            targets[k] = d.GetRasterBand(1)
    for k, band in targets.items():
        _tag(band, k, roles, labels)
    for d in dsts:
        d.SetMetadata({"SOURCE": os.path.basename(path), "VALUES": "spectral index from surface reflectance",
                       "SCALING": "; ".join(f"band {i}: × {s:g} {o:+g}" for i, (s, o) in sorted(scaling.items())
                                            if i in need)})

    rows = max(16, min(h, 4_000_000 // max(1, w)))
    nblocks = math.ceil(h / rows)
    for bi in range(nblocks):
        if cancel_cb is not None and cancel_cb():
            targets.clear()
            dsts.clear()                     # closes the (incomplete) files
            raise InterruptedError("Cancelled")
        if progress_cb is not None:
            progress_cb(bi, nblocks + (1 if overviews else 0), "Computing the indices")
        y = bi * rows
        hh = min(rows, h - y)
        arr = {}
        for i in need:
            a = src[i].ReadAsArray(xoff, yoff + y, w, hh).astype(np.float32)
            bad_px = ~np.isfinite(a)
            if nodata[i] is not None and not (isinstance(nodata[i], float) and math.isnan(nodata[i])):
                bad_px |= a == nodata[i]
            if is_int[i]:
                bad_px |= a == 0
            s, o = scaling.get(i, (1.0, 0.0))
            a = a * np.float32(s) + np.float32(o)
            a[bad_px] = np.nan
            arr[i] = a
        b = {r: arr[roles[r]] for r in ROLES if roles.get(r) in arr}
        with np.errstate(all="ignore"):
            for k in keys:
                v = np.asarray(LIBRARY[k][4](b), dtype=np.float32)
                v[~np.isfinite(v)] = np.nan
                targets[k].WriteArray(v, 0, y)
    for d in dsts:
        d.FlushCache()
    if overviews and max(w, h) > 2048:
        if progress_cb is not None:
            progress_cb(nblocks, nblocks + 1, "Building overviews")
        levels = [f for f in (2, 4, 8, 16, 32, 64) if max(w, h) / f >= 256]
        for d in dsts:
            d.BuildOverviews("AVERAGE", levels)
    targets = dsts = None
    return paths
