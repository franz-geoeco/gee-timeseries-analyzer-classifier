"""
Earth Engine side of the GEE Time Series Analyzer.

Pure earthengine-api + pandas (no Qt / QGIS imports), so everything here can
be exercised from a plain Python interpreter with `ee` initialised. Ports the
collection logic of the two GEE Code Editor apps this plugin replaces
(a time series analyzer and a field inspector):

  * Sentinel-1 GRD IW, VV + VH, descending and ascending passes, with the
    Field Inspector's < -30 dB border-noise mask
  * Sentinel-2 SR Harmonized, SCL cloud mask (classes 4-7 kept)
  * Landsat 8/9 Collection 2 Level-2, QA_PIXEL cloud/shadow/dilated mask

Instead of one ui.Chart.image.series() call per polygon and product, every
product is extracted with ONE server-side reduceRegions() over all polygons,
then fetched page by page via ee.data.computeFeatures(). Each row also
carries the polygon's valid-pixel fraction for that image, so the "at least
X % of the polygon must be cloud-free" filter of the original apps is applied
client-side (filter_series) and can be changed without re-querying GEE.
"""
import json
import os

import pandas as pd

S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
S1_COLLECTION = "COPERNICUS/S1_GRD"
L8_COLLECTION = "LANDSAT/LC08/C02/T1_L2"
L9_COLLECTION = "LANDSAT/LC09/C02/T1_L2"

MIN_YEAR = 2015   # Sentinel-1 / Landsat 8; Sentinel-2 SR starts 2017-03
MAX_YEAR = 2026

# Same order and colours as the charts of the original apps.
SERIES = {
    "S1_VV_DESC": {"label": "S1 VV Descending", "group": "Sentinel-1 Descending", "color": "#e67e22", "unit": "VV (dB)"},
    "S1_VH_DESC": {"label": "S1 VH Descending", "group": "Sentinel-1 Descending", "color": "#d35400", "unit": "VH (dB)"},
    "S1_VV_ASC":  {"label": "S1 VV Ascending",  "group": "Sentinel-1 Ascending",  "color": "#2980b9", "unit": "VV (dB)"},
    "S1_VH_ASC":  {"label": "S1 VH Ascending",  "group": "Sentinel-1 Ascending",  "color": "#1a5276", "unit": "VH (dB)"},
    "S2_INDEX":   {"label": "Sentinel-2",       "group": "Optical",               "color": "#27ae60", "unit": None},
    "LS_INDEX":   {"label": "Landsat 8/9",      "group": "Optical",               "color": "#8e44ad", "unit": None},
}

# Products are what gets queried; each yields one or two SERIES.
PRODUCTS = {
    "s1_desc": {"label": "Sentinel-1 VV+VH Descending", "series": ["S1_VV_DESC", "S1_VH_DESC"]},
    "s1_asc":  {"label": "Sentinel-1 VV+VH Ascending",  "series": ["S1_VV_ASC", "S1_VH_ASC"]},
    "s2":      {"label": "Sentinel-2 optical index",    "series": ["S2_INDEX"]},
    "landsat": {"label": "Landsat 8/9 optical index",   "series": ["LS_INDEX"]},
}

OPTICAL_INDICES = {
    "NDVI": "Normalized Difference Vegetation Index",
    "EVI":  "Enhanced Vegetation Index",
    "SAVI": "Soil-Adjusted Vegetation Index",
    "NDWI": "Normalized Difference Water Index (McFeeters)",
    "NDMI": "Normalized Difference Moisture Index",
}
INDEX_RANGE = {"NDVI": (-0.2, 1.0), "EVI": (-0.2, 1.0), "SAVI": (-0.2, 1.0),
               "NDWI": (-1.0, 0.6), "NDMI": (-0.6, 0.8)}

# Default polygon colours: the three of the original app first.
POLYGON_COLOURS = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#F7B731", "#A55EEA",
                   "#26DE81", "#FD9644", "#2D98DA", "#EB3B5A", "#778CA3"]

COMPOSITE_MODES = {
    "true":  {"label": "True colour (B4/B3/B2)",
              "vis": {"bands": ["B4", "B3", "B2"], "min": 0, "max": 3000, "gamma": 1.4}},
    "false": {"label": "False colour NIR (B8/B4/B3)",
              "vis": {"bands": ["B8", "B4", "B3"], "min": 0, "max": 5000, "gamma": 1.2}},
    "ndvi":  {"label": "NDVI",
              "vis": {"min": -0.2, "max": 0.8,
                      "palette": ["d73027", "f46d43", "fdae61", "fee08b",
                                  "d9ef8b", "a6d96a", "66bd63", "1a9850"]}},
}
MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December"]

_initialised = False


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------
def import_ee():
    """Returns the `ee` module. The official "Google Earth Engine" QGIS
    plugin ships earthengine-api in its extlibs folder and puts it on
    sys.path when it loads -- if it hasn't loaded yet, do that ourselves."""
    try:
        import ee
        return ee
    except ImportError:
        pass
    try:
        import ee_plugin
        ee_plugin.add_ee_dependencies()
        import ee
        return ee
    except Exception as e:
        raise ImportError(
            "The earthengine-api package was not found. Install and enable the "
            "'Google Earth Engine' plugin (Plugins -> Manage and Install Plugins), "
            "or install it into QGIS's Python: python -m pip install earthengine-api"
        ) from e


def ee_plugin_project():
    """The Cloud project the Google Earth Engine plugin was set up with
    (stored next to the OAuth credentials), or None."""
    ee = import_ee()
    try:
        with open(ee.oauth.get_credentials_path()) as f:
            return json.load(f).get("project")
    except Exception:
        return None


def ensure_initialised(project=None, force=False):
    """Makes sure ee is initialised. If the Google Earth Engine plugin is
    active it has already done that at QGIS start-up and this is a cheap
    no-op; otherwise initialise with the given / stored project."""
    global _initialised
    ee = import_ee()
    if _initialised and not force:
        return ee
    if not force:
        try:
            ee.Number(1).getInfo()
            _initialised = True
            return ee
        except Exception:
            pass
    ee.Initialize(project=project or ee_plugin_project())
    ee.Number(1).getInfo()
    _initialised = True
    return ee


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def regions_fc(polygons):
    """polygons: list of dicts {pid, name, geojson} (geojson in EPSG:4326).
    Only `pid` is kept as a property, so reduceRegions' output columns are
    exactly pid + band means."""
    ee = import_ee()
    feats = [ee.Feature(ee.Geometry(p["geojson"], None, False), {"pid": int(p["pid"])})
             for p in polygons]
    return ee.FeatureCollection(feats)


def bbox_geometry(bbox, buffer_m=0):
    """bbox = (xmin, ymin, xmax, ymax) in EPSG:4326."""
    ee = import_ee()
    geom = ee.Geometry.Rectangle(list(bbox), None, False)
    if buffer_m:
        geom = geom.buffer(buffer_m).bounds()
    return geom


# ---------------------------------------------------------------------------
# Collections -- each image gets a 0/1 'valid' band (unmasked, so pixels
# outside the scene footprint count as invalid) next to its value band(s).
# ---------------------------------------------------------------------------
def mask_s2_clouds(img):
    scl = img.select("SCL")
    clear = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7))
    return img.updateMask(clear)


def _optical_index(img, index):
    """img has common-name reflectance bands blue/green/red/nir/swir1 (0-1)."""
    if index == "NDVI":
        out = img.normalizedDifference(["nir", "red"])
    elif index == "NDWI":
        out = img.normalizedDifference(["green", "nir"])
    elif index == "NDMI":
        out = img.normalizedDifference(["nir", "swir1"])
    elif index == "EVI":
        out = img.expression(
            "2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)",
            {"nir": img.select("nir"), "red": img.select("red"), "blue": img.select("blue")})
    elif index == "SAVI":
        out = img.expression(
            "1.5 * (nir - red) / (nir + red + 0.5)",
            {"nir": img.select("nir"), "red": img.select("red")})
    else:
        raise ValueError(f"Unknown index {index}")
    return out.rename("value")


def sentinel2_collection(region, start, end, index="NDVI", max_cloud=70):
    ee = import_ee()

    def prep(img):
        scl = img.select("SCL")
        clear = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7))
        refl = (img.select(["B2", "B3", "B4", "B8", "B11"],
                           ["blue", "green", "red", "nir", "swir1"])
                .divide(10000).updateMask(clear))
        valid = clear.unmask(0).rename("valid")
        return (_optical_index(refl, index).addBands(valid)
                .copyProperties(img, ["system:time_start"]))

    return (ee.ImageCollection(S2_COLLECTION)
            .filterBounds(region)
            .filterDate(start, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
            .map(prep))


def landsat_collection(region, start, end, index="NDVI", max_cloud=70):
    ee = import_ee()

    def prep(img):
        qa = img.select("QA_PIXEL")
        clear = (qa.bitwiseAnd(1 << 1).eq(0)          # dilated cloud
                 .And(qa.bitwiseAnd(1 << 3).eq(0))    # cloud
                 .And(qa.bitwiseAnd(1 << 4).eq(0)))   # cloud shadow
        refl = (img.select(["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6"],
                           ["blue", "green", "red", "nir", "swir1"])
                .multiply(0.0000275).add(-0.2).updateMask(clear))
        valid = clear.unmask(0).rename("valid")
        return (_optical_index(refl, index).addBands(valid)
                .copyProperties(img, ["system:time_start"]))

    l8 = (ee.ImageCollection(L8_COLLECTION).filterBounds(region).filterDate(start, end)
          .filter(ee.Filter.lt("CLOUD_COVER", max_cloud)))
    l9 = (ee.ImageCollection(L9_COLLECTION).filterBounds(region).filterDate(start, end)
          .filter(ee.Filter.lt("CLOUD_COVER", max_cloud)))
    return l8.merge(l9).map(prep)


def sentinel1_collection(region, start, end, orbit):
    """Raw GRD backscatter (dB), VV + VH, one pass direction. Pixels below
    -30 dB (GRD border noise) are masked, as in the Field Inspector."""
    ee = import_ee()

    def prep(img):
        good = img.select("VV").gt(-30).And(img.select("VH").gt(-30))
        bands = img.select(["VV", "VH"]).updateMask(good)
        valid = good.unmask(0).rename("valid")
        return bands.addBands(valid).copyProperties(img, ["system:time_start"])

    return (ee.ImageCollection(S1_COLLECTION)
            .filterBounds(region)
            .filterDate(start, end)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.eq("orbitProperties_pass", orbit))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .map(prep))


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
def _reduce_to_table(collection, fc, scale, tile_scale=4):
    ee = import_ee()

    def per_image(img):
        stats = img.reduceRegions(collection=fc, reducer=ee.Reducer.mean(),
                                  scale=scale, tileScale=tile_scale)
        t = img.get("system:time_start")
        return stats.map(lambda f: ee.Feature(None, f.toDictionary().set("t", t)))

    return collection.map(per_image).flatten()


def _fetch_table(fc):
    ee = import_ee()
    df = ee.data.computeFeatures({"expression": fc, "fileFormat": "PANDAS_DATAFRAME"})
    if "geo" in df.columns:
        df = df.drop(columns=["geo"])
    return df


def _melt(df, mapping):
    """df has columns t, pid, valid + value columns; mapping {column: series_key}."""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["date", "pid", "series", "value", "valid"])
    rows = []
    for col, key in mapping.items():
        if col not in df.columns:
            continue
        part = pd.DataFrame({
            "date": pd.to_datetime(df["t"], unit="ms"),
            "pid": df["pid"].astype(int),
            "series": key,
            "value": pd.to_numeric(df[col], errors="coerce"),
            "valid": pd.to_numeric(df.get("valid"), errors="coerce"),
        })
        rows.append(part)
    if not rows:
        return pd.DataFrame(columns=["date", "pid", "series", "value", "valid"])
    return pd.concat(rows, ignore_index=True)


def extract_product(product, polygons, start, end, index="NDVI", max_cloud=70):
    """One product -> long-format DataFrame [date, pid, series, value, valid]
    with one row per image x polygon x series (unfiltered)."""
    fc = regions_fc(polygons)
    if product in ("s1_desc", "s1_asc"):
        orbit = "DESCENDING" if product == "s1_desc" else "ASCENDING"
        suffix = "DESC" if product == "s1_desc" else "ASC"
        coll = sentinel1_collection(fc, start, end, orbit)
        df = _fetch_table(_reduce_to_table(coll, fc, scale=10))
        return _melt(df, {"VV": f"S1_VV_{suffix}", "VH": f"S1_VH_{suffix}"})
    if product == "s2":
        coll = sentinel2_collection(fc, start, end, index, max_cloud)
        df = _fetch_table(_reduce_to_table(coll, fc, scale=10))
        return _melt(df, {"value": "S2_INDEX"})
    if product == "landsat":
        coll = landsat_collection(fc, start, end, index, max_cloud)
        df = _fetch_table(_reduce_to_table(coll, fc, scale=30))
        return _melt(df, {"value": "LS_INDEX"})
    raise ValueError(f"Unknown product {product}")


def extract_time_series(polygons, start, end, products, index="NDVI", max_cloud=70,
                        progress_cb=None, cancel_cb=None):
    """Runs extract_product for each selected product. progress_cb(i, n, msg)
    is called before each request; cancel_cb() -> True aborts between
    requests. Returns (DataFrame, {product: error message})."""
    ensure_initialised()
    frames, errors = [], {}
    for i, product in enumerate(products):
        if cancel_cb is not None and cancel_cb():
            break
        if progress_cb is not None:
            progress_cb(i, len(products), f"Extracting {PRODUCTS[product]['label']}")
        try:
            frames.append(extract_product(product, polygons, start, end, index, max_cloud))
        except Exception as e:
            errors[product] = str(e)
    frames = [f for f in frames if len(f)]
    if frames:
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.DataFrame(columns=["date", "pid", "series", "value", "valid"])
    return df, errors


def filter_series(df, min_valid=0.5):
    """Drops images where less than `min_valid` of the polygon was usable
    (clouds / outside the scene / S1 border noise), then merges same-day
    acquisitions (overlapping S2 tiles, adjacent S1 slices) into one mean
    value per day -- the charts of the original apps showed these as
    duplicate points."""
    if df is None or len(df) == 0:
        return df
    sub = df[(df["valid"] >= min_valid) & df["value"].notna()].copy()
    if len(sub) == 0:
        return sub
    sub["date"] = sub["date"].dt.normalize()
    return (sub.groupby(["series", "pid", "date"], as_index=False)
            .agg(value=("value", "mean"), valid=("valid", "max"))
            .sort_values(["series", "pid", "date"]))


def deviation_scores(df, series_keys, pids, min_polygons=3):
    """Outlier score per polygon: mean |z-score| of its value against all
    `pids` on the same acquisition date, averaged over `series_keys`. Dates
    with fewer than `min_polygons` observations are ignored. Polygons without
    any comparable date get NaN."""
    collected = {pid: [] for pid in pids}
    if df is None or len(df) == 0 or len(pids) < min_polygons:
        return {pid: float("nan") for pid in pids}
    for key in series_keys:
        part = df[(df["series"] == key) & df["pid"].isin(pids)]
        if len(part) == 0:
            continue
        piv = part.pivot_table(index="date", columns="pid", values="value", aggfunc="mean")
        piv = piv[piv.count(axis=1) >= min_polygons]
        if len(piv) == 0:
            continue
        sd = piv.std(axis=1).replace(0, float("nan"))
        z = piv.sub(piv.mean(axis=1), axis=0).div(sd, axis=0).abs()
        for pid in z.columns:
            m = z[pid].mean()
            if pd.notna(m):
                collected[int(pid)].append(float(m))
    return {pid: (sum(v) / len(v) if v else float("nan")) for pid, v in collected.items()}


# ---------------------------------------------------------------------------
# Monthly Sentinel-2 composites (the May / July / September RGB layers)
# ---------------------------------------------------------------------------
def s2_monthly_composite(region, year, month, mode="true", max_cloud=60):
    """Cloud-masked median composite of one calendar month, clipped to
    `region`. Returns (ee.Image, vis_params)."""
    ee = import_ee()
    start = ee.Date.fromYMD(year, month, 1)
    coll = (ee.ImageCollection(S2_COLLECTION)
            .filterBounds(region)
            .filterDate(start, start.advance(1, "month"))
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
            .map(mask_s2_clouds))
    img = coll.median()
    if mode == "ndvi":
        img = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
    return img.clip(region), COMPOSITE_MODES[mode]["vis"]


def composite_layer_name(mode, year, month):
    prefix = {"true": "S2 RGB", "false": "S2 NIR", "ndvi": "S2 NDVI"}[mode]
    return f"{prefix} {MONTH_NAMES[month - 1]} {year}"


# ---------------------------------------------------------------------------
# GEE FeatureCollection assets -> GeoJSON features (Field Inspector mode)
# ---------------------------------------------------------------------------
def fetch_asset_features(asset_id, progress_cb=None, page_size=2000):
    """Downloads a whole table asset as a list of GeoJSON features (EPSG:4326).
    Uses computeFeatures paging, so it isn't capped at getInfo()'s 5000."""
    ee = ensure_initialised()
    fc = ee.FeatureCollection(asset_id)
    features, token, page = [], None, 0
    while True:
        params = {"expression": fc, "pageSize": page_size}
        if token:
            params["pageToken"] = token
        resp = ee.data.computeFeatures(params)
        features.extend(resp.get("features", []))
        page += 1
        if progress_cb is not None:
            progress_cb(page, page + 1, f"Downloaded {len(features)} features")
        token = resp.get("nextPageToken") or resp.get("next_page_token")
        if not token:
            break
    return features


# ---------------------------------------------------------------------------
# Local GeoTIFF downloads (computePixels in tiles) and Drive exports -- the
# Earth Engine map layers in QGIS are tile services and can't be saved.
# ---------------------------------------------------------------------------
MAX_DOWNLOAD_PIXELS = 400_000_000
_DTYPES = {"uint8": ("GDT_Byte", 1), "uint16": ("GDT_UInt16", 2), "int16": ("GDT_Int16", 2),
           "float32": ("GDT_Float32", 4)}


def utm_epsg(region):
    """'EPSG:326xx' / 'EPSG:327xx' of the region's centroid (export default)."""
    lon, lat = region.centroid(100).coordinates().getInfo()
    return f"EPSG:{(32600 if lat >= 0 else 32700) + int((lon + 180) // 6) % 60 + 1}"


def download_grid(region, scale=10, crs=None):
    """Pixel grid of the region's bounding box in `crs` (default: UTM of the
    region) -> dict(crs, x0, y1, width, height, scale)."""
    import math
    ee = ensure_initialised()
    crs = crs or utm_epsg(region)
    ring = region.bounds(1, ee.Projection(crs)).coordinates().getInfo()[0]
    xs, ys = [q[0] for q in ring], [q[1] for q in ring]
    x0 = math.floor(min(xs) / scale) * scale
    y1 = math.ceil(max(ys) / scale) * scale
    width = int(math.ceil((max(xs) - x0) / scale))
    height = int(math.ceil((y1 - min(ys)) / scale))
    return {"crs": crs, "x0": x0, "y1": y1, "width": width, "height": height, "scale": scale}


def _fetch_tile(image, grid, bands, xoff, yoff, w, h, retries=4):
    """One block of pixels -> list of 2-D arrays, one per band."""
    import time
    ee = import_ee()
    s = grid["scale"]
    request = {
        "expression": image, "fileFormat": "NUMPY_NDARRAY",
        "grid": {"dimensions": {"width": w, "height": h},
                 "affineTransform": {"scaleX": s, "shearX": 0, "translateX": grid["x0"] + xoff * s,
                                     "shearY": 0, "scaleY": -s, "translateY": grid["y1"] - yoff * s},
                 "crsCode": grid["crs"]},
    }
    for attempt in range(retries):
        try:
            arr = ee.data.computePixels(request)
            if arr.dtype.names:
                return [arr[b] for b in bands]
            return [arr] if arr.ndim == 2 else [arr[..., i] for i in range(arr.shape[-1])]
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))


def download_image(image, region, path, bands, dtype="float32", nodata=None, scale=10, crs=None,
                   palette=None, metadata=None, band_meta=None, workers=4, progress_cb=None, cancel_cb=None):
    """Computes `image` (already cast to `dtype`, masked pixels unmasked to
    `nodata`) over the region's bounding box tile by tile, in parallel, and
    writes a tiled, compressed GeoTIFF with overviews. Band descriptions =
    `bands`, so the file works as a raster stack for local classification.
    palette = optional [(value, (r, g, b), name)] -> colour table + category
    names (single band); metadata = optional dataset tags; band_meta =
    optional per-band dicts (key "scale" -> GDAL scale factor, the rest ->
    band metadata). Returns path."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from osgeo import gdal, osr

    grid = download_grid(region, scale, crs)
    W, H = grid["width"], grid["height"]
    if W * H > MAX_DOWNLOAD_PIXELS:
        raise RuntimeError(f"The area is too large for a direct download ({W} × {H} pixels at {scale} m). "
                           "Use a coarser scale or the export to Drive.")
    gdt, nbytes = _DTYPES[dtype]
    # computePixels returns at most ~48 MB per request
    tile = 1024
    while tile > 128 and tile * tile * nbytes * len(bands) > 32_000_000:
        tile //= 2

    ds = gdal.GetDriverByName("GTiff").Create(
        path, W, H, len(bands), getattr(gdal, gdt),
        ["TILED=YES", "COMPRESS=DEFLATE", "BIGTIFF=IF_SAFER",
         "PREDICTOR=3" if dtype == "float32" else "PREDICTOR=2"])
    if ds is None:
        raise RuntimeError(f"Could not create {path}")
    ds.SetGeoTransform((grid["x0"], scale, 0, grid["y1"], 0, -scale))
    srs = osr.SpatialReference()
    srs.SetFromUserInput(grid["crs"])
    ds.SetProjection(srs.ExportToWkt())
    if metadata:
        ds.SetMetadata({key: str(v) for key, v in metadata.items()})
    for i, name in enumerate(bands):
        band = ds.GetRasterBand(i + 1)
        band.SetDescription(name)
        if nodata is not None:
            band.SetNoDataValue(nodata)
        if band_meta and i < len(band_meta) and band_meta[i]:
            extra = dict(band_meta[i])
            if "scale" in extra:
                band.SetScale(float(extra.pop("scale")))
                band.SetOffset(0.0)
            if extra:
                band.SetMetadata({key: str(v) for key, v in extra.items()})
    if palette:
        band = ds.GetRasterBand(1)
        ct = gdal.ColorTable()
        names = [""] * 256
        for value, rgb, name in palette:
            ct.SetColorEntry(int(value), tuple(rgb) + (255,))
            names[int(value)] = name
        band.SetRasterColorTable(ct)
        band.SetRasterColorInterpretation(gdal.GCI_PaletteIndex)
        band.SetCategoryNames(names)
    band = None

    jobs = [(x, y, min(tile, W - x), min(tile, H - y))
            for y in range(0, H, tile) for x in range(0, W, tile)]
    done = 0
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_fetch_tile, image, grid, bands, *j): j for j in jobs}
            for fut in as_completed(futures):
                if cancel_cb is not None and cancel_cb():
                    for f in futures:
                        f.cancel()
                    raise InterruptedError("Cancelled")
                x, y, _, _ = futures[fut]
                for i, arr in enumerate(fut.result()):
                    ds.GetRasterBand(i + 1).WriteArray(arr, x, y)
                done += 1
                if progress_cb is not None:
                    progress_cb(done, len(jobs) + 1, f"Downloaded tile {done}/{len(jobs)}")
        if progress_cb is not None:
            progress_cb(len(jobs), len(jobs) + 1, "Building overviews")
        ds.BuildOverviews("NEAREST" if palette else "AVERAGE",
                          [f for f in (2, 4, 8, 16, 32) if min(W, H) // f >= 64])
    except BaseException:
        ds = None
        try:
            os.remove(path)
        except OSError:
            pass
        raise
    ds = None
    return path
