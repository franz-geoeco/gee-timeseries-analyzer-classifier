"""
Spectra from local Sentinel-2 L2A scenes (Spectra section, offline): a
.SAFE.zip as downloaded from the Copernicus Data Space, an unzipped .SAFE
folder or its MTD_MSIL2A.xml.

The JP2 bands are read directly (GDAL /vsizip/, no unpacking), each at its
finest resolution, converted to surface reflectance with the product's
BOA_ADD_OFFSET / BOA_QUANTIFICATION_VALUE (processing baseline >= 04.00 has
an offset of -1000) and masked with the scene classification (SCL) like the
Earth Engine version. Results have the same structure as
spectral.closest_spectrum. GDAL + numpy, no Qt / QGIS imports.
"""
import datetime as dt
import json
import os
import re
import zipfile
from xml.parsers import expat

import numpy as np

from .spectral import SENSORS

BANDS = SENSORS["s2"]["bands"]                     # B1 ... B12 (Earth Engine names)
FILE_BAND = {b: (f"B{int(b[1:]):02d}" if b[1:].isdigit() else b) for b in BANDS}   # B1 -> B01, B8A -> B8A
SCL_BAD = (1, 3, 8, 9, 10)                         # saturated, cloud shadow, cloud medium / high, cirrus
_BAND_FILE = re.compile(r"_(B\d{2}|B8A|SCL|TCI)_(10|20|60)m\.jp2$", re.I)


class _El:
    __slots__ = ("tag", "attrib", "text")

    def __init__(self, tag, attrib):
        self.tag, self.attrib, self.text = tag, attrib, ""

    def get(self, key, default=None):
        return self.attrib.get(key, default)


def _parse_xml(data):
    """Elements of an XML document in document order (tag without namespace
    prefix, attributes, own text). Plain expat with every DTD / entity
    declaration refused, so entity expansion ("billion laughs") and external
    entities are impossible -- the protection of defusedxml without the extra
    dependency. Sentinel-2 metadata never contains a DTD."""
    def refuse(*_):
        raise ValueError("XML with a DTD or entity declarations is not accepted.")

    els, stack = [], []

    def start(name, attrs):
        el = _El(name.split(":")[-1], attrs)
        els.append(el)
        stack.append(el)

    def end(_name):
        stack.pop()

    def chars(text):
        if stack:
            stack[-1].text += text

    p = expat.ParserCreate()
    p.SetParamEntityParsing(expat.XML_PARAM_ENTITY_PARSING_NEVER)
    p.StartDoctypeDeclHandler = refuse
    p.EntityDeclHandler = refuse
    p.ExternalEntityRefHandler = refuse
    p.StartElementHandler, p.EndElementHandler, p.CharacterDataHandler = start, end, chars
    p.Parse(data, True)
    return els


def _xml_text(els, tag):
    el = next((e for e in els if e.tag == tag), None)
    return el.text.strip() if el is not None and el.text.strip() else None


def open_scene(path):
    """Reads the file list and metadata of an L2A product -> scene dict.
    Any other raster file (GeoTIFF / VRT stack, e.g. from export_scene) is
    opened with open_raster."""
    path = os.path.abspath(path)
    low = path.lower()
    if os.path.isfile(path) and not low.endswith((".zip", ".xml")):
        return open_raster(path)
    if low.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            mtd = next((n for n in names if n.endswith("MTD_MSIL2A.xml")), None)
            if mtd is None:
                raise ValueError("No MTD_MSIL2A.xml in the zip — is this a Sentinel-2 Level-2A product?")
            xml = z.read(mtd)
        prefix = "/vsizip/" + path.replace("\\", "/") + "/"
        files = [prefix + n for n in names]
    else:
        safe = os.path.dirname(path) if path.lower().endswith(".xml") else path
        mtd_path = os.path.join(safe, "MTD_MSIL2A.xml")
        if not os.path.exists(mtd_path):
            raise ValueError("No MTD_MSIL2A.xml found — choose a Sentinel-2 L2A .SAFE.zip, .SAFE folder or its "
                             "MTD_MSIL2A.xml.")
        xml = open(mtd_path, "rb").read()
        files = [os.path.join(r, f) for r, _, fs in os.walk(safe) for f in fs]
    root = _parse_xml(xml)

    best = {}                                       # band -> (resolution, file), finest wins
    for f in files:
        m = _BAND_FILE.search(f)
        if m and "IMG_DATA" in f.replace("\\", "/"):
            band, res = m.group(1).upper(), int(m.group(2))
            if band not in best or res < best[band][0]:
                best[band] = (res, f)
    missing = [b for b in BANDS if FILE_BAND[b] not in best] + ([] if "SCL" in best else ["SCL"])
    if missing:
        raise ValueError(f"Bands missing in the product: {', '.join(missing)}")

    phys = {}                                       # bandId -> physical band (B1 ... B12)
    for e in root:
        if e.tag == "Spectral_Information" and e.get("physicalBand"):
            phys[e.get("bandId")] = e.get("physicalBand")
    offsets = {}
    for e in root:
        if e.tag == "BOA_ADD_OFFSET":
            offsets[phys.get(e.get("band_id"), "")] = float(e.text)
    quant = float(_xml_text(root, "BOA_QUANTIFICATION_VALUE") or 10000)
    start = _xml_text(root, "PRODUCT_START_TIME") or ""
    product = os.path.basename(path).replace(".zip", "").replace(".SAFE", "") if not path.endswith(".xml") \
        else os.path.basename(os.path.dirname(path)).replace(".SAFE", "")
    tile = re.search(r"_T(\d{2}[A-Z]{3})_", product)
    return {"kind": "safe", "path": path, "product": product, "date": start[:10], "time": start,
            "spacecraft": _xml_text(root, "SPACECRAFT_NAME") or "Sentinel-2",
            "baseline": _xml_text(root, "PROCESSING_BASELINE"), "tile": tile.group(1) if tile else "",
            "cloud": float(_xml_text(root, "Cloud_Coverage_Assessment") or "nan"),
            "files": {k: v for k, v in best.items()}, "offsets": offsets, "quant": quant, "_ds": {}}


_CRAFT = (("S2A", "Sentinel-2A"), ("S2B", "Sentinel-2B"), ("S2C", "Sentinel-2C"), ("S2D", "Sentinel-2D"),
          ("LC09", "Landsat 9"), ("LC08", "Landsat 8"), ("LE07", "Landsat 7"), ("LT05", "Landsat 5"),
          ("LT04", "Landsat 4"))


def _date_from_name(text):
    for m in re.finditer(r"(?<!\d)((?:19|20)\d{2})(\d{2})(\d{2})(?!\d)", text):
        try:
            return dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3))).isoformat()
        except ValueError:
            continue
    return ""


def _clean_name(label):
    """'B8 NIR (842 nm)' / 'B4, central wavelength 665 nm' -> 'B8 NIR' / 'B4'."""
    label = re.sub(r",?\s*central wavelength\s*\d+(\.\d+)?\s*nm", "", label, flags=re.I)
    label = re.sub(r"\s*\(\s*\d{3,4}(\.\d+)?\s*nm\s*\)", "", label)
    return label.strip() or label


def open_raster(path):
    """A GeoTIFF / VRT stack as a local scene -- e.g. the GeoTIFF of
    export_scene, but any Sentinel-2 / Landsat stack works whose bands can
    be recognised (indices.describe_bands: wavelength metadata, band
    names, or the sensor's band order). Reflectance scaling as in
    indices.resolve_scaling; an SCL band (name 'SCL') gives the cloud mask;
    product, date, spacecraft, tile and cloud cover from the metadata
    (export_scene writes them) or the file name."""
    from osgeo import gdal
    from . import indices as ix
    ds = gdal.Open(path)
    if ds is None:
        raise ValueError("GDAL cannot open this file.")
    sensor, bands = ix.describe_bands(path)
    spec = sorted((b for b in bands if b["wavelength"] and 400 <= b["wavelength"] <= 2500),
                  key=lambda b: b["wavelength"])
    if len(spec) < 3:
        raise ValueError("No spectral bands recognised. The raster needs band names or wavelengths (e.g. "
                         "'Export…' of a local scene) or a Sentinel-2 / Landsat product name in the file name, so "
                         "that the usual band order can be used.")
    scl = None
    for i in range(ds.RasterCount):
        b = ds.GetRasterBand(i + 1)
        if re.search(r"\bSCL\b", f"{b.GetMetadataItem('BANDNAME') or ''} {b.GetDescription() or ''}", re.I):
            scl = i + 1
    scaling, how = ix.resolve_scaling(path, {f"b{b['index']}": b["index"] for b in spec}, sensor)
    md = {k.upper(): v for k, v in (ds.GetMetadata() or {}).items()}
    base = os.path.splitext(os.path.basename(path))[0]
    product = md.get("PRODUCT") or md.get("SENTINEL2_PRODUCT") or md.get("PRODUCT_URI", "").replace(".SAFE", "") \
        or base
    text = f"{product} {base}"
    t = md.get("SENSING_TIME") or md.get("PRODUCT_START_TIME") or md.get("DATATAKE_1_DATATAKE_SENSING_START") or ""
    date = t[:10] if re.match(r"\d{4}-\d{2}-\d{2}", t) else _date_from_name(text)
    craft = md.get("SPACECRAFT") or md.get("SPACECRAFT_NAME") or md.get("DATATAKE_1_SPACECRAFT_NAME") or next(
        (name for code, name in _CRAFT if code in text.upper()),
        ix.SENSORS[sensor]["label"] if sensor in ix.SENSORS else "Raster")
    tile = md.get("TILE") or (re.search(r"_T(\d{2}[A-Z]{3})_", text) or [None, ""])[1]
    try:
        cloud = float(md.get("CLOUD_COVER_PERCENT") or md.get("CLOUD_COVERAGE_ASSESSMENT") or "nan")
    except ValueError:
        cloud = float("nan")
    ids, names = [], []
    known = {}                                       # band id -> name of the plugin's sensor tables
    if sensor == "s2":
        known = {b: n for b, n in zip(SENSORS["s2"]["bands"], SENSORS["s2"]["names"])}
    elif sensor == "l89":
        known = {b.replace("SR_", ""): n for b, n in zip(SENSORS["landsat"]["bands"], SENSORS["landsat"]["names"])}
    for b in spec:
        band = ds.GetRasterBand(b["index"])
        assumed = re.search(r"\(= (B\w+)\?\)", b["label"])           # assigned by the usual band order
        bid = band.GetMetadataItem("BANDNAME") or ix._band_id(band.GetDescription()) or \
            (assumed.group(1) if assumed else f"band {b['index']}")
        ids.append(bid)
        names.append(known.get(bid.replace("SR_", "")) or _clean_name(b["label"]))
    return {"kind": "raster", "path": path, "product": product, "date": date, "time": t, "spacecraft": craft,
            "baseline": md.get("PROCESSING_BASELINE"), "tile": tile or "", "cloud": cloud, "files": {}, "_ds": {},
            "sensor": sensor, "spec": [b["index"] for b in spec], "band_ids": ids,
            "names": names, "wavelengths": [b["wavelength"] for b in spec],
            "scaling": scaling, "scaling_how": how, "bands_how": ", ".join(sorted({b["how"] for b in spec})),
            "scl": scl}


def raster_rgb(scene):
    """[(band, min, max)] for red / green / blue of an open_raster scene:
    reflectance 0-0.3 expressed in the values QGIS shows (QGIS applies a
    stored scale / offset itself; otherwise the stored values). None if
    the raster has no red, green and blue bands."""
    from osgeo import gdal
    ds = gdal.Open(scene["path"])
    out = []
    for target, lo_w, hi_w in ((665, 625, 690), (560, 525, 600), (490, 455, 515)):
        cands = [(abs(w - target), i) for i, w in zip(scene["spec"], scene["wavelengths"]) if lo_w <= w <= hi_w]
        if not cands:
            return None
        i = min(cands)[1]
        b = ds.GetRasterBand(i)
        stored = (b.GetScale() not in (None, 1.0)) or (b.GetOffset() not in (None, 0.0))
        s, o = (1.0, 0.0) if stored else scene["scaling"].get(i, (1.0, 0.0))
        out.append((i, (0.0 - o) / s, (0.3 - o) / s))
    return out


def scene_label(scene):
    tile = f" T{scene['tile']}" if scene.get("tile") else ""
    return f"{scene['spacecraft'].replace('Sentinel-', 'S')} {scene['date'] or 'date?'}{tile} (local)"


def scene_info(scene):
    """One line for tooltips."""
    if scene.get("kind") == "raster":
        return (f"{os.path.basename(scene['path'])} · {len(scene['spec'])} spectral bands "
                f"({scene['bands_how']}) · reflectance: {scene['scaling_how']} · "
                + ("SCL cloud mask" if scene["scl"] else "no SCL band: no cloud mask"))
    return f"{scene['product']} · processing baseline {scene['baseline']}"


def _mask(geom, srs, wkt, x0, y1, w, h, cx, cy):
    """Pixels of a w x h window (top-left x0, y1; cell cx, cy) inside the polygon."""
    from osgeo import gdal, ogr
    mem = gdal.GetDriverByName("MEM").Create("", w, h, 1, gdal.GDT_Byte)
    mem.SetGeoTransform((x0, cx, 0, y1, 0, -cy))
    mem.SetProjection(wkt)
    ods = (ogr.GetDriverByName("MEM") or ogr.GetDriverByName("Memory")).CreateDataSource("p")
    lyr = ods.CreateLayer("p", srs, ogr.wkbMultiPolygon)
    f = ogr.Feature(lyr.GetLayerDefn())
    f.SetGeometry(geom)
    lyr.CreateFeature(f)
    gdal.RasterizeLayer(mem, [1], lyr, burn_values=[1])
    inside = mem.GetRasterBand(1).ReadAsArray() > 0
    if not inside.any():                             # polygon smaller than a pixel: its centre pixel
        inside[h // 2, w // 2] = True
    return inside


def _to_raster_crs(geojson, wkt):
    from osgeo import ogr, osr
    srs = osr.SpatialReference(wkt=wkt)
    wgs = osr.SpatialReference()
    wgs.ImportFromEPSG(4326)
    for s in (srs, wgs):
        s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    geom = ogr.CreateGeometryFromJson(json.dumps(geojson))
    geom.Transform(osr.CoordinateTransformation(wgs, srs))
    return geom, srs


def _days(scene, target_date):
    if target_date and scene.get("date"):
        return (dt.date.fromisoformat(scene["date"]) - dt.date.fromisoformat(target_date)).days
    return 0


def _sample_raster(scene, geojson, target_date=None, cloud_mask=True, min_clear=0.5):
    """sample_scene for open_raster scenes: on the raster's own grid."""
    from osgeo import gdal
    ds = scene["_ds"].get("raster")
    if ds is None:
        ds = scene["_ds"]["raster"] = gdal.Open(scene["path"])
    wkt = ds.GetProjection()
    geom, srs = _to_raster_crs(geojson, wkt)
    is_point = geojson.get("type") == "Point"
    gt = ds.GetGeoTransform()
    cx, cy = gt[1], -gt[5]
    minx, maxx, miny, maxy = geom.GetEnvelope()
    c0 = int(np.floor((minx - gt[0]) / cx))
    r0 = int(np.floor((gt[3] - maxy) / cy))
    c1 = max(c0 + 1, int(np.ceil((maxx - gt[0]) / cx)))
    r1 = max(r0 + 1, int(np.ceil((gt[3] - miny) / cy)))
    c0, r0 = max(0, c0), max(0, r0)
    c1, r1 = min(ds.RasterXSize, c1), min(ds.RasterYSize, r1)
    if c1 <= c0 or r1 <= r0:
        return None                                  # outside the raster
    w, h = c1 - c0, r1 - r0
    inside = np.ones((h, w), bool) if is_point else _mask(geom, srs, wkt, gt[0] + c0 * cx, gt[3] - r0 * cy,
                                                          w, h, cx, cy)
    arrays = {}
    for i in scene["spec"]:
        band = ds.GetRasterBand(i)
        a = band.ReadAsArray(c0, r0, w, h).astype(np.float64)
        bad = ~np.isfinite(a)
        nd = band.GetNoDataValue()
        if nd is not None and np.isfinite(nd):
            bad |= a == nd
        if band.DataType not in (gdal.GDT_Float32, gdal.GDT_Float64):
            bad |= a == 0
        s, o = scene["scaling"].get(i, (1.0, 0.0))
        a = a * s + o
        a[bad] = np.nan
        arrays[i] = a
    valid = np.any([np.isfinite(a) for a in arrays.values()], axis=0)
    covered = inside & valid
    if scene["scl"]:
        scl = ds.GetRasterBand(scene["scl"]).ReadAsArray(c0, r0, w, h)
        covered &= np.isfinite(scl) & (scl > 0)
        clear = covered & ~np.isin(scl, SCL_BAD) if cloud_mask else covered
    else:
        clear = covered
    if not covered.any():
        return None
    share = float(clear.sum() / inside.sum())
    if share < (1.0 if is_point else float(min_clear)):
        return None
    values, sd = [], []
    for i in scene["spec"]:
        v = arrays[i][clear]
        v = v[np.isfinite(v)]
        values.append(float(v.mean()) if v.size else None)
        sd.append(None if is_point or v.size < 2 else float(v.std(ddof=1)))
    return {"sensor": "local", "date": scene["date"] or "date?", "days": _days(scene, target_date),
            "image": scene["product"], "spacecraft": scene["spacecraft"], "cloud": scene["cloud"],
            "clear": share, "values": values, "sd": sd, "wavelengths": list(scene["wavelengths"]),
            "bands": list(scene["band_ids"]), "names": list(scene["names"])}


def _ds(scene, band):
    from osgeo import gdal
    if band not in scene["_ds"]:
        ds = gdal.Open(scene["files"][band][1])
        if ds is None:
            raise RuntimeError(f"Could not read {band} of {scene['product']}")
        scene["_ds"][band] = ds
    return scene["_ds"][band]


def _read_on_grid(ds, x0, y1, w, h, cell):
    """Values of `ds` (same CRS) on a grid of w x h cells of size `cell`
    starting at the top-left corner (x0, y1); nearest neighbour; 0 outside."""
    gt = ds.GetGeoTransform()
    xs = x0 + (np.arange(w) + 0.5) * cell
    ys = y1 - (np.arange(h) + 0.5) * cell
    cols = np.floor((xs - gt[0]) / gt[1]).astype(int)
    rows = np.floor((ys - gt[3]) / gt[5]).astype(int)
    c0, c1 = max(cols.min(), 0), min(cols.max(), ds.RasterXSize - 1)
    r0, r1 = max(rows.min(), 0), min(rows.max(), ds.RasterYSize - 1)
    out = np.zeros((h, w), dtype=np.float64)
    if c0 > c1 or r0 > r1:
        return out
    block = ds.GetRasterBand(1).ReadAsArray(int(c0), int(r0), int(c1 - c0 + 1), int(r1 - r0 + 1)).astype(np.float64)
    ok_c = (cols >= c0) & (cols <= c1)
    ok_r = (rows >= r0) & (rows <= r1)
    rr, cc = np.meshgrid(np.where(ok_r)[0], np.where(ok_c)[0], indexing="ij")
    out[rr, cc] = block[rows[rr] - r0, cols[cc] - c0]
    return out


def sample_scene(scene, geojson, target_date=None, cloud_mask=True, min_clear=0.5):
    """Spectrum of a point / polygon (GeoJSON, EPSG:4326) in one local scene
    -> dict like spectral.closest_spectrum, or None (outside the scene / not
    clear enough)."""
    if scene.get("kind") == "raster":
        return _sample_raster(scene, geojson, target_date, cloud_mask, min_clear)
    ref = _ds(scene, "SCL")
    geom, srs = _to_raster_crs(geojson, ref.GetProjection())
    is_point = geojson.get("type") == "Point"
    cell = 10.0
    gt = ref.GetGeoTransform()
    ext = (gt[0], gt[3] + gt[5] * ref.RasterYSize, gt[0] + gt[1] * ref.RasterXSize, gt[3])
    minx, maxx, miny, maxy = geom.GetEnvelope()
    if maxx < ext[0] or minx > ext[2] or maxy < ext[1] or miny > ext[3]:
        return None                                  # not in this tile
    # 10 m grid aligned to the tile
    x0 = gt[0] + np.floor((minx - gt[0]) / cell) * cell
    y1 = gt[3] - np.floor((gt[3] - maxy) / cell) * cell
    w = max(1, int(np.ceil((maxx - x0) / cell)))
    h = max(1, int(np.ceil((y1 - miny) / cell)))
    inside = np.ones((h, w), dtype=bool) if is_point else _mask(geom, srs, ref.GetProjection(), x0, y1, w, h,
                                                                 cell, cell)
    scl = _read_on_grid(ref, x0, y1, w, h, cell)
    covered = inside & (scl > 0)
    if not covered.any():
        return None
    clear = covered & ~np.isin(scl, SCL_BAD) if cloud_mask else covered
    share = float(clear.sum() / inside.sum())
    if share < (1.0 if is_point else float(min_clear)):
        return None
    values, sd = [], []
    for b in BANDS:
        fb = FILE_BAND[b]
        dn = _read_on_grid(_ds(scene, fb), x0, y1, w, h, cell)
        ok = clear & (dn > 0)
        refl = (dn[ok] + scene["offsets"].get(b, 0.0)) / scene["quant"]
        values.append(float(refl.mean()) if refl.size else None)
        sd.append(None if is_point or refl.size < 2 else float(refl.std(ddof=1)))
    return {"sensor": "local", "date": scene["date"], "days": _days(scene, target_date),
            "image": scene["product"], "spacecraft": scene["spacecraft"], "cloud": scene["cloud"],
            "clear": share, "values": values, "sd": sd, "wavelengths": SENSORS["s2"]["wavelengths"],
            "bands": BANDS, "names": SENSORS["s2"]["names"]}


def tci_path(scene):
    """True-colour image (TCI) at the finest resolution, for the map."""
    return scene["files"]["TCI"][1] if "TCI" in scene["files"] else None


RGB_BANDS = (4, 3, 2)                               # B4, B3, B2 in the all-bands VRT (1-based)
VRT_SUBDIR = ("gee_timeseries_analyzer", "local_s2")   # below the QGIS profile folder
SCL_CLASSES = ("0 no data, 1 saturated / defective, 2 dark area, 3 cloud shadow, 4 vegetation, 5 not vegetated, "
               "6 water, 7 unclassified, 8 cloud medium probability, 9 cloud high probability, 10 thin cirrus, "
               "11 snow / ice")


def build_vrt(scene, out_dir):
    """All spectral bands (B1 ... B12, each from its finest file, resampled
    to 10 m by nearest neighbour) plus SCL as one virtual raster in
    `out_dir` -> path. No pixels are copied; the VRT points into the .zip /
    .SAFE. Every band is named ("B8 NIR (842 nm)") and carries the product's
    scale (1 / quantification) and offset, so QGIS shows and identifies
    surface reflectance; DN 0 is nodata. Band order = BANDS, then SCL.

    Each band is first wrapped in its own 10 m VRT (in a sub-folder) and
    those are stacked: a stack that resamples the 20 / 60 m files itself
    bypasses the JPEG 2000 overviews, and drawing the whole tile then takes
    half a minute instead of a fraction of a second."""
    from osgeo import gdal
    work = os.path.join(out_dir, scene["product"])
    os.makedirs(work, exist_ok=True)
    path = os.path.join(out_dir, f"{scene['product']}_all_bands.vrt")
    keys = [FILE_BAND[b] for b in BANDS] + ["SCL"]
    ref = gdal.Open(scene["files"][FILE_BAND["B2"]][1])      # 10 m grid
    gt = ref.GetGeoTransform()
    size = (ref.RasterXSize, ref.RasterYSize)
    bounds = (gt[0], gt[3] + gt[5] * size[1], gt[0] + gt[1] * size[0], gt[3])
    ref = None
    singles = []
    for k in keys:
        src = scene["files"][k][1]
        single = os.path.join(work, f"{k}.vrt")
        if scene["files"][k][0] == 10:
            ok = gdal.Translate(single, src, format="VRT")
        else:
            # 20 / 60 m -> 10 m with the warper: a plain resampling VRT (Translate / BuildVRT)
            # rounds the source offset of windows that start at an odd 10 m pixel and then
            # reads the neighbouring 20 m pixel for ~3/4 of the pixels; the warper is exact
            ok = gdal.Warp(single, src, format="VRT", outputBounds=bounds, width=size[0], height=size[1],
                           resampleAlg="near", srcNodata=0, dstNodata=0)
        if ok is None:
            raise RuntimeError(f"Could not read {k} of {scene['product']}")
        ok = None
        singles.append(single)
    opts = gdal.BuildVRTOptions(separate=True, srcNodata=0, VRTNodata=0)
    ds = gdal.BuildVRT(path, singles, options=opts)
    if ds is None:
        raise RuntimeError(f"Could not build the band stack of {scene['product']}")
    names = SENSORS["s2"]["names"]
    waves = SENSORS["s2"]["wavelengths"]
    for i, b in enumerate(BANDS):
        band = ds.GetRasterBand(i + 1)
        band.SetDescription(f"{names[i]} ({waves[i]} nm)")
        band.SetScale(1.0 / scene["quant"])
        band.SetOffset(scene["offsets"].get(b, 0.0) / scene["quant"])
    ds.GetRasterBand(len(keys)).SetDescription("SCL scene classification")
    ds.SetMetadataItem("SENTINEL2_PRODUCT", scene["product"])
    ds.FlushCache()
    ds = None
    return path


def export_scene(scene, vrt_path, out_path, bbox=None, reflectance=True, include_scl=True,
                 progress=None, cancel=None):
    """Writes the all-bands VRT (build_vrt) as a GeoTIFF with its metadata:
    band names, central wavelengths (also as GDAL's IMAGERY domain
    CENTRAL_WAVELENGTH_UM), product, date, sensor, processing baseline.

    bbox: (xmin, ymin, xmax, ymax) in the scene CRS, snapped outwards to the
    10 m grid; None = whole tile (several GB). reflectance=True: Float32
    surface reflectance (0-1), NoData NaN; False: the original UInt16 values
    with scale / offset stored, so GIS software still reads reflectance
    (half the size). progress(percent), cancel() -> True to stop. -> out_path"""
    import math
    from osgeo import gdal
    src = gdal.Open(vrt_path)
    if src is None:
        raise RuntimeError(f"Could not open {vrt_path}")
    n = src.RasterCount if include_scl else src.RasterCount - 1
    gt = src.GetGeoTransform()
    cell = gt[1]
    kw = {}
    if bbox is not None:
        xmin, ymin, xmax, ymax = bbox
        x0 = max(gt[0], gt[0] + math.floor((xmin - gt[0]) / cell) * cell)
        x1 = min(gt[0] + src.RasterXSize * cell, gt[0] + math.ceil((xmax - gt[0]) / cell) * cell)
        y1 = min(gt[3], gt[3] - math.floor((gt[3] - ymax) / cell) * cell)
        y0 = max(gt[3] - src.RasterYSize * cell, gt[3] - math.ceil((gt[3] - ymin) / cell) * cell)
        if x1 <= x0 or y1 <= y0:
            raise ValueError("The extent does not overlap the scene.")
        kw["projWin"] = [x0, y1, x1, y0]
    if reflectance:
        kw.update(outputType=gdal.GDT_Float32, unscale=True, noData="nan")
        co = ["PREDICTOR=3"]
    else:
        kw.update(outputType=gdal.GDT_UInt16, noData=0)
        co = ["PREDICTOR=2"]
    co += ["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=IF_SAFER", "NUM_THREADS=ALL_CPUS"]

    def _cb(pct, _msg, _data):
        if progress:
            progress(100.0 * pct)
        return 0 if cancel and cancel() else 1

    out = gdal.Translate(out_path, src, format="GTiff", bandList=list(range(1, n + 1)), creationOptions=co,
                         callback=_cb, **kw)
    if out is None:
        raise RuntimeError("Export failed or was cancelled.")
    names = SENSORS["s2"]["names"]
    waves = SENSORS["s2"]["wavelengths"]
    for i in range(n):
        band = out.GetRasterBand(i + 1)
        if i < len(BANDS):
            band.SetDescription(f"{names[i]} ({waves[i]} nm)")
            band.SetMetadataItem("BANDNAME", BANDS[i])
            band.SetMetadataItem("WAVELENGTH_NM", str(waves[i]))
            band.SetMetadataItem("CENTRAL_WAVELENGTH_UM", f"{waves[i] / 1000:.3f}", "IMAGERY")
            band.SetUnitType("reflectance" if reflectance else "")
            if reflectance:
                band.SetScale(1.0)
                band.SetOffset(0.0)
        else:
            band.SetDescription("SCL scene classification")
            band.SetMetadataItem("BANDNAME", "SCL")
            band.SetMetadataItem("CLASSES", SCL_CLASSES)   # (GeoTIFF: one NoData value for all bands)
    out.SetMetadata({
        "PRODUCT": scene["product"], "SENSING_TIME": scene.get("time") or scene["date"],
        "SPACECRAFT": scene.get("spacecraft") or "", "PROCESSING_BASELINE": scene.get("baseline") or "",
        "TILE": scene.get("tile") or "", "CLOUD_COVER_PERCENT": f"{scene.get('cloud', float('nan')):.2f}",
        "VALUES": ("surface reflectance 0-1 (BOA_ADD_OFFSET and BOA_QUANTIFICATION_VALUE applied)" if reflectance
                   else "original L2A values; reflectance = value * scale + offset (stored per band)"),
        "RESAMPLING": "all bands at 10 m, nearest neighbour",
    })
    out.FlushCache()
    out = None
    return out_path
