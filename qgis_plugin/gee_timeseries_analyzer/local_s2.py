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
    """Reads the file list and metadata of an L2A product -> scene dict."""
    path = os.path.abspath(path)
    if path.lower().endswith(".zip"):
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
    return {"path": path, "product": product, "date": start[:10], "time": start,
            "spacecraft": _xml_text(root, "SPACECRAFT_NAME") or "Sentinel-2",
            "baseline": _xml_text(root, "PROCESSING_BASELINE"), "tile": tile.group(1) if tile else "",
            "cloud": float(_xml_text(root, "Cloud_Coverage_Assessment") or "nan"),
            "files": {k: v for k, v in best.items()}, "offsets": offsets, "quant": quant, "_ds": {}}


def scene_label(scene):
    return f"{scene['spacecraft'].replace('Sentinel-', 'S')} {scene['date']} T{scene['tile']} (local)"


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
    from osgeo import gdal, ogr, osr
    ref = _ds(scene, "SCL")
    srs = osr.SpatialReference(wkt=ref.GetProjection())
    wgs = osr.SpatialReference()
    wgs.ImportFromEPSG(4326)
    for s in (srs, wgs):
        s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    geom = ogr.CreateGeometryFromJson(json.dumps(geojson))
    geom.Transform(osr.CoordinateTransformation(wgs, srs))
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
    if is_point:
        inside = np.ones((h, w), dtype=bool)
    else:
        mem = gdal.GetDriverByName("MEM").Create("", w, h, 1, gdal.GDT_Byte)
        mem.SetGeoTransform((x0, cell, 0, y1, 0, -cell))
        mem.SetProjection(ref.GetProjection())
        ods = (ogr.GetDriverByName("MEM") or ogr.GetDriverByName("Memory")).CreateDataSource("p")
        lyr = ods.CreateLayer("p", srs, ogr.wkbMultiPolygon)
        f = ogr.Feature(lyr.GetLayerDefn())
        f.SetGeometry(geom)
        lyr.CreateFeature(f)
        gdal.RasterizeLayer(mem, [1], lyr, burn_values=[1])
        inside = mem.GetRasterBand(1).ReadAsArray() > 0
        if not inside.any():                         # polygon smaller than a pixel: its centre pixel
            inside[h // 2, w // 2] = True
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
    days = None
    if target_date:
        days = (dt.date.fromisoformat(scene["date"]) - dt.date.fromisoformat(target_date)).days
    return {"sensor": "local", "date": scene["date"], "days": days if days is not None else 0,
            "image": scene["product"], "spacecraft": scene["spacecraft"], "cloud": scene["cloud"],
            "clear": share, "values": values, "sd": sd, "wavelengths": SENSORS["s2"]["wavelengths"],
            "bands": BANDS, "names": SENSORS["s2"]["names"]}


def tci_path(scene):
    """True-colour image (TCI) at the finest resolution, for the map."""
    return scene["files"]["TCI"][1] if "TCI" in scene["files"] else None
