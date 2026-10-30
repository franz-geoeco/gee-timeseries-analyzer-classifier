"""
Earth Engine side of the Spectra section: surface-reflectance spectra of a
point or polygon from the Sentinel-2 / Landsat 8-9 image closest in time to
a target date, anywhere on Earth.

For every sample and sensor, all images within ± window days that cover the
geometry are reduced over it (point: pixel value; polygon: mean and
standard deviation) together with the clear share of the geometry. The
image closest to the target date with enough clear pixels wins -- one
request per sample and sensor. Pure earthengine-api (no Qt / QGIS imports).
"""
from . import gee_client as c

SENSORS = {
    "s2": {
        "label": "Sentinel-2", "collection": [c.S2_COLLECTION], "scale": 10,
        "bands": ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"],
        "names": ["B1 coastal", "B2 blue", "B3 green", "B4 red", "B5 red edge 1", "B6 red edge 2",
                  "B7 red edge 3", "B8 NIR", "B8A narrow NIR", "B9 water vapour", "B11 SWIR 1", "B12 SWIR 2"],
        "wavelengths": [443, 490, 560, 665, 705, 740, 783, 842, 865, 945, 1610, 2190],
        "cloud_prop": "CLOUDY_PIXEL_PERCENTAGE", "rgb": ["B4", "B3", "B2"],
    },
    "landsat": {
        "label": "Landsat 8/9", "collection": [c.L8_COLLECTION, c.L9_COLLECTION], "scale": 30,
        "bands": ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"],
        "names": ["B1 coastal", "B2 blue", "B3 green", "B4 red", "B5 NIR", "B6 SWIR 1", "B7 SWIR 2"],
        "wavelengths": [443, 482, 561, 655, 865, 1609, 2201],
        "cloud_prop": "CLOUD_COVER", "rgb": ["SR_B4", "SR_B3", "SR_B2"],
    },
}
MASKS = {"s2": "SCL: cloud, cloud shadow, cirrus, saturated pixels removed",
         "landsat": "QA_PIXEL: cloud, dilated cloud, cloud shadow, cirrus removed"}


def _collection(sensor, region, start, end, max_cloud):
    ee = c.import_ee()
    meta = SENSORS[sensor]
    coll = None
    for cid in meta["collection"]:
        part = (ee.ImageCollection(cid).filterBounds(region).filterDate(start, end)
                .filter(ee.Filter.lte(meta["cloud_prop"], max_cloud)))
        coll = part if coll is None else coll.merge(part)
    return coll


def _reflectance(img, sensor, cloud_mask):
    """-> (reflectance 0-1 of the spectral bands, clear 0/1 band)."""
    if sensor == "s2":
        refl = img.select(SENSORS["s2"]["bands"]).divide(10000)
        scl = img.select("SCL")
        clear = (scl.neq(1).And(scl.neq(3)).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
                 if cloud_mask else scl.gte(0))
    else:
        refl = img.select(SENSORS["landsat"]["bands"]).multiply(0.0000275).add(-0.2)
        qa = img.select("QA_PIXEL")
        clear = (qa.bitwiseAnd(0b11110).eq(0) if cloud_mask      # dilated cloud, cirrus, cloud, shadow
                 else qa.gte(0))
    return refl, clear


def closest_spectrum(geojson, sensor, date, window_days=30, max_cloud=60, cloud_mask=True,
                     min_clear=0.5):
    """Spectrum of the image of `sensor` closest to `date` (YYYY-MM-DD).
    Returns None if no image within the window has enough clear pixels, else
    {sensor, date, days, image, spacecraft, cloud, clear, values, sd,
    wavelengths, bands, names}."""
    ee = c.ensure_initialised()
    meta = SENSORS[sensor]
    is_point = geojson.get("type") == "Point"
    geom = ee.Geometry(geojson) if is_point else ee.Geometry(geojson, None, False)   # points: no geodesic flag
    t0 = ee.Date(date)
    coll = _collection(sensor, geom, t0.advance(-window_days, "day"), t0.advance(window_days + 1, "day"),
                       max_cloud)
    bands = meta["bands"]
    if is_point:
        reducer = ee.Reducer.first()
    else:
        reducer = ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True)

    def per_image(img):
        refl, clear = _reflectance(img, sensor, cloud_mask)
        refl = refl.updateMask(clear)
        stats = refl.reduceRegion(reducer=reducer, geometry=geom, scale=meta["scale"],
                                  bestEffort=True, maxPixels=1e8)
        share = clear.unmask(0).rename("clear").reduceRegion(
            reducer=ee.Reducer.mean(), geometry=geom, scale=meta["scale"], bestEffort=True, maxPixels=1e8)
        first = bands[0] if is_point else f"{bands[0]}_mean"
        return ee.Feature(None, stats).set({
            "gta_days": img.date().difference(t0, "day"),
            "gta_absdays": img.date().difference(t0, "day").abs(),
            "gta_time": img.get("system:time_start"),
            "gta_image": img.get("system:index"),
            "gta_cloud": img.get(meta["cloud_prop"]),
            "gta_craft": ee.Algorithms.If(img.get("SPACECRAFT_NAME"), img.get("SPACECRAFT_NAME"),
                                          img.get("SPACECRAFT_ID")),
            "gta_clear": ee.Number(share.get("clear", 0)),
            "gta_v": stats.get(first),         # null if every pixel is masked
        })

    fc = coll.map(per_image).filter(ee.Filter.notNull(["gta_v"]))
    fc = fc.filter(ee.Filter.gte("gta_clear", 1.0 if is_point else float(min_clear)))
    best = fc.sort("gta_absdays").first()
    info = ee.Algorithms.If(fc.size().gt(0), best, None).getInfo()
    if not info:
        return None
    p = info["properties"]
    if is_point:
        values = [p.get(b) for b in bands]
        sd = [None] * len(bands)
    else:
        values = [p.get(f"{b}_mean") for b in bands]
        sd = [p.get(f"{b}_stdDev") for b in bands]
    import datetime as dt
    when = dt.datetime.fromtimestamp(p["gta_time"] / 1000, dt.timezone.utc).strftime("%Y-%m-%d")
    return {"sensor": sensor, "date": when, "days": int(round(p["gta_days"])), "image": p.get("gta_image"),
            "spacecraft": p.get("gta_craft") or "", "cloud": p.get("gta_cloud"), "clear": p.get("gta_clear"),
            "values": values, "sd": sd, "wavelengths": meta["wavelengths"], "bands": bands,
            "names": meta["names"]}


def sample(geojson, sensors, date, window_days=30, max_cloud=60, cloud_mask=True, min_clear=0.5,
           progress_cb=None, cancel_cb=None):
    """closest_spectrum for every selected sensor -> {sensor: result or None}."""
    out = {}
    for i, s in enumerate(sensors):
        if cancel_cb is not None and cancel_cb():
            raise InterruptedError("Cancelled")
        if progress_cb is not None:
            progress_cb(i, len(sensors), f"{SENSORS[s]['label']}: searching the closest clear image")
        out[s] = closest_spectrum(geojson, s, date, window_days, max_cloud, cloud_mask, min_clear)
    return out


def closest_image(sensor, region_bbox, date, window_days=30, max_cloud=60, cloud_mask=True):
    """True-colour image of the acquisition day closest to `date` over a
    bbox (all scenes of that day mosaicked) -> (ee.Image, vis, day) or None."""
    ee = c.ensure_initialised()
    meta = SENSORS[sensor]
    region = c.bbox_geometry(region_bbox)
    t0 = ee.Date(date)
    coll = _collection(sensor, region, t0.advance(-window_days, "day"), t0.advance(window_days + 1, "day"),
                       max_cloud)
    coll = coll.map(lambda img: img.set("gta_absdays", img.date().difference(t0, "day").abs()))
    first = coll.sort("gta_absdays").first()
    day = ee.Algorithms.If(coll.size().gt(0), ee.Date(first.get("system:time_start")).format("YYYY-MM-dd"),
                           None).getInfo()
    if not day:
        return None
    d0 = ee.Date(day)
    same_day = coll.filterDate(d0, d0.advance(1, "day"))

    def prep(img):
        refl, clear = _reflectance(img, sensor, cloud_mask)
        return refl.updateMask(clear) if cloud_mask else refl
    rgb = [b for b in meta["rgb"]]
    img = same_day.map(prep).mosaic().select(rgb)
    return img, {"bands": rgb, "min": 0.0, "max": 0.3, "gamma": 1.2}, day
