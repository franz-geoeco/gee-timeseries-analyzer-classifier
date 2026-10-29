"""
Earth Engine side of the Classification tab: supervised crop classification
from multitemporal Sentinel-2 / Sentinel-1 composites, ported from the
Code Editor crop-classification script and generalised:

  * any training polygons (QGIS layer or GEE table asset) with any class
    attribute (numeric codes or strings), optional train/test split by an
    attribute or at random by polygon
  * any classification area (QGIS layer, GEE asset or map extent), optional
    ESA WorldCover mask
  * selectable features: S2 indices / bands (Cloud Score+ or SCL mask) and
    S1 VV / VH / VH-VV, aggregated per period (monthly, semi-monthly, every
    2 / 3 months) with one or more reducers, optional gap filling
  * Random Forest, Gradient Tree Boost or CART

Like the original script, training samples are polygon means (or, optionally,
pixels). They are fetched to the client once and the classifier is trained on
that uploaded table. This has two benefits. The map tiles don't recompute the
training samples. And a model is fully described by its feature recipe plus
the sample table: a saved model (JSON) rebuilds the same features for ANY
year and retrains the identical classifier (fixed seed), which is how it is
"applied to another year".

Pure earthengine-api + pandas/numpy (no Qt / QGIS imports).
"""
import datetime as dt
import json
import os

import numpy as np
import pandas as pd

from . import gee_client as c

CLASS_PROP = "gta_class"    # class code written onto the training features
X_PROP, Y_PROP = "gta_x", "gta_y"   # polygon centroid of a sample (spatial cross-validation)
SPLIT_PROP = "gta_split"    # split attribute value (QGIS-layer training data)
ID_PROP = "gta_id"          # polygon id, kept in the per-sample tables
NODATA = 255
MODEL_FORMAT = "gee_timeseries_analyzer.classification_model"
MODEL_VERSION = 1

PERIODS = {
    "dekad":  "Dekads (1st-10th, 11th-20th, 21st-end)",
    "half":   "Semi-monthly (1st-15th, 16th-end)",
    "month":  "Monthly",
    "2month": "Every 2 months",
    "3month": "Every 3 months",
    "season": "Whole season (one composite)",
}
REDUCERS = {
    "median": "Median", "mean": "Mean", "min": "Minimum", "max": "Maximum",
    "p10": "10th percentile", "p25": "25th percentile",
    "p75": "75th percentile", "p90": "90th percentile", "stdDev": "Standard deviation",
}
GAP_FILL = {
    "window": "Median of the neighbouring periods (± 1 period)",
    "none":   "None (gaps stay masked)",
}
S2_INDICES = {
    "NDVI": "NDVI", "EVI": "EVI", "SAVI": "SAVI", "NDWI": "NDWI (McFeeters)",
    "NDMI": "NDMI", "NDRE": "NDRE (red edge, B8/B5)",
}
S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
S2_MASKS = {"csplus": "Cloud Score+ (cs_cdf)", "scl": "Scene classification (SCL)"}
S1_FEATURES = {"VV": "VV (dB)", "VH": "VH (dB)", "VH_VV": "VH − VV (dB ratio)"}
S1_ORBITS = {"BOTH": "Both passes", "DESCENDING": "Descending", "ASCENDING": "Ascending"}
CLASSIFIERS = {"rf": "Random Forest", "gtb": "Gradient Tree Boost", "cart": "CART (single tree)"}
SAMPLING = {"mean": "One sample per polygon (polygon mean)", "pixels": "Pixels inside the polygons"}

WORLDCOVER = "ESA/WorldCover/v200"
CS_PLUS = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"

DEFAULT_FEATURES = {
    "start_month": 3, "end_month": 10, "period": "month", "reducers": ["median"],
    "gap_fill": "window",
    "s2_features": ["NDVI"], "s2_mask": "csplus", "cs_threshold": 0.60, "s2_max_cloud": 60,
    "s1_features": [], "s1_orbit": "BOTH",
    "scale": 10,
}
DEFAULT_CLASSIFIER = {"type": "rf", "trees": 300, "variables_per_split": 0,
                      "min_leaf": 1, "bag_fraction": 0.5, "max_nodes": 0,
                      "shrinkage": 0.05, "seed": 42}

# Crop palette for codes 0-9 (cotton, winter wheat, wheat-rice, wheat-maize,
# wheat-other, orchard, vegetables, alfalfa, rice, other), then generated.
DEFAULT_CLASS_COLOURS = ["#993C8C", "#E0A800", "#2D7AAB", "#BB2B16", "#F28E2B",
                  "#368429", "#E377C2", "#9BE36B", "#17BECF", "#A6A6A6"]


# ---------------------------------------------------------------------------
# Periods and band names (client-side, so they are known before any request)
# ---------------------------------------------------------------------------
def _add_months(d, n):
    m = d.month - 1 + n
    return dt.date(d.year + m // 12, m % 12 + 1, 1)


def periods(year, features):
    """-> list of (label, start date, end date exclusive). The season runs
    from start_month to end_month (inclusive) of `year`; if start_month >
    end_month it starts in the previous year (e.g. Oct-Jun winter crops).
    Labels only use calendar months, so band names don't depend on the
    year and a model can be applied to any other year."""
    sm, em = int(features["start_month"]), int(features["end_month"])
    first = dt.date(year if sm <= em else year - 1, sm, 1)
    stop = _add_months(dt.date(year, em, 1), 1)
    kind = features["period"]
    out = []
    d = first
    if kind == "season":
        last = _add_months(stop, -1)
        return [(f"{first.month:02d}-{last.month:02d}", first, stop)]
    if kind == "dekad":
        while d < stop:
            nxt = _add_months(d, 1)
            for tag, a, b in (("d1", d, d.replace(day=11)), ("d2", d.replace(day=11), d.replace(day=21)),
                              ("d3", d.replace(day=21), nxt)):
                out.append((f"{d.month:02d}{tag}", a, b))
            d = nxt
        return out
    if kind == "half":
        while d < stop:
            mid = d.replace(day=16)
            out.append((f"{d.month:02d}a", d, mid))
            out.append((f"{d.month:02d}b", mid, _add_months(d, 1)))
            d = _add_months(d, 1)
        return out
    step = {"month": 1, "2month": 2, "3month": 3}[kind]
    while d < stop:
        e = min(_add_months(d, step), stop)
        last = _add_months(e, -1)
        label = f"{d.month:02d}" if last == d else f"{d.month:02d}-{last.month:02d}"
        out.append((label, d, e))
        d = e
    return out


def _sensor_features(features):
    return list(features.get("s2_features", [])), list(features.get("s1_features", []))


def band_names(features, year=2020):
    """Names of all bands of the feature stack: <feature>_<period>_<reducer>."""
    s2f, s1f = _sensor_features(features)
    names = []
    for label, _, _ in periods(year, features):
        for red in features["reducers"]:
            for f in s2f + s1f:
                names.append(f"{f}_{label}_{red}")
    return names


def band_table(features, year):
    """One row per band of the feature stack: name, variable, sensor,
    period label, first / last day, reducer and the int16 scale factor."""
    s2f, s1f = _sensor_features(features)
    rows = []
    for label, s, e in periods(year, features):
        for red in features["reducers"]:
            for f in s2f + s1f:
                rows.append({"name": f"{f}_{label}_{red}", "variable": f,
                             "sensor": "Sentinel-2" if f in s2f else "Sentinel-1",
                             "period": label, "start": s.isoformat(),
                             "end": (e - dt.timedelta(days=1)).isoformat(), "reducer": red,
                             "int16_scale": 100 if f in s1f else 10000})
    return rows


def describe_periods(features, year):
    """Readable list of the periods of `year`, e.g. '03 (2026-03-01 – 03-31)'."""
    out = []
    for label, s, e in periods(year, features):
        last = e - dt.timedelta(days=1)
        out.append(f"{label} ({s.isoformat()} – {last.strftime('%m-%d') if last.year == s.year else last.isoformat()})")
    return out


def describe_features(features):
    n_per = len(periods(2020, features))
    s2f, s1f = _sensor_features(features)
    parts = []
    if s2f:
        parts.append("S2 " + "/".join(s2f))
    if s1f:
        parts.append("S1 " + "/".join(s1f))
    return (f"{len(band_names(features))} features = {len(s2f) + len(s1f)} variable(s) "
            f"({', '.join(parts) or 'none'}) × {n_per} period(s) × "
            f"{len(features['reducers'])} reducer(s)")


# ---------------------------------------------------------------------------
# Collections and the feature stack
# ---------------------------------------------------------------------------
def _reducer(key):
    ee = c.import_ee()
    if key.startswith("p") and key[1:].isdigit():
        return ee.Reducer.percentile([int(key[1:])])
    return {"median": ee.Reducer.median, "mean": ee.Reducer.mean, "min": ee.Reducer.min,
            "max": ee.Reducer.max, "stdDev": ee.Reducer.stdDev}[key]()


def _s2_collection(region, start, end, features):
    ee = c.import_ee()
    wanted = features["s2_features"]
    mask_kind = features.get("s2_mask", "csplus")
    thr = float(features.get("cs_threshold", 0.6))

    coll = (ee.ImageCollection(c.S2_COLLECTION)
            .filterBounds(region)
            .filterDate(start, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", int(features.get("s2_max_cloud", 60)))))
    if mask_kind == "csplus":
        coll = coll.linkCollection(ee.ImageCollection(CS_PLUS), ["cs_cdf"])

    def prep(img):
        if mask_kind == "csplus":
            clear = img.select("cs_cdf").gte(thr)
        else:
            scl = img.select("SCL")
            clear = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7))
        refl = img.select(S2_BANDS).divide(10000)
        common = refl.select(["B2", "B3", "B4", "B8", "B11"],
                             ["blue", "green", "red", "nir", "swir1"])
        out = []
        for f in wanted:
            if f in S2_BANDS:
                out.append(refl.select(f))
            elif f == "NDRE":
                out.append(refl.normalizedDifference(["B8", "B5"]).rename("NDRE"))
            else:
                out.append(c._optical_index(common, f).rename(f))
        return (ee.Image.cat(out).toFloat().updateMask(clear)
                .copyProperties(img, ["system:time_start"]))

    return coll.map(prep)


def _s1_collection(region, start, end, features):
    ee = c.import_ee()
    wanted = features["s1_features"]
    coll = (ee.ImageCollection(c.S1_COLLECTION)
            .filterBounds(region)
            .filterDate(start, end)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))
    orbit = features.get("s1_orbit", "BOTH")
    if orbit != "BOTH":
        coll = coll.filter(ee.Filter.eq("orbitProperties_pass", orbit))

    def prep(img):
        good = img.select("VV").gt(-30).And(img.select("VH").gt(-30))   # GRD border noise
        vv, vh = img.select("VV"), img.select("VH")
        bands = {"VV": vv, "VH": vh, "VH_VV": vh.subtract(vv).rename("VH_VV")}
        return (ee.Image.cat([bands[f] for f in wanted]).toFloat().updateMask(good)
                .copyProperties(img, ["system:time_start"]))

    return coll.map(prep)


def _composite(coll, start, end, reducer_key, names):
    """Reduce one period; a fully masked image with the right band names if
    no scene falls into it (instead of an image without bands)."""
    ee = c.import_ee()
    part = coll.filterDate(start, end)
    empty = ee.Image.constant([0] * len(names)).rename(names).toFloat().updateMask(0)
    reduced = part.reduce(_reducer(reducer_key)).rename(names).toFloat()
    return ee.Image(ee.Algorithms.If(part.size().gt(0), reduced, empty))


def feature_stack(region, year, features):
    """Multitemporal feature image for one year (band order = band_names)."""
    ee = c.import_ee()
    pers = periods(year, features)
    s2f, s1f = _sensor_features(features)
    if not s2f and not s1f:
        raise ValueError("Select at least one Sentinel-2 or Sentinel-1 feature.")
    # collections cover the season plus one period on either side (gap fill)
    pad = dt.timedelta(days=95)
    t0 = (pers[0][1] - pad).isoformat()
    t1 = (pers[-1][2] + pad).isoformat()
    sensors = []
    if s2f:
        sensors.append((s2f, _s2_collection(region, t0, t1, features)))
    if s1f:
        sensors.append((s1f, _s1_collection(region, t0, t1, features)))

    gap = features.get("gap_fill", "window") == "window"
    images = []
    for label, s, e in pers:
        length = e - s
        for red in features["reducers"]:
            for feats, coll in sensors:
                names = [f"{f}_{label}_{red}" for f in feats]
                img = _composite(coll, s.isoformat(), e.isoformat(), red, names)
                if gap:
                    wide = _composite(coll, (s - length).isoformat(), (e + length).isoformat(), red, names)
                    img = img.unmask(wide, False)
                images.append(img)
    return ee.Image.cat(images).select(band_names(features, year)).toFloat()


# ---------------------------------------------------------------------------
# Training data, AOI and mask
# ---------------------------------------------------------------------------
def _value_variants(value):
    """A split value typed in the dock matches the string and the number."""
    out = [str(value)]
    try:
        f = float(value)
        out += [f, int(f)] if f == int(f) else [f]
    except (TypeError, ValueError):
        pass
    return out


def training_collection(source, classes):
    """source:
         {"type": "features", "features": [{"geojson", "value", "split", "id"}]}
           (a QGIS layer, already reprojected to EPSG:4326)
         {"type": "asset", "asset_id", "class_prop", "split_prop"}
    classes: list of {"value", "code", ...}. Returns an ee.FeatureCollection
    with CLASS_PROP (code), ID_PROP and -- if a split attribute is used --
    the split value; polygons whose class is not in `classes` are dropped."""
    ee = c.import_ee()
    if source["type"] == "features":
        code_of = {_key(k["value"]): int(k["code"]) for k in classes}
        feats = []
        for f in source["features"]:
            code = code_of.get(_key(f["value"]))
            if code is None:
                continue
            # system:index = polygon id -> randomColumn (random split) is stable per polygon
            props = {CLASS_PROP: code, ID_PROP: str(f["id"]), "system:index": str(f["id"])}
            if f.get("split") is not None:
                props[SPLIT_PROP] = str(f["split"])
            feats.append(ee.Feature(ee.Geometry(f["geojson"], None, False), props))
        return ee.FeatureCollection(feats)

    fc = ee.FeatureCollection(source["asset_id"])
    prop = source["class_prop"]
    fc = fc.filter(ee.Filter.notNull([prop]))
    if source.get("exclude"):
        fc = fc.filter(ee.Filter.inList("system:index", list(source["exclude"])).Not())
    lookup_in = [k["value"] for k in classes]
    lookup_out = [int(k["code"]) for k in classes]
    split_prop = source.get("split_prop")

    def tidy(f):
        props = {CLASS_PROP: f.get(prop), ID_PROP: f.get("system:index")}
        if split_prop:
            props[SPLIT_PROP] = f.get(split_prop)
        return ee.Feature(f.geometry(), props)

    return fc.remap(lookup_in, lookup_out, prop).map(tidy)


def split_collection(fc, split):
    """split: {"mode": "none" | "random" | "attribute", "test_fraction",
    "test_value", "seed"} -> (train fc, test fc or None)."""
    ee = c.import_ee()
    mode = split.get("mode", "none")
    if mode == "random":
        frac = float(split.get("test_fraction", 0.3))
        fc = fc.randomColumn("gta_r", int(split.get("seed", 42)))
        return fc.filter(ee.Filter.gte("gta_r", frac)), fc.filter(ee.Filter.lt("gta_r", frac))
    if mode == "attribute":
        test = ee.Filter.inList(SPLIT_PROP, _value_variants(split.get("test_value", "test")))
        return fc.filter(test.Not()), fc.filter(test)
    return fc, None


def aoi_geometry(aoi):
    """aoi: {"type": "geojson", "geojson"} | {"type": "asset", "asset_id"} |
    {"type": "bbox", "bbox": (xmin, ymin, xmax, ymax)} -> ee.Geometry."""
    ee = c.import_ee()
    if aoi["type"] == "geojson":
        return ee.Geometry(aoi["geojson"], None, False)
    if aoi["type"] == "asset":
        return ee.FeatureCollection(aoi["asset_id"]).geometry()
    return c.bbox_geometry(aoi["bbox"])


def _extent(region, fc=None):
    """Bounding box of the AOI (+ training polygons): only used to pick the
    scenes, so a box is enough and much cheaper than a union of polygons."""
    box = region.bounds(10)
    if fc is not None:
        box = box.union(fc.geometry(10).bounds(10), 10).bounds(10)
    return box


def worldcover_mask(classes):
    """1 where ESA WorldCover 2021 is one of `classes` (e.g. 40 cropland,
    10 tree cover -- the original script's agriculture mask)."""
    ee = c.import_ee()
    wc = ee.ImageCollection(WORLDCOVER).first()
    m = wc.eq(int(classes[0]))
    for k in classes[1:]:
        m = m.Or(wc.eq(int(k)))
    return m


# ---------------------------------------------------------------------------
# Samples
# ---------------------------------------------------------------------------
def sample_collection(stack, fc, bands, sampling, scale, max_per_class=200, seed=42, codes=()):
    """Server-side sample table: polygon means (as in the original script) or
    pixels (at most `max_per_class` per class, random)."""
    ee = c.import_ee()

    def centroid(f):
        xy = f.geometry().centroid(10).coordinates()
        return f.set({X_PROP: xy.get(0), Y_PROP: xy.get(1)})
    fc = fc.select([CLASS_PROP, ID_PROP]).map(centroid)
    if sampling == "pixels":
        px = stack.sampleRegions(collection=fc, properties=[CLASS_PROP, ID_PROP, X_PROP, Y_PROP],
                                 scale=scale, tileScale=8, geometries=False)
        px = px.randomColumn("gta_r2", seed)
        parts = [px.filter(ee.Filter.eq(CLASS_PROP, int(k))).limit(int(max_per_class), "gta_r2")
                 for k in codes]
        out = parts[0]
        for part in parts[1:]:
            out = out.merge(part)
        return out
    return (stack.reduceRegions(collection=fc, reducer=ee.Reducer.mean(),
                                scale=scale, tileScale=8)
            .filter(ee.Filter.notNull(bands)))


def fetch_samples(fc, bands):
    """-> DataFrame [gta_class, gta_id, gta_x, gta_y, <bands>] (rows with
    missing bands dropped; x / y = polygon centroid, for spatial CV)."""
    ee = c.import_ee()
    keep = [CLASS_PROP, ID_PROP, X_PROP, Y_PROP]
    fc = fc.map(lambda f: ee.Feature(None, f.toDictionary(f.propertyNames().filter(
        ee.Filter.inList("item", keep + bands)))))
    df = c._fetch_table(fc)
    for col in keep + bands:
        if col not in df.columns:
            df[col] = np.nan
    df = df[keep + bands].dropna(subset=[CLASS_PROP] + bands)
    df[CLASS_PROP] = df[CLASS_PROP].astype(int)
    df[ID_PROP] = df[ID_PROP].astype(str)
    return df.reset_index(drop=True)


def samples_fc(df, bands):
    """Uploads a sample table as an ee.FeatureCollection (no geometry)."""
    ee = c.import_ee()
    feats = []
    for row in df[[CLASS_PROP] + bands].itertuples(index=False):
        props = {CLASS_PROP: int(row[0])}
        props.update({b: float(v) for b, v in zip(bands, row[1:])})
        feats.append(ee.Feature(None, props))
    return ee.FeatureCollection(feats)


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------
def make_classifier(cfg):
    ee = c.import_ee()
    kind = cfg.get("type", "rf")
    seed = int(cfg.get("seed", 42))
    max_nodes = int(cfg.get("max_nodes", 0)) or None
    if kind == "rf":
        return ee.Classifier.smileRandomForest(
            numberOfTrees=int(cfg.get("trees", 300)),
            variablesPerSplit=int(cfg.get("variables_per_split", 0)) or None,
            minLeafPopulation=int(cfg.get("min_leaf", 1)),
            bagFraction=float(cfg.get("bag_fraction", 0.5)),
            maxNodes=max_nodes, seed=seed)
    if kind == "gtb":
        return ee.Classifier.smileGradientTreeBoost(
            numberOfTrees=int(cfg.get("trees", 300)),
            shrinkage=float(cfg.get("shrinkage", 0.05)),
            samplingRate=float(cfg.get("bag_fraction", 0.7)),
            maxNodes=max_nodes, seed=seed)
    if kind == "cart":
        return ee.Classifier.smileCart(maxNodes=max_nodes,
                                       minLeafPopulation=int(cfg.get("min_leaf", 1)))
    raise ValueError(f"Unknown classifier {kind}")


def train(df, bands, cfg):
    return make_classifier(cfg).train(features=samples_fc(df, bands), classProperty=CLASS_PROP,
                                      inputProperties=bands)


def classify_map(stack, classifier, region, mask=None):
    """Classified image (byte, class codes); background masked."""
    img = stack.clip(region)
    if mask is not None:
        img = img.updateMask(mask)
    return img.classify(classifier).rename("class").toByte()


def predict(classifier, df, bands):
    """Server-side prediction of a sample table -> numpy array of codes."""
    if df is None or len(df) == 0:
        return np.array([], dtype=int)
    out = c._fetch_table(samples_fc(df, bands).classify(classifier, "classification")
                         .select(["classification"]))
    return out["classification"].astype(int).to_numpy()


def importance(classifier):
    """Variable importance (+ the RF out-of-bag error) without fetching the
    trees, which explain() also contains."""
    ee = c.import_ee()
    try:
        info = (ee.Dictionary(classifier.explain())
                .select(["importance", "outOfBagErrorEstimate"], True).getInfo())
    except Exception:
        return {}, None
    imp = info.get("importance") or {}
    return dict(sorted(imp.items(), key=lambda kv: -kv[1])), info.get("outOfBagErrorEstimate")


def pixel_predictions(class_map, test_fc, scale, fraction=0.2, seed=1):
    """Reference vs. map at the pixels of the test polygons (the original
    script's 8d), subsampled to `fraction` for memory limits."""
    ee = c.import_ee()
    px = (class_map.rename("classification")
          .sampleRegions(collection=test_fc.select([CLASS_PROP]), properties=[CLASS_PROP],
                         scale=scale, tileScale=8, geometries=False)
          .randomColumn("gta_r3", seed).filter(ee.Filter.lt("gta_r3", fraction))
          .select([CLASS_PROP, "classification"]))
    df = c._fetch_table(px)
    if len(df) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    return df[CLASS_PROP].astype(int).to_numpy(), df["classification"].astype(int).to_numpy()


# ---------------------------------------------------------------------------
# Accuracy (client-side, from reference / predicted codes)
# ---------------------------------------------------------------------------
def accuracy_metrics(ref, pred, codes):
    """-> dict with the confusion matrix (rows = reference, cols = map, in
    the order of `codes`), overall accuracy, kappa and per-class producer's
    / user's accuracy and F1 -- the same numbers as ee.ConfusionMatrix."""
    codes = [int(k) for k in codes]
    idx = {k: i for i, k in enumerate(codes)}
    n = len(codes)
    cm = np.zeros((n, n), dtype=int)
    for r, p in zip(ref, pred):
        if int(r) in idx and int(p) in idx:
            cm[idx[int(r)], idx[int(p)]] += 1
    total = cm.sum()
    diag = np.diag(cm).astype(float)
    rows, cols = cm.sum(axis=1).astype(float), cm.sum(axis=0).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        pa = np.where(rows > 0, diag / rows, np.nan)
        ua = np.where(cols > 0, diag / cols, np.nan)
        f1 = np.where((pa + ua) > 0, 2 * pa * ua / (pa + ua), np.nan)
    oa = diag.sum() / total if total else float("nan")
    pe = (rows * cols).sum() / (total * total) if total else float("nan")
    kappa = (oa - pe) / (1 - pe) if total and pe != 1 else float("nan")
    return {"codes": codes, "matrix": cm.tolist(), "n": int(total),
            "overall_accuracy": float(oa), "kappa": float(kappa),
            "producers": [_num(v) for v in pa], "users": [_num(v) for v in ua],
            "f1": [_num(v) for v in f1],
            "n_reference": rows.astype(int).tolist(), "n_mapped": cols.astype(int).tolist()}


def _num(v):
    return None if v != v else float(v)


def metrics_table(metrics, classes):
    """Per-class metrics + summary row (as the script's validation_metrics CSV)."""
    names = {int(k["code"]): k["name"] for k in classes}
    rows = []
    for i, code in enumerate(metrics["codes"]):
        rows.append({"class_code": code, "class_name": names.get(code, str(code)),
                     "n_reference": metrics["n_reference"][i], "n_mapped": metrics["n_mapped"][i],
                     "producers_acc": metrics["producers"][i], "users_acc": metrics["users"][i],
                     "f1": metrics["f1"][i]})
    rows.append({"class_code": -1, "class_name": "OVERALL (PA=OA, UA=kappa)",
                 "n_reference": metrics["n"], "n_mapped": metrics["n"],
                 "producers_acc": metrics["overall_accuracy"], "users_acc": metrics["kappa"], "f1": None})
    return pd.DataFrame(rows)


def confusion_table(metrics, classes):
    names = {int(k["code"]): k["name"] for k in classes}
    labels = [names.get(k, str(k)) for k in metrics["codes"]]
    df = pd.DataFrame(metrics["matrix"], columns=labels)
    df.insert(0, "reference", labels)
    return df


# ---------------------------------------------------------------------------
# Whole runs (called from a QgsTask)
# ---------------------------------------------------------------------------
def _step(progress_cb, cancel_cb, i, n, msg):
    if cancel_cb is not None and cancel_cb():
        raise InterruptedError("Cancelled")
    if progress_cb is not None:
        progress_cb(i, n, msg)


def pool_samples(train_df, extra):
    """Adds samples of other model files (multi-year training). extra =
    {"columns", "rows", "sources"} with codes already mapped to `classes`."""
    if not extra or not extra.get("rows"):
        return train_df
    ex = pd.DataFrame(extra["rows"], columns=extra["columns"])
    ex[CLASS_PROP] = ex[CLASS_PROP].astype(int)
    ex[ID_PROP] = [f"pooled:{i}" for i in range(len(ex))]
    return pd.concat([train_df, ex], ignore_index=True)


def run_cv(own_df, bands, codes, cv, fit_predict, progress_cb=None, cancel_cb=None):
    """Spatial k-fold cross-validation of the run's own samples (train + test)."""
    from . import analysis_tools as at
    return at.cross_validate(own_df, bands, codes, fit_predict, int(cv.get("k", 5)),
                             float(cv.get("block_km", 5)), int(cv.get("seed", 42)), progress_cb, cancel_cb)


def run_training(year, features, classifier_cfg, classes, training, split, aoi,
                 mask_classes=None, sampling="mean", max_per_class=200, pixel_accuracy=True,
                 extra_samples=None, cv=None, progress_cb=None, cancel_cb=None):
    """Builds the feature stack, samples the training (and test) polygons,
    trains, evaluates and builds the classified map. Returns a result dict;
    result["model"] is what save_model writes."""
    c.ensure_initialised()
    n = 5
    bands = band_names(features, year)
    codes = [int(k["code"]) for k in classes]
    region = aoi_geometry(aoi)
    fc = training_collection(training, classes)
    train_fc, test_fc = split_collection(fc, split)
    stack = feature_stack(_extent(region, fc), year, features)
    scale = int(features.get("scale", 10))
    seed = int(classifier_cfg.get("seed", 42))

    _step(progress_cb, cancel_cb, 0, n, "Sampling training polygons")
    train_df = fetch_samples(sample_collection(stack, train_fc, bands, sampling, scale,
                                               max_per_class, seed, codes), bands)
    if len(train_df) == 0:
        raise RuntimeError("No training samples with complete features -- check the season, "
                           "the cloud settings and that the training polygons overlap the data.")
    if train_df[CLASS_PROP].nunique() < 2:
        raise RuntimeError("Training samples contain fewer than two classes.")
    test_df = None
    if test_fc is not None:
        _step(progress_cb, cancel_cb, 1, n, "Sampling test polygons")
        test_df = fetch_samples(sample_collection(stack, test_fc, bands, sampling, scale,
                                                  max_per_class, seed, codes), bands)
    own_df = train_df
    train_df = pool_samples(train_df, extra_samples)

    _step(progress_cb, cancel_cb, 2, n, "Training the classifier")
    clf = train(train_df, bands, classifier_cfg)
    imp, oob = importance(clf)
    cv_res = None
    if cv:
        all_df = pd.concat([d for d in (own_df, test_df) if d is not None], ignore_index=True)
        cv_res = run_cv(all_df, bands, codes, cv,
                        lambda tr, te: predict(train(pool_samples(tr, extra_samples), bands, classifier_cfg),
                                               te, bands), progress_cb, cancel_cb)

    mask = worldcover_mask(mask_classes) if mask_classes else None
    class_map = classify_map(stack, clf, region, mask)

    metrics, pixel_metrics, per_sample = None, None, None
    if test_df is not None and len(test_df):
        _step(progress_cb, cancel_cb, 3, n, "Accuracy of the test samples")
        pred = predict(clf, test_df, bands)
        metrics = accuracy_metrics(test_df[CLASS_PROP], pred, codes)
        per_sample = test_df[[ID_PROP, CLASS_PROP]].assign(classification=pred)
        if pixel_accuracy and sampling == "mean":
            _step(progress_cb, cancel_cb, 4, n, "Pixel-level accuracy of the map")
            ref, pp = pixel_predictions(class_map, test_fc, scale)
            pixel_metrics = accuracy_metrics(ref, pp, codes)

    model = {
        "format": MODEL_FORMAT, "version": MODEL_VERSION,
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "trained_year": int(year), "features": features, "classifier": classifier_cfg,
        "classes": classes, "bands": bands, "sampling": sampling,
        "training": {k: v for k, v in training.items() if k != "features"},
        "split": split,
        "samples": {"columns": [CLASS_PROP] + bands,
                    "rows": train_df[[CLASS_PROP] + bands].values.tolist()},
        "pooled_from": (extra_samples or {}).get("sources", []),
        "accuracy": None if metrics is None else
        {k: metrics[k] for k in ("overall_accuracy", "kappa", "n")},
    }
    return {"year": int(year), "map": class_map, "region": region, "classes": classes,
            "metrics": metrics, "pixel_metrics": pixel_metrics, "importance": imp,
            "oob_error": oob, "per_sample": per_sample, "n_train": len(train_df),
            "utm": utm_epsg(region), "cv": cv_res, "train_df": own_df, "test_df": test_df,
            "n_pooled": len(train_df) - len(own_df),
            "n_test": 0 if test_df is None else len(test_df),
            "train_counts": train_df[CLASS_PROP].value_counts().to_dict(), "model": model}


def run_apply(model, year, aoi, mask_classes=None, reference=None, progress_cb=None, cancel_cb=None):
    """Applies a saved model to another year: same feature recipe for `year`,
    classifier retrained from the stored samples. `reference` = optional
    (training source, split, classes from match_classes) of THAT year to
    validate the map."""
    c.ensure_initialised()
    features, bands, classes = model["features"], model["bands"], model["classes"]
    codes = [int(k["code"]) for k in classes]
    train_df = model_samples(model)
    region = aoi_geometry(aoi)
    ref_fc = None
    if reference is not None:
        fc = training_collection(reference[0], reference[2])
        train_part, test_part = split_collection(fc, reference[1])
        ref_fc = test_part if test_part is not None else train_part
    stack = feature_stack(_extent(region, ref_fc), year, features)
    scale = int(features.get("scale", 10))

    _step(progress_cb, cancel_cb, 0, 3, "Retraining the stored model")
    clf = train(train_df, bands, model["classifier"])
    mask = worldcover_mask(mask_classes) if mask_classes else None
    class_map = classify_map(stack, clf, region, mask)

    metrics, pixel_metrics, per_sample = None, None, None
    if ref_fc is not None:
        _step(progress_cb, cancel_cb, 1, 3, f"Sampling the {year} reference polygons")
        ref_df = fetch_samples(sample_collection(stack, ref_fc, bands, "mean", scale), bands)
        if len(ref_df):
            pred = predict(clf, ref_df, bands)
            metrics = accuracy_metrics(ref_df[CLASS_PROP], pred, codes)
            per_sample = ref_df[[ID_PROP, CLASS_PROP]].assign(classification=pred)
            _step(progress_cb, cancel_cb, 2, 3, "Pixel-level accuracy of the map")
            ref, pp = pixel_predictions(class_map, ref_fc, scale)
            pixel_metrics = accuracy_metrics(ref, pp, codes)
    imp, oob = importance(clf)
    return {"year": int(year), "applied": True, "map": class_map, "region": region, "classes": classes,
            "metrics": metrics, "pixel_metrics": pixel_metrics,
            "importance": imp, "oob_error": oob, "per_sample": per_sample, "utm": utm_epsg(region),
            "n_train": len(train_df), "n_test": 0 if per_sample is None else len(per_sample),
            "train_counts": train_df[CLASS_PROP].value_counts().to_dict(), "model": model}


# ---------------------------------------------------------------------------
# Class lists, palette, model files
# ---------------------------------------------------------------------------
def _key(v):
    """Comparable key for class values (1, 1.0 and '1' are the same class)."""
    try:
        f = float(v)
        return str(int(f)) if f == int(f) else str(f)
    except (TypeError, ValueError):
        return str(v)


def build_classes(value_counts, value_names=None):
    """value_counts: {value: n polygons}. Integer values 0-254 keep their
    code (e.g. the 'class' attribute 0-9), anything else gets 0..n-1 in
    sorted order. -> list of {value, code, name, color, n}."""
    value_names = value_names or {}
    values = list(value_counts)
    numeric = all(_is_int(v) for v in values)
    if numeric and all(0 <= int(float(v)) < NODATA for v in values):
        ordered = sorted(values, key=lambda v: int(float(v)))
        codes = [int(float(v)) for v in ordered]
    else:
        ordered = sorted(values, key=lambda v: (float(v) if numeric else str(v)))
        codes = list(range(len(ordered)))
    out = []
    for v, code in zip(ordered, codes):
        name = value_names.get(_key(v)) or str(v)
        out.append({"value": _plain(v), "code": int(code), "name": str(name),
                    "color": default_colour(code), "n": int(value_counts[v])})
    return out


def match_classes(model_classes, value_counts, value_names=None):
    """Reference data of another year -> the model's codes, matched by class
    value or, failing that, by class name (e.g. a 2025 layer with only a
    'crop' column against a model trained on 'class' codes with 'crop'
    names). -> (classes for training_collection, unmatched values)."""
    value_names = value_names or {}
    by_val = {_key(k["value"]): k for k in model_classes}
    by_name = {str(k["name"]).lower(): k for k in model_classes}
    out, unmatched = [], []
    for v, n in value_counts.items():
        k = (by_val.get(_key(v)) or by_name.get(str(v).lower())
             or by_name.get(str(value_names.get(_key(v), "")).lower()))
        if k is None:
            unmatched.append(v)
            continue
        out.append(dict(k, value=_plain(v), n=int(n)))
    return out, unmatched


utm_epsg = c.utm_epsg


def _is_int(v):
    try:
        f = float(v)
        return f == int(f) and not isinstance(v, bool)
    except (TypeError, ValueError):
        return False


def _plain(v):
    """numpy scalars -> plain Python (JSON / ee friendly); 1.0 -> 1."""
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, float) and v == int(v):
        return int(v)
    return v


def default_colour(code):
    if 0 <= code < len(DEFAULT_CLASS_COLOURS):
        return DEFAULT_CLASS_COLOURS[code]
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(((code * 47) % 360) / 360.0, 0.63, 0.86)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def map_vis(classes):
    """Palette over 0..max code (unused codes grey)."""
    top = max(int(k["code"]) for k in classes)
    pal = ["cccccc"] * (top + 1)
    for k in classes:
        pal[int(k["code"])] = k["color"].lstrip("#")
    return {"min": 0, "max": top, "palette": pal}


def asset_classes(asset_id, class_prop, name_prop=None):
    """Distinct class values (+ polygon counts and names) of a table asset."""
    ee = c.ensure_initialised()
    props = [class_prop] + ([name_prop] if name_prop else [])
    fc = (ee.FeatureCollection(asset_id).filter(ee.Filter.notNull([class_prop]))
          .filter(ee.Filter.neq(class_prop, "")))
    df = c._fetch_table(fc.map(lambda f: ee.Feature(None, f.toDictionary(props))))
    if class_prop not in df.columns or len(df) == 0:
        raise RuntimeError(f"No values in property '{class_prop}' of {asset_id}.")
    counts = df[class_prop].value_counts().to_dict()
    names = {}
    if name_prop and name_prop in df.columns:
        for v, nm in df[[class_prop, name_prop]].dropna().drop_duplicates(class_prop).itertuples(index=False):
            names[_key(v)] = nm
    return build_classes(counts, names)


def asset_property_names(asset_id):
    ee = c.ensure_initialised()
    return sorted(ee.FeatureCollection(asset_id).first().propertyNames().getInfo())


def model_samples(model):
    s = model["samples"]
    return pd.DataFrame(s["rows"], columns=s["columns"]).astype({CLASS_PROP: int})


def save_model(model, path):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(model, fh, indent=1, default=_plain)


def load_model(path):
    with open(path, encoding="utf-8") as fh:
        model = json.load(fh)
    if model.get("format") != MODEL_FORMAT:
        raise ValueError("Not a GEE Time Series Analyzer classification model file.")
    if model.get("version", 0) > MODEL_VERSION:
        raise ValueError("The model was saved by a newer plugin version.")
    missing = [b for b in model["bands"] if b not in model["samples"]["columns"]]
    if missing:
        raise ValueError(f"Model file is incomplete (missing sample columns {missing[:3]} …).")
    return model


def model_summary(model):
    acc = model.get("accuracy") or {}
    oa = f", OA {acc['overall_accuracy']:.1%}" if acc.get("overall_accuracy") is not None else ""
    where = "a local stack" if model.get("engine") == "local" else "Earth Engine"
    year = model.get("trained_year") or (model.get("stack") or {}).get("file", "?")
    return (f"{CLASSIFIERS.get(model['classifier']['type'], '?')} trained on {year} ({where}) · "
            f"{len(model['classes'])} classes · {len(model['bands'])} features · "
            f"{len(model['samples']['rows'])} samples{oa}")


# ---------------------------------------------------------------------------
# Direct download (GeoTIFF with class codes) -- the map layer in QGIS is only
# an Earth Engine tile service and can't be saved as a raster.
# ---------------------------------------------------------------------------
def download_map(class_map, region, classes, path, scale=10, crs=None, progress_cb=None, cancel_cb=None):
    """Classified map -> local GeoTIFF: class codes, background 255 = NoData,
    colour table and class names (gee_client.download_image)."""
    palette = [(int(k["code"]), _hex_rgb(k["color"]), str(k["name"])) for k in classes]
    return c.download_image(class_map.unmask(NODATA, False).toByte().rename("class"), region, path,
                            ["class"], "uint8", NODATA, scale, crs, palette=palette,
                            progress_cb=progress_cb, cancel_cb=cancel_cb)


def _hex_rgb(hexc):
    hexc = hexc.lstrip("#")
    return tuple(int(hexc[i:i + 2], 16) for i in (0, 2, 4))


# ---------------------------------------------------------------------------
# Field-level results of an Earth Engine map
# ---------------------------------------------------------------------------
def field_majority_ee(class_map, polygons, scale=10, progress_cb=None, cancel_cb=None):
    """Class histogram of the map per polygon (frequencyHistogram) ->
    DataFrame like analysis_tools.field_majority."""
    ee = c.ensure_initialised()
    rows = []
    chunk = 2000
    for i0 in range(0, len(polygons), chunk):
        _step(progress_cb, cancel_cb, i0 // chunk, (len(polygons) + chunk - 1) // chunk,
              f"Field statistics {i0 + 1}-{min(i0 + chunk, len(polygons))} of {len(polygons)}")
        part = polygons[i0:i0 + chunk]
        fc = ee.FeatureCollection([ee.Feature(ee.Geometry(q["geojson"], None, False), {"gta_i": i0 + j})
                                   for j, q in enumerate(part)])
        stats = class_map.rename("class").reduceRegions(collection=fc, reducer=ee.Reducer.frequencyHistogram(),
                                                         scale=scale, tileScale=8)
        df = c._fetch_table(stats.map(lambda f: ee.Feature(None, {"gta_i": f.get("gta_i"),
                                                                  "hist": f.get("histogram")})))
        hists = dict(zip(df["gta_i"].astype(int), df["hist"])) if len(df) else {}
        for j, q in enumerate(part):
            h = hists.get(i0 + j) or {}
            h = {int(float(kk)): float(v) for kk, v in h.items() if float(v) > 0} if isinstance(h, dict) else {}
            n0 = sum(h.values())
            h = {kk: v for kk, v in h.items() if n0 and v / n0 >= 0.001}   # partial edge-pixel weights
            n = sum(h.values())
            if not n:
                rows.append({"id": q["id"], "class_code": None, "share": None, "second_code": None,
                             "second_share": None, "n_pixels": 0})
                continue
            order = sorted(h, key=lambda kk: -h[kk])
            rows.append({"id": q["id"], "class_code": order[0], "share": h[order[0]] / n,
                         "second_code": order[1] if len(order) > 1 else None,
                         "second_share": h[order[1]] / n if len(order) > 1 else None, "n_pixels": int(round(n))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Feature stack download -- input for the offline classification
# (local_classifier.py): same bands as the online run + WorldCover band.
# ---------------------------------------------------------------------------
STACK_NODATA = -9999.0
STACK_EXTENTS = {
    "aoi": "Classification area",
    "training": "Training polygons (bounding box)",
    "both": "Classification area + training polygons (bounding box)",
}


def download_feature_stack(year, features, path, aoi=None, training=None, classes=None, extent="aoi",
                           scale=10, crs=None, progress_cb=None, cancel_cb=None):
    """Downloads the feature stack of `year` (bands = band_names(features),
    float32, NoData -9999) plus the ESA WorldCover classes as band
    'worldcover', tagged with year and feature recipe. extent 'aoi' clips to
    the classification area; 'training' / 'both' cover the bounding box of
    the training polygons (and the area)."""
    ee = c.ensure_initialised()
    bands = band_names(features, year)
    region = aoi_geometry(aoi) if aoi is not None and extent in ("aoi", "both") else None
    fc = training_collection(training, classes) if extent in ("training", "both") else None
    if fc is not None:
        box = fc.geometry(10).bounds(10)
        region = box if region is None else _extent(region, fc)
    if region is None:
        raise ValueError("No area for the stack.")
    stack = feature_stack(region.bounds(10), year, features)
    if extent == "aoi":
        stack = stack.clip(region)
    wc = ee.ImageCollection(WORLDCOVER).first().rename("worldcover").toFloat()
    image = stack.unmask(STACK_NODATA, False).addBands(wc).toFloat()
    meta = {"GTA_KIND": "feature_stack", "GTA_YEAR": int(year), "GTA_RECIPE": json.dumps(features)}
    return c.download_image(image, region, path, bands + ["worldcover"], "float32", STACK_NODATA,
                            scale, crs, metadata=meta, progress_cb=progress_cb, cancel_cb=cancel_cb)


# ---------------------------------------------------------------------------
# Composite / time-series export (Export tab): any feature recipe, written as
# stacks, per-period or per-variable files. Every file is a valid stack for
# the offline classification (band names = feature names, recipe stored).
# ---------------------------------------------------------------------------
EXPORT_LAYOUTS = {
    "stack":    "One multi-band stack per year (all periods and variables)",
    "period":   "One file per period (all variables)",
    "variable": "One file per variable (time series of all periods)",
}
EXPORT_DTYPES = {
    "float32": "Float32 — physical values (reflectance 0–1, index, dB)",
    "int16":   "Int16 — scaled (optical × 10000, SAR dB × 100), GDAL scale factor set",
}
EXPORT_CONTENT = {
    "data":      "Data: the selected variables",
    "rgb_true":  "RGB quicklook, true colour B4/B3/B2 (8-bit, one file per period)",
    "rgb_false": "RGB quicklook, false colour B8/B4/B3 (8-bit, one file per period)",
}
INT16_NODATA = -32768
_QUICKLOOK = {"rgb_true": (["B4", "B3", "B2"], 0.3, 1.4), "rgb_false": (["B8", "B4", "B3"], 0.5, 1.2)}


def _sanitize(name):
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(name))


def export_plan(years, features, content="data", layout="stack", prefix="s2"):
    """Client-side plan of an export (no Earth Engine): list of files
    {name, year, kind, rows (band table rows), recipe}. Quicklooks are one
    3-band file per period."""
    out = []
    tag = PERIODS_SHORT.get(features["period"], features["period"])
    reds = "-".join(features["reducers"])
    for year in years:
        if content in _QUICKLOOK:
            qbands = _QUICKLOOK[content][0]
            f = dict(features, s2_features=qbands, s1_features=[], reducers=features["reducers"][:1])
            for row in [r for r in band_table(f, year) if r["variable"] == qbands[0]]:
                rows = [r for r in band_table(f, year) if r["period"] == row["period"]]
                out.append({"name": _sanitize(f"{prefix}_{year}_{row['period']}_{tag}_{content}"),
                            "year": year, "kind": "quicklook", "rows": rows, "recipe": f})
            continue
        table = band_table(features, year)
        if layout in ("period", "variable"):
            key = "period" if layout == "period" else "variable"
            groups = {}
            for row in table:
                groups.setdefault(row[key], []).append(row)
            parts = [(f"{year}_{g}", rows) for g, rows in groups.items()]
        else:
            parts = [(f"{year}", table)]
        for suffix, rows in parts:
            out.append({"name": _sanitize(f"{prefix}_{suffix}_{tag}_{reds}"), "year": year,
                        "kind": "feature_stack", "rows": rows, "recipe": features})
    return out


def export_jobs(region, years, features, content="data", layout="stack", dtype="float32",
                prefix="s2", worldcover=False, clip=True):
    """export_plan + the Earth Engine images: list of jobs {name, image,
    bands, dtype, nodata, metadata, band_meta, recipe, year, table}."""
    ee = c.import_ee()
    jobs = []
    stacks = {}
    for item in export_plan(years, features, content, layout, prefix):
        year, f, rows = item["year"], item["recipe"], item["rows"]
        key = (year, item["kind"])
        if key not in stacks:
            st = feature_stack(region.bounds(10), year, f)
            stacks[key] = st.clip(region) if clip else st
        stack = stacks[key]
        names = [r["name"] for r in rows]
        if item["kind"] == "quicklook":
            _, vmax, gamma = _QUICKLOOK[content]
            img = (stack.select(names).visualize(min=0, max=vmax, gamma=gamma)
                   .rename(["red", "green", "blue"]).unmask(0, False).toUint8())
            meta = {"GTA_KIND": "quicklook", "GTA_YEAR": year,
                    "GTA_PERIOD": f"{rows[0]['start']}/{rows[0]['end']}", "GTA_RECIPE": json.dumps(f)}
            jobs.append({"name": item["name"], "image": img, "bands": ["red", "green", "blue"],
                         "dtype": "uint8", "nodata": 0, "metadata": meta,
                         "band_meta": [dict({kk: str(v) for kk, v in r.items() if kk not in ("name", "int16_scale")},
                                            source_band=r["name"]) for r in rows],
                         "recipe": f, "year": year, "table": rows})
            continue
        img = stack.select(names)
        band_meta = [{kk: str(v) for kk, v in r.items() if kk not in ("name", "int16_scale")} for r in rows]
        if dtype == "int16":
            factors = [r["int16_scale"] for r in rows]
            img = (img.multiply(ee.Image.constant(factors)).round().clamp(-32767, 32767)
                   .unmask(INT16_NODATA, False).toInt16())
            for bm, fac in zip(band_meta, factors):
                bm["scale"] = 1.0 / fac
            nodata, out_type = INT16_NODATA, "int16"
        else:
            img = img.unmask(STACK_NODATA, False).toFloat()
            nodata, out_type = STACK_NODATA, "float32"
        bands = list(names)
        if worldcover:
            wc = ee.ImageCollection(WORLDCOVER).first().rename("worldcover")
            img = img.addBands(wc.toInt16() if out_type == "int16" else wc.toFloat())
            bands.append("worldcover")
            band_meta.append({"variable": "ESA WorldCover 2021 class"})
        meta = {"GTA_KIND": "feature_stack", "GTA_YEAR": year, "GTA_RECIPE": json.dumps(f), "GTA_LAYOUT": layout}
        jobs.append({"name": item["name"], "image": img, "bands": bands, "dtype": out_type, "nodata": nodata,
                     "metadata": meta, "band_meta": band_meta, "recipe": f, "year": year, "table": rows})
    return jobs


PERIODS_SHORT = {"dekad": "dekad", "half": "semimonthly", "month": "monthly", "2month": "2monthly",
                 "3month": "3monthly", "season": "season"}


def _sidecar(path, job, scale, crs):
    side = {"file": os.path.basename(path), "year": job["year"], "kind": job["metadata"].get("GTA_KIND"),
            "scale_m": scale, "crs": crs, "dtype": job["dtype"], "nodata": job["nodata"],
            "recipe": job["recipe"], "bands": job["table"] or job["bands"]}
    with open(os.path.splitext(path)[0] + ".json", "w", encoding="utf-8") as fh:
        json.dump(side, fh, indent=1)


def download_export(region, years, features, folder, content="data", layout="stack", dtype="float32",
                    prefix="s2", worldcover=False, clip=True, scale=10, crs=None,
                    progress_cb=None, cancel_cb=None):
    """Downloads every export job as a GeoTIFF (+ JSON sidecar with the
    recipe and band table) into `folder`; returns the paths."""
    c.ensure_initialised()
    crs = crs or c.utm_epsg(region)
    jobs = export_jobs(region, years, features, content, layout, dtype, prefix, worldcover, clip)
    paths = []
    for i, job in enumerate(jobs):
        path = os.path.join(folder, job["name"] + ".tif")

        def sub(j, n, msg, i=i, name=job["name"]):
            if progress_cb is not None:
                progress_cb(i * n + j, len(jobs) * n, f"{name} ({i + 1}/{len(jobs)}): {msg}")
        c.download_image(job["image"], region, path, job["bands"], job["dtype"], job["nodata"], scale, crs,
                         metadata=job["metadata"], band_meta=job["band_meta"],
                         progress_cb=sub, cancel_cb=cancel_cb)
        _sidecar(path, job, scale, crs)
        paths.append(path)
    return paths


def drive_export(region, years, features, folder="GEE_Exports", content="data", layout="stack",
                 dtype="float32", prefix="s2", worldcover=False, clip=True, scale=10, crs=None):
    """One Earth Engine batch task per export job (GeoTIFF to Drive); task ids."""
    ee = c.ensure_initialised()
    crs = crs or c.utm_epsg(region)
    ids = []
    for job in export_jobs(region, years, features, content, layout, dtype, prefix, worldcover, clip):
        task = ee.batch.Export.image.toDrive(
            image=job["image"], description=job["name"][:100], folder=folder or None,
            fileNamePrefix=job["name"], region=region.bounds(10), scale=scale, crs=crs, maxPixels=1e13,
            fileFormat="GeoTIFF", formatOptions={"cloudOptimized": True, "noData": job["nodata"]})
        task.start()
        ids.append(task.id)
    return ids


# ---------------------------------------------------------------------------
# Batch exports
# ---------------------------------------------------------------------------
def export_map(class_map, region, kind, name, scale=10, crs=None, folder="GEE_Exports", asset_id=None):
    """Starts an Earth Engine batch export; returns the task id.
    kind 'drive' -> Cloud-Optimised GeoTIFF with background = 255 = NoData
    (a GeoTIFF has no mask; masked pixels would otherwise become class 0);
    kind 'asset' -> image asset with masked background, mode pyramids."""
    ee = c.ensure_initialised()
    region = region.bounds(10)
    if kind == "drive":
        params = dict(image=class_map.unmask(NODATA, False).toByte(), description=name,
                      folder=folder or None, fileNamePrefix=name, region=region, scale=scale,
                      maxPixels=1e13, fileFormat="GeoTIFF",
                      formatOptions={"cloudOptimized": True, "noData": NODATA})
        if crs:
            params["crs"] = crs
        task = ee.batch.Export.image.toDrive(**params)
    else:
        params = dict(image=class_map, description=name, assetId=asset_id, region=region,
                      scale=scale, maxPixels=1e13, pyramidingPolicy={".default": "mode"})
        if crs:
            params["crs"] = crs
        task = ee.batch.Export.image.toAsset(**params)
    task.start()
    return task.id
