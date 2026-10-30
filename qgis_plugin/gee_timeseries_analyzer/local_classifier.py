"""
Offline classification on local raster stacks (GeoTIFF), the counterpart of
classification.py without Earth Engine:

  * a stack is any multi-band raster whose band descriptions are the feature
    names -- e.g. a feature stack downloaded by the Classification tab
    (classification.download_feature_stack, same band names as the online
    run) or a monthly composite stack from the Analysis tab. An optional
    auxiliary band "worldcover" (ESA WorldCover classes) provides the mask.
  * training polygons are rasterised onto the stack grid block by block;
    samples are polygon means or pixels, exactly like the online sampling
  * scikit-learn Random Forest / Gradient Boosting / decision tree
  * prediction block by block into a paletted GeoTIFF (255 = NoData)

Models use the same JSON format as online models (samples + band names +
classes): a model trained online can classify a local stack with the same
band names and vice versa. GDAL (osgeo) + numpy + pandas + scikit-learn; no
Qt / QGIS imports.
"""
import datetime as dt
import json
import os
import warnings

import numpy as np
import pandas as pd

from . import classification as k

AUX_BANDS = ("worldcover",)
META_KIND = "GTA_KIND"
META_YEAR = "GTA_YEAR"
META_RECIPE = "GTA_RECIPE"
BLOCK_BYTES = 64_000_000     # stack values read per block


def require_sklearn():
    try:
        import sklearn  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "Offline classification needs scikit-learn. Install it into QGIS's Python, e.g. in the "
            "OSGeo4W Shell: python -m pip install scikit-learn") from e


# ---------------------------------------------------------------------------
# Stacks
# ---------------------------------------------------------------------------
def stack_info(path):
    """-> dict(path, bands (feature band names), aux {name: band index},
    index {name: band index}, nodata, width, height, gt, wkt, year, recipe, kind)."""
    from osgeo import gdal
    ds = gdal.Open(path)
    if ds is None:
        raise RuntimeError(f"Could not open {path}")
    names = []
    for i in range(ds.RasterCount):
        d = ds.GetRasterBand(i + 1).GetDescription().strip()
        names.append(d or f"band_{i + 1}")
    if len(set(names)) != len(names):
        raise RuntimeError(f"{os.path.basename(path)}: band names are not unique.")
    meta = ds.GetMetadata() or {}
    recipe = None
    if meta.get(META_RECIPE):
        try:
            recipe = json.loads(meta[META_RECIPE])
        except ValueError:
            recipe = None
    year = meta.get(META_YEAR)
    index = {n: i + 1 for i, n in enumerate(names)}
    scales = {}
    for i, n in enumerate(names):
        b = ds.GetRasterBand(i + 1)
        s, o = b.GetScale(), b.GetOffset()
        if (s not in (None, 1.0)) or (o not in (None, 0.0)):
            scales[n] = (s if s is not None else 1.0, o if o is not None else 0.0)
    return {"path": path, "bands": [n for n in names if n not in AUX_BANDS],
            "aux": {n: index[n] for n in names if n in AUX_BANDS}, "index": index,
            "nodata": ds.GetRasterBand(1).GetNoDataValue(),
            "width": ds.RasterXSize, "height": ds.RasterYSize,
            "gt": ds.GetGeoTransform(), "wkt": ds.GetProjection(),
            "year": int(year) if year and year.isdigit() else None, "scales": scales,
            "recipe": recipe, "kind": meta.get(META_KIND, "")}


def describe_stack(info):
    parts = [f"{len(info['bands'])} feature bands", f"{info['width']} × {info['height']} px"]
    if info["year"]:
        parts.insert(0, str(info["year"]))
    if info["aux"]:
        parts.append("+ " + ", ".join(info["aux"]))
    return " · ".join(parts)


def write_stack_metadata(path, kind, year=None, recipe=None, extra=None):
    """Tags a GeoTIFF as stack (kind, year, feature recipe)."""
    from osgeo import gdal
    ds = gdal.Open(path, gdal.GA_Update)
    meta = {META_KIND: kind}
    if year is not None:
        meta[META_YEAR] = str(int(year))
    if recipe is not None:
        meta[META_RECIPE] = json.dumps(recipe)
    meta.update(extra or {})
    ds.SetMetadata(meta)
    ds = None


def _read_bands(ds, info, bands, xoff, yoff, w, h):
    """[n_bands, h, w] float32 in physical units (GDAL scale / offset
    applied, e.g. int16 exports); NoData -> NaN."""
    out = np.empty((len(bands), h, w), dtype=np.float32)
    for j, b in enumerate(bands):
        raw = ds.GetRasterBand(info["index"][b]).ReadAsArray(xoff, yoff, w, h).astype(np.float32)
        bad = ~_valid(raw, info["nodata"])
        if b in info["scales"]:
            s, o = info["scales"][b]
            raw = raw * s + o
        raw[bad] = np.nan
        out[j] = raw
    return out


def _valid(arr, nodata):
    ok = np.isfinite(arr)
    if nodata is not None:
        ok &= arr != nodata
    return ok


def _rows_per_block(width, nbands, itemsize=4):
    return max(1, int(BLOCK_BYTES // max(1, width * nbands * itemsize)))


# ---------------------------------------------------------------------------
# Rasterising training polygons onto the stack grid
# ---------------------------------------------------------------------------
def _polygon_layer(polygons, wkt):
    """polygons: [(geojson in EPSG:4326, integer id >= 1)] -> in-memory OGR
    layer in the stack's CRS (+ its envelope)."""
    from osgeo import ogr, osr
    src = osr.SpatialReference()
    src.ImportFromEPSG(4326)
    dst = osr.SpatialReference()
    dst.ImportFromWkt(wkt)
    for s in (src, dst):
        s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    xform = osr.CoordinateTransformation(src, dst)
    ds = ogr.GetDriverByName("Memory").CreateDataSource("polys")
    lyr = ds.CreateLayer("polys", dst, ogr.wkbMultiPolygon)
    lyr.CreateField(ogr.FieldDefn("pid", ogr.OFTInteger))
    env = None
    for geojson, pid in polygons:
        g = ogr.CreateGeometryFromJson(json.dumps(geojson))
        if g is None or g.IsEmpty():
            continue
        g.Transform(xform)
        f = ogr.Feature(lyr.GetLayerDefn())
        f.SetField("pid", int(pid))
        f.SetGeometry(g)
        lyr.CreateFeature(f)
        e = g.GetEnvelope()
        env = e if env is None else (min(env[0], e[0]), max(env[1], e[1]), min(env[2], e[2]), max(env[3], e[3]))
    return ds, lyr, env


def polygon_pixels(info, polygons, bands, cancel_cb=None):
    """Values of all stack pixels whose centre lies in a polygon ->
    (pids [n], values [n, bands], worldcover [n] or None). Pixels with
    NoData in some band keep NaN there."""
    from osgeo import gdal
    ogr_ds, lyr, env = _polygon_layer(polygons, info["wkt"])
    empty = (np.zeros(0, int), np.zeros((0, len(bands)), np.float32), None)
    if env is None:
        return empty
    x0, px, _, y0, _, py = info["gt"]
    c0 = max(0, int(np.floor((env[0] - x0) / px)))
    c1 = min(info["width"], int(np.ceil((env[1] - x0) / px)) + 1)
    r0 = max(0, int(np.floor((env[3] - y0) / py)))
    r1 = min(info["height"], int(np.ceil((env[2] - y0) / py)) + 1)
    if c0 >= c1 or r0 >= r1:
        return empty
    ds = gdal.Open(info["path"])
    wc_idx = info["aux"].get("worldcover")
    W = c1 - c0
    step = _rows_per_block(W, len(bands))
    out_pid, out_val, out_wc = [], [], []
    for r in range(r0, r1, step):
        if cancel_cb is not None and cancel_cb():
            raise InterruptedError("Cancelled")
        h = min(step, r1 - r)
        mem = gdal.GetDriverByName("MEM").Create("", W, h, 1, gdal.GDT_Int32)
        mem.SetGeoTransform((x0 + c0 * px, px, 0, y0 + r * py, 0, py))
        mem.SetProjection(info["wkt"])
        gdal.RasterizeLayer(mem, [1], lyr, options=["ATTRIBUTE=pid"])
        pid = mem.GetRasterBand(1).ReadAsArray()
        inside = pid > 0
        if not inside.any():
            continue
        vals = _read_bands(ds, info, bands, c0, r, W, h)[:, inside].T
        out_pid.append(pid[inside])
        out_val.append(vals)
        if wc_idx:
            out_wc.append(ds.GetRasterBand(wc_idx).ReadAsArray(c0, r, W, h)[inside])
    if not out_pid:
        return empty
    return (np.concatenate(out_pid), np.concatenate(out_val),
            np.concatenate(out_wc) if out_wc else None)


def sample_polygons(info, polys, bands, sampling="mean", max_per_class=200, seed=42, cancel_cb=None):
    """polys: [{"geojson", "code", "id"}] -> DataFrame [gta_class, gta_id,
    <bands>] like classification.fetch_samples (polygon means: per-band mean
    of the valid pixels, polygons with an empty band dropped; pixels: complete
    pixels only, at most max_per_class random pixels per class)."""
    cols = [k.CLASS_PROP, k.ID_PROP, k.X_PROP, k.Y_PROP] + bands
    if not polys:
        return pd.DataFrame(columns=cols)
    pids, vals, _ = polygon_pixels(info, [(q["geojson"], i + 1) for i, q in enumerate(polys)], bands,
                                   cancel_cb)
    if len(pids) == 0:
        return pd.DataFrame(columns=cols)
    codes = np.array([int(q["code"]) for q in polys])
    ids = np.array([str(q["id"]) for q in polys], dtype=object)
    from .analysis_tools import geojson_centroid
    cxy = np.array([geojson_centroid(q["geojson"]) for q in polys], dtype=float)
    df = pd.DataFrame(vals, columns=bands)
    df.insert(0, "_pid", pids)
    if sampling == "mean":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            df = df.groupby("_pid", sort=True)[bands].mean().reset_index()
    df = df.dropna(subset=bands)
    pidx = df["_pid"].to_numpy() - 1
    df.insert(0, k.Y_PROP, cxy[pidx, 1])
    df.insert(0, k.X_PROP, cxy[pidx, 0])
    df.insert(0, k.ID_PROP, ids[pidx])
    df.insert(0, k.CLASS_PROP, codes[pidx])
    df = df.drop(columns="_pid")
    if sampling == "pixels":
        rng = np.random.default_rng(seed)
        parts = []
        for _, part in df.groupby(k.CLASS_PROP, sort=True):
            if len(part) > max_per_class:
                part = part.iloc[np.sort(rng.choice(len(part), max_per_class, replace=False))]
            parts.append(part)
        df = pd.concat(parts) if parts else df
    return df[cols].reset_index(drop=True)


def split_polygons(features, classes, split):
    """QGIS-layer training features -> (train polys, test polys or None);
    the same split rules as classification.split_collection."""
    code_of = {k._key(q["value"]): int(q["code"]) for q in classes}
    polys = []
    for f in features:
        code = code_of.get(k._key(f["value"]))
        if code is not None:
            polys.append({"geojson": f["geojson"], "code": code, "id": f["id"], "split": f.get("split")})
    mode = split.get("mode", "none")
    if mode == "random":
        # stable per polygon id: excluding polygons doesn't reshuffle the others
        from zlib import crc32
        seed = int(split.get("seed", 42))
        r = [crc32(f"{seed}:{q['id']}".encode()) / 2 ** 32 for q in polys]
        frac = float(split.get("test_fraction", 0.3))
        return ([q for q, v in zip(polys, r) if v >= frac], [q for q, v in zip(polys, r) if v < frac])
    if mode == "attribute":
        test_keys = {k._key(v) for v in k._value_variants(split.get("test_value", "test"))}
        is_test = [q["split"] is not None and k._key(q["split"]) in test_keys for q in polys]
        return ([q for q, t in zip(polys, is_test) if not t], [q for q, t in zip(polys, is_test) if t])
    return polys, None


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------
def make_estimator(cfg, n_features, n_jobs=-1):
    """scikit-learn counterpart of classification.make_classifier. n_jobs =
    CPU cores for Random Forest training and prediction (-1 = all); boosting
    and single trees run on one core."""
    require_sklearn()
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.tree import DecisionTreeClassifier
    kind = cfg.get("type", "rf")
    seed = int(cfg.get("seed", 42))
    max_nodes = int(cfg.get("max_nodes", 0)) or None
    if max_nodes is not None:
        max_nodes = max(2, max_nodes)
    if kind == "rf":
        vps = int(cfg.get("variables_per_split", 0))
        bag = float(cfg.get("bag_fraction", 0.5))
        return RandomForestClassifier(
            n_estimators=int(cfg.get("trees", 300)),
            max_features=min(vps, n_features) if vps else "sqrt",
            min_samples_leaf=int(cfg.get("min_leaf", 1)), max_leaf_nodes=max_nodes,
            bootstrap=True, max_samples=bag if bag < 1 else None, oob_score=True,
            random_state=seed, n_jobs=n_jobs)
    if kind == "gtb":
        return GradientBoostingClassifier(
            n_estimators=int(cfg.get("trees", 300)), learning_rate=float(cfg.get("shrinkage", 0.05)),
            subsample=float(cfg.get("bag_fraction", 0.7)), max_leaf_nodes=max_nodes, random_state=seed)
    if kind == "cart":
        return DecisionTreeClassifier(min_samples_leaf=int(cfg.get("min_leaf", 1)),
                                      max_leaf_nodes=max_nodes, random_state=seed)
    raise ValueError(f"Unknown classifier {kind}")


def train(df, bands, cfg, n_jobs=-1):
    est = make_estimator(cfg, len(bands), n_jobs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")     # OOB warnings for tiny classes
        est.fit(df[bands].to_numpy(np.float32), df[k.CLASS_PROP].to_numpy(int))
    imp = dict(sorted(zip(bands, (float(v) for v in est.feature_importances_)), key=lambda kv: -kv[1]))
    oob = 1.0 - est.oob_score_ if getattr(est, "oob_score_", None) is not None else None
    return est, imp, oob


def _predict(est, X, chunk=200_000):
    out = np.empty(len(X), dtype=np.int32)
    for i in range(0, len(X), chunk):
        out[i:i + chunk] = est.predict(X[i:i + chunk])
    return out


def predict_samples(est, df, bands):
    if df is None or len(df) == 0:
        return np.array([], dtype=int)
    return _predict(est, df[bands].to_numpy(np.float32))


def predict_stack(est, info, bands, out_path, classes, mask_classes=None, progress_cb=None, cancel_cb=None):
    """Classifies every pixel of the stack (block by block) into a tiled,
    compressed GeoTIFF: class codes, 255 = NoData (missing features or
    masked), colour table + class names, overviews."""
    from osgeo import gdal
    missing = [b for b in bands if b not in info["index"]]
    if missing:
        raise RuntimeError(f"The stack lacks {len(missing)} band(s) of the model, e.g. {', '.join(missing[:4])}. "
                           "Use a stack built with the same features.")
    wc_idx = info["aux"].get("worldcover")
    if mask_classes and not wc_idx:
        raise RuntimeError("The WorldCover mask is on, but the stack has no 'worldcover' band. "
                           "Switch the mask off or download the stack with this plugin.")
    W, H = info["width"], info["height"]
    src = gdal.Open(info["path"])
    out = gdal.GetDriverByName("GTiff").Create(out_path, W, H, 1, gdal.GDT_Byte,
                                               ["TILED=YES", "COMPRESS=DEFLATE", "BIGTIFF=IF_SAFER"])
    if out is None:
        raise RuntimeError(f"Could not create {out_path}")
    out.SetGeoTransform(info["gt"])
    out.SetProjection(info["wkt"])
    band = out.GetRasterBand(1)
    band.SetNoDataValue(k.NODATA)
    band.SetDescription("class")
    ct = gdal.ColorTable()
    names = [""] * 256
    for q in classes:
        ct.SetColorEntry(int(q["code"]), k._hex_rgb(q["color"]) + (255,))
        names[int(q["code"])] = str(q["name"])
    band.SetRasterColorTable(ct)
    band.SetRasterColorInterpretation(gdal.GCI_PaletteIndex)
    band.SetCategoryNames(names)
    step = _rows_per_block(W, len(bands))
    n_blocks = (H + step - 1) // step
    try:
        for bi, r in enumerate(range(0, H, step)):
            if cancel_cb is not None and cancel_cb():
                raise InterruptedError("Cancelled")
            h = min(step, H - r)
            arr = _read_bands(src, info, bands, 0, r, W, h)
            ok = np.isfinite(arr).all(axis=0)
            if mask_classes:
                ok &= np.isin(src.GetRasterBand(wc_idx).ReadAsArray(0, r, W, h), mask_classes)
            res = np.full((h, W), k.NODATA, dtype=np.uint8)
            if ok.any():
                res[ok] = _predict(est, arr[:, ok].T)
            band.WriteArray(res, 0, r)
            if progress_cb is not None:
                progress_cb(bi + 1, n_blocks + 1, f"Classified block {bi + 1}/{n_blocks}")
        band = None
        out.BuildOverviews("NEAREST", [f for f in (2, 4, 8, 16, 32) if min(W, H) // f >= 64])
    except BaseException:
        band = out = None
        try:
            os.remove(out_path)
        except OSError:
            pass
        raise
    band = out = None
    return out_path


def _pixel_check(est, info, test_polys, bands, mask_classes, fraction=0.2, seed=1):
    """Reference vs. map at the pixels of the test polygons (as the online
    'map pixels in test polygons'): the map's mask applies, 20 % subsample."""
    if not test_polys:
        return None
    pids, vals, wc = polygon_pixels(info, [(q["geojson"], i + 1) for i, q in enumerate(test_polys)], bands)
    ok = ~np.isnan(vals).any(axis=1)
    if mask_classes and wc is not None:
        ok &= np.isin(wc, mask_classes)
    rng = np.random.default_rng(seed)
    ok &= rng.random(len(ok)) < fraction
    if not ok.any():
        return None
    codes = np.array([int(q["code"]) for q in test_polys])
    return codes[pids[ok] - 1], _predict(est, vals[ok])


# ---------------------------------------------------------------------------
# Whole runs (called from a QgsTask)
# ---------------------------------------------------------------------------
def _stack_label(info):
    return info["year"] if info["year"] else os.path.splitext(os.path.basename(info["path"]))[0]


def run_local_training(train_stack, predict_stack_path, out_path, classifier_cfg, classes, training, split,
                       mask_classes=None, sampling="mean", max_per_class=200, pixel_accuracy=True,
                       extra_samples=None, cv=None, n_jobs=-1, progress_cb=None, cancel_cb=None):
    """Offline counterpart of classification.run_training. training =
    {"type": "features", "features": [...]} (a QGIS layer)."""
    require_sklearn()
    if training.get("type") != "features":
        raise RuntimeError("Offline classification needs the training polygons as a QGIS layer.")
    info_t = stack_info(train_stack)
    info_p = info_t if predict_stack_path in (None, "", train_stack) else stack_info(predict_stack_path)
    bands = info_t["bands"]
    if not bands:
        raise RuntimeError("The training stack has no feature bands.")
    codes = [int(q["code"]) for q in classes]
    seed = int(classifier_cfg.get("seed", 42))
    train_polys, test_polys = split_polygons(training["features"], classes, split)
    n = 5

    k._step(progress_cb, cancel_cb, 0, n, "Sampling training polygons")
    train_df = sample_polygons(info_t, train_polys, bands, sampling, max_per_class, seed, cancel_cb)
    if len(train_df) == 0:
        raise RuntimeError("No training samples: the training polygons don't overlap valid pixels of the stack.")
    if train_df[k.CLASS_PROP].nunique() < 2:
        raise RuntimeError("Training samples contain fewer than two classes.")
    test_df = None
    if test_polys:
        k._step(progress_cb, cancel_cb, 1, n, "Sampling test polygons")
        test_df = sample_polygons(info_t, test_polys, bands, sampling, max_per_class, seed, cancel_cb)
    own_df = train_df
    train_df = k.pool_samples(train_df, extra_samples)

    k._step(progress_cb, cancel_cb, 2, n, "Training the classifier")
    est, imp, oob = train(train_df, bands, classifier_cfg, n_jobs)
    cv_res = None
    if cv:
        # folds one after another: each Random Forest already uses n_jobs cores
        all_df = pd.concat([d for d in (own_df, test_df) if d is not None], ignore_index=True)
        cv_res = k.run_cv(all_df, bands, codes, cv,
                          lambda tr, te: predict_samples(train(k.pool_samples(tr, extra_samples), bands,
                                                               classifier_cfg, n_jobs)[0], te, bands),
                          progress_cb, cancel_cb)

    metrics = pixel_metrics = per_sample = None
    if test_df is not None and len(test_df):
        k._step(progress_cb, cancel_cb, 3, n, "Accuracy of the test samples")
        pred = predict_samples(est, test_df, bands)
        metrics = k.accuracy_metrics(test_df[k.CLASS_PROP], pred, codes)
        per_sample = test_df[[k.ID_PROP, k.CLASS_PROP]].assign(classification=pred)
        if pixel_accuracy and sampling == "mean":
            pc = _pixel_check(est, info_t, test_polys, bands, mask_classes)
            if pc is not None:
                pixel_metrics = k.accuracy_metrics(pc[0], pc[1], codes)

    k._step(progress_cb, cancel_cb, 4, n, "Classifying the stack")
    predict_stack(est, info_p, bands, out_path, classes, mask_classes, cancel_cb=cancel_cb)

    model = {
        "format": k.MODEL_FORMAT, "version": k.MODEL_VERSION, "engine": "local",
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "trained_year": info_t["year"], "features": info_t["recipe"], "classifier": classifier_cfg,
        "classes": classes, "bands": bands, "sampling": sampling,
        "training": {kk: v for kk, v in training.items() if kk != "features"},
        "stack": {"kind": info_t["kind"], "file": os.path.basename(train_stack)},
        "split": split,
        "samples": {"columns": [k.CLASS_PROP] + bands,
                    "rows": train_df[[k.CLASS_PROP] + bands].values.tolist()},
        "pooled_from": (extra_samples or {}).get("sources", []),
        "accuracy": None if metrics is None else
        {kk: metrics[kk] for kk in ("overall_accuracy", "kappa", "n")},
    }
    return {"local": True, "year": _stack_label(info_p), "map_path": out_path, "classes": classes,
            "cv": cv_res, "train_df": own_df, "test_df": test_df, "n_pooled": len(train_df) - len(own_df),
            "metrics": metrics, "pixel_metrics": pixel_metrics, "importance": imp, "oob_error": oob,
            "per_sample": per_sample, "n_train": len(train_df),
            "n_test": 0 if test_df is None else len(test_df),
            "train_counts": train_df[k.CLASS_PROP].value_counts().to_dict(), "model": model}


def run_local_apply(model, stack_path, out_path, mask_classes=None, reference=None, n_jobs=-1,
                    progress_cb=None, cancel_cb=None):
    """Applies a saved model (online or offline) to a local stack: the
    classifier is retrained from the stored samples with scikit-learn.
    reference = optional (training source, split, matched classes)."""
    require_sklearn()
    info = stack_info(stack_path)
    bands, classes = model["bands"], model["classes"]
    codes = [int(q["code"]) for q in classes]
    k._step(progress_cb, cancel_cb, 0, 3, "Retraining the stored model")
    train_df = k.model_samples(model)
    est, imp, oob = train(train_df, bands, model["classifier"], n_jobs)

    metrics = pixel_metrics = per_sample = None
    if reference is not None:
        k._step(progress_cb, cancel_cb, 1, 3, "Sampling the reference polygons")
        train_part, test_part = split_polygons(reference[0]["features"], reference[2], reference[1])
        ref_polys = test_part if test_part is not None else train_part
        missing = [b for b in bands if b not in info["index"]]
        if missing:
            raise RuntimeError(f"The stack lacks band(s) of the model, e.g. {', '.join(missing[:4])}.")
        ref_df = sample_polygons(info, ref_polys, bands, "mean", cancel_cb=cancel_cb)
        if len(ref_df):
            pred = predict_samples(est, ref_df, bands)
            metrics = k.accuracy_metrics(ref_df[k.CLASS_PROP], pred, codes)
            per_sample = ref_df[[k.ID_PROP, k.CLASS_PROP]].assign(classification=pred)
            pc = _pixel_check(est, info, ref_polys, bands, mask_classes)
            if pc is not None:
                pixel_metrics = k.accuracy_metrics(pc[0], pc[1], codes)

    k._step(progress_cb, cancel_cb, 2, 3, "Classifying the stack")
    predict_stack(est, info, bands, out_path, classes, mask_classes, cancel_cb=cancel_cb)
    return {"local": True, "applied": True, "year": _stack_label(info), "map_path": out_path,
            "classes": classes, "metrics": metrics, "pixel_metrics": pixel_metrics, "importance": imp,
            "oob_error": oob, "per_sample": per_sample, "n_train": len(train_df),
            "n_test": 0 if per_sample is None else len(per_sample),
            "train_counts": train_df[k.CLASS_PROP].value_counts().to_dict(), "model": model}
