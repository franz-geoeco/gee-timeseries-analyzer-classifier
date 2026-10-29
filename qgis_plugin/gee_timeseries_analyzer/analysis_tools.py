"""
Engine-independent analysis of classification results (numpy / pandas /
GDAL, no Qt / QGIS imports):

  * class profiles      -- mean ± sd per class and period from the samples
  * training check      -- suspicious training samples (far from their own
                           class or closer to another class)
  * spatial k-fold CV   -- folds of spatial blocks; any fit / predict
                           function (Earth Engine or scikit-learn)
  * field majority      -- classified raster -> majority class per polygon
  * crop rotation       -- areas per year and transition matrices of several
                           classified rasters, change map
"""
import json
import math
import os
import re

import numpy as np
import pandas as pd

from . import classification as k

X_PROP, Y_PROP = "gta_x", "gta_y"     # polygon centroid (lon / lat) of a sample
_BAND = re.compile(r"^(.+?)_(\d{2}[A-Za-z0-9-]*)(?:_([A-Za-z0-9]+))?$")


# ---------------------------------------------------------------------------
# Band names -> variable / period / reducer
# ---------------------------------------------------------------------------
def parse_bands(bands, recipe=None, year=None):
    """-> DataFrame [band, variable, reducer, period, x, x_label] in band
    order; x = period midpoint as day of the season (with a recipe) or the
    period's rank."""
    rows = []
    table = {}
    if recipe:
        year = year or 2020
        table = {r["name"]: r for r in k.band_table(recipe, year)}
        t0 = min((r["start"] for r in table.values()), default=None)
    for b in bands:
        r = table.get(b)
        if r is not None:
            s, e = pd.Timestamp(r["start"]), pd.Timestamp(r["end"])
            mid = s + (e - s) / 2
            rows.append({"band": b, "variable": r["variable"], "reducer": r["reducer"], "period": r["period"],
                         "x": (mid - pd.Timestamp(t0)).days, "x_label": mid.strftime("%d %b")})
            continue
        mt = _BAND.match(b)       # e.g. NDVI_05_median, VH_VV_04a_p90, NDVI_07 (composite stack)
        var, period, red = (mt.group(1), mt.group(2), mt.group(3) or "value") if mt else (b, "", "value")
        rows.append({"band": b, "variable": var, "reducer": red, "period": period, "x": None, "x_label": period})
    df = pd.DataFrame(rows)
    if len(df) and df["x"].isna().any():
        for _, part in df.groupby(["variable", "reducer"]):
            df.loc[part.index, "x"] = range(len(part))
    return df


# ---------------------------------------------------------------------------
# Class profiles (item 4)
# ---------------------------------------------------------------------------
def class_profiles(samples, bands, recipe=None, year=None):
    """samples: DataFrame [gta_class, <bands>] -> {(variable, reducer):
    DataFrame [code, x, x_label, period, mean, sd, n]}."""
    meta = parse_bands(bands, recipe, year)
    out = {}
    for (var, red), part in meta.groupby(["variable", "reducer"], sort=False):
        frames = []
        for _, row in part.iterrows():
            g = samples.groupby(k.CLASS_PROP)[row["band"]].agg(["mean", "std", "count"]).reset_index()
            g.columns = ["code", "mean", "sd", "n"]
            g["x"], g["x_label"], g["period"] = row["x"], row["x_label"], row["period"]
            frames.append(g)
        out[(var, red)] = pd.concat(frames, ignore_index=True)
    return out


# ---------------------------------------------------------------------------
# Training check (item 5)
# ---------------------------------------------------------------------------
def suspicious_samples(df, bands, threshold=2.5, min_class=3):
    """Per sample (pixels averaged per polygon): mean |z| of its features
    against its own class (robust: class median / MAD-based sd), and the
    class whose centroid is nearest in standardised feature space.
    -> DataFrame [gta_id, gta_class, score, nearest, nearest_dist,
    own_dist, n, flag, reason] sorted by severity."""
    if df is None or len(df) == 0:
        return pd.DataFrame()
    d = df.groupby([k.ID_PROP, k.CLASS_PROP], as_index=False)[bands].mean()
    n_px = df.groupby([k.ID_PROP, k.CLASS_PROP]).size().reset_index(name="n")
    d = d.merge(n_px, on=[k.ID_PROP, k.CLASS_PROP])
    X = d[bands].to_numpy(float)
    gsd = np.nanstd(X, axis=0)
    gsd[gsd == 0] = 1
    Z = (X - np.nanmean(X, axis=0)) / gsd
    codes = d[k.CLASS_PROP].to_numpy()
    cents, score = {}, np.full(len(d), np.nan)
    for c_ in np.unique(codes):
        idx = codes == c_
        cents[c_] = np.nanmean(Z[idx], axis=0)
        if idx.sum() >= min_class:
            med = np.nanmedian(X[idx], axis=0)
            mad = 1.4826 * np.nanmedian(np.abs(X[idx] - med), axis=0)
            sd = np.where(mad > 0, mad, np.nanstd(X[idx], axis=0))
            sd[sd == 0] = 1
            score[idx] = np.nanmean(np.abs((X[idx] - med) / sd), axis=1)
    cc = np.array(list(cents))
    C = np.stack([cents[c_] for c_ in cc])
    dist = np.sqrt(((Z[:, None, :] - C[None, :, :]) ** 2).mean(axis=2))
    nearest = cc[np.argmin(dist, axis=1)]
    own = dist[np.arange(len(d)), np.searchsorted(cc, codes)]
    d["score"], d["nearest"], d["nearest_dist"], d["own_dist"] = score, nearest, dist.min(axis=1), own
    far = d["score"] > threshold
    other = (d["nearest"] != d[k.CLASS_PROP]) & (d["own_dist"] > 1.15 * d["nearest_dist"])
    d["flag"] = far | other
    d["reason"] = np.where(far & other, "far from own class + closer to another",
                           np.where(far, "far from own class", np.where(other, "closer to another class", "")))
    d["severity"] = d["score"].fillna(0) + np.where(other, d["own_dist"] / d["nearest_dist"].clip(lower=1e-6), 0)
    return d.drop(columns=bands).sort_values(["flag", "severity"], ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Spatial cross-validation (item 6)
# ---------------------------------------------------------------------------
def spatial_folds(df, k_folds=5, block_km=5.0, seed=42):
    """Fold number per sample: samples are grouped into square blocks of
    block_km (from the polygon centroid), the blocks are split randomly into
    k folds -- so neighbouring fields and the pixels of one field stay in
    the same fold."""
    lat0 = float(np.nanmean(df[Y_PROP])) if len(df) else 0.0
    bx = np.floor(df[X_PROP].to_numpy(float) * 111.32 * math.cos(math.radians(lat0)) / block_km)
    by = np.floor(df[Y_PROP].to_numpy(float) * 110.57 / block_km)
    blocks = pd.Series(list(zip(bx, by)))
    uniq = blocks.drop_duplicates().tolist()
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    fold_of = {b: int(i % k_folds) for i, b in zip(range(len(uniq)), [uniq[j] for j in order])}
    return blocks.map(fold_of).to_numpy(), len(uniq)


def cross_validate(df, bands, codes, fit_predict, k_folds=5, block_km=5.0, seed=42,
                   progress_cb=None, cancel_cb=None):
    """fit_predict(train_df, test_df) -> predicted codes. Returns {folds:
    [metrics...], pooled: metrics, oa_mean, oa_sd, kappa_mean, kappa_sd,
    f1_mean [per class], f1_sd, k, block_km, n_blocks}."""
    if X_PROP not in df.columns or df[X_PROP].isna().all():
        raise RuntimeError("The samples have no coordinates for a spatial cross-validation.")
    df = df.dropna(subset=[X_PROP, Y_PROP]).reset_index(drop=True)
    fold, n_blocks = spatial_folds(df, k_folds, block_km, seed)
    folds, ref_all, pred_all = [], [], []
    for f in range(k_folds):
        if cancel_cb is not None and cancel_cb():
            raise InterruptedError("Cancelled")
        if progress_cb is not None:
            progress_cb(f, k_folds, f"Cross-validation fold {f + 1}/{k_folds}")
        test, train = df[fold == f], df[fold != f]
        if len(test) == 0 or train[k.CLASS_PROP].nunique() < 2:
            continue
        pred = np.asarray(fit_predict(train, test))
        m = k.accuracy_metrics(test[k.CLASS_PROP], pred, codes)
        m["fold"] = f + 1
        folds.append(m)
        ref_all += list(test[k.CLASS_PROP])
        pred_all += list(pred)
    if not folds:
        raise RuntimeError("Cross-validation failed: too few samples / blocks for the folds.")
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
    oa = np.array([m["overall_accuracy"] for m in folds])
    ka = np.array([m["kappa"] for m in folds])
    f1 = np.array([[np.nan if v is None else v for v in m["f1"]] for m in folds], dtype=float)
    return {"folds": folds, "pooled": k.accuracy_metrics(ref_all, pred_all, codes),
            "oa_mean": float(oa.mean()), "oa_sd": float(oa.std(ddof=1)) if len(oa) > 1 else 0.0,
            "kappa_mean": float(np.nanmean(ka)), "kappa_sd": float(np.nanstd(ka, ddof=1)) if len(ka) > 1 else 0.0,
            "f1_mean": [None if np.isnan(v) else float(v) for v in np.nanmean(f1, axis=0)],
            "f1_sd": [None if np.isnan(v) else float(v) for v in np.nanstd(f1, axis=0)],
            "k": k_folds, "block_km": block_km, "n_blocks": n_blocks, "codes": [int(c_) for c_ in codes]}


def geojson_centroid(geojson):
    """Rough centroid (mean of the outer ring vertices) of a (Multi)Polygon."""
    coords = geojson.get("coordinates") or []
    ring = coords[0][0] if geojson.get("type") == "MultiPolygon" and coords and coords[0] else (coords[0] if coords else [])
    if not ring:
        return None, None
    xs, ys = [p_[0] for p_ in ring], [p_[1] for p_ in ring]
    return sum(xs) / len(xs), sum(ys) / len(ys)


# ---------------------------------------------------------------------------
# Classified rasters: classes, field majority (item 2), rotation (item 10)
# ---------------------------------------------------------------------------
def raster_classes(path):
    """{code: {"name", "color"}} from the colour table / category names."""
    from osgeo import gdal
    ds = gdal.Open(path)
    b = ds.GetRasterBand(1)
    ct, names = b.GetColorTable(), b.GetCategoryNames() or []
    out = {}
    for code in range(256):
        name = names[code] if code < len(names) and names[code] else None
        col = None
        if ct is not None and code < ct.GetCount():
            r, g, b_, a = ct.GetColorEntry(code)
            if a and (r, g, b_) != (0, 0, 0):
                col = f"#{r:02x}{g:02x}{b_:02x}"
        if name or col:
            out[code] = {"name": name or str(code), "color": col or k.default_colour(code)}
    return out


def _grid_ds(path, ref=None):
    """Opens `path`; if `ref` (a GDAL dataset) is given and the grids
    differ, returns a nearest-neighbour VRT on ref's grid."""
    from osgeo import gdal
    ds = gdal.Open(path)
    if ref is None:
        return ds
    if (ds.GetGeoTransform() == ref.GetGeoTransform() and ds.RasterXSize == ref.RasterXSize
            and ds.RasterYSize == ref.RasterYSize and ds.GetProjection() == ref.GetProjection()):
        return ds
    gt = ref.GetGeoTransform()
    bounds = (gt[0], gt[3] + gt[5] * ref.RasterYSize, gt[0] + gt[1] * ref.RasterXSize, gt[3])
    return gdal.Warp("", ds, format="VRT", outputBounds=bounds, xRes=gt[1], yRes=abs(gt[5]),
                     dstSRS=ref.GetProjection(), resampleAlg="near")


def _pixel_area(ds, rows):
    """Area in ha of one pixel for each row index (geographic CRS: by latitude)."""
    from osgeo import osr
    gt = ds.GetGeoTransform()
    srs = osr.SpatialReference(wkt=ds.GetProjection())
    if srs.IsGeographic():
        lat = gt[3] + (rows + 0.5) * gt[5]
        return abs(gt[1] * 111320 * np.cos(np.radians(lat)) * gt[5] * 110574) / 1e4
    return np.full(len(rows), abs(gt[1] * gt[5]) / 1e4)


def _mask_band(ref, polygons):
    """Rasterised (geojson in EPSG:4326) polygons on ref's grid -> 0/1 array, or None."""
    if not polygons:
        return None
    from osgeo import gdal
    from . import local_classifier as lc
    ogr_ds, lyr, _ = lc._polygon_layer([(g, 1) for g in polygons], ref.GetProjection())   # keep ogr_ds alive
    mem = gdal.GetDriverByName("MEM").Create("", ref.RasterXSize, ref.RasterYSize, 1, gdal.GDT_Byte)
    mem.SetGeoTransform(ref.GetGeoTransform())
    mem.SetProjection(ref.GetProjection())
    gdal.RasterizeLayer(mem, [1], lyr, burn_values=[1])
    return mem.GetRasterBand(1).ReadAsArray()


def field_majority(raster_path, polygons, nodata=k.NODATA, cancel_cb=None):
    """Majority class per polygon of a classified raster (pixel centres in
    the polygon). polygons: [{"geojson" (EPSG:4326), "id"}] -> DataFrame
    [id, class_code, share, second_code, second_share, n_pixels]."""
    from osgeo import gdal
    from . import local_classifier as lc
    ds = gdal.Open(raster_path)
    info = {"path": raster_path, "wkt": ds.GetProjection(), "gt": ds.GetGeoTransform(),
            "width": ds.RasterXSize, "height": ds.RasterYSize}
    nd = ds.GetRasterBand(1).GetNoDataValue()
    nd = nodata if nd is None else nd
    ogr_ds, lyr, env = lc._polygon_layer([(q["geojson"], i + 1) for i, q in enumerate(polygons)], info["wkt"])
    counts = np.zeros((len(polygons) + 1, 256), dtype=np.int64)
    if env is not None:
        x0, px, _, y0, _, py = info["gt"]
        c0 = max(0, int(np.floor((env[0] - x0) / px)))
        c1 = min(info["width"], int(np.ceil((env[1] - x0) / px)) + 1)
        r0 = max(0, int(np.floor((env[3] - y0) / py)))
        r1 = min(info["height"], int(np.ceil((env[2] - y0) / py)) + 1)
        W = max(0, c1 - c0)
        step = max(1, 20_000_000 // max(1, W * 4))
        band = ds.GetRasterBand(1)
        for r in range(r0, r1, step):
            if cancel_cb is not None and cancel_cb():
                raise InterruptedError("Cancelled")
            h = min(step, r1 - r)
            mem = gdal.GetDriverByName("MEM").Create("", W, h, 1, gdal.GDT_Int32)
            mem.SetGeoTransform((x0 + c0 * px, px, 0, y0 + r * py, 0, py))
            mem.SetProjection(info["wkt"])
            gdal.RasterizeLayer(mem, [1], lyr, options=["ATTRIBUTE=pid"])
            pid = mem.GetRasterBand(1).ReadAsArray()
            cls = band.ReadAsArray(c0, r, W, h)
            ok = (pid > 0) & (cls != nd)
            np.add.at(counts, (pid[ok], cls[ok].astype(np.int64)), 1)
    rows = []
    for i, q in enumerate(polygons):
        cnt = counts[i + 1]
        n = int(cnt.sum())
        if n == 0:
            rows.append({"id": q["id"], "class_code": None, "share": None, "second_code": None,
                         "second_share": None, "n_pixels": 0})
            continue
        order = np.argsort(cnt)[::-1]
        second = order[1] if cnt[order[1]] > 0 else None
        rows.append({"id": q["id"], "class_code": int(order[0]), "share": float(cnt[order[0]] / n),
                     "second_code": None if second is None else int(second),
                     "second_share": None if second is None else float(cnt[second] / n), "n_pixels": n})
    return pd.DataFrame(rows)


def rotation(entries, mask_polygons=None, progress_cb=None, cancel_cb=None):
    """entries: [(year, path)] sorted by year. All rasters are read on the
    grid of the first one. -> {years, classes {code: {name, color}}, areas
    DataFrame [year, code, ha], transitions {(y0, y1): ha matrix 256x256},
    pixel_ha}."""
    from osgeo import gdal
    entries = sorted(entries, key=lambda e: e[0])
    ref = gdal.Open(entries[0][1])
    dss = [_grid_ds(p, ref) for _, p in entries]
    classes = {}
    for _, p in entries:
        for code, q in raster_classes(p).items():
            classes.setdefault(code, q)
    nds = []
    for ds in dss:
        nd = ds.GetRasterBand(1).GetNoDataValue()
        nds.append(k.NODATA if nd is None else nd)
    mask = _mask_band(ref, mask_polygons)
    W, H = ref.RasterXSize, ref.RasterYSize
    step = max(1, 40_000_000 // max(1, W * len(dss)))
    areas = np.zeros((len(dss), 256))
    trans = {(entries[i][0], entries[i + 1][0]): np.zeros((256, 256)) for i in range(len(entries) - 1)}
    blocks = (H + step - 1) // step
    for bi, r in enumerate(range(0, H, step)):
        if cancel_cb is not None and cancel_cb():
            raise InterruptedError("Cancelled")
        h = min(step, H - r)
        pa = _pixel_area(ref, np.arange(r, r + h))[:, None] * np.ones((1, W))
        arrs = [ds.GetRasterBand(1).ReadAsArray(0, r, W, h).astype(np.int64) for ds in dss]
        valid = [(a != nd) & (a >= 0) & (a < 256) for a, nd in zip(arrs, nds)]
        if mask is not None:
            valid = [v & (mask[r:r + h] > 0) for v in valid]
        for i, (a, v) in enumerate(zip(arrs, valid)):
            areas[i] += np.bincount(a[v], weights=pa[v], minlength=256)[:256]
        for i in range(len(arrs) - 1):
            v = valid[i] & valid[i + 1]
            idx = arrs[i][v] * 256 + arrs[i + 1][v]
            trans[(entries[i][0], entries[i + 1][0])] += np.bincount(idx, weights=pa[v], minlength=65536).reshape(256, 256)
        if progress_cb is not None:
            progress_cb(bi + 1, blocks + 1, f"Block {bi + 1}/{blocks}")
    used = sorted(set(np.nonzero(areas.sum(axis=0))[0].tolist()))
    for code in used:
        classes.setdefault(code, {"name": str(code), "color": k.default_colour(code)})
    rows = [{"year": y, "code": code, "ha": float(areas[i, code])} for i, (y, _) in enumerate(entries) for code in used]
    return {"years": [y for y, _ in entries], "paths": [p for _, p in entries], "codes": used,
            "classes": {c_: classes[c_] for c_ in used}, "areas": pd.DataFrame(rows), "transitions": trans}


def transition_table(rot, y0, y1):
    """Long table [from_code, from_name, to_code, to_name, ha, pct_of_from]."""
    m = rot["transitions"][(y0, y1)]
    names = {c_: q["name"] for c_, q in rot["classes"].items()}
    rows = []
    for a in rot["codes"]:
        tot = m[a].sum()
        for b in rot["codes"]:
            if m[a, b] > 0:
                rows.append({"from_code": a, "from_name": names[a], "to_code": b, "to_name": names[b],
                             "ha": float(m[a, b]), "pct_of_from": float(m[a, b] / tot) if tot else None})
    return pd.DataFrame(rows)


def write_change_map(path_a, path_b, out_path, classes, mask_polygons=None, cancel_cb=None):
    """Change map A -> B: each observed transition gets its own value
    (1..n, 0 = NoData), named 'cotton -> rice'; unchanged pixels grey,
    changes in the colour of the new class. Returns the legend list."""
    from osgeo import gdal
    ref = gdal.Open(path_a)
    b_ds = _grid_ds(path_b, ref)
    nd_a = ref.GetRasterBand(1).GetNoDataValue()
    nd_b = b_ds.GetRasterBand(1).GetNoDataValue()
    nd_a = k.NODATA if nd_a is None else nd_a
    nd_b = k.NODATA if nd_b is None else nd_b
    mask = _mask_band(ref, mask_polygons)
    W, H = ref.RasterXSize, ref.RasterYSize
    step = max(1, 40_000_000 // max(1, W * 2))
    seen = np.zeros(65536, dtype=bool)
    for r in range(0, H, step):
        h = min(step, H - r)
        a = ref.GetRasterBand(1).ReadAsArray(0, r, W, h).astype(np.int64)
        b = b_ds.GetRasterBand(1).ReadAsArray(0, r, W, h).astype(np.int64)
        v = (a != nd_a) & (b != nd_b)
        if mask is not None:
            v &= mask[r:r + h] > 0
        seen[np.unique(a[v] * 256 + b[v])] = True
    codes = np.nonzero(seen)[0]
    if len(codes) > 65534:
        raise RuntimeError("Too many transitions.")
    lut = np.zeros(65536, dtype=np.uint16)
    lut[codes] = np.arange(1, len(codes) + 1)
    out = gdal.GetDriverByName("GTiff").Create(out_path, W, H, 1, gdal.GDT_UInt16,
                                               ["TILED=YES", "COMPRESS=DEFLATE"])
    out.SetGeoTransform(ref.GetGeoTransform())
    out.SetProjection(ref.GetProjection())
    band = out.GetRasterBand(1)
    band.SetNoDataValue(0)
    for r in range(0, H, step):
        if cancel_cb is not None and cancel_cb():
            raise InterruptedError("Cancelled")
        h = min(step, H - r)
        a = ref.GetRasterBand(1).ReadAsArray(0, r, W, h).astype(np.int64)
        b = b_ds.GetRasterBand(1).ReadAsArray(0, r, W, h).astype(np.int64)
        v = (a != nd_a) & (b != nd_b)
        if mask is not None:
            v &= mask[r:r + h] > 0
        res = np.zeros((h, W), dtype=np.uint16)
        res[v] = lut[a[v] * 256 + b[v]]
        band.WriteArray(res, 0, r)
    legend = []
    for i, code in enumerate(codes, start=1):
        fa, fb = int(code // 256), int(code % 256)
        na = classes.get(fa, {}).get("name", str(fa))
        nb = classes.get(fb, {}).get("name", str(fb))
        col = "#c8c8c8" if fa == fb else classes.get(fb, {}).get("color", "#999999")
        legend.append({"value": i, "from": fa, "to": fb, "label": f"{na} → {nb}" if fa != fb else f"{na} (unchanged)",
                       "color": col})
    band.SetCategoryNames([""] + [q["label"] for q in legend])
    band = out = None
    with open(os.path.splitext(out_path)[0] + "_legend.json", "w", encoding="utf-8") as fh:
        json.dump(legend, fh, indent=1)
    return legend
