"""
Matplotlib figure builders for the GEE Time Series Analyzer dock widget.

Pure matplotlib (no Qt / QGIS imports) so each function can be exercised
from a plain Python interpreter. Every function receives an already-created
Figure, clears it, and draws into it -- the caller (dock_widget.py) owns
the FigureCanvasQTAgg and calls canvas.draw() afterwards.

Both chart types draw the six panels of the original GEE apps, top to
bottom: S1 VV Desc, S1 VH Desc, S1 VV Asc, S1 VH Asc, Sentinel-2 index,
Landsat 8/9 index.
"""
import warnings

import matplotlib.dates as mdates

from .gee_client import SERIES, INDEX_RANGE

SOURCE_TXT = "Data: Copernicus Sentinel-1/-2, USGS Landsat 8/9 via Google Earth Engine"
EMPTY_COL = "#888888"


def _series_title(key, index):
    meta = SERIES[key]
    return f"{meta['label']} {index}" if meta["unit"] is None else meta["label"]


def _series_unit(key, index):
    unit = SERIES[key]["unit"]
    return index if unit is None else unit


def _style_axis(ax, key, index):
    ax.grid(color="#eeeeee")
    ax.set_ylabel(_series_unit(key, index), fontsize=8)
    ax.tick_params(labelsize=8)
    if SERIES[key]["unit"] is None:
        lo, hi = INDEX_RANGE.get(index, (-1, 1))
        ax.set_ylim(lo, hi)
        if lo < 0 < hi:
            ax.axhline(0, color="#333333", lw=1.1, ls="--", zorder=1.5)
    locator = mdates.AutoDateLocator(minticks=3, maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def _empty(ax, msg="No valid observations."):
    ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha="center", va="center",
            color=EMPTY_COL, fontsize=9)


def _attribute(fig):
    fig.text(0.99, 0.002, SOURCE_TXT, ha="right", va="bottom",
             fontsize=7, style="italic", color="#777777")


def _make_axes(fig, n):
    axes = fig.subplots(n, 1, sharex=True, squeeze=False)[:, 0]
    return list(axes)


def plot_single_polygon(fig, df, series_keys, pid, name, index, date_range):
    """The six charts the original apps drew per polygon / clicked field:
    one colour per product, one panel per series."""
    fig.clear()
    if not series_keys:
        ax = fig.add_subplot(111)
        _empty(ax, "No products selected.")
        return
    axes = _make_axes(fig, len(series_keys))
    sub_pid = df[df["pid"] == pid] if df is not None and len(df) else None

    for ax, key in zip(axes, series_keys):
        ax.set_title(f"{_series_title(key, index)} — {name}", loc="left", fontsize=9,
                     fontweight="bold", color=SERIES[key]["color"])
        _style_axis(ax, key, index)
        part = None if sub_pid is None else sub_pid[sub_pid["series"] == key]
        if part is None or len(part) == 0:
            _empty(ax)
            continue
        ax.plot(part["date"], part["value"], color=SERIES[key]["color"],
                lw=1.6, marker="o", ms=3)

    axes[0].set_xlim(*date_range)
    _attribute(fig)
    fig.tight_layout(rect=(0, 0.015, 1, 1))


def plot_combined(fig, df, series_keys, polygons, index, date_range, max_legend=12):
    """The 'Combined Comparison' block of the Time Series Analyzer: one panel
    per series, one line per polygon in the polygon's own colour. Landsat
    lines are dashed, as in the original."""
    fig.clear()
    if not series_keys:
        ax = fig.add_subplot(111)
        _empty(ax, "No products selected.")
        return
    axes = _make_axes(fig, len(series_keys))

    for ax, key in zip(axes, series_keys):
        ax.set_title(f"Combined {_series_title(key, index)} — all polygons", loc="left",
                     fontsize=9, fontweight="bold", color=SERIES[key]["color"])
        _style_axis(ax, key, index)
        part = None if df is None or len(df) == 0 else df[df["series"] == key]
        if part is None or len(part) == 0:
            _empty(ax)
            continue
        dashed = key == "LS_INDEX"
        drawn = 0
        for poly in polygons:
            pp = part[part["pid"] == poly["pid"]]
            if len(pp) == 0:
                continue
            ax.plot(pp["date"], pp["value"], color=poly["color"], lw=1.4, marker="o", ms=2.5,
                    ls="--" if dashed else "-",
                    label=poly["name"] if drawn < max_legend else None)
            drawn += 1
        if drawn == 0:
            _empty(ax)
        elif drawn > 1:
            ax.legend(fontsize=7, loc="best", frameon=False,
                      ncol=min(3, max(1, min(drawn, max_legend) // 4 + 1)))

    axes[0].set_xlim(*date_range)
    _attribute(fig)
    fig.tight_layout(rect=(0, 0.015, 1, 1))


UNIFORM_COL = "#7f8c8d"
HIGHLIGHT_COL = "#e31a1c"


def plot_compare(fig, df, series_keys, polygons, index, date_range, uniform=False, show_band=True):
    """Many polygons on shared axes for spotting outliers. Lines are pickable:
    each carries gid=str(pid). Returns the artist bookkeeping that
    highlight_compare() needs: {"lines": {pid: [Line2D]}, "base": {Line2D:
    colour}, "label": Text, "uniform": bool}."""
    fig.clear()
    out = {"lines": {}, "base": {}, "label": None, "uniform": uniform}
    if not series_keys or not polygons:
        plot_placeholder(fig, "Check at least one polygon." if series_keys else "No products selected.")
        return out
    axes = _make_axes(fig, len(series_keys))
    pids = [q["pid"] for q in polygons]
    sub = None if df is None or len(df) == 0 else df[df["pid"].isin(pids)]

    for ax, key in zip(axes, series_keys):
        ax.set_title(f"{_series_title(key, index)} — {len(polygons)} polygon(s)", loc="left",
                     fontsize=9, fontweight="bold", color=SERIES[key]["color"])
        _style_axis(ax, key, index)
        part = None if sub is None else sub[sub["series"] == key]
        if part is None or len(part) == 0:
            _empty(ax)
            continue
        if show_band and len(polygons) > 1:
            st = part.groupby("date")["value"].agg(["mean", "std", "count"])
            st = st[st["count"] >= 2]
            if len(st):
                ax.fill_between(st.index, st["mean"] - st["std"], st["mean"] + st["std"],
                                color="#95a5a6", alpha=0.22, lw=0, zorder=1)
                ax.plot(st.index, st["mean"], color="#2c3e50", lw=1.3, ls=":", zorder=2)
        for q in polygons:
            pp = part[part["pid"] == q["pid"]]
            if len(pp) == 0:
                continue
            colour = UNIFORM_COL if uniform else q["color"]
            line, = ax.plot(pp["date"], pp["value"], color=colour, lw=1.1, marker="o", ms=2,
                            alpha=0.85, zorder=3, picker=True, pickradius=5)
            line.set_gid(str(q["pid"]))
            out["lines"].setdefault(q["pid"], []).append(line)
            out["base"][line] = colour

    axes[0].set_xlim(*date_range)
    out["label"] = fig.text(0.01, 0.998, "", ha="left", va="top", fontsize=9,
                            fontweight="bold", color=HIGHLIGHT_COL)
    _attribute(fig)
    fig.tight_layout(rect=(0, 0.015, 1, 0.985))
    return out


def highlight_compare(artists, pid, label=""):
    """Emphasises the lines of `pid` (None = reset) without redrawing the
    figure from scratch; the caller only needs canvas.draw_idle()."""
    for p, lines in artists.get("lines", {}).items():
        for line in lines:
            base = artists["base"][line]
            if pid is None:
                line.set(color=base, alpha=0.85, lw=1.1, ms=2, zorder=3)
            elif p == pid:
                line.set(color=HIGHLIGHT_COL if artists["uniform"] else base,
                         alpha=1.0, lw=2.8, ms=4, zorder=10)
            else:
                line.set(color=base, alpha=0.18, lw=0.9, ms=1.5, zorder=3)
    if artists.get("label") is not None:
        artists["label"].set_text(label if pid is not None else "")


def plot_placeholder(fig, msg):
    fig.clear()
    ax = fig.add_subplot(111)
    ax.axis("off")
    _empty(ax, msg)


# ---------------------------------------------------------------------------
# Classification tab (also used by report.py). Every panel's height grows
# with the number of classes / features, so many classes stay readable.
# ---------------------------------------------------------------------------
def _short(name, n=14):
    name = str(name)
    return name if len(name) <= n else name[:n - 1] + "…"


def _fontsize(n):
    return 8 if n <= 10 else (7 if n <= 18 else 6)


def confusion_axis(ax, metrics, names, title, annotate="count"):
    """Heatmap: colour = share of the reference row, diagonal blue,
    confusions red; cells show counts (or row shares), omitted if too many
    classes to be legible."""
    import numpy as np
    m = np.array(metrics["matrix"], dtype=float)
    n = m.shape[0]
    tot = m.sum(axis=1, keepdims=True)
    share = m / np.where(tot > 0, tot, 1)
    rgb = np.ones(m.shape + (3,))
    for i in range(n):
        for j in range(n):
            s = share[i, j]
            rgb[i, j] = (1 - 0.85 * s, 1 - 0.55 * s, 1) if i == j else (1, 1 - 0.85 * s, 1 - 0.85 * s)
    ax.imshow(rgb, aspect="auto", interpolation="nearest")
    fs = _fontsize(n)
    if n <= 30:
        for i in range(n):
            for j in range(n):
                if m[i, j]:
                    txt = f"{share[i, j]:.0%}" if annotate == "share" else f"{int(m[i, j])}"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=fs - 1,
                            color="white" if share[i, j] > 0.6 else "black")
    labels = [_short(x) for x in names]
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=fs)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=fs)
    ax.set_xticks(np.arange(-0.5, n), minor=True)
    ax.set_yticks(np.arange(-0.5, n), minor=True)
    ax.grid(which="minor", color="#ffffff", lw=0.6)
    ax.tick_params(which="minor", length=0)
    ax.set_xlabel("map", fontsize=8)
    ax.set_ylabel("reference", fontsize=8)
    ax.set_title(f"{title}\nOA {metrics['overall_accuracy']:.1%} · κ {metrics['kappa']:.2f} · "
                 f"n = {metrics['n']}", loc="left", fontsize=9, fontweight="bold")


def class_accuracy_axis(ax, metrics, names, colours=None, title="Accuracy per class"):
    """Horizontal grouped bars (PA / UA / F1) -- one row per class, so the
    labels stay readable however many classes there are."""
    n = len(names)
    y = list(range(n))
    h = 0.26
    for off, key, label, col in ((-h, "producers", "Producer's", "#1f77b4"),
                                 (0, "users", "User's", "#ff7f0e"),
                                 (h, "f1", "F1", "#2ca02c")):
        vals = [v if v is not None else 0 for v in metrics[key]]
        ax.barh([i + off for i in y], vals, h, label=label, color=col)
    ax.set_yticks(y)
    ax.set_yticklabels([_short(x, 18) for x in names], fontsize=_fontsize(n))
    if colours:
        for tick, col in zip(ax.get_yticklabels(), colours):
            tick.set_bbox({"facecolor": col, "alpha": 0.35, "edgecolor": "none", "pad": 1})
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="x", color="#eeeeee")
    ax.legend(fontsize=7, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    ax.set_title(title, loc="left", fontsize=9, fontweight="bold", pad=18)


def importance_axis(ax, importance, max_features=20):
    top = list(importance.items())[:max_features][::-1]
    ax.barh([_short(kk, 24) for kk, _ in top], [v for _, v in top], color="#27ae60")
    ax.tick_params(labelsize=7)
    ax.grid(axis="x", color="#eeeeee")
    ax.set_title(f"Variable importance (top {len(top)} of {len(importance)})", loc="left",
                 fontsize=9, fontweight="bold")


def classification_heights(result, max_features=20):
    """Pixel heights of the panels plot_classification draws (grow with the
    number of classes / features)."""
    h = []
    m = result.get("metrics")
    if m:
        n = len(m["codes"])
        h += [max(320, 140 + 24 * n), max(220, 90 + 30 * n)]
    pm = result.get("pixel_metrics")
    if pm and pm["n"]:
        h.append(max(320, 140 + 24 * len(pm["codes"])))
    if result.get("importance"):
        h.append(70 + 16 * min(len(result["importance"]), max_features))
    return h or [200]


def plot_classification(fig, result, max_features=20, ref_label="test"):
    """Confusion matrices (test samples, map pixels), producer's / user's
    accuracy + F1 per class and the top variable importances. The figure is
    resized to the panels' pixel heights (the caller sizes the canvas)."""
    fig.clear()
    heights = classification_heights(result, max_features)
    fig.set_size_inches(fig.get_figwidth(), (sum(heights) + 40) / fig.dpi)
    if not (result.get("metrics") or result.get("importance")):
        plot_placeholder(fig, "No accuracy assessment (no test set).")
        return
    by_code = {int(q["code"]): q for q in result["classes"]}
    axes = fig.subplots(len(heights), 1, squeeze=False,
                        gridspec_kw={"height_ratios": heights})[:, 0]
    ai = 0
    m = result.get("metrics")
    if m:
        names = [by_code.get(c_, {}).get("name", str(c_)) for c_ in m["codes"]]
        colours = [by_code.get(c_, {}).get("color", "#999999") for c_ in m["codes"]]
        confusion_axis(axes[ai], m, names, f"{ref_label.capitalize()} samples")
        class_accuracy_axis(axes[ai + 1], m, names, colours, f"Accuracy per class ({ref_label} samples)")
        ai += 2
    pm = result.get("pixel_metrics")
    if pm and pm["n"]:
        confusion_axis(axes[ai], pm, [by_code.get(c_, {}).get("name", str(c_)) for c_ in pm["codes"]],
                       f"Map pixels in {ref_label} polygons")
        ai += 1
    if result.get("importance"):
        importance_axis(axes[ai], result["importance"], max_features)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout(h_pad=2.0)


# ---------------------------------------------------------------------------
# Class profiles (training samples) and crop-rotation flows
# ---------------------------------------------------------------------------
def class_profiles_axis(ax, df, classes, codes=None, band=True, title=""):
    """Mean ± 1σ per class over the periods (df from analysis_tools.class_profiles)."""
    by_code = {int(q["code"]): q for q in classes}
    codes = [c_ for c_ in sorted(df["code"].unique()) if codes is None or c_ in codes]
    xs = sorted(df["x"].unique())
    labels = df.drop_duplicates("x").set_index("x")["x_label"]
    for code in codes:
        part = df[df["code"] == code].sort_values("x")
        q = by_code.get(int(code), {"name": str(code), "color": "#999999"})
        ax.plot(part["x"], part["mean"], color=q["color"], lw=1.8, marker="o", ms=3,
                label=f"{q['name']} (n={int(part['n'].max())})")
        if band:
            sd = part["sd"].fillna(0)
            ax.fill_between(part["x"], part["mean"] - sd, part["mean"] + sd, color=q["color"], alpha=0.13, lw=0)
    ax.set_xticks(xs)
    tilt = len(xs) > 5
    ax.set_xticklabels([labels.get(x_, "") for x_ in xs], rotation=45 if tilt else 0,
                       ha="right" if tilt else "center", fontsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(color="#eeeeee")
    lo, hi = ax.get_ylim()
    if lo < 0 < hi:
        ax.axhline(0, color="#333333", lw=1.0, ls="--", zorder=1.5)
    ax.set_title(title, loc="left", fontsize=9, fontweight="bold")
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False,
              ncol=1 if len(codes) <= 14 else 2)


def plot_class_profiles(fig, df, classes, codes=None, band=True, title=""):
    fig.clear()
    if df is None or len(df) == 0:
        plot_placeholder(fig, "No profiles — train or load a model first.")
        return
    ax = fig.add_subplot(111)
    class_profiles_axis(ax, df, classes, codes, band, title)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout()


def plot_alluvial(fig, rot, min_share=0.002, title="Crop rotation"):
    """Flow (alluvial) diagram: one column per year (stacked class areas),
    bands between consecutive years = transitions (area)."""
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch, Rectangle
    fig.clear()
    ax = fig.add_subplot(111)
    years, codes, classes = rot["years"], rot["codes"], rot["classes"]
    areas = rot["areas"].pivot(index="code", columns="year", values="ha").reindex(index=codes).fillna(0)
    total = max(float(areas.sum().max()), 1e-9)
    gap = 0.012 * total
    colw = 0.12
    pos = {}                                     # (year, code) -> [bottom, top]
    for xi, y in enumerate(years):
        top = total + gap * len(codes)
        for code in codes:
            h = float(areas.loc[code, y])
            pos[(y, code)] = [top - h, top]
            if h > 0:
                ax.add_patch(Rectangle((xi - colw / 2, top - h), colw, h, color=classes[code]["color"], lw=0))
                if h > 0.03 * total:
                    ax.text(xi, top - h / 2, f"{h:,.0f}", ha="center", va="center", fontsize=6, color="#111",
                            rotation=90)
            top -= h + gap
    for i in range(len(years) - 1):
        y0, y1 = years[i], years[i + 1]
        m = rot["transitions"][(y0, y1)]
        out_off = {code: pos[(y0, code)][1] for code in codes}
        in_off = {code: pos[(y1, code)][1] for code in codes}
        for a in codes:
            for b in codes:
                v = float(m[a, b])
                if v <= 0:
                    continue
                ya0, ya1 = out_off[a], out_off[a] - v
                yb0, yb1 = in_off[b], in_off[b] - v
                out_off[a], in_off[b] = ya1, yb1
                if v < min_share * total:
                    continue
                x0, x1 = i + colw / 2, i + 1 - colw / 2
                xm = (x0 + x1) / 2
                verts = [(x0, ya0), (xm, ya0), (xm, yb0), (x1, yb0), (x1, yb1), (xm, yb1), (xm, ya1), (x0, ya1), (x0, ya0)]
                codes_p = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO,
                           Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY]
                ax.add_patch(PathPatch(Path(verts, codes_p), facecolor=classes[a]["color"],
                                       alpha=0.18 if a == b else 0.45, lw=0))
    for code in codes:
        b0, t0 = pos[(years[0], code)]
        if t0 - b0 > 0:
            ax.text(-colw / 2 - 0.03, (b0 + t0) / 2, classes[code]["name"], ha="right", va="center", fontsize=7)
        b1, t1 = pos[(years[-1], code)]
        if t1 - b1 > 0:
            ax.text(len(years) - 1 + colw / 2 + 0.03, (b1 + t1) / 2, classes[code]["name"], ha="left",
                    va="center", fontsize=7)
    ax.set_xlim(-0.6, len(years) - 0.4)
    ax.set_ylim(0, total + gap * len(codes) * 1.02)
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels([str(y) for y in years], fontsize=9, fontweight="bold")
    ax.set_yticks([])
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.set_title(f"{title} — areas in ha; bands = transitions (pale = unchanged)", loc="left", fontsize=9,
                 fontweight="bold")
    fig.tight_layout()
