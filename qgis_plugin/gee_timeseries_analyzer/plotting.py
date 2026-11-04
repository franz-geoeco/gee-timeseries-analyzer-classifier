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
import numpy as np
from matplotlib.lines import Line2D

from .gee_client import SERIES, INDEX_RANGE

SOURCE_TXT = "Data: Copernicus Sentinel-1/-2, USGS Landsat 8/9 via Google Earth Engine"
EMPTY_COL = "#888888"


# ---------------------------------------------------------------------------
# Hover: per-point information that snaps to the data
# ---------------------------------------------------------------------------
def set_hover(line, labels):
    """Attaches one hover text per data point (same order as the line's
    data) to a Line2D; HoverTip shows it when the cursor is near the point."""
    line.gee_hover = list(labels)
    return line


class HoverTip:
    """Snap-to-point tooltip for a FigureCanvas: the data point nearest to
    the cursor (within `radius` px) on any line carrying set_hover() labels
    is ringed in the line's colour and its text shown beside it. Pure
    matplotlib; survives fig.clear() (re-created on the next hover)."""

    def __init__(self, canvas, radius=12):
        self.canvas, self.radius = canvas, radius
        self._ann = self._ring = None
        self._key = None
        canvas.mpl_connect("motion_notify_event", self._on_move)
        canvas.mpl_connect("figure_leave_event", lambda _e: self.hide())

    def _nearest(self, event):
        ax, best = event.inaxes, None
        for ln in ax.get_lines():
            labels = getattr(ln, "gee_hover", None)
            if not labels or not ln.get_visible():
                continue
            xy = np.asarray(ln.get_xydata(), dtype=float)
            if len(xy) != len(labels) or not len(xy):
                continue
            px = ax.transData.transform(xy)
            d = np.hypot(px[:, 0] - event.x, px[:, 1] - event.y)
            if np.isnan(d).all():
                continue
            i = int(np.nanargmin(d))
            if d[i] <= self.radius and (best is None or d[i] < best[0] - 0.5
                                        or (abs(d[i] - best[0]) <= 0.5 and ln.get_zorder() > best[3].get_zorder())):
                best = (d[i], i, xy[i], ln)
        return best

    def _artists(self, ax):
        if self._ann is not None and self._ann.axes is ax and self._ann.figure is not None:
            return
        self._ann = ax.annotate("", (0, 0), xytext=(12, 12), textcoords="offset points", fontsize=8,
                                zorder=100, annotation_clip=False,
                                bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#999999", alpha=0.96))
        self._ring = Line2D([], [], marker="o", ms=10, mfc="none", mew=2, ls="none", zorder=99)
        ax.add_artist(self._ring)

    def _on_move(self, event):
        if event.inaxes is None or event.button is not None:
            return self.hide()
        hit = self._nearest(event)
        if hit is None:
            return self.hide()
        _, i, (x, y), ln = hit
        if self._key == (id(ln), i) and self._ann is not None and self._ann.get_visible():
            return
        self._key = (id(ln), i)
        ax = event.inaxes
        self._artists(ax)
        bb = ax.bbox                                  # flip the box away from the nearer edge
        right = event.x > bb.x0 + 0.6 * bb.width
        top = event.y > bb.y0 + 0.6 * bb.height
        self._ann.xy = (x, y)
        self._ann.set_text(ln.gee_hover[i])
        self._ann.set_position((-12 if right else 12, -12 if top else 12))
        self._ann.set_ha("right" if right else "left")
        self._ann.set_va("top" if top else "bottom")
        self._ann.set_visible(True)
        self._ring.set_data([x], [y])
        self._ring.set_markeredgecolor(ln.get_color())
        self._ring.set_visible(True)
        self.canvas.draw_idle()

    def hide(self):
        self._key = None
        if self._ann is not None and self._ann.get_visible():
            self._ann.set_visible(False)
            self._ring.set_visible(False)
            self.canvas.draw_idle()


def _fmt_date(d):
    return d.strftime("%d %b %Y") if hasattr(d, "strftime") else str(d)[:10]


def _ts_labels(part, key, index, name):
    """Hover texts of a time-series line: polygon, series, date, value and
    the share of valid pixels of that acquisition."""
    title, unit = _series_title(key, index), _series_unit(key, index)
    fmt = "{:.2f}" if SERIES[key]["unit"] else "{:.3f}"
    valid = part["valid"] if "valid" in part.columns else [None] * len(part)
    out = []
    for d, v, c in zip(part["date"], part["value"], valid):
        txt = f"{name}\n{title}\n{_fmt_date(d)}\n{unit}: {fmt.format(v)}"
        if c is not None and c == c:
            txt += f"\nvalid pixels: {c:.0%}"
        out.append(txt)
    return out


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
        line, = ax.plot(part["date"], part["value"], color=SERIES[key]["color"],
                        lw=1.6, marker="o", ms=3)
        set_hover(line, _ts_labels(part, key, index, name))

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
            line, = ax.plot(pp["date"], pp["value"], color=poly["color"], lw=1.4, marker="o", ms=2.5,
                            ls="--" if dashed else "-",
                            label=poly["name"] if drawn < max_legend else None)
            set_hover(line, _ts_labels(pp, key, index, poly["name"]))
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
                mean, = ax.plot(st.index, st["mean"], color="#2c3e50", lw=1.3, ls=":", zorder=2)
                fmt = "{:.2f}" if SERIES[key]["unit"] else "{:.3f}"
                set_hover(mean, [f"Mean of {int(n)} polygons\n{_series_title(key, index)}\n{_fmt_date(d)}\n"
                                 f"{_series_unit(key, index)}: {fmt.format(m)} ± {fmt.format(s)}"
                                 for d, m, s, n in zip(st.index, st["mean"], st["std"], st["count"])])
        for q in polygons:
            pp = part[part["pid"] == q["pid"]]
            if len(pp) == 0:
                continue
            colour = UNIFORM_COL if uniform else q["color"]
            line, = ax.plot(pp["date"], pp["value"], color=colour, lw=1.1, marker="o", ms=2,
                            alpha=0.85, zorder=3, picker=True, pickradius=5)
            line.set_gid(str(q["pid"]))
            set_hover(line, _ts_labels(pp, key, index, q["name"]))
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


# ---------------------------------------------------------------------------
# Spectra (Spectra section)
# ---------------------------------------------------------------------------
_REGIONS = [(400, 700, "VIS", "#f3f6fb"), (700, 750, "red edge", "#fbf3f3"), (750, 1000, "NIR", "#f4faf3"),
            (1000, 2300, "SWIR", "#faf7f0")]
SENSOR_STYLE = {"s2": {"ls": "-", "marker": "o", "label": "Sentinel-2"},
                "landsat": {"ls": "--", "marker": "s", "label": "Landsat 8/9"}}


def _spectrum_axis(ax, title):
    for lo, hi, name, col in _REGIONS:
        ax.axvspan(lo, hi, color=col, zorder=0, lw=0)
        ax.text((lo + hi) / 2, 1.0, name, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=6.5, color="#888888")
    ax.set_xlim(400, 2300)
    ax.set_xlabel("Wavelength (nm)", fontsize=8)
    ax.set_ylabel("Surface reflectance", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(color="#e6e6e6", lw=0.6)
    ax.set_title(title, loc="left", fontsize=9, fontweight="bold", pad=12)


def _draw_spectrum(ax, res, colour, style, label, sample="", source=""):
    wl, val, sd = res["wavelengths"], res["values"], res["sd"]
    names = res.get("names") or res.get("bands") or [""] * len(wl)
    pts = [(w, v, s, n) for w, v, s, n in zip(wl, val, sd, names) if v is not None]
    if not pts:
        return
    xs, ys, ss, ns = zip(*pts)
    line, = ax.plot(xs, ys, color=colour, ls=style["ls"], marker=style["marker"], ms=3.5, lw=1.6, label=label)
    when = f"{res['date']} ({res['days']:+d} d)" if res.get("date") else ""
    set_hover(line, [f"{sample}\n{n} · {w:g} nm\nreflectance: {v:.4f}" + (f" ± {s:.4f}" if s is not None else "")
                     + f"\n{source}{', ' if source and when else ''}{when}" for w, v, s, n in pts])
    if any(s is not None for s in ss):
        lo = [y - (s or 0) for y, s in zip(ys, ss)]
        hi = [y + (s or 0) for y, s in zip(ys, ss)]
        ax.fill_between(xs, lo, hi, color=colour, alpha=0.12, lw=0)


LOCAL_COLOURS = ["#8e44ad", "#d35400", "#16a085", "#2c3e50", "#c0392b", "#7f8c8d"]


def _sensor_style(s, labels, i):
    st = SENSOR_STYLE.get(s)
    if st is not None:
        return dict(st, label=(labels or {}).get(s, st["label"]))
    return {"ls": ":", "marker": "^", "label": (labels or {}).get(s, s), "colour": LOCAL_COLOURS[i % len(LOCAL_COLOURS)]}


def plot_spectra(fig, samples, sensors, single=None, labels=None):
    """samples: [{id, name, color, results: {sensor: closest_spectrum dict
    or None}}]; sensors: keys ("s2", "landsat", local scene keys); labels:
    {key: panel / legend label}. single=None -> one panel per sensor with
    all samples; single=<sample> -> all sensors of one sample in one panel."""
    fig.clear()
    if not samples:
        plot_placeholder(fig, "Click a point or draw a polygon on the map to sample a spectrum.")
        return
    if single is not None:
        ax = fig.add_subplot(111)
        _spectrum_axis(ax, f"{single['name']} — all sensors")
        drawn = 0
        for i, s in enumerate(sensors):
            res = (single.get("results") or {}).get(s)
            if res:
                st = _sensor_style(s, labels, i)
                colour = single["color"] if s == "s2" else ("#555555" if s == "landsat" else st["colour"])
                name = st["label"] if s not in SENSOR_STYLE else (res["spacecraft"] or st["label"])
                _draw_spectrum(ax, res, colour, st, f"{name} · {res['date']} ({res['days']:+d} d)",
                               single["name"], name)
                drawn += 1
        if drawn:
            ax.legend(fontsize=7, loc="upper left", frameon=False)
        else:
            _empty(ax, "No clear image for this sample.")
    else:
        axes = fig.subplots(len(sensors), 1, squeeze=False)[:, 0]
        for i, (ax, s) in enumerate(zip(axes, sensors)):
            st = _sensor_style(s, labels, i)
            _spectrum_axis(ax, st["label"])
            drawn = 0
            for q in samples:
                res = (q.get("results") or {}).get(s)
                if res:
                    _draw_spectrum(ax, res, q["color"], {"ls": "-", "marker": st["marker"]},
                                   f"{q['name']} · {res['date']} ({res['days']:+d} d)",
                                   q["name"], res.get("spacecraft") or st["label"])
                    drawn += 1
            if drawn:
                ax.legend(fontsize=7, loc="upper left", frameon=False, ncol=1 if drawn <= 5 else 2)
            else:
                _empty(ax, "No clear data for the samples (outside the scene / window, or masked).")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout(rect=(0, 0, 1, 0.97))      # room for the title above the region labels
