"""
Self-contained HTML accuracy report of a classification run (Earth Engine or
offline): summary cards, kappa scale, confusion matrices as colour-coded HTML
tables (readable and selectable for many classes), the most frequent
confusions, a sortable per-class table with sample-size check, charts
(embedded PNG) and the full run settings.

Pure Python + matplotlib (no Qt / QGIS imports): build_report(result) works
on the result dicts of classification.run_* and local_classifier.run_*.
"""
import base64
import os
import datetime as dt
import html
import io
import math

from matplotlib.figure import Figure

from . import classification as k
from . import plotting as p

KAPPA_SCALE = [(0.0, "Slight", "#e15759"), (0.2, "Fair", "#f28e2b"), (0.4, "Moderate", "#edc948"),
               (0.6, "Substantial", "#59a14f"), (0.8, "Almost perfect", "#2e7d32")]
ADEQUATE, MARGINAL = 50, 20     # training samples per class (Congalton & Green 2019: >= 50)


# ---------------------------------------------------------------------------
# Extra metrics
# ---------------------------------------------------------------------------
def kappa_label(kappa):
    """Landis & Koch (1977)."""
    if kappa is None or kappa != kappa:
        return "–"
    if kappa < 0:
        return "Poor (worse than chance)"
    label = KAPPA_SCALE[0][1]
    for lo, name, _ in KAPPA_SCALE:
        if kappa >= lo:
            label = name
    return label


def summary_metrics(m):
    """OA with 95 % CI, balanced accuracy, macro F1 and Pontius & Millones
    (2011) quantity / allocation disagreement."""
    n = m["n"]
    oa = m["overall_accuracy"]
    ci = 1.96 * math.sqrt(oa * (1 - oa) / n) if n else float("nan")
    pa = [v for v, r in zip(m["producers"], m["n_reference"]) if r and v is not None]
    f1 = [v for v, r in zip(m["f1"], m["n_reference"]) if r and v is not None]
    q = (0.5 * sum(abs(r - c) for r, c in zip(m["n_reference"], m["n_mapped"])) / n) if n else float("nan")
    return {"oa": oa, "oa_ci": ci, "kappa": m["kappa"], "balanced": sum(pa) / len(pa) if pa else float("nan"),
            "macro_f1": sum(f1) / len(f1) if f1 else float("nan"),
            "quantity": q, "allocation": (1 - oa) - q if n else float("nan")}


def top_confusions(m, names, limit=15):
    """Off-diagonal cells sorted by count: (reference, map, n, % of the
    reference class)."""
    out = []
    for i, row in enumerate(m["matrix"]):
        tot = sum(row)
        for j, v in enumerate(row):
            if i != j and v:
                out.append((names[i], names[j], v, v / tot if tot else 0))
    return sorted(out, key=lambda t: -t[2])[:limit]


# ---------------------------------------------------------------------------
# HTML pieces
# ---------------------------------------------------------------------------
def _e(v):
    return html.escape(str(v))


def _pct(v, digits=1):
    return "–" if v is None or v != v else f"{v * 100:.{digits}f} %"


def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _swatch(colour):
    return f"<span class='sw' style='background:{_e(colour)}'></span>"


def _card(label, value, sub=""):
    return (f"<div class='card'><div class='cl'>{_e(label)}</div><div class='cv'>{value}</div>"
            f"<div class='cs'>{sub}</div></div>")


def _kappa_scale(kappa):
    segs = "".join(f"<div class='ks' style='background:{c}'><span>{_e(n)}</span></div>"
                   for _, n, c in KAPPA_SCALE)
    pos = max(0.0, min(1.0, kappa if kappa == kappa else 0.0)) * 100
    return (f"<div class='kscale'>{segs}<div class='kmark' style='left:{pos:.1f}%'>"
            f"<b>κ = {kappa:.3f}</b></div></div>"
            "<div class='kaxis'><span>0</span><span>0.2</span><span>0.4</span><span>0.6</span>"
            "<span>0.8</span><span>1</span></div>")


def _confusion_table(m, classes_by_code):
    """Colour-coded matrix: rows = reference, columns = map; colour = share
    of the reference row (diagonal blue, confusion red); totals, PA, UA."""
    codes = m["codes"]
    names = [classes_by_code.get(c_, {}).get("name", str(c_)) for c_ in codes]
    head = "".join(f"<th class='vh' title='{_e(n)}'><div>{_swatch(classes_by_code.get(c_, {}).get('color', '#ccc'))}"
                   f"{_e(n)}</div></th>" for c_, n in zip(codes, names))
    rows = []
    for i, (c_, n) in enumerate(zip(codes, names)):
        row = m["matrix"][i]
        tot = sum(row)
        cells = []
        for j, v in enumerate(row):
            share = v / tot if tot else 0
            if v == 0:
                style = ""
            elif i == j:
                style = f"background:rgba(31,119,180,{0.12 + 0.88 * share:.2f});"
            else:
                style = f"background:rgba(214,39,40,{0.10 + 0.90 * share:.2f});"
            if share > 0.55:
                style += "color:#fff;"
            title = f"{n} → {names[j]}: {v} ({share:.0%} of {n})"
            cells.append(f"<td style='{style}' title='{_e(title)}'>{v if v else ''}</td>")
        pa = m["producers"][i]
        rows.append(f"<tr><th class='rh'>{_swatch(classes_by_code.get(c_, {}).get('color', '#ccc'))}{_e(n)}</th>"
                    f"{''.join(cells)}<td class='tot'>{tot}</td><td class='acc'>{_pct(pa, 0)}</td></tr>")
    col_tot = "".join(f"<td class='tot'>{v}</td>" for v in m["n_mapped"])
    ua = "".join(f"<td class='acc'>{_pct(v, 0)}</td>" for v in m["users"])
    return (f"<div class='scroll'><table class='cm'><thead><tr><th class='corner'>reference ↓ / map →</th>{head}"
            f"<th class='vh'><div>Total</div></th><th class='vh'><div>Producer's acc.</div></th></tr></thead>"
            f"<tbody>{''.join(rows)}<tr><th class='rh'>Total</th>{col_tot}<td class='tot'><b>{m['n']}</b></td><td></td></tr>"
            f"<tr><th class='rh'>User's accuracy</th>{ua}<td></td><td class='acc'><b>{_pct(m['overall_accuracy'], 1)}</b></td>"
            f"</tr></tbody></table></div>")


def _bar(v, colour):
    if v is None or v != v:
        return "–"
    return (f"<div class='bar'><div style='width:{v * 100:.0f}%;background:{colour}'></div>"
            f"<span>{v * 100:.1f}</span></div>")


def _class_table(result, m):
    classes = sorted(result["classes"], key=lambda q: int(q["code"]))
    idx = {c_: i for i, c_ in enumerate(m["codes"])} if m else {}
    train = result.get("train_counts", {})
    rows = []
    for q in classes:
        code = int(q["code"])
        n_train = int(train.get(code, 0))
        status = ("ok", "Adequate") if n_train >= ADEQUATE else (("warn", "Marginal") if n_train >= MARGINAL
                                                                   else ("bad", "Insufficient"))
        cells = [f"<td data-v='{code}'>{code}</td>",
                 f"<td data-v='{_e(q['name'])}'>{_swatch(q['color'])}{_e(q['name'])}</td>",
                 f"<td data-v='{n_train}'>{n_train} <span class='chip {status[0]}'>{status[1]}</span></td>"]
        if m:
            i = idx.get(code)
            vals = [None] * 5 if i is None else [m["producers"][i], m["users"][i], m["f1"][i]]
            if i is not None:
                vals += [None if vals[0] is None else 1 - vals[0], None if vals[1] is None else 1 - vals[1]]
            nref = 0 if i is None else m["n_reference"][i]
            nmap = 0 if i is None else m["n_mapped"][i]
            cells += [f"<td data-v='{nref}'>{nref}</td>", f"<td data-v='{nmap}'>{nmap}</td>"]
            for v, col in zip(vals, ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd")):
                cells.append(f"<td data-v='{-1 if v is None else v:.4f}'>{_bar(v, col)}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    head = ["Code", "Class", "Training samples"]
    if m:
        head += ["Reference (test)", "Mapped (test)", "Producer's acc.", "User's acc.", "F1",
                 "Omission error", "Commission error"]
    ths = "".join(f"<th onclick='sortTable(this)'>{h}</th>" for h in head)
    return f"<div class='scroll'><table class='ct sortable'><thead><tr>{ths}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"


def _settings(result):
    model = result["model"]
    rows = []
    rows.append(("Engine", "Offline (scikit-learn, local raster stack)" if result.get("local")
                 else "Google Earth Engine"))
    if result.get("map_path"):
        rows.append(("Map file", os.path.basename(result["map_path"])))   # no local paths in shared reports
    if model.get("stack"):
        rows.append(("Training stack", f"{model['stack'].get('file', '')} ({model['stack'].get('kind', '')})"))
    rows.append(("Trained on", str(model.get("trained_year") or "a local stack")))
    if result.get("applied"):
        rows.append(("Applied to", str(result["year"])))
    f = model.get("features")
    if f:
        year = model.get("trained_year") or 2020
        mask = (f"Cloud Score+ cs_cdf ≥ {f.get('cs_threshold')}" if f.get("s2_mask") == "csplus"
                else "SCL (classes 4–7 kept)")
        rows += [("Features", k.describe_features(f)),
                 ("Periods", "; ".join(k.describe_periods(f, year))),
                 ("Aggregation", ", ".join(k.REDUCERS.get(r, r) for r in f["reducers"])
                  + (" · gaps filled from neighbouring periods" if f.get("gap_fill") == "window" else "")),
                 ("Sentinel-2", f"{', '.join(f['s2_features']) or '–'} · mask {mask} · scenes ≤ "
                                f"{f.get('s2_max_cloud')} % cloud"),
                 ("Sentinel-1", f"{', '.join(f['s1_features'])} · {k.S1_ORBITS.get(f.get('s1_orbit'), '')}"
                  if f.get("s1_features") else "–"),
                 ("Scale", f"{f.get('scale')} m")]
    else:
        rows.append(("Features", f"{len(model['bands'])} stack bands: {', '.join(model['bands'][:12])}"
                                 + (" …" if len(model["bands"]) > 12 else "")))
    cfg = model["classifier"]
    rows.append(("Classifier", f"{k.CLASSIFIERS.get(cfg['type'], cfg['type'])} · "
                 + ", ".join(f"{kk.replace('_', ' ')} = {v}" for kk, v in cfg.items() if kk != "type")))
    rows.append(("Samples", k.SAMPLING.get(model.get("sampling", "mean"), model.get("sampling", ""))))
    if model.get("pooled_from"):
        rows.append(("Pooled samples", "; ".join(model["pooled_from"])))
    split = model.get("split") or {}
    mode = split.get("mode", "none")
    rows.append(("Train / test split", {"attribute": f"by attribute (test value '{split.get('test_value')}')",
                                        "random": f"random by polygon, {split.get('test_fraction', 0):.0%} test",
                                        "none": "no test set"}.get(mode, mode)))
    rows.append(("Model created", model.get("created", "")))
    return "<table class='kv'>" + "".join(f"<tr><th>{_e(a)}</th><td>{_e(b)}</td></tr>" for a, b in rows) + "</table>"


def _cv_section(cv, classes_by_code):
    cards = "".join([
        _card("Spatial CV overall accuracy", f"{_pct(cv['oa_mean'])}", f"± {_pct(cv['oa_sd'])} (sd over folds)"),
        _card("Spatial CV kappa", f"{cv['kappa_mean']:.3f}", f"± {cv['kappa_sd']:.3f} · {kappa_label(cv['kappa_mean'])}"),
        _card("Folds / blocks", f"{cv['k']} / {cv['n_blocks']}", f"{cv['block_km']:g} km blocks"),
        _card("Pooled CV accuracy", _pct(cv["pooled"]["overall_accuracy"]), f"n = {cv['pooled']['n']}")])
    folds = "".join(f"<tr><td>{m['fold']}</td><td>{m['n']}</td><td>{_pct(m['overall_accuracy'])}</td>"
                    f"<td>{m['kappa']:.3f}</td></tr>" for m in cv["folds"])
    f1 = "".join(
        f"<tr><td>{_swatch(classes_by_code.get(c_, {}).get('color', '#ccc'))}"
        f"{_e(classes_by_code.get(c_, {}).get('name', c_))}</td><td>{_bar(mn, '#2ca02c')}</td>"
        f"<td>{'–' if sd is None else f'± {sd * 100:.1f}'}</td></tr>"
        for c_, mn, sd in zip(cv["codes"], cv["f1_mean"], cv["f1_sd"]))
    return (f"<section><h2>Spatial cross-validation</h2><p class='note'>Samples (train + test) are grouped into "
            f"{cv['block_km']:g} km blocks by their polygon centroid; the blocks are split randomly into {cv['k']} "
            "folds, so neighbouring fields and the pixels of one field are never in training and test at the "
            "same time. Each fold is predicted by a model trained on the other folds.</p>"
            f"<div class='cards'>{cards}</div><h3>Folds</h3><table class='conf'><tr><th>Fold</th><th>n</th>"
            f"<th>Overall accuracy</th><th>Kappa</th></tr>{folds}</table>"
            f"<h3>F1 per class (mean ± sd over folds)</h3><table class='conf'>{f1}</table></section>")


def _figures(result, m, names, colours):
    out = []
    model = result["model"]
    try:
        from . import analysis_tools as at
        prof = at.class_profiles(k.model_samples(model), model["bands"], model.get("features"),
                                 model.get("trained_year"))
        for (var, red), df in list(prof.items())[:6]:
            fig = Figure(figsize=(9, 3.6))
            p.class_profiles_axis(fig.add_subplot(111), df, model["classes"], None, True,
                                  f"Class profile: {var} ({red}), mean ± 1σ of the training samples")
            fig.tight_layout()
            out.append((f"Class profile {var} ({red})", _png(fig)))
    except Exception:
        pass
    if m:
        n = len(m["codes"])
        fig = Figure(figsize=(8, max(2.5, 0.85 + 0.32 * n)))
        p.class_accuracy_axis(fig.add_subplot(111), m, names, colours)
        fig.tight_layout()
        out.append(("Accuracy per class", _png(fig)))
        fig = Figure(figsize=(max(6, 2.5 + 0.42 * n), max(5, 2 + 0.42 * n)))
        p.confusion_axis(fig.add_subplot(111), m, names, "Confusion (row-normalised)", annotate="share")
        fig.tight_layout()
        out.append(("Confusion matrix (share of each reference class)", _png(fig)))
    imp = result.get("importance")
    if imp:
        top = list(imp.items())[:30]
        fig = Figure(figsize=(8, max(2.5, 0.6 + 0.24 * len(top))))
        p.importance_axis(fig.add_subplot(111), imp, 30)
        fig.tight_layout()
        out.append((f"Variable importance (top {len(top)} of {len(imp)})", _png(fig)))
    return "".join(f"<figure><img src='data:image/png;base64,{b64}' alt='{_e(t)}'><figcaption>{_e(t)}</figcaption>"
                   f"</figure>" for t, b64 in out)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
CSS = """
:root{--ink:#1f2937;--muted:#6b7280;--line:#e5e7eb;--bg:#f6f7f9;--card:#fff;--accent:#1f6f8b}
*{box-sizing:border-box}body{margin:0;font:14px/1.5 'Segoe UI',system-ui,-apple-system,sans-serif;color:var(--ink);background:var(--bg)}
.wrap{max-width:1280px;margin:0 auto;padding:24px 20px 40px}
header{background:linear-gradient(120deg,#1f6f8b,#2a9d8f);color:#fff;border-radius:10px;padding:22px 26px;margin-bottom:22px}
header h1{margin:0 0 4px;font-size:1.7em}header p{margin:0;opacity:.9}
section{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:18px 22px;margin-bottom:20px}
h2{margin:0 0 12px;font-size:1.25em;color:var(--accent)}h3{margin:18px 0 8px;font-size:1.05em}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:12px}
.card{border:1px solid var(--line);border-radius:8px;padding:10px 14px;background:#fbfcfd}
.cl{font-size:.8em;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}.cv{font-size:1.6em;font-weight:600}
.cs{font-size:.82em;color:var(--muted)}
.kscale{position:relative;display:flex;height:34px;border-radius:6px;overflow:visible;margin:28px 0 2px}
.ks{flex:1;display:flex;align-items:center;justify-content:center;color:#fff;font-size:.85em;font-weight:600}
.ks:first-child{border-radius:6px 0 0 6px}.ks:nth-child(5){border-radius:0 6px 6px 0}
.kmark{position:absolute;top:-24px;bottom:-6px;border-left:3px solid #111;transform:translateX(-1px)}
.kmark b{position:absolute;top:-2px;left:6px;white-space:nowrap;font-size:.9em}
.kaxis{display:flex;justify-content:space-between;font-size:.8em;color:var(--muted)}
.scroll{overflow-x:auto}
table{border-collapse:collapse}
.cm td,.cm th{border:1px solid var(--line);padding:3px 6px;text-align:center;min-width:34px;font-variant-numeric:tabular-nums}
.cm th.rh{text-align:left;white-space:nowrap;position:sticky;left:0;background:#fff;font-weight:500}
.cm th.vh{vertical-align:bottom;height:140px;padding:4px 2px}
.cm th.vh div{writing-mode:vertical-rl;transform:rotate(180deg);white-space:nowrap;max-height:136px;overflow:hidden;
  text-overflow:ellipsis;font-weight:500;margin:0 auto}
.cm th.corner{font-size:.75em;color:var(--muted);font-weight:400;position:sticky;left:0;background:#fff}
.cm td.tot{background:#f3f4f6;font-weight:500}.cm td.acc{background:#eef6fb;font-weight:500}
.sw{display:inline-block;width:11px;height:11px;border:1px solid #0003;border-radius:2px;margin-right:6px;vertical-align:-1px}
.ct{width:100%}.ct th,.ct td{border-bottom:1px solid var(--line);padding:5px 8px;text-align:left;white-space:nowrap}
.ct th{cursor:pointer;background:#f3f4f6;position:sticky;top:0;user-select:none}.ct th:hover{background:#e5e7eb}
.ct th.asc:after{content:' ▲'}.ct th.desc:after{content:' ▼'}
.bar{position:relative;width:110px;height:16px;background:#f1f3f5;border-radius:3px}
.bar div{height:100%;border-radius:3px;opacity:.75}.bar span{position:absolute;left:6px;top:-1px;font-size:.82em}
.chip{display:inline-block;padding:0 7px;border-radius:9px;font-size:.75em;margin-left:6px}
.chip.ok{background:#dcfce7;color:#166534}.chip.warn{background:#fef9c3;color:#854d0e}.chip.bad{background:#fee2e2;color:#991b1b}
.conf td,.conf th{border-bottom:1px solid var(--line);padding:4px 10px;text-align:left}
.kv th{text-align:left;padding:4px 14px 4px 0;color:var(--muted);font-weight:500;vertical-align:top;white-space:nowrap}
.kv td{padding:4px 0}
figure{margin:14px 0}figure img{max-width:100%;border:1px solid var(--line);border-radius:6px}
figcaption{font-size:.85em;color:var(--muted)}
.note{font-size:.88em;color:var(--muted)}
footer{text-align:center;color:var(--muted);font-size:.85em;margin-top:26px}
@media print{body{background:#fff}section{break-inside:avoid;border:none}}
"""

JS = """
function sortTable(th){const t=th.closest('table'),i=[...th.parentNode.children].indexOf(th),b=t.tBodies[0];
const asc=!th.classList.contains('asc');[...th.parentNode.children].forEach(h=>h.classList.remove('asc','desc'));
th.classList.add(asc?'asc':'desc');const v=r=>{const c=r.children[i],d=c.dataset.v;const n=parseFloat(d);
return isNaN(n)?(d||c.textContent).toLowerCase():n};
[...b.rows].sort((a,c)=>{const x=v(a),y=v(c);return (x>y?1:x<y?-1:0)*(asc?1:-1)}).forEach(r=>b.appendChild(r));}
"""


def build_report(result, title=None):
    classes_by_code = {int(q["code"]): q for q in result["classes"]}
    m = result.get("metrics")
    pm = result.get("pixel_metrics") if result.get("pixel_metrics") and result["pixel_metrics"]["n"] else None
    applied = result.get("applied", False)
    ref = "reference" if applied else "test"
    title = title or (f"Classification {result['year']} — accuracy report" if not applied
                      else f"Model applied to {result['year']} — accuracy report")
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    parts = [f"<header><h1>{_e(title)}</h1><p>{len(result['classes'])} classes · "
             f"{result['n_train']} training samples · {result['n_test']} {ref} samples · "
             f"{'offline (scikit-learn)' if result.get('local') else 'Google Earth Engine'} · generated {now}</p></header>"]

    cards = []
    if m:
        s = summary_metrics(m)
        cards += [_card("Overall accuracy", _pct(s["oa"]), f"95 % CI ± {_pct(s['oa_ci'])} · n = {m['n']}"),
                  _card("Cohen's kappa", f"{s['kappa']:.3f}", kappa_label(s["kappa"])),
                  _card("Balanced accuracy", _pct(s["balanced"]), "mean producer's accuracy"),
                  _card("Macro F1", f"{s['macro_f1']:.3f}" if s["macro_f1"] == s["macro_f1"] else "–", "mean over classes"),
                  _card("Quantity disagreement", _pct(s["quantity"]), "wrong class proportions"),
                  _card("Allocation disagreement", _pct(s["allocation"]), "wrong locations")]
    if pm:
        sp = summary_metrics(pm)
        cards.append(_card(f"Map pixels in {ref} polygons", _pct(sp["oa"]), f"κ {sp['kappa']:.3f} · n = {pm['n']}"))
    if result.get("oob_error") is not None:
        cards.append(_card("Out-of-bag error", _pct(result["oob_error"]), "Random Forest, training data"))
    cards.append(_card("Classes", str(len(result["classes"])), f"{result['n_train']} training samples"))
    parts.append(f"<section><h2>Summary</h2><div class='cards'>{''.join(cards)}</div>")
    if m:
        parts.append(f"<h3>Kappa interpretation (Landis &amp; Koch 1977)</h3>{_kappa_scale(m['kappa'])}")
        parts.append("<p class='note'>Quantity and allocation disagreement (Pontius &amp; Millones 2011) split the "
                     "total error 1 − OA into wrong class proportions and wrong spatial allocation.</p>")
    else:
        parts.append(f"<p class='note'>No {ref} data — no independent accuracy assessment.</p>")
    parts.append("</section>")

    parts.append("<section><h2>Per-class accuracy</h2><p class='note'>Click a column header to sort. Sample size: "
                 f"≥ {ADEQUATE} training samples per class adequate, {MARGINAL}–{ADEQUATE - 1} marginal "
                 "(Congalton &amp; Green 2019).</p>" + _class_table(result, m) + "</section>")

    if m:
        names = [classes_by_code.get(c_, {}).get("name", str(c_)) for c_ in m["codes"]]
        conf = top_confusions(m, names)
        parts.append(f"<section><h2>Confusion matrix — {ref} samples</h2><p class='note'>Rows = reference, "
                     "columns = map. Colour = share of the reference class (blue = correct, red = confused); "
                     "hover a cell for details.</p>" + _confusion_table(m, classes_by_code))
        if conf:
            rows = "".join(f"<tr><td>{_e(a)}</td><td>→</td><td>{_e(b)}</td><td>{n}</td><td>{_pct(s_, 0)}</td></tr>"
                           for a, b, n, s_ in conf)
            parts.append("<h3>Most frequent confusions</h3><table class='conf'><tr><th>Reference</th><th></th>"
                         f"<th>mapped as</th><th>n</th><th>% of reference class</th></tr>{rows}</table>")
        parts.append("</section>")
    if result.get("cv"):
        parts.append(_cv_section(result["cv"], classes_by_code))
    if pm:
        parts.append(f"<section><h2>Confusion matrix — map pixels in {ref} polygons</h2><p class='note'>Pixels are "
                     "spatially autocorrelated; report the number of polygons as the sample size.</p>"
                     + _confusion_table(pm, classes_by_code) + "</section>")

    if True:
        names = [classes_by_code.get(c_, {}).get("name", str(c_)) for c_ in (m["codes"] if m else [])]
        colours = [classes_by_code.get(c_, {}).get("color", "#999") for c_ in (m["codes"] if m else [])]
        parts.append("<section><h2>Charts</h2>" + _figures(result, m, names, colours) + "</section>")

    parts.append("<section><h2>Settings</h2>" + _settings(result) + "</section>")
    parts.append("""<section><h2>References</h2><ul class='note'>
<li>Congalton, R.G. &amp; Green, K. (2019). <i>Assessing the Accuracy of Remotely Sensed Data: Principles and Practices</i> (3rd ed.). CRC Press.</li>
<li>Landis, J.R. &amp; Koch, G.G. (1977). The measurement of observer agreement for categorical data. <i>Biometrics</i> 33(1), 159–174.</li>
<li>Pontius, R.G. &amp; Millones, M. (2011). Death to Kappa. <i>International Journal of Remote Sensing</i> 32(15), 4407–4429.</li>
<li>Olofsson, P. et al. (2014). Good practices for estimating area and assessing accuracy of land change. <i>Remote Sensing of Environment</i> 148, 42–57.</li>
<li>Foody, G.M. (2002). Status of land cover classification accuracy assessment. <i>Remote Sensing of Environment</i> 80(1), 185–201.</li>
</ul></section>""")
    parts.append("<footer>Generated with the GEE Time Series Analyzer &amp; Classifier QGIS plugin</footer>")
    return (f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'><title>{_e(title)}</title>"
            f"<style>{CSS}</style><script>{JS}</script></head><body><div class='wrap'>{''.join(parts)}</div></body></html>")


def save_report(result, path, title=None):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(build_report(result, title))
    return path
