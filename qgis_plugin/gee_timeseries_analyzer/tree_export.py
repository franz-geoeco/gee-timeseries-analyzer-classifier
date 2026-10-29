"""
Random Forest / CART trees of a classification model as an interactive,
self-contained HTML viewer (no internet needed).

The trees are rebuilt from the model's stored samples with the same
settings and seed, so they are the trees of the classification:
  * Earth Engine: Classifier.explain() -> 'trees' (RF) / 'tree' (CART), the
    rpart-like text format ("2) NDVI_05_median<=0.27 225 95.3 0 (p0 p1 ...)")
  * offline: scikit-learn estimators_ / tree_
Gradient boosting exposes no classification trees and is not supported.

Viewer: forest overview (variable usage, root splits, tree list) and a tree
view with pan / zoom, click-to-expand, depth control, variable highlight and
leaf colours = class colours. Pure Python (no Qt / QGIS imports).
"""
import datetime as dt
import html
import json
import re

from . import classification as k

MAX_TOTAL_NODES = 1_500_000      # keeps the HTML below ~100 MB

_LINE = re.compile(r"^\s*(\d+)\)\s+(?:root|(.+?)(<=|>)(\S+))\s+(\S+)\s+(\S+)\s+(\S+)\s+\(([^)]*)\)\s*(\*)?\s*$")


class TreesUnavailable(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Common format: tree = {"nodes": [[parent, feature index or -1, threshold,
# n, class index, [probabilities]], ...]} (node 0 = root, children follow
# their parent's left / right lists), plus "left" / "right" index lists.
# ---------------------------------------------------------------------------
def parse_ee_tree(text, features):
    """Parses one tree of Earth Engine's explain() text format."""
    fidx = {f: i for i, f in enumerate(features)}
    raw = {}
    for line in text.splitlines():
        mt = _LINE.match(line)
        if not mt:
            continue
        nid = int(mt.group(1))
        raw[nid] = {"feat": mt.group(2), "op": mt.group(3), "thr": mt.group(4), "n": float(mt.group(5)),
                    "y": int(float(mt.group(7))), "p": [float(v) for v in mt.group(8).split()],
                    "leaf": bool(mt.group(9))}
    if 1 not in raw:
        raise ValueError("Unexpected tree format.")
    order, index = [], {}

    def walk(nid, parent):
        i = len(order)
        index[nid] = i
        order.append((nid, parent))
        if not raw[nid]["leaf"] and 2 * nid in raw:
            walk(2 * nid, i)
            walk(2 * nid + 1, i)
    walk(1, -1)
    nodes, left, right = [], [-1] * len(order), [-1] * len(order)
    for nid, parent in order:
        r = raw[nid]
        feat, thr = -1, 0.0
        if not r["leaf"] and 2 * nid in raw:
            lc_, rc_ = raw[2 * nid], raw[2 * nid + 1]
            le = lc_ if lc_["op"] == "<=" else rc_
            feat = fidx.get(le["feat"], -1)
            thr = float(le["thr"])
            left[index[nid]] = index[2 * nid] if lc_["op"] == "<=" else index[2 * nid + 1]
            right[index[nid]] = index[2 * nid + 1] if lc_["op"] == "<=" else index[2 * nid]
        nodes.append([parent, feat, thr, r["n"], r["y"], [round(v, 4) for v in r["p"]]])
    return {"nodes": nodes, "left": left, "right": right}


def parse_sklearn_tree(est):
    t = est.tree_
    nodes = []
    parent = [-1] * t.node_count
    for i in range(t.node_count):
        for ch in (t.children_left[i], t.children_right[i]):
            if ch >= 0:
                parent[ch] = i
    for i in range(t.node_count):
        v = t.value[i][0]
        s = float(v.sum()) or 1.0
        probs = [round(float(x) / s, 4) for x in v]
        leaf = t.children_left[i] < 0
        nodes.append([parent[i], -1 if leaf else int(t.feature[i]), 0.0 if leaf else float(t.threshold[i]),
                      float(t.n_node_samples[i]), int(max(range(len(probs)), key=probs.__getitem__)), probs])
    return {"nodes": nodes, "left": [int(x) for x in t.children_left], "right": [int(x) for x in t.children_right]}


def _tree_stats(tree):
    nodes, left = tree["nodes"], tree["left"]
    depth = [0] * len(nodes)
    for i in range(1, len(nodes)):
        depth[i] = depth[nodes[i][0]] + 1
    leaves = sum(1 for i in range(len(nodes)) if left[i] < 0)
    return {"n_nodes": len(nodes), "n_leaves": leaves, "depth": max(depth) if depth else 0}


# ---------------------------------------------------------------------------
# Building the forest of a result
# ---------------------------------------------------------------------------
def forest_for_result(result, progress_cb=None, cancel_cb=None):
    """-> forest dict (features, classes, trees). Retrains from the model's
    samples: Earth Engine for online models, scikit-learn offline."""
    model = result["model"]
    cfg = model["classifier"]
    kind = cfg.get("type", "rf")
    if kind not in ("rf", "cart"):
        raise TreesUnavailable("Tree export is available for Random Forest and CART. Gradient boosting builds "
                               "regression trees per class that can't be shown as classification trees.")
    bands = model["bands"]
    classes = sorted(model["classes"], key=lambda q: int(q["code"]))
    df = k.model_samples(model)
    k._step(progress_cb, cancel_cb, 0, 3, "Rebuilding the classifier")
    if result.get("local"):
        from . import local_classifier as lc
        est, _, _ = lc.train(df, bands, cfg)
        code_order = [int(x) for x in est.classes_]
        ests = est.estimators_ if kind == "rf" else [est]
        k._step(progress_cb, cancel_cb, 1, 3, "Reading the trees")
        trees = [parse_sklearn_tree(e) for e in ests]
        engine = "scikit-learn (offline)"
    else:
        c_ = k.c.ensure_initialised()
        clf = k.train(df, bands, cfg)
        k._step(progress_cb, cancel_cb, 1, 3, "Fetching the trees from Earth Engine")
        info = c_.Dictionary(clf.explain()).select(["trees", "tree", "classes"], True).getInfo()
        texts = info.get("trees") or ([info["tree"]] if info.get("tree") else [])
        if not texts:
            raise TreesUnavailable("Earth Engine returned no trees for this classifier.")
        code_order = [int(x) for x in info.get("classes", [q["code"] for q in classes])]
        trees = [parse_ee_tree(t, bands) for t in texts]
        engine = "Google Earth Engine (smile)"
    k._step(progress_cb, cancel_cb, 2, 3, "Writing the viewer")
    by_code = {int(q["code"]): q for q in classes}
    cls = [{"code": c2, "name": by_code.get(c2, {}).get("name", str(c2)),
            "color": by_code.get(c2, {}).get("color", "#999999")} for c2 in code_order]
    for i, t in enumerate(trees):
        t.update(_tree_stats(t))
        t["id"] = i + 1
    total = sum(t["n_nodes"] for t in trees)
    note = ""
    if total > MAX_TOTAL_NODES:
        keep, acc = [], 0
        for t in trees:
            if acc + t["n_nodes"] > MAX_TOTAL_NODES:
                break
            keep.append(t)
            acc += t["n_nodes"]
        note = f"Only the first {len(keep)} of {len(trees)} trees are included ({total:,} nodes in total)."
        trees = keep
    return {"engine": engine, "kind": k.CLASSIFIERS.get(kind, kind), "features": bands, "classes": cls,
            "trees": trees, "note": note, "settings": cfg,
            "trained": str(model.get("trained_year") or (model.get("stack") or {}).get("file", "")),
            "samples": len(df), "created": dt.datetime.now().strftime("%Y-%m-%d %H:%M")}


def save_trees_html(result, path, progress_cb=None, cancel_cb=None):
    forest = forest_for_result(result, progress_cb, cancel_cb)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(build_html(forest))
    return path, len(forest["trees"])


# ---------------------------------------------------------------------------
# Viewer
# ---------------------------------------------------------------------------
_CSS = """
:root{--ink:#1f2937;--muted:#6b7280;--line:#e5e7eb;--bg:#f6f7f9;--accent:#1f6f8b}
*{box-sizing:border-box}html,body{height:100%;margin:0}
body{font:13px/1.45 'Segoe UI',system-ui,sans-serif;color:var(--ink);background:var(--bg);display:flex;flex-direction:column}
header{background:linear-gradient(120deg,#1f6f8b,#2a9d8f);color:#fff;padding:10px 18px;display:flex;align-items:center;gap:18px}
header h1{font-size:1.25em;margin:0}header .sub{opacity:.9;font-size:.92em}
nav{display:flex;gap:4px;margin-left:auto}nav button{background:#ffffff26;color:#fff;border:0;padding:6px 14px;border-radius:5px;cursor:pointer}
nav button.on{background:#fff;color:var(--accent);font-weight:600}
main{flex:1;display:flex;min-height:0}
#side{width:300px;border-right:1px solid var(--line);background:#fff;display:flex;flex-direction:column}
#side .tools{padding:8px;border-bottom:1px solid var(--line);display:flex;gap:6px;flex-wrap:wrap}
#side input,#side select{flex:1;min-width:0;padding:4px 6px;border:1px solid var(--line);border-radius:4px}
#tlist{overflow:auto;flex:1}#tlist div{padding:6px 10px;border-bottom:1px solid #f1f1f1;cursor:pointer}
#tlist div:hover{background:#f3f7fa}#tlist div.on{background:#dbeafe}
#tlist b{display:inline-block;width:56px}#tlist small{color:var(--muted)}
#view{flex:1;display:flex;flex-direction:column;min-width:0}
.bar{display:flex;gap:8px;align-items:center;padding:6px 10px;background:#fff;border-bottom:1px solid var(--line);flex-wrap:wrap}
.bar button{padding:4px 10px;border:1px solid var(--line);background:#fff;border-radius:4px;cursor:pointer}
.bar button:hover{background:#f3f4f6}.bar label{color:var(--muted)}
#svgwrap{flex:1;position:relative;overflow:hidden;background:#fbfcfd;cursor:grab}
#svgwrap.drag{cursor:grabbing}svg{width:100%;height:100%;display:block}
.nd rect{stroke:#9ca3af;stroke-width:1;rx:5}.nd.int rect{fill:#fff}.nd.int{cursor:pointer}
.nd.int:hover rect{stroke:#1f6f8b;stroke-width:2}.nd text{font-size:11px;pointer-events:none}
.nd .f{font-weight:600}.nd .m{fill:#6b7280;font-size:10px}.nd.hl rect{stroke:#e11d48;stroke-width:3}
.nd.col rect{stroke-dasharray:4 2}
.edge{fill:none;stroke:#b6bcc6;stroke-width:1.3}.el{font-size:10px;fill:#6b7280}
#tip{position:fixed;pointer-events:none;background:#111827ee;color:#fff;padding:7px 9px;border-radius:5px;font-size:12px;display:none;max-width:320px;z-index:9}
#tip .r{display:flex;align-items:center;gap:6px}#tip .s{width:10px;height:10px;border-radius:2px;display:inline-block}
#legend{padding:8px 10px;border-top:1px solid var(--line);max-height:30%;overflow:auto}
#legend div{display:flex;align-items:center;gap:6px;margin:2px 0}#legend span{width:12px;height:12px;border-radius:2px;display:inline-block}
#overview{flex:1;overflow:auto;padding:18px 24px;display:none}
section{background:#fff;border:1px solid var(--line);border-radius:10px;padding:14px 18px;margin-bottom:16px}
h2{color:var(--accent);font-size:1.15em;margin:0 0 10px}
table{border-collapse:collapse}td,th{padding:4px 10px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
th{background:#f3f4f6;cursor:pointer}.hb{height:12px;background:#2a9d8f;border-radius:2px;display:inline-block;vertical-align:middle}
.note{color:var(--muted)}.cards{display:flex;gap:12px;flex-wrap:wrap}
.card{border:1px solid var(--line);border-radius:8px;padding:8px 14px;min-width:140px}.card b{display:block;font-size:1.4em}
"""

_JS = r"""
const F = JSON.parse(document.getElementById('forest').textContent);
const C = F.classes, FE = F.features;
let cur = 0, open = new Set(), hlFeat = -1, maxDepth = 3, scale = 1, tx = 40, ty = 30;
const NW = 168, NH = 50, GX = 16, GY = 74;
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmtN = v => Number.isInteger(v) ? String(v) : v.toFixed(1);
const fmt = v => Math.abs(v) >= 100 ? v.toFixed(1) : Math.abs(v) >= 1 ? v.toFixed(3) : v.toPrecision(3);

function depths(t){const d=new Array(t.nodes.length).fill(0);for(let i=1;i<t.nodes.length;i++)d[i]=d[t.nodes[i][0]]+1;return d;}
function openTo(t, maxD){open=new Set();const d=depths(t);t.nodes.forEach((n,i)=>{if(t.left[i]>=0&&d[i]<maxD)open.add(i)});}

function layout(t){
  const pos={};let slot=0;
  (function rec(i,d){
    const isOpen=t.left[i]>=0&&open.has(i);
    if(isOpen){rec(t.left[i],d+1);rec(t.right[i],d+1);pos[i]={x:(pos[t.left[i]].x+pos[t.right[i]].x)/2,y:d*(NH+GY)};}
    else{pos[i]={x:slot*(NW+GX),y:d*(NH+GY)};slot++;}
  })(0,0);
  return pos;
}

function drawTree(){
  const t=F.trees[cur],pos=layout(t);let edges='',nodes='';
  for(const k in pos){const i=+k,n=t.nodes[i],p=pos[i];
    if(t.left[i]>=0&&open.has(i)){
      [[t.left[i],'≤'],[t.right[i],'>']].forEach(([c,lab])=>{const q=pos[c];
        const x1=p.x+NW/2,y1=p.y+NH,x2=q.x+NW/2,y2=q.y,my=(y1+y2)/2;
        edges+=`<path class="edge" d="M${x1},${y1} C${x1},${my} ${x2},${my} ${x2},${y2}"/>`;
        edges+=`<text class="el" x="${(x1+x2)/2+(lab=='≤'?-14:6)}" y="${my}">${lab}</text>`;});}
    const leaf=t.left[i]<0,col=C[n[4]]?C[n[4]].color:'#999',pm=Math.max(...n[5]);
    let cls='nd '+(leaf?'leaf':'int')+(!leaf&&!open.has(i)?' col':'')+(!leaf&&n[1]==hlFeat?' hl':'');
    let body;
    if(leaf){body=`<rect width="${NW}" height="${NH}" fill="${col}" fill-opacity="0.85"/>`+
      `<text class="f" x="8" y="18" fill="${lum(col)}">${esc(cut(C[n[4]].name,22))}</text>`+
      `<text x="8" y="34" fill="${lum(col)}">p = ${(pm*100).toFixed(0)} % · n = ${fmtN(n[3])}</text>`;}
    else{body=`<rect width="${NW}" height="${NH}"/>`+
      `<text class="f" x="8" y="17">${esc(cut(FE[n[1]]||'?',20))} ≤ ${fmt(n[2])}</text>`+
      `<text class="m" x="8" y="31">n = ${fmtN(n[3])} · ${esc(cut(C[n[4]].name,14))} ${(pm*100).toFixed(0)} %</text>`+
      distBar(n[5])+(open.has(i)?'':`<text class="m" x="${NW-14}" y="31">＋</text>`);}
    nodes+=`<g class="${cls}" data-i="${i}" transform="translate(${p.x},${p.y})">${body}</g>`;}
  $('g').innerHTML=edges+nodes;applyT();
  $('info').textContent=`Tree ${t.id} · ${t.n_nodes} nodes · ${t.n_leaves} leaves · depth ${t.depth} · showing ${Object.keys(pos).length} nodes`;
  document.querySelectorAll('#tlist div').forEach(d=>d.classList.toggle('on',+d.dataset.i===cur));
}
function distBar(p){let x=8,s='';const w=NW-16;p.forEach((v,j)=>{if(v<=0)return;const ww=v*w;
  s+=`<rect x="${x}" y="38" width="${ww}" height="7" fill="${C[j]?C[j].color:'#999'}" style="stroke:none"/>`;x+=ww;});return s;}
function cut(s,n){s=String(s);return s.length>n?s.slice(0,n-1)+'…':s;}
function lum(h){const c=h.replace('#','');const r=parseInt(c.substr(0,2),16),g=parseInt(c.substr(2,2),16),b=parseInt(c.substr(4,2),16);
  return (0.299*r+0.587*g+0.114*b)>150?'#111':'#fff';}
function applyT(){$('g').setAttribute('transform',`translate(${tx},${ty}) scale(${scale})`);}
function fit(){const b=$('g').getBBox(),W=$('svgwrap').clientWidth,H=$('svgwrap').clientHeight;if(!b.width)return;
  scale=Math.min(1.5,Math.min((W-40)/b.width,(H-40)/b.height));
  if(scale<0.55){scale=0.55;const r=layout(F.trees[cur])[0];tx=W/2-(r.x+NW/2)*scale;}   // readable: centre on the root
  else tx=(W-b.width*scale)/2-b.x*scale;
  ty=20-b.y*scale;applyT();}

function selectTree(i){cur=i;openTo(F.trees[cur],maxDepth);drawTree();fit();}
function tip(e,i){const t=F.trees[cur],n=t.nodes[i];const rows=n[5].map((v,j)=>[v,j]).filter(a=>a[0]>0.001).sort((a,b)=>b[0]-a[0]).slice(0,8)
  .map(([v,j])=>`<div class="r"><span class="s" style="background:${C[j].color}"></span>${esc(C[j].name)}: ${(v*100).toFixed(1)} %</div>`).join('');
  const head=t.left[i]<0?'<b>Leaf</b>':`<b>${esc(FE[n[1]])} ≤ ${fmt(n[2])}</b><br><span style="opacity:.75">left = yes, right = no · click to ${open.has(i)?'collapse':'expand'}</span>`;
  $('tip').innerHTML=`${head}<br>n = ${fmtN(n[3])} samples<br>${rows}`;$('tip').style.display='block';
  $('tip').style.left=(e.clientX+14)+'px';$('tip').style.top=(e.clientY+14)+'px';}

function buildList(){const q=$('tq').value.toLowerCase(),sort=$('ts').value;
  let idx=F.trees.map((t,i)=>i);
  if(q)idx=idx.filter(i=>(FE[F.trees[i].nodes[0][1]]||'').toLowerCase().includes(q)||String(F.trees[i].id)===q);
  const key={id:t=>t.id,depth:t=>-t.depth,nodes:t=>-t.n_nodes,leaves:t=>t.n_leaves}[sort];idx.sort((a,b)=>key(F.trees[a])-key(F.trees[b]));
  $('tlist').innerHTML=idx.map(i=>{const t=F.trees[i];return `<div data-i="${i}"><b>Tree ${t.id}</b>`+
    `<small>depth ${t.depth} · ${t.n_leaves} leaves</small><br><small>root: ${esc(FE[t.nodes[0][1]]||'leaf')}</small></div>`}).join('');
  document.querySelectorAll('#tlist div').forEach(d=>d.onclick=()=>selectTree(+d.dataset.i));
  document.querySelectorAll('#tlist div').forEach(d=>d.classList.toggle('on',+d.dataset.i===cur));}

function overview(){
  const use=FE.map(()=>({n:0,root:0,d:0}));let total=0;
  F.trees.forEach(t=>{const d=depths(t);t.nodes.forEach((n,i)=>{if(t.left[i]>=0&&n[1]>=0){use[n[1]].n++;use[n[1]].d+=d[i];total++;if(i===0)use[n[1]].root++;}})});
  const rows=FE.map((f,i)=>({f,i,...use[i]})).filter(r=>r.n).sort((a,b)=>b.n-a.n);const mx=rows.length?rows[0].n:1;
  const nt=F.trees.length,dep=F.trees.map(t=>t.depth),lv=F.trees.map(t=>t.n_leaves),avg=a=>(a.reduce((x,y)=>x+y,0)/a.length).toFixed(1);
  $('overview').innerHTML=`<section><h2>Forest</h2><div class="cards">
   <div class="card">Trees<b>${nt}</b></div><div class="card">Mean depth<b>${avg(dep)}</b></div>
   <div class="card">Mean leaves<b>${avg(lv)}</b></div><div class="card">Splits<b>${total.toLocaleString()}</b></div>
   <div class="card">Training samples<b>${F.samples}</b></div></div>
   <p class="note">${esc(F.kind)} · ${esc(F.engine)} · trained on ${esc(F.trained)} · ${esc(Object.entries(F.settings).map(([a,b])=>a+'='+b).join(', '))}</p>
   ${F.note?`<p><b>${esc(F.note)}</b></p>`:''}
   <p class="note">How to read: every internal node tests one variable (left branch = value ≤ threshold). The coloured bar shows the class mix of the samples reaching the node; leaves are coloured by their class. The forest votes over all trees, so single trees are only one view — the variable usage below summarises them all.</p></section>
   <section><h2>Variable usage across all trees</h2><p class="note">Click a variable to highlight its splits in the tree view.</p>
   <table><tr><th>Variable</th><th>Splits</th><th></th><th>% of splits</th><th>Root splits</th><th>Mean depth</th></tr>
   ${rows.map(r=>`<tr data-f="${r.i}" style="cursor:pointer"><td>${esc(r.f)}</td><td>${r.n}</td><td><span class="hb" style="width:${(160*r.n/mx).toFixed(0)}px"></span></td>
   <td>${(100*r.n/total).toFixed(1)} %</td><td>${r.root}</td><td>${(r.d/r.n).toFixed(1)}</td></tr>`).join('')}</table></section>`;
  document.querySelectorAll('#overview tr[data-f]').forEach(tr=>tr.onclick=()=>{hlFeat=+tr.dataset.f;$('hl').value=hlFeat;show('tree');drawTree();});}

function show(which){$('overview').style.display=which==='ov'?'block':'none';$('view').style.display=which==='ov'?'none':'flex';
  $('bov').classList.toggle('on',which==='ov');$('btr').classList.toggle('on',which!=='ov');if(which!=='ov')fit();}

window.onload=()=>{
  $('legend').innerHTML='<b>Classes</b>'+C.map(c=>`<div><span style="background:${c.color}"></span>${esc(c.code)} ${esc(c.name)}</div>`).join('');
  $('hl').innerHTML='<option value="-1">— highlight variable —</option>'+FE.map((f,i)=>`<option value="${i}">${esc(f)}</option>`).join('');
  $('hl').onchange=()=>{hlFeat=+$('hl').value;drawTree();};
  $('dep').oninput=()=>{maxDepth=+$('dep').value;$('depv').textContent=maxDepth;openTo(F.trees[cur],maxDepth);drawTree();fit();};
  $('tq').oninput=buildList;$('ts').onchange=buildList;
  $('prev').onclick=()=>selectTree((cur-1+F.trees.length)%F.trees.length);
  $('next').onclick=()=>selectTree((cur+1)%F.trees.length);
  $('fit').onclick=fit;$('all').onclick=()=>{openTo(F.trees[cur],999);drawTree();fit();};
  $('none').onclick=()=>{openTo(F.trees[cur],0);drawTree();fit();};
  $('bov').onclick=()=>show('ov');$('btr').onclick=()=>show('tree');
  const w=$('svgwrap');let drag=null,moved=false;
  w.onwheel=e=>{e.preventDefault();const r=w.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top,f=Math.exp(-e.deltaY*0.0015);
    const ns=Math.max(0.05,Math.min(4,scale*f));tx=mx-(mx-tx)*ns/scale;ty=my-(my-ty)*ns/scale;scale=ns;applyT();};
  w.onmousedown=e=>{drag={x:e.clientX,y:e.clientY,tx,ty};moved=false;w.classList.add('drag');};
  window.onmousemove=e=>{if(drag){const dx=e.clientX-drag.x,dy=e.clientY-drag.y;if(Math.abs(dx)+Math.abs(dy)>3)moved=true;
    tx=drag.tx+dx;ty=drag.ty+dy;applyT();}
    const g=e.target.closest&&e.target.closest('.nd');if(g&&!drag)tip(e,+g.dataset.i);else $('tip').style.display='none';};
  window.onmouseup=e=>{if(drag&&!moved){const g=e.target.closest&&e.target.closest('.nd.int');
    if(g){const i=+g.dataset.i;open.has(i)?open.delete(i):open.add(i);drawTree();}}drag=null;w.classList.remove('drag');};
  buildList();overview();selectTree(0);show('ov');
};
"""


def build_html(forest):
    title = f"{forest['kind']} — {len(forest['trees'])} tree(s)"
    data = json.dumps(forest, separators=(",", ":")).replace("</", "<\\/")
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>
<style>{_CSS}</style></head><body>
<header><h1>{html.escape(title)}</h1><span class="sub">{html.escape(forest['engine'])} · {len(forest['features'])} variables ·
{len(forest['classes'])} classes · generated {html.escape(forest['created'])}</span>
<nav><button id="bov">Forest overview</button><button id="btr">Tree view</button></nav></header>
<main><div id="side"><div class="tools"><input id="tq" placeholder="Filter: root variable or tree no.">
<select id="ts"><option value="id">by number</option><option value="depth">deepest first</option>
<option value="nodes">largest first</option><option value="leaves">fewest leaves</option></select></div>
<div id="tlist"></div><div id="legend"></div></div>
<div id="overview"></div>
<div id="view"><div class="bar"><button id="prev">◀</button><button id="next">▶</button>
<label>open to depth</label><input id="dep" type="range" min="0" max="12" value="3"><span id="depv">3</span>
<button id="none">Collapse</button><button id="all">Expand all</button><button id="fit">Fit</button>
<select id="hl"></select><span id="info" class="note"></span></div>
<div id="svgwrap"><svg><g id="g"></g></svg></div></div></main>
<div id="tip"></div>
<script id="forest" type="application/json">{data}</script>
<script>{_JS}</script></body></html>"""
