"""The Spectra section: surface-reflectance spectra of points and polygons
from the Sentinel-2 / Landsat 8-9 image closest to a chosen date, anywhere
on Earth (spectral.py does the Earth Engine work), and from local Sentinel-2
L2A scenes (local_s2.py, offline).

Samples come from a click on the map, a drawn polygon or the selected
features of a layer; each is sampled in a background task (further clicks
are queued). Charts: all samples per sensor, or all sensors of one sample.
"""
import csv
import json
import os

from qgis.core import (
    QgsApplication, QgsProject, QgsGeometry, QgsCoordinateTransform, QgsWkbTypes, QgsPointXY,
    QgsMapLayerProxyModel,
)
from qgis.gui import QgsMapLayerComboBox, QgsFieldComboBox, QgsMapToolEmitPoint, QgsRubberBand, QgsVertexMarker
from qgis.PyQt.QtCore import Qt, QDate
from qgis.PyQt.QtGui import QColor, QIcon
from qgis.PyQt.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QPushButton, QLabel, QSpinBox,
    QCheckBox, QComboBox, QDateEdit, QTreeWidget, QTreeWidgetItem, QAbstractItemView, QFileDialog,
    QProgressBar, QHeaderView, QListWidget, QListWidgetItem,
)
from qgis.core import QgsRasterLayer, NULL

from . import gee_client as c
from . import help_texts as H
from . import local_s2 as ls
from . import plotting as p
from . import polygon_layer as pl
from . import spectral as sp
from . import style
from .tasks import CallTask

MAX_SAMPLES = 10


def _sample_all(geojson, sensors, date, window_days, max_cloud, cloud_mask, min_clear, scenes,
                progress_cb=None, cancel_cb=None):
    """Earth Engine sensors (closest image) + local scenes -> {key: result or None}."""
    out = {}
    ee_keys = [s for s in sensors if s in sp.SENSORS]
    if ee_keys:
        out.update(sp.sample(geojson, ee_keys, date, window_days, max_cloud, cloud_mask, min_clear,
                             progress_cb, cancel_cb))
    for key in [s for s in sensors if s in scenes]:
        if cancel_cb is not None and cancel_cb():
            raise InterruptedError("Cancelled")
        out[key] = ls.sample_scene(scenes[key], geojson, date, cloud_mask, min_clear)
    return out
IMAGE_GROUP = "GEE spectra images"
try:
    from qgis.core import Qgis
    POINT_POLY_FILTER = Qgis.LayerFilter.PointLayer | Qgis.LayerFilter.PolygonLayer
except AttributeError:
    POINT_POLY_FILTER = QgsMapLayerProxyModel.PointLayer | QgsMapLayerProxyModel.PolygonLayer


class SpectraTab(QWidget):
    def __init__(self, dock):
        super().__init__()
        self.dock = dock
        self.iface = dock.iface
        self.canvas = dock.canvas
        self.samples = []          # {id, name, color, geojson, kind, results, marker}
        self.scenes = {}           # key -> local Sentinel-2 L2A scene (local_s2.open_scene)
        self._next_scene = 1
        self._next_id = 1
        self._queue = []
        self._task = None
        self._prev_tool = None
        self.point_tool = QgsMapToolEmitPoint(self.canvas)
        self.point_tool.canvasClicked.connect(self._on_point)
        self.poly_tool = pl.PolygonDrawTool(self.canvas)
        self.poly_tool.polygonDrawn.connect(self._on_polygon)
        self.canvas.mapToolSet.connect(self._on_map_tool_set)
        self._build_ui()

    # ------------------------------------------------------------------
    def _build_ui(self):
        from .dock_widget import PlotPanel
        lay = QVBoxLayout(self)

        box = QGroupBox("Date and sensors")
        form = QFormLayout(box)
        drow = QHBoxLayout()
        self.date = QDateEdit(QDate.currentDate())
        self.date.setCalendarPopup(True)
        self.date.setDisplayFormat("yyyy-MM-dd")
        self.date.setMinimumDate(QDate(2013, 4, 1))
        self.window = QSpinBox()
        self.window.setRange(1, 365)
        self.window.setValue(30)
        self.window.setPrefix("± ")
        self.window.setSuffix(" days")
        drow.addWidget(self.date)
        drow.addWidget(QLabel("search"))
        drow.addWidget(self.window)
        drow.addStretch(1)
        form.addRow("Target date:", drow)
        srow = QHBoxLayout()
        self.chk_s2 = QCheckBox("Sentinel-2 (10 m)")
        self.chk_ls = QCheckBox("Landsat 8/9 (30 m)")
        for cb in (self.chk_s2, self.chk_ls):
            cb.setChecked(True)
            srow.addWidget(cb)
        srow.addStretch(1)
        form.addRow("Earth Engine:", srow)
        self.scene_list = QListWidget()
        self.scene_list.setMaximumHeight(70)
        self.scene_list.setToolTip("Local Sentinel-2 L2A scenes: tick to include them in the sampling")
        self.scene_list.itemChanged.connect(self._mark_stale)
        lrow2 = QHBoxLayout()
        b_add = QPushButton("Add scenes…")
        b_add.setToolTip("Sentinel-2 L2A .SAFE.zip, or MTD_MSIL2A.xml of an unzipped .SAFE folder")
        b_add.clicked.connect(self.on_add_scenes)
        b_rm = QPushButton("Remove")
        b_rm.clicked.connect(self.on_remove_scenes)
        lrow2.addWidget(b_add)
        lrow2.addWidget(b_rm)
        lrow2.addStretch(1)
        lrow2.addWidget(style.InfoButton(H.SPECTRA_LOCAL))
        lbox = QVBoxLayout()
        lbox.setSpacing(3)
        lbox.addWidget(self.scene_list)
        lbox.addLayout(lrow2)
        lw = QWidget()
        lw.setLayout(lbox)
        lbox.setContentsMargins(0, 0, 0, 0)
        form.addRow("Local scenes:", lw)
        crow = QHBoxLayout()
        self.chk_mask = QCheckBox("Cloud mask")
        self.chk_mask.setChecked(True)
        self.min_clear = QSpinBox()
        self.min_clear.setRange(1, 100)
        self.min_clear.setValue(50)
        self.min_clear.setSuffix(" % clear (polygons)")
        self.max_cloud = QSpinBox()
        self.max_cloud.setRange(1, 100)
        self.max_cloud.setValue(60)
        self.max_cloud.setPrefix("scenes ≤ ")
        self.max_cloud.setSuffix(" % cloud")
        crow.addWidget(self.chk_mask)
        crow.addWidget(self.min_clear)
        crow.addWidget(self.max_cloud)
        crow.addWidget(style.InfoButton(H.SPECTRA_CLOUD))
        form.addRow(crow)
        for w in (self.date, self.window, self.min_clear, self.max_cloud):
            (w.dateChanged if w is self.date else w.valueChanged).connect(self._mark_stale)
        for cb in (self.chk_s2, self.chk_ls, self.chk_mask):
            cb.toggled.connect(self._mark_stale)
        lay.addWidget(box)

        sbox = QGroupBox("Samples")
        sl = QVBoxLayout(sbox)
        trow = QHBoxLayout()
        self.btn_point = QPushButton("Click a point")
        self.btn_poly = QPushButton("Draw a polygon")
        for b in (self.btn_point, self.btn_poly):
            b.setCheckable(True)
            trow.addWidget(b)
        self.btn_point.toggled.connect(lambda on: self._toggle_tool(self.point_tool, on, self.btn_poly))
        self.btn_poly.toggled.connect(lambda on: self._toggle_tool(self.poly_tool, on, self.btn_point))
        sl.addLayout(trow)
        lrow = QHBoxLayout()
        self.layer = QgsMapLayerComboBox()
        self.layer.setFilters(POINT_POLY_FILTER)
        self.layer.setAllowEmptyLayer(True)
        self.layer.setLayer(None)
        btn_sel = QPushButton("Add selected features")
        btn_sel.clicked.connect(self.on_add_selected)
        lrow.addWidget(self.layer, 1)
        lrow.addWidget(btn_sel)
        sl.addLayout(lrow)
        nrow = QHBoxLayout()
        nrow.addWidget(QLabel("Name from column:"))
        self.name_field = QgsFieldComboBox()
        self.name_field.setAllowEmptyFieldName(True)
        self.name_field.setToolTip("Column used to name the added features (empty: layer name + feature id)")
        self.layer.layerChanged.connect(self.name_field.setLayer)
        nrow.addWidget(self.name_field, 1)
        sl.addLayout(nrow)
        self.tree = QTreeWidget()
        self.tree.setColumnCount(3)
        self.tree.setHeaderLabels(["Sample", "Sentinel-2", "Landsat 8/9"])
        self.tree.setRootIsDecorated(False)
        self.tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tree.setMinimumHeight(110)
        self.tree.setMaximumHeight(190)
        self.tree.itemClicked.connect(self._on_tree_click)
        self.tree.itemChanged.connect(self._on_tree_renamed)
        self.tree.setToolTip("Double click a name to rename the sample")
        sl.addWidget(self.tree)
        brow = QHBoxLayout()
        for label, slot in (("Remove selected", self.on_remove), ("Clear all", self.on_clear),
                            ("Re-sample all", self.on_resample)):
            b = QPushButton(label)
            b.clicked.connect(slot)
            brow.addWidget(b)
        sl.addLayout(brow)
        brow2 = QHBoxLayout()
        b_img = QPushButton("Show images on map")
        b_img.setToolTip("Adds the true-colour scene of the closest acquisition day (map view) for each sensor")
        b_img.clicked.connect(self.on_show_images)
        b_csv = QPushButton("Export CSV…")
        b_csv.clicked.connect(self.on_export_csv)
        brow2.addWidget(b_img)
        brow2.addWidget(b_csv)
        sl.addLayout(brow2)
        lay.addWidget(sbox)

        vrow = QHBoxLayout()
        vrow.addWidget(QLabel("Chart:"))
        self.view = QComboBox()
        self.view.currentIndexChanged.connect(lambda *_: self._redraw())
        vrow.addWidget(self.view, 1)
        lay.addLayout(vrow)
        self.plot = PlotPanel()
        self.plot.canvas.setMinimumHeight(460)
        p.plot_placeholder(self.plot.figure, "Click a point or draw a polygon on the map to sample a spectrum.")
        lay.addWidget(self.plot, 1)

        row = QHBoxLayout()
        self.status = QLabel(f"Up to {MAX_SAMPLES} samples. The closest clear image per sensor is used.")
        self.status.setWordWrap(True)
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setMaximumWidth(140)
        self.progress.hide()
        row.addWidget(self.status, 1)
        row.addWidget(self.progress)
        lay.addLayout(row)
        self._fill_view()

    # ------------------------------------------------------------------
    # Map tools
    # ------------------------------------------------------------------
    def _toggle_tool(self, tool, on, other):
        if on:
            other.setChecked(False)
            current = self.canvas.mapTool()
            if current not in (self.point_tool, self.poly_tool):
                self._prev_tool = current
            self.canvas.setMapTool(tool)
        elif self.canvas.mapTool() is tool:
            if self._prev_tool is not None:
                self.canvas.setMapTool(self._prev_tool)
            else:
                self.canvas.unsetMapTool(tool)

    def _on_map_tool_set(self, new_tool, *_):
        if new_tool is not self.point_tool and self.btn_point.isChecked():
            self.btn_point.setChecked(False)
        if new_tool is not self.poly_tool and self.btn_poly.isChecked():
            self.btn_poly.setChecked(False)

    def _to_wgs84(self, geom, crs=None):
        g = QgsGeometry(geom)
        src = crs or self.canvas.mapSettings().destinationCrs()
        if src != pl.WGS84:
            g.transform(QgsCoordinateTransform(src, pl.WGS84, QgsProject.instance()))
        return g

    def _on_point(self, point, button):
        if button != Qt.LeftButton:
            return
        g = self._to_wgs84(QgsGeometry.fromPointXY(QgsPointXY(point)))
        self._add_sample(g, None)

    def _on_polygon(self, geom):
        self._add_sample(self._to_wgs84(geom), None)

    def on_add_selected(self):
        layer = self.layer.currentLayer()
        if layer is None or not layer.selectedFeatureCount():
            self.dock._warn("Select points or polygons of the chosen layer first.")
            return
        for f in layer.getSelectedFeatures():
            if len(self.samples) >= MAX_SAMPLES:
                break
            g = f.geometry()
            if g is None or g.isEmpty():
                continue
            if g.type() == QgsWkbTypes.PointGeometry:
                g = QgsGeometry.fromPointXY(g.asMultiPoint()[0] if g.isMultipart() else g.asPoint())
            field = self.name_field.currentField()
            value = f[field] if field and layer.fields().indexOf(field) >= 0 else None
            name = str(value) if value is not None and value != NULL and str(value).strip() else \
                f"{layer.name()} #{f.id()}"
            self._add_sample(self._to_wgs84(g, layer.crs()), name)

    # ------------------------------------------------------------------
    # Samples
    # ------------------------------------------------------------------
    def _add_sample(self, geom_wgs84, name):
        if len(self.samples) >= MAX_SAMPLES:
            self.dock._warn(f"At most {MAX_SAMPLES} samples — remove some first.")
            return
        is_point = geom_wgs84.type() == QgsWkbTypes.PointGeometry
        if not is_point and not geom_wgs84.isGeosValid():
            geom_wgs84 = geom_wgs84.makeValid()
        sid = self._next_id
        self._next_id += 1
        colour = c.POLYGON_COLOURS[(sid - 1) % len(c.POLYGON_COLOURS)]
        if is_point:
            pt = geom_wgs84.asPoint()
            gj = {"type": "Point", "coordinates": [round(pt.x(), 7), round(pt.y(), 7)]}
            label = name or f"Point {sid} ({pt.y():.4f}, {pt.x():.4f})"
        else:
            gj = json.loads(geom_wgs84.asJson(7))
            label = name or f"Polygon {sid}"
        s = {"id": sid, "name": label, "color": colour, "geojson": gj, "kind": "point" if is_point else "polygon",
             "results": None, "geom": geom_wgs84}
        s["marker"] = self._marker(s)
        self.samples.append(s)
        self._refresh_tree()
        self._enqueue(sid)

    def _marker(self, s):
        xform = QgsCoordinateTransform(pl.WGS84, self.canvas.mapSettings().destinationCrs(), QgsProject.instance())
        g = QgsGeometry(s["geom"])
        g.transform(xform)
        col = QColor(s["color"])
        if s["kind"] == "point":
            m = QgsVertexMarker(self.canvas)
            m.setCenter(g.asPoint())
            m.setColor(col)
            m.setFillColor(col)
            m.setIconType(QgsVertexMarker.ICON_CIRCLE)
            m.setIconSize(11)
            m.setPenWidth(2)
            return m
        rb = QgsRubberBand(self.canvas, QgsWkbTypes.PolygonGeometry)
        rb.setToGeometry(g, None)
        rb.setStrokeColor(col)
        fill = QColor(col)
        fill.setAlpha(50)
        rb.setFillColor(fill)
        rb.setWidth(2)
        return rb

    def _remove_marker(self, s):
        m = s.get("marker")
        if m is not None:
            self.canvas.scene().removeItem(m)
            s["marker"] = None

    def _settings(self):
        sensors = [k for k, cb in (("s2", self.chk_s2), ("landsat", self.chk_ls)) if cb.isChecked()]
        for i in range(self.scene_list.count()):
            it = self.scene_list.item(i)
            if it.checkState() == Qt.Checked:
                sensors.append(it.data(Qt.UserRole))
        return {"sensors": sensors, "date": self.date.date().toString("yyyy-MM-dd"),
                "window_days": self.window.value(), "max_cloud": self.max_cloud.value(),
                "cloud_mask": self.chk_mask.isChecked(), "min_clear": self.min_clear.value() / 100.0}

    def _enqueue(self, sid):
        if sid not in self._queue:
            self._queue.append(sid)
        self._process_next()

    def _process_next(self):
        if self._task is not None or not self._queue:
            return
        settings = self._settings()
        if not settings["sensors"]:
            self.dock._warn("Tick at least one sensor or local scene.")
            self._queue = []
            return
        if any(k in sp.SENSORS for k in settings["sensors"]) and not self.dock._ensure_ee():
            self._queue = []
            return
        sid = self._queue.pop(0)
        s = self._sample(sid)
        if s is None:
            self._process_next()
            return
        s["settings"] = settings
        scenes = {k: v for k, v in self.scenes.items() if k in settings["sensors"]}
        self._task = CallTask(f"Spectrum of {s['name']}", _sample_all,
                              dict(settings, geojson=s["geojson"], scenes=scenes),
                              lambda task, ok, sid=sid: self._on_done(task, ok, sid))
        self._task.progressText.connect(self.status.setText)
        self.progress.setRange(0, 0)
        self.progress.show()
        self.status.setText(f"Sampling {s['name']} … ({len(self._queue)} queued)")
        QgsApplication.taskManager().addTask(self._task)

    def _on_done(self, task, ok, sid):
        self._task = None
        self.progress.hide()
        s = self._sample(sid)
        if task.exception is not None:
            self.dock._error(f"Sampling failed: {task.exception}")
        elif ok and s is not None:
            s["results"] = task.result
            missing = [self._label(k) for k, v in task.result.items() if v is None]
            self.status.setText(f"{s['name']} sampled." + (f" No clear data for: {', '.join(missing)} "
                                                           "(outside the scene / window or masked)." if missing else ""))
        self._refresh_tree()
        self._fill_view()
        self._redraw()
        self._process_next()

    def _sample(self, sid):
        return next((s for s in self.samples if s["id"] == sid), None)

    def _label(self, key):
        if key in sp.SENSORS:
            return sp.SENSORS[key]["label"]
        return ls.scene_label(self.scenes[key]) if key in self.scenes else key

    def _columns(self):
        keys = []
        for s in self.samples:
            for k in (s.get("results") or {}):
                if k not in keys:
                    keys.append(k)
        for k in self._settings()["sensors"]:
            if k not in keys:
                keys.append(k)
        return keys

    def _refresh_tree(self):
        keys = self._columns()
        self.tree.blockSignals(True)
        self.tree.clear()
        self.tree.setColumnCount(1 + len(keys))
        self.tree.setHeaderLabels(["Sample"] + [self._label(k) for k in keys])
        for s in self.samples:
            cells = [s["name"]]
            for k in keys:
                res = (s.get("results") or {}).get(k) if s.get("results") else None
                if s.get("results") is None:
                    cells.append("…" if s["id"] in self._queue or self._task else "")
                elif k not in s["results"]:
                    cells.append("–")
                elif res is None:
                    cells.append("no data")
                else:
                    cells.append(f"{res['date']} ({res['days']:+d} d)")
            it = QTreeWidgetItem(cells)
            it.setIcon(0, QIcon(self._swatch(s["color"])))
            it.setData(0, Qt.UserRole, s["id"])
            it.setFlags(it.flags() | Qt.ItemIsEditable)
            self.tree.addTopLevelItem(it)
        self.tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tree.blockSignals(False)

    def _on_tree_renamed(self, item, col):
        if col != 0:
            return
        s = self._sample(item.data(0, Qt.UserRole))
        name = item.text(0).strip()
        if s is None or not name or name == s["name"]:
            return
        s["name"] = name
        self._fill_view()
        self._redraw()

    @staticmethod
    def _swatch(colour):
        from .dock_widget import _swatch
        return _swatch(colour)

    def _on_tree_click(self, item, _col):
        i = self.view.findData(item.data(0, Qt.UserRole))
        if i >= 0:
            self.view.setCurrentIndex(i)
        s = self._sample(item.data(0, Qt.UserRole))
        if s is not None:
            xform = QgsCoordinateTransform(pl.WGS84, self.canvas.mapSettings().destinationCrs(), QgsProject.instance())
            g = QgsGeometry(s["geom"])
            g.transform(xform)
            self.canvas.flashGeometries([g])

    def on_remove(self):
        ids = {it.data(0, Qt.UserRole) for it in self.tree.selectedItems()}
        for s in [s for s in self.samples if s["id"] in ids]:
            self._remove_marker(s)
        self.samples = [s for s in self.samples if s["id"] not in ids]
        self._queue = [q for q in self._queue if q not in ids]
        self._refresh_tree()
        self._fill_view()
        self._redraw()

    def on_clear(self):
        for s in self.samples:
            self._remove_marker(s)
        self.samples = []
        self._queue = []
        self._next_id = 1
        self._refresh_tree()
        self._fill_view()
        self._redraw()

    def on_resample(self):
        for s in self.samples:
            s["results"] = None
            self._enqueue(s["id"])
        self._refresh_tree()

    def on_add_scenes(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Sentinel-2 L2A scenes", "", "Sentinel-2 L2A (*.zip MTD_MSIL2A.xml);;All files (*)")
        for path in paths:
            try:
                scene = self.dock._run_busy(ls.open_scene, path)
            except Exception as e:
                self.dock._error(f"{os.path.basename(path)}: {e}")
                continue
            if any(sc["path"] == scene["path"] for sc in self.scenes.values()):
                continue
            key = f"local{self._next_scene}"
            self._next_scene += 1
            self.scenes[key] = scene
            it = QListWidgetItem(f"{ls.scene_label(scene)} · {scene['cloud']:.0f} % cloud")
            it.setData(Qt.UserRole, key)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked)
            it.setToolTip(f"{scene['product']} · processing baseline {scene['baseline']}")
            self.scene_list.addItem(it)
        self._refresh_tree()

    def on_remove_scenes(self):
        for it in self.scene_list.selectedItems():
            self.scenes.pop(it.data(Qt.UserRole), None)
            self.scene_list.takeItem(self.scene_list.row(it))
        self._refresh_tree()

    def _mark_stale(self, *_):
        if self.samples:
            self.status.setText("Settings changed — press 'Re-sample all' to update the spectra.")

    # ------------------------------------------------------------------
    # Charts
    # ------------------------------------------------------------------
    def _fill_view(self):
        current = self.view.currentData()
        self.view.blockSignals(True)
        self.view.clear()
        self.view.addItem("All samples — one panel per sensor", None)
        for s in self.samples:
            self.view.addItem(QIcon(self._swatch(s["color"])), f"{s['name']} — all sensors", s["id"])
        i = self.view.findData(current)
        self.view.setCurrentIndex(i if i >= 0 else 0)
        self.view.blockSignals(False)

    def _redraw(self):
        done = [s for s in self.samples if s.get("results")]
        sensors = [k for k in self._columns() if any(k in s["results"] for s in done)] or \
            self._settings()["sensors"] or ["s2"]
        sid = self.view.currentData()
        single = self._sample(sid) if sid is not None else None
        if single is not None and not single.get("results"):
            p.plot_placeholder(self.plot.figure, "Still sampling …")
        else:
            p.plot_spectra(self.plot.figure, done, sensors, single, {k: self._label(k) for k in sensors})
        self.plot.canvas.setMinimumHeight(460 if single is not None or len(sensors) == 1
                                          else 310 * len(sensors))
        self.plot.redraw()

    # ------------------------------------------------------------------
    # Images and export
    # ------------------------------------------------------------------
    def on_show_images(self):
        settings = self._settings()
        if not settings["sensors"]:
            return
        xform = QgsCoordinateTransform(self.canvas.mapSettings().destinationCrs(), pl.WGS84, QgsProject.instance())
        r = xform.transformBoundingBox(self.canvas.extent())
        bbox = (r.xMinimum(), r.yMinimum(), r.xMaximum(), r.yMaximum())
        project = QgsProject.instance()
        root = project.layerTreeRoot()
        group = root.findGroup(IMAGE_GROUP) or root.insertGroup(0, IMAGE_GROUP)
        added = []
        for k in [k for k in settings["sensors"] if k in self.scenes]:
            tci = ls.tci_path(self.scenes[k])
            name = f"{ls.scene_label(self.scenes[k])} true colour"
            if tci and not project.mapLayersByName(name):
                layer = QgsRasterLayer(tci, name)
                if layer.isValid():
                    project.addMapLayer(layer, False)
                    group.insertLayer(0, layer)
                    added.append(name)
        for k in [k for k in settings["sensors"] if k in sp.SENSORS]:
            if not self.dock._ensure_ee():
                break
            try:
                res = self.dock._run_busy(sp.closest_image, k, bbox, settings["date"], settings["window_days"],
                                          settings["max_cloud"], settings["cloud_mask"])
            except Exception as e:
                self.dock._error(f"{sp.SENSORS[k]['label']}: {e}")
                continue
            if res is None:
                self.dock._warn(f"{sp.SENSORS[k]['label']}: no image within ± {settings['window_days']} days.")
                continue
            img, vis, day = res
            name = f"{sp.SENSORS[k]['label']} {day} (closest to {settings['date']})"
            for lyr in project.mapLayersByName(name):
                project.removeMapLayer(lyr.id())
            try:
                layer = self.dock._run_busy(self.dock._add_ee_layer, img, vis, name)
            except Exception as e:
                self.dock._error(f"{name}: {e}")
                continue
            self.dock._move_into_group(layer, group, 0)
            added.append(name)
        if added:
            self.status.setText("Added: " + "; ".join(added))

    def on_export_csv(self):
        done = [s for s in self.samples if s.get("results")]
        if not done:
            self.dock._warn("No spectra to export yet.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save spectra", "spectra.csv", "CSV (*.csv)")
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["sample", "geometry", "lon", "lat", "target_date", "sensor", "spacecraft", "image",
                        "image_date", "days_from_target", "scene_cloud_pct", "clear_share", "band", "band_name",
                        "wavelength_nm", "reflectance", "reflectance_sd"])
            for s in done:
                cen = s["geom"].centroid().asPoint()
                for k, res in s["results"].items():
                    if res is None:
                        continue
                    for b, nm, wl, v, sd in zip(res["bands"], res["names"], res["wavelengths"], res["values"],
                                                res["sd"]):
                        w.writerow([s["name"], s["kind"], round(cen.x(), 6), round(cen.y(), 6),
                                    s["settings"]["date"], self._label(k), res["spacecraft"], res["image"],
                                    res["date"], res["days"], res["cloud"], res["clear"], b, nm, wl, v, sd])
        self.iface.messageBar().pushSuccess("Spectra", f"Saved {os.path.basename(path)}")

    def cleanup(self):
        for tool in (self.point_tool, self.poly_tool):
            if self.canvas.mapTool() is tool:
                self.canvas.unsetMapTool(tool)
        try:
            self.canvas.mapToolSet.disconnect(self._on_map_tool_set)
        except TypeError:
            pass
        if self._task is not None:
            self._task.cancel()
        for s in self.samples:
            self._remove_marker(s)
        self.poly_tool.reset()
