"""
Indices section: spectral indices from a local Sentinel-2 / Landsat raster
(a GeoTIFF / VRT stack, the all-bands layer of a local scene, or a
Sentinel-2 L2A .SAFE.zip directly). The bands are found from the raster's
metadata (indices.py) and can be changed by hand; the result is one
GeoTIFF stack with named bands or one GeoTIFF per index.
"""
import os

from qgis.core import (
    QgsApplication, QgsProject, QgsRasterLayer, QgsCoordinateTransform, QgsMapLayerProxyModel,
    QgsColorRampShader, QgsRasterShader, QgsSingleBandPseudoColorRenderer, QgsStyle, QgsGradientColorRamp,
)
from qgis.gui import QgsMapLayerComboBox
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QColor, QBrush
from qgis.PyQt.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout, QGroupBox, QPushButton, QLabel, QComboBox,
    QTreeWidget, QTreeWidgetItem, QRadioButton, QButtonGroup, QLineEdit, QCheckBox, QFileDialog, QProgressBar,
    QHeaderView,
)

from . import help_texts as H
from . import indices as ix
from . import local_s2 as ls
from . import style
from .tasks import CallTask

GROUP = "Spectral indices"
RASTER_FILTER = QgsMapLayerProxyModel.RasterLayer
NO_BAND = "—"


class IndicesTab(QWidget):
    def __init__(self, dock):
        super().__init__()
        self.dock = dock
        self.iface = dock.iface
        self.canvas = dock.canvas
        self.source = None           # path of the raster read
        self.origin = None           # the file / layer the user chose (a .zip is read through a VRT)
        self.source_name = ""
        self.sensor = None
        self.bands = []              # indices.describe_bands
        self._task = None
        self._filling = False
        self._build_ui()
        self._on_layer(self.layer.currentLayer())

    # ------------------------------------------------------------------
    def _build_ui(self):
        lay = QVBoxLayout(self)

        box = QGroupBox("Input")
        form = QFormLayout(box)
        row = QHBoxLayout()
        self.layer = QgsMapLayerComboBox()
        self.layer.setFilters(RASTER_FILTER)
        self.layer.setAllowEmptyLayer(True)
        self.layer.layerChanged.connect(self._on_layer)
        b_file = QPushButton("File…")
        b_file.setToolTip("A GeoTIFF / VRT stack, a Sentinel-2 L2A .SAFE.zip or the MTD_MSIL2A.xml of a .SAFE folder")
        b_file.clicked.connect(self.on_browse)
        row.addWidget(self.layer, 1)
        row.addWidget(b_file)
        form.addRow("Raster:", row)
        srow = QHBoxLayout()
        self.sensor_combo = QComboBox()
        self.sensor_combo.addItem("Detect automatically", None)
        for key, meta in ix.SENSORS.items():
            self.sensor_combo.addItem(meta["label"], key)
        self.sensor_combo.currentIndexChanged.connect(lambda *_: self._detect())
        srow.addWidget(self.sensor_combo, 1)
        srow.addWidget(style.InfoButton(H.INDICES_INPUT))
        form.addRow("Sensor:", srow)
        self.input_info = style.WrapLabel("Choose a raster.")
        self.input_info.setStyleSheet("color:#666")
        form.addRow(self.input_info)
        lay.addWidget(box)

        bbox = QGroupBox("Bands")
        bl = QVBoxLayout(bbox)
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        self.role_combos = {}
        for n, (role, (label, target, _lo, _hi)) in enumerate(ix.ROLES.items()):
            cb = QComboBox()
            cb.setToolTip(f"{label}, about {target} nm")
            cb.currentIndexChanged.connect(lambda *_: self._on_roles_changed())
            self.role_combos[role] = cb
            r, c = divmod(n, 2)
            grid.addWidget(QLabel(f"{label}:"), r, 2 * c)
            grid.addWidget(cb, r, 2 * c + 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        bl.addLayout(grid)
        scrow = QHBoxLayout()
        scrow.addWidget(QLabel("Reflectance:"))
        self.scaling = QComboBox()
        for key, label in ix.SCALINGS.items():
            self.scaling.addItem(label, key)
        self.scaling.currentIndexChanged.connect(lambda *_: self._update_scaling())
        scrow.addWidget(self.scaling, 1)
        scrow.addWidget(style.InfoButton(H.INDICES_SCALING))
        bl.addLayout(scrow)
        self.scaling_info = style.WrapLabel("")
        self.scaling_info.setStyleSheet("color:#666")
        bl.addWidget(self.scaling_info)
        lay.addWidget(bbox)

        ibox = QGroupBox("Indices")
        il = QVBoxLayout(ibox)
        self.tree = QTreeWidget()
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels(["Index", "Formula"])
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.tree.setMinimumHeight(220)
        self.items = {}
        for cat in ix.CATEGORIES:
            top = QTreeWidgetItem([cat])
            top.setFlags(top.flags() | Qt.ItemIsAutoTristate | Qt.ItemIsUserCheckable)
            top.setCheckState(0, Qt.Unchecked)
            f = top.font(0)
            f.setBold(True)
            top.setFont(0, f)
            self.tree.addTopLevelItem(top)
            for key in [k for k, v in ix.LIBRARY.items() if v[0] == cat]:
                i = ix.info(key)
                it = QTreeWidgetItem([key, i["formula"]])
                it.setData(0, Qt.UserRole, key)
                it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                it.setCheckState(0, Qt.Checked if key in ix.COMMON else Qt.Unchecked)
                top.addChild(it)
                self.items[key] = it
            top.setExpanded(True)
        self.tree.itemChanged.connect(lambda *_: self._update_count())
        il.addWidget(self.tree, 1)
        brow = QHBoxLayout()
        for label, keys in (("Common", ix.COMMON), ("All available", None), ("None", ())):
            b = QPushButton(label)
            b.clicked.connect(lambda _=False, ks=keys: self._check(ks))
            brow.addWidget(b)
        brow.addStretch(1)
        self.count_label = QLabel("")
        brow.addWidget(self.count_label)
        il.addLayout(brow)
        lay.addWidget(ibox, 1)

        obox = QGroupBox("Output")
        of = QFormLayout(obox)
        mrow = QHBoxLayout()
        self.rb_stack = QRadioButton("Raster stack (one GeoTIFF, one named band per index)")
        self.rb_single = QRadioButton("Single-band GeoTIFFs (one per index)")
        self.rb_stack.setChecked(True)
        grp = QButtonGroup(self)
        grp.addButton(self.rb_stack)
        grp.addButton(self.rb_single)
        mcol = QVBoxLayout()
        mcol.setSpacing(2)
        mcol.addWidget(self.rb_stack)
        mcol.addWidget(self.rb_single)
        mrow.addLayout(mcol, 1)
        of.addRow("Write:", mrow)
        self.rb_stack.toggled.connect(lambda *_: self._suggest_output())
        self.extent = QComboBox()
        self.extent.addItem("Whole raster", "all")
        self.extent.addItem("Current map view", "view")
        of.addRow("Extent:", self.extent)
        prow = QHBoxLayout()
        self.out_path = QLineEdit()
        self.out_path.setPlaceholderText("output GeoTIFF / folder")
        self.out_path.textEdited.connect(lambda *_: setattr(self, "_auto_out", False))
        b_out = QPushButton("…")
        b_out.setFixedWidth(30)
        b_out.clicked.connect(self.on_choose_output)
        prow.addWidget(self.out_path, 1)
        prow.addWidget(b_out)
        of.addRow("Save to:", prow)
        self.chk_add = QCheckBox("Add to the map (colour ramp per index)")
        self.chk_add.setChecked(True)
        of.addRow(self.chk_add)
        lay.addWidget(obox)

        run = QHBoxLayout()
        self.btn_run = QPushButton("Calculate indices")
        style.primary(self.btn_run)
        self.btn_run.clicked.connect(self.on_run)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(lambda: self._task and self._task.cancel())
        run.addWidget(self.btn_run, 1)
        run.addWidget(self.btn_cancel)
        lay.addLayout(run)
        srow2 = QHBoxLayout()
        self.status = style.WrapLabel("Indices are computed locally (no Earth Engine).")
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setMaximumWidth(140)
        self.progress.hide()
        srow2.addWidget(self.status, 1)
        srow2.addWidget(self.progress)
        lay.addLayout(srow2)

    # ------------------------------------------------------------------
    # Input and detection
    # ------------------------------------------------------------------
    def _on_layer(self, layer):
        if layer is None:
            return
        src = layer.source().split("|")[0]
        self._set_source(src, layer.name())

    def on_browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Raster for the indices", "",
            "Rasters and Sentinel-2 products (*.tif *.tiff *.vrt *.img *.jp2 *.zip MTD_MSIL2A.xml);;All files (*)")
        if not path:
            return
        self.layer.blockSignals(True)
        self.layer.setLayer(None)
        self.layer.blockSignals(False)
        self._set_source(path, os.path.splitext(os.path.basename(path))[0])

    def _set_source(self, path, name):
        origin = path
        try:
            if path.lower().endswith(".zip") or os.path.basename(path).upper() == "MTD_MSIL2A.XML":
                scene = ls.open_scene(path)
                path = ls.build_vrt(scene, os.path.join(QgsApplication.qgisSettingsDirPath(), *ls.VRT_SUBDIR))
                name = scene["product"]
        except Exception as e:
            self.dock._error(f"{os.path.basename(path)}: {e}")
            return
        self.source, self.source_name, self.origin = path, name.replace(".SAFE", ""), origin
        self._detect()
        self._suggest_output(force=True)

    def _detect(self):
        if not self.source:
            return
        try:
            sensor, self.bands = ix.describe_bands(self.source, self.sensor_combo.currentData())
        except Exception as e:
            self.input_info.setText(f"Could not read the raster: {e}")
            self.bands = []
            return
        self.sensor = sensor
        roles = ix.assign_roles(self.bands)
        self._filling = True
        for role, cb in self.role_combos.items():
            cb.clear()
            cb.addItem(NO_BAND, None)
            for b in self.bands:
                wl = f" · {b['wavelength']:.0f} nm" if b["wavelength"] and "nm" not in b["label"] else ""
                cb.addItem(f"{b['index']}: {b['label']}{wl}", b["index"])
            cb.setCurrentIndex(max(0, cb.findData(roles.get(role))))
        self._filling = False
        hows = sorted({b["how"] for b in self.bands if b["how"]})
        n_found = sum(1 for v in roles.values() if v)
        sensor_txt = ix.SENSORS[sensor]["label"] if sensor in ix.SENSORS else "unknown sensor"
        warn = ""
        if "band order (assumed)" in hows:
            warn = " The bands were assigned by their usual order — please check them."
        elif n_found == 0:
            warn = (" No band could be identified: choose the sensor above (band order) or assign the bands "
                    "by hand.")
        name = os.path.basename(self.origin or self.source)
        self.input_info.setText(f"{name}: {len(self.bands)} band(s) · {sensor_txt} · {n_found} of {len(ix.ROLES)} "
                                f"band roles found ({', '.join(hows) or 'nothing found'}).{warn}")
        self._on_roles_changed()

    def roles(self):
        return {role: cb.currentData() for role, cb in self.role_combos.items()}

    def _on_roles_changed(self):
        if self._filling:
            return
        roles = self.roles()
        grey = QBrush(QColor("#9aa0a6"))
        self.tree.blockSignals(True)
        for key, it in self.items.items():
            ok = ix.available(key, roles)
            it.setDisabled(not ok)
            i = ix.info(key)
            tip = f"<b>{key}</b> — {i['name']}<br>{i['formula']}<br><i>{i['reference']}</i>"
            if not ok:
                tip += f"<br><span style='color:#b00'>Needs: {', '.join(ix.missing_roles(key, roles))}</span>"
                it.setForeground(0, grey)
                it.setForeground(1, grey)
            else:
                it.setForeground(0, QBrush(self.palette().text().color()))
                it.setForeground(1, QBrush(self.palette().text().color()))
            it.setToolTip(0, tip)
            it.setToolTip(1, tip)
        self.tree.blockSignals(False)
        self._update_scaling()
        self._update_count()

    def _update_scaling(self):
        if not self.source or self._filling:
            return
        try:
            _so, how = ix.resolve_scaling(self.source, self.roles(), self.sensor, self.scaling.currentData())
        except Exception as e:
            how = f"could not check: {e}"
        self.scaling_info.setText(f"Conversion to reflectance: {how}.")

    def selected(self):
        roles = self.roles()
        return [k for k, it in self.items.items()
                if it.checkState(0) == Qt.Checked and ix.available(k, roles)]

    def _check(self, keys):
        roles = self.roles()
        for k, it in self.items.items():
            on = (keys is None and ix.available(k, roles)) or (keys is not None and k in keys)
            it.setCheckState(0, Qt.Checked if on else Qt.Unchecked)
        self._update_count()

    def _update_count(self):
        n = len(self.selected())
        self.count_label.setText(f"{n} selected")
        self._suggest_output()

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    def _default_dir(self):
        src = self.origin or self.source
        if src and os.path.isfile(src) and not src.endswith("_all_bands.vrt"):
            return os.path.dirname(src)
        home = QgsProject.instance().homePath()
        return home or os.path.expanduser("~")

    def _suggest_output(self, force=False):
        if not self.source_name:
            return
        cur = self.out_path.text().strip()
        if cur and not force and not getattr(self, "_auto_out", False):
            return
        d = os.path.dirname(cur) if cur and not force else self._default_dir()
        if self.rb_stack.isChecked():
            keys = self.selected()
            tag = keys[0] if len(keys) == 1 else "indices"
            self.out_path.setText(os.path.join(d, f"{self.source_name}_{tag}.tif"))
        else:
            self.out_path.setText(os.path.join(d, f"{self.source_name}_indices"))
        self._auto_out = True

    def on_choose_output(self):
        if self.rb_stack.isChecked():
            path, _ = QFileDialog.getSaveFileName(self, "Index stack", self.out_path.text(), "GeoTIFF (*.tif)")
        else:
            path = QFileDialog.getExistingDirectory(self, "Folder for the index GeoTIFFs", self.out_path.text())
        if path:
            self.out_path.setText(path)
            self._auto_out = False

    def _window(self):
        if self.extent.currentData() != "view":
            return None
        layer = QgsRasterLayer(self.source, "tmp")
        if not layer.isValid():
            return None
        xf = QgsCoordinateTransform(self.canvas.mapSettings().destinationCrs(), layer.crs(), QgsProject.instance())
        r = xf.transformBoundingBox(self.canvas.extent())
        return ix.pixel_window(self.source, (r.xMinimum(), r.yMinimum(), r.xMaximum(), r.yMaximum()))

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def on_run(self):
        if self._task is not None:
            return
        if not self.source:
            self.dock._warn("Choose a raster first.")
            return
        keys = self.selected()
        if not keys:
            self.dock._warn("Tick at least one available index.")
            return
        out = self.out_path.text().strip()
        if not out:
            self.dock._warn("Choose where to save the result.")
            return
        roles = self.roles()
        try:
            scaling, how = ix.resolve_scaling(self.source, roles, self.sensor, self.scaling.currentData())
            window = self._window()
        except Exception as e:
            self.dock._warn(str(e))
            return
        stack = self.rb_stack.isChecked()
        if stack and not out.lower().endswith((".tif", ".tiff")):
            out += ".tif"
        target = out if stack else (out, f"{self.source_name}_")
        labels = {b["index"]: b["label"] for b in self.bands}
        self._task = CallTask(f"Spectral indices ({len(keys)})", ix.compute,
                              {"path": self.source, "keys": keys, "roles": roles, "scaling": scaling, "out": target,
                               "mode": "stack" if stack else "single", "window": window, "band_labels": labels},
                              lambda task, ok, keys=keys: self._on_done(task, ok, keys))
        self._task.progressText.connect(self.status.setText)
        self._task.progressChanged.connect(lambda v: self.progress.setValue(int(v)))
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.show()
        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.status.setText(f"Computing {', '.join(keys)} … (reflectance: {how})")
        QgsApplication.taskManager().addTask(self._task)

    def _on_done(self, task, ok, keys):
        self._task = None
        self.progress.hide()
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        if task.exception is not None:
            self.dock._error(f"Indices failed: {task.exception}")
            self.status.setText("Failed.")
            return
        if not ok:
            self.status.setText("Cancelled.")
            return
        paths = task.result
        if self.chk_add.isChecked():
            self._add_layers(paths, keys)
        where = paths[0] if len(paths) == 1 else os.path.dirname(paths[0])
        self.status.setText(f"{len(keys)} index/indices written: {where}")
        self.iface.messageBar().pushSuccess("Spectral indices", f"Saved {where}")

    def _add_layers(self, paths, keys):
        project = QgsProject.instance()
        root = project.layerTreeRoot()
        group = root.findGroup(GROUP) or root.insertGroup(0, GROUP)
        if len(paths) == 1 and len(keys) > 1:
            items = [(paths[0], os.path.splitext(os.path.basename(paths[0]))[0], keys[0])]
        else:
            items = [(p, os.path.splitext(os.path.basename(p))[0], k) for p, k in zip(paths, keys)]
        for path, name, key in reversed(items):
            for lyr in project.mapLayersByName(name):
                project.removeMapLayer(lyr.id())
            layer = QgsRasterLayer(path, name)
            if not layer.isValid():
                continue
            self._style(layer, 1, key)
            project.addMapLayer(layer, False)
            group.insertLayer(0, layer)

    @staticmethod
    def _style(layer, band, key):
        """Pseudo-colour with the index's typical range and colour ramp."""
        i = ix.info(key)
        vmin, vmax = i["range"]
        st = QgsStyle.defaultStyle()
        name = next((n for n in st.colorRampNames() if n.lower() == i["ramp"].lower()), None)
        ramp = (st.colorRamp(name) if name else None) or QgsGradientColorRamp(QColor("#f7fcb9"), QColor("#006837"))
        fcn = QgsColorRampShader(vmin, vmax)
        fcn.setColorRampType(QgsColorRampShader.Interpolated)
        n = 10
        fcn.setColorRampItemList([QgsColorRampShader.ColorRampItem(vmin + (vmax - vmin) * j / n, ramp.color(j / n),
                                                                   f"{vmin + (vmax - vmin) * j / n:.3g}")
                                  for j in range(n + 1)])
        shader = QgsRasterShader()
        shader.setRasterShaderFunction(fcn)
        ren = QgsSingleBandPseudoColorRenderer(layer.dataProvider(), band, shader)
        ren.setClassificationMin(vmin)
        ren.setClassificationMax(vmax)
        layer.setRenderer(ren)

    def cleanup(self):
        if self._task is not None:
            self._task.cancel()
