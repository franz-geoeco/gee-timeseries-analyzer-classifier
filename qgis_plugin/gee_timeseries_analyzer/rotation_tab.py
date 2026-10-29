"""The Crop rotation tab (Classification section): rotation / change between
yearly classified maps.

The user lists one classified GeoTIFF per year (class codes, 255 = NoData,
e.g. maps downloaded in the Classification section or offline classifications);
the tab computes the class areas per year, the transition matrix of every
pair of consecutive years and a flow diagram, writes a change map for a
chosen pair and exports the tables as CSV. analysis_tools.py does the work.
"""
import json
import os
import re

import pandas as pd
from qgis.core import QgsApplication, QgsProject, QgsRasterLayer, QgsPalettedRasterRenderer
from qgis.gui import QgsMapLayerComboBox
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QColor, QBrush
from qgis.PyQt.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QPushButton, QLabel, QSpinBox,
    QCheckBox, QComboBox, QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QFileDialog, QProgressBar, QTabWidget,
)
from matplotlib.figure import Figure

from . import analysis_tools as at
from . import gee_client as c
from . import plotting as p
from . import polygon_layer as pl
from . import style
from .tasks import CallTask


def _rotation_task(entries, mask_polygons, progress_cb=None, cancel_cb=None):
    return at.rotation(entries, mask_polygons, progress_cb, cancel_cb)


def _change_task(path_a, path_b, out_path, classes, mask_polygons, progress_cb=None, cancel_cb=None):
    return out_path, at.write_change_map(path_a, path_b, out_path, classes, mask_polygons, cancel_cb)


class RotationTab(QWidget):
    def __init__(self, dock):
        super().__init__()
        self.dock = dock
        self.iface = dock.iface
        self.rot = None
        self._task = None
        self._build_ui()

    # ------------------------------------------------------------------
    def _build_ui(self):
        from .dock_widget import PlotPanel, POLYGON_FILTER
        lay = QVBoxLayout(self)

        box = QGroupBox("Yearly classified maps")
        bl = QVBoxLayout(box)
        hint = QLabel("One classified GeoTIFF per year (class codes, 255 = NoData) — e.g. maps saved with "
                      "'Download GeoTIFF' or offline classifications. Class names and colours are read from the "
                      "files; all maps are compared on the grid of the earliest year.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#666; font-size:11px")
        bl.addWidget(hint)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Year", "Classified map"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setMinimumHeight(130)
        bl.addWidget(self.table)
        row = QHBoxLayout()
        for label, slot in (("Add row", lambda: self._add_row()), ("Add files…", self.on_add_files),
                            ("Remove selected", self.on_remove)):
            b = QPushButton(label)
            b.clicked.connect(slot)
            row.addWidget(b)
        bl.addLayout(row)
        lay.addWidget(box)

        area = QGroupBox("Area (optional)")
        af = QFormLayout(area)
        arow = QHBoxLayout()
        self.area_layer = QgsMapLayerComboBox()
        self.area_layer.setFilters(POLYGON_FILTER)
        self.area_layer.setAllowEmptyLayer(True)
        self.area_layer.setLayer(None)
        self.chk_area_sel = QCheckBox("Selected only")
        arow.addWidget(self.area_layer, 1)
        arow.addWidget(self.chk_area_sel)
        af.addRow("Restrict to:", arow)
        lay.addWidget(area)

        run = QHBoxLayout()
        self.btn_run = QPushButton("Compute rotation")
        style.primary(self.btn_run)
        self.btn_run.clicked.connect(self.on_compute)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(lambda: self._task and self._task.cancel())
        run.addWidget(self.btn_run, 1)
        run.addWidget(self.btn_cancel)
        lay.addLayout(run)

        self.res_tabs = QTabWidget()
        # areas
        aw = QWidget()
        al = QVBoxLayout(aw)
        self.area_table = QTableWidget(0, 0)
        self.area_table.verticalHeader().setVisible(False)
        self.area_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.area_table.setMaximumHeight(220)
        al.addWidget(self.area_table)
        self.area_plot = PlotPanel()
        self.area_plot.canvas.setMinimumHeight(320)
        p.plot_placeholder(self.area_plot.figure, "Add yearly maps and press 'Compute rotation'.")
        al.addWidget(self.area_plot, 1)
        self.res_tabs.addTab(aw, "Areas")
        # transitions
        tw = QWidget()
        tl = QVBoxLayout(tw)
        trow = QHBoxLayout()
        trow.addWidget(QLabel("From → to:"))
        self.pair = QComboBox()
        self.pair.currentIndexChanged.connect(lambda *_: self._show_pair())
        trow.addWidget(self.pair, 1)
        self.pair_unit = QComboBox()
        self.pair_unit.addItems(["ha", "% of the earlier year's class"])
        self.pair_unit.currentIndexChanged.connect(lambda *_: self._show_pair())
        trow.addWidget(self.pair_unit)
        tl.addLayout(trow)
        self.matrix = QTableWidget(0, 0)
        self.matrix.setEditTriggers(QAbstractItemView.NoEditTriggers)
        tl.addWidget(self.matrix, 1)
        self.top_label = QLabel()
        self.top_label.setWordWrap(True)
        self.top_label.setTextFormat(Qt.RichText)
        tl.addWidget(self.top_label)
        btn_change = QPushButton("Write change map for this pair…")
        btn_change.setToolTip("GeoTIFF with one value per observed transition ('cotton → rice'); unchanged "
                              "pixels grey, changes in the colour of the later class")
        btn_change.clicked.connect(self.on_change_map)
        tl.addWidget(btn_change)
        self.res_tabs.addTab(tw, "Transitions")
        # flows
        self.flow_plot = PlotPanel()
        self.flow_plot.canvas.setMinimumHeight(460)
        p.plot_placeholder(self.flow_plot.figure, "No rotation computed yet.")
        self.res_tabs.addTab(self.flow_plot, "Flow diagram")
        lay.addWidget(self.res_tabs, 1)

        brow = QHBoxLayout()
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.progress = QProgressBar()
        self.progress.setMaximumHeight(14)
        self.progress.setTextVisible(False)
        self.progress.hide()
        btn_csv = QPushButton("Save tables (CSV)…")
        btn_csv.clicked.connect(self.on_save_csv)
        brow.addWidget(self.status, 1)
        brow.addWidget(self.progress)
        brow.addWidget(btn_csv)
        lay.addLayout(brow)

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------
    def _add_row(self, year=None, layer=None):
        from .dock_widget import RASTER_FILTER
        r = self.table.rowCount()
        self.table.insertRow(r)
        spin = QSpinBox()
        spin.setRange(1980, 2100)
        years = [self.table.cellWidget(i, 0).value() for i in range(r)]
        spin.setValue(year or (max(years) + 1 if years else c.MAX_YEAR))
        combo = QgsMapLayerComboBox()
        combo.setFilters(RASTER_FILTER)
        combo.setAllowEmptyLayer(True)
        combo.setLayer(layer)
        self.table.setCellWidget(r, 0, spin)
        self.table.setCellWidget(r, 1, combo)

    def on_add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Classified maps (one per year)", "",
                                                "GeoTIFF (*.tif *.tiff);;All files (*)")
        for path in paths:
            layer = None
            for lyr in QgsProject.instance().mapLayers().values():
                if isinstance(lyr, QgsRasterLayer) and os.path.normcase(lyr.source()) == os.path.normcase(path):
                    layer = lyr
            if layer is None:
                layer = QgsRasterLayer(path, os.path.splitext(os.path.basename(path))[0])
                if not layer.isValid():
                    self.dock._error(f"Could not open {path}")
                    continue
                QgsProject.instance().addMapLayer(layer)
            years = re.findall(r"(?<!\d)((?:19|20)\d{2})(?!\d)", os.path.basename(path))
            self._add_row(int(years[-1]) if years else None, layer)

    def on_remove(self):
        for r in sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True):
            self.table.removeRow(r)

    def _entries(self):
        out = []
        for r in range(self.table.rowCount()):
            layer = self.table.cellWidget(r, 1).currentLayer()
            if layer is None:
                continue
            out.append((self.table.cellWidget(r, 0).value(), layer.source().split("|")[0]))
        years = [y for y, _ in out]
        if len(out) < 2:
            raise ValueError("Add at least two yearly maps.")
        if len(set(years)) != len(years):
            raise ValueError("Each year may appear only once.")
        return sorted(out)

    def _mask_polygons(self):
        layer = self.area_layer.currentLayer()
        if layer is None:
            return None
        return [pl.layer_union_geojson(layer, self.chk_area_sel.isChecked())]

    # ------------------------------------------------------------------
    # Runs
    # ------------------------------------------------------------------
    def _start(self, desc, fn, kwargs, on_done):
        self._task = CallTask(desc, fn, kwargs, on_done)
        self._task.progressText.connect(self.status.setText)
        self._task.progressChanged.connect(lambda v: self.progress.setValue(int(v)))
        self.progress.setValue(0)
        self.progress.show()
        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.status.setText(f"{desc} …")
        QgsApplication.taskManager().addTask(self._task)

    def _finish(self, task, ok, what):
        self._task = None
        self.progress.hide()
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        if task.exception is not None:
            self.dock._error(f"{what} failed: {task.exception}")
            self.status.setText(f"{what} failed.")
            return False
        if not ok:
            self.status.setText(f"{what} cancelled.")
            return False
        return True

    def on_compute(self):
        if self._task is not None:
            return
        try:
            entries = self._entries()
            mask = self._mask_polygons()
        except ValueError as e:
            self.dock._warn(str(e))
            return
        self._start("Computing the crop rotation", _rotation_task, {"entries": entries, "mask_polygons": mask},
                    self._on_done)

    def _on_done(self, task, ok):
        if not self._finish(task, ok, "Rotation"):
            return
        self.rot = task.result
        self._show_areas()
        self.pair.blockSignals(True)
        self.pair.clear()
        for (y0, y1) in self.rot["transitions"]:
            self.pair.addItem(f"{y0} → {y1}", (y0, y1))
        self.pair.blockSignals(False)
        self._show_pair()
        p.plot_alluvial(self.flow_plot.figure, self.rot)
        self.flow_plot.canvas.setMinimumHeight(max(460, 30 * len(self.rot["codes"]) + 120))
        self.flow_plot.redraw()
        self.status.setText(f"{len(self.rot['years'])} years, {len(self.rot['codes'])} classes.")

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------
    def _name(self, code):
        return self.rot["classes"][code]["name"]

    def _show_areas(self):
        rot = self.rot
        piv = rot["areas"].pivot(index="code", columns="year", values="ha").reindex(rot["codes"]).fillna(0)
        t = self.area_table
        t.clear()
        t.setRowCount(len(piv) + 1)
        t.setColumnCount(len(rot["years"]) + 1)
        t.setHorizontalHeaderLabels(["Class"] + [f"{y} (ha)" for y in rot["years"]])
        for r, code in enumerate(piv.index):
            it = QTableWidgetItem(f"{code} {self._name(code)}")
            it.setBackground(QBrush(QColor(rot["classes"][code]["color"]).lighter(160)))
            t.setItem(r, 0, it)
            for j, y in enumerate(rot["years"]):
                t.setItem(r, j + 1, QTableWidgetItem(f"{piv.loc[code, y]:,.1f}"))
        t.setItem(len(piv), 0, QTableWidgetItem("Total"))
        for j, y in enumerate(rot["years"]):
            t.setItem(len(piv), j + 1, QTableWidgetItem(f"{piv[y].sum():,.1f}"))
        t.resizeColumnsToContents()

        fig = self.area_plot.figure
        fig.clear()
        ax = fig.add_subplot(111)
        bottom = [0.0] * len(rot["years"])
        xs = list(range(len(rot["years"])))
        for code in piv.index:
            vals = [float(piv.loc[code, y]) for y in rot["years"]]
            ax.bar(xs, vals, 0.6, bottom=bottom, color=rot["classes"][code]["color"], label=self._name(code))
            bottom = [b + v for b, v in zip(bottom, vals)]
        ax.set_xticks(xs)
        ax.set_xticklabels([str(y) for y in rot["years"]])
        ax.set_ylabel("ha", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", color="#eeeeee")
        ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
        ax.set_title("Class areas per year", loc="left", fontsize=9, fontweight="bold")
        fig.tight_layout()
        self.area_plot.redraw()

    def _show_pair(self):
        if self.rot is None or self.pair.currentData() is None:
            return
        y0, y1 = self.pair.currentData()
        m = self.rot["transitions"][(y0, y1)]
        codes = self.rot["codes"]
        pct = self.pair_unit.currentIndex() == 1
        t = self.matrix
        t.clear()
        t.setRowCount(len(codes) + 1)
        t.setColumnCount(len(codes) + 1)
        t.setHorizontalHeaderLabels([f"→ {self._name(c_)} ({y1})" for c_ in codes] + ["Total"])
        t.setVerticalHeaderLabels([f"{self._name(c_)} ({y0})" for c_ in codes] + ["Total"])
        for i, a in enumerate(codes):
            tot = m[a, codes].sum()
            for j, b in enumerate(codes):
                v = m[a, b]
                share = v / tot if tot else 0
                it = QTableWidgetItem("" if v == 0 else (f"{share:.0%}" if pct else f"{v:,.1f}"))
                it.setTextAlignment(Qt.AlignCenter)
                if v:
                    col = QColor(31, 119, 180) if a == b else QColor(214, 39, 40)
                    col.setAlphaF(0.12 + 0.75 * share)
                    it.setBackground(QBrush(col))
                    it.setToolTip(f"{self._name(a)} {y0} → {self._name(b)} {y1}: {v:,.1f} ha ({share:.1%})")
                t.setItem(i, j, it)
            t.setItem(i, len(codes), QTableWidgetItem(f"{tot:,.1f}"))
        for j, b in enumerate(codes):
            t.setItem(len(codes), j, QTableWidgetItem(f"{m[codes, b].sum():,.1f}"))
        t.setItem(len(codes), len(codes), QTableWidgetItem(f"{m[codes][:, codes].sum():,.1f}"))
        t.resizeColumnsToContents()
        tt = at.transition_table(self.rot, y0, y1)
        ch = tt[tt["from_code"] != tt["to_code"]].sort_values("ha", ascending=False).head(8)
        total = tt["ha"].sum()
        stable = tt[tt["from_code"] == tt["to_code"]]["ha"].sum()
        items = "".join(f"<li>{r.from_name} → {r.to_name}: {r.ha:,.0f} ha ({r.pct_of_from:.0%} of {r.from_name})</li>"
                        for r in ch.itertuples())
        self.top_label.setText(f"<b>{stable / total:.0%} unchanged</b> ({stable:,.0f} of {total:,.0f} ha with a class "
                               f"in both years). Largest changes:<ul>{items}</ul>" if total else "No overlap.")

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------
    def on_change_map(self):
        if self.rot is None or self.pair.currentData() is None or self._task is not None:
            return
        y0, y1 = self.pair.currentData()
        pa = self.rot["paths"][self.rot["years"].index(y0)]
        pb = self.rot["paths"][self.rot["years"].index(y1)]
        path, _ = QFileDialog.getSaveFileName(self, "Save change map", f"change_{y0}_{y1}.tif",
                                              "GeoTIFF (*.tif *.tiff)")
        if not path:
            return
        try:
            mask = self._mask_polygons()
        except ValueError as e:
            self.dock._warn(str(e))
            return
        self._start(f"Writing the change map {y0} → {y1}", _change_task,
                    {"path_a": pa, "path_b": pb, "out_path": path, "classes": self.rot["classes"],
                     "mask_polygons": mask}, self._on_change_done)

    def _on_change_done(self, task, ok):
        if not self._finish(task, ok, "Change map"):
            return
        path, legend = task.result
        layer = QgsRasterLayer(path, os.path.splitext(os.path.basename(path))[0])
        if layer.isValid():
            cats = [QgsPalettedRasterRenderer.Class(q["value"], QColor(q["color"]), q["label"]) for q in legend]
            layer.setRenderer(QgsPalettedRasterRenderer(layer.dataProvider(), 1, cats))
            QgsProject.instance().addMapLayer(layer)
        self.status.setText(f"Saved {path} ({len(legend)} transitions; legend in *_legend.json).")

    def on_save_csv(self):
        if self.rot is None:
            self.dock._warn("Compute the rotation first.")
            return
        folder = QFileDialog.getExistingDirectory(self, "Folder for the rotation tables")
        if not folder:
            return
        rot = self.rot
        areas = rot["areas"].copy()
        areas.insert(2, "class_name", areas["code"].map(lambda c_: rot["classes"][c_]["name"]))
        areas.to_csv(os.path.join(folder, "rotation_areas.csv"), index=False)
        frames = []
        for (y0, y1) in rot["transitions"]:
            tt = at.transition_table(rot, y0, y1)
            tt.insert(0, "year_to", y1)
            tt.insert(0, "year_from", y0)
            frames.append(tt)
        pd.concat(frames, ignore_index=True).to_csv(os.path.join(folder, "rotation_transitions.csv"), index=False)
        fig = Figure(figsize=(10, max(5, 0.3 * len(rot["codes"]) + 2)), dpi=150)
        p.plot_alluvial(fig, rot)
        fig.savefig(os.path.join(folder, "rotation_flows.png"))
        self.iface.messageBar().pushSuccess("Crop rotation", f"Saved areas, transitions and flow diagram to {folder}")

    def cleanup(self):
        if self._task is not None:
            self._task.cancel()
