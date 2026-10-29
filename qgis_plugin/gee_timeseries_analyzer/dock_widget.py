"""The GEE Time Series Analyzer & Classifier dock widget. A header with
three sections drives the real QGIS map canvas and hosts embedded
matplotlib charts:

  Time series     -- Polygons, Analysis, Charts, Compare (below)
  Classification  -- classify_tab.py (+ the crop rotation, rotation_tab.py)
  Export imagery  -- composites / time-series stacks as GeoTIFF

  Polygons  -- where polygons come from: drawn on the canvas, imported from
               any polygon layer / shapefile, clicked on a field layer
               (Field Inspector mode), or downloaded from a GEE table asset
  Analysis  -- time window, products, optical index, cloud thresholds and
               the monthly Sentinel-2 composite layers
  Charts    -- combined (all polygons) and per-polygon charts, CSV / PNG export
  Compare   -- many polygons on shared axes, outlier ranking
  Export    -- composites / time-series stacks with any recipe as GeoTIFF
  Rotation  -- crop rotation / change between yearly classified maps
  Classify  -- supervised crop classification with savable / re-applicable
               models (classify_tab.py)

Short Earth Engine calls (initialising, asset download, composite map ids)
run synchronously with a wait cursor like the DWD plugin's fetches; the
time-series extraction itself runs as a QgsTask (see tasks.py).
"""
import math
import os
from urllib.parse import quote

import pandas as pd
from qgis.core import (
    QgsApplication, QgsProject, QgsVectorLayer, QgsRasterLayer, QgsMapLayerProxyModel,
    QgsCoordinateTransform, QgsRectangle, QgsLayerTreeLayer, QgsGeometry, QgsWkbTypes,
)
from qgis.gui import QgsMapLayerComboBox, QgsFieldExpressionWidget, QgsMapToolExtent, QgsRubberBand
from qgis.PyQt.QtCore import Qt, QDate
from qgis.PyQt.QtGui import QColor, QPixmap, QPainter, QIcon
from qgis.PyQt.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout,
    QTabWidget, QPushButton, QLabel, QLineEdit, QSpinBox, QDoubleSpinBox,
    QCheckBox, QRadioButton, QButtonGroup, QComboBox, QGroupBox, QListWidget,
    QListWidgetItem, QAbstractItemView, QFileDialog, QApplication, QDateEdit,
    QProgressBar, QScrollArea, QSplitter, QTreeWidget, QTreeWidgetItem, QStackedWidget,
    QFrame, QToolButton, QSizePolicy,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from . import gee_client as c
from . import plotting as p
from . import polygon_layer as pl
from .tasks import ExtractTask, CallTask
from .classify_tab import ClassificationTab
from .rotation_tab import RotationTab
from .recipe_widget import FeatureRecipeWidget
from . import style
from . import help_texts as H
from . import classification as k

TITLE = "GEE Time Series Analyzer & Classifier"
SECTIONS = ("Time series", "Classification", "Export imagery")
CURRENT_YEAR = 2026
MAX_POLYGONS = 300
COMPOSITE_GROUP = "GEE Sentinel-2 composites"
COMPOSITE_EXPORT_GROUP = "GEE exports (GeoTIFF)"
try:
    from qgis.core import Qgis
    POLYGON_FILTER = Qgis.LayerFilter.PolygonLayer      # QGIS >= 3.34
    RASTER_FILTER = Qgis.LayerFilter.RasterLayer
except AttributeError:
    POLYGON_FILTER = QgsMapLayerProxyModel.PolygonLayer
    RASTER_FILTER = QgsMapLayerProxyModel.RasterLayer
PANEL_HEIGHT_PX = 190

# GEE table assets offered in the asset combo: (asset id, crop column, year).
# Empty on purpose -- users type their own asset ids.
ASSET_PRESETS = []


def _swatch(colour):
    pix = QPixmap(12, 12)
    pix.fill(Qt.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setBrush(QColor(colour))
    painter.setPen(QColor("#666666"))
    painter.drawRect(1, 1, 10, 10)
    painter.end()
    return pix


class _ScoreItem(QTreeWidgetItem):
    """Compare-tab row; sorts the deviation column numerically (UserRole)."""

    def __lt__(self, other):
        tree = self.treeWidget()
        col = tree.sortColumn() if tree is not None else 0
        if col == 1:
            a, b = self.data(1, Qt.UserRole), other.data(1, Qt.UserRole)
            return (-1.0 if a is None else a) < (-1.0 if b is None else b)
        return self.text(col).lower() < other.text(col).lower()


class PlotPanel(QWidget):
    """A matplotlib canvas + interactive navigation toolbar (pan / zoom /
    save-as-PNG), inside a scroll area because six stacked panels are taller
    than a typical dock."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(6, 8), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.canvas)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(scroll)

    def set_panels(self, n):
        self.canvas.setMinimumHeight(max(1, n) * PANEL_HEIGHT_PX + 40)

    def redraw(self):
        self.canvas.draw_idle()


class GEEDockWidget(QDockWidget):
    def __init__(self, iface, parent=None):
        super().__init__(TITLE, parent)
        self.setObjectName("GEETimeSeriesAnalyzerDock")
        self.iface = iface
        self.canvas = iface.mapCanvas()

        self._prev_map_tool = None
        self._task = None
        self._pending_analyze = False
        self._composite_key = None

        self.raw_df = None              # long DataFrame from the last extraction (unfiltered)
        self._filtered_df = None        # raw_df after the valid-pixel filter / same-day merge
        self.analysis_polygons = []     # polygon snapshot of the last extraction
        self.analysis_meta = {}         # products / index / date window of the last extraction
        self._pending_keep = set()      # pids whose cached rows are reused by the running task
        self._focus_pid = None          # polygon to show once the running analysis is done
        self._syncing = False           # True while we change the map selection ourselves
        self._hooked_layer_id = None    # analysis layer whose selectionChanged we listen to
        self._cmp_artists = {}          # plotting.plot_compare bookkeeping
        self._cmp_highlight = None      # highlighted pid on the Compare tab
        self._cmp_hover = None
        self._cmp_unchecked = set()     # pids unticked on the Compare tab

        self.draw_tool = pl.PolygonDrawTool(self.canvas)
        self.draw_tool.polygonDrawn.connect(self.on_polygon_drawn)
        self.pick_tool = pl.FeaturePickTool(self.canvas, lambda: self.source_combo.currentLayer())
        self.pick_tool.featurePicked.connect(self.on_feature_picked)
        self.rect_tool = QgsMapToolExtent(self.canvas)
        self.rect_tool.extentChanged.connect(self.on_export_rect_drawn)
        self._export_rect = None        # drawn export rectangle (EPSG:4326)
        self._export_task = None
        self._rect_band = None
        self.pick_tool.nothingPicked.connect(
            lambda: self.poly_status.setText("No field found here — click inside a polygon of the selected layer."))

        self._build_ui()
        self.canvas.mapToolSet.connect(self._on_map_tool_set)
        QgsProject.instance().layersRemoved.connect(lambda _ids: self.refresh_polygon_list())
        self.refresh_polygon_list()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        root = QWidget()
        root.setObjectName("gtaRoot")
        root.setStyleSheet(style.QSS)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(6, 4, 6, 6)
        root_layout.setSpacing(6)
        root_layout.addWidget(self._build_header())

        self.sections = QStackedWidget()
        root_layout.addWidget(self.sections, 1)

        # Time series: the four tabs of the original apps
        ts = QWidget()
        ts_layout = QVBoxLayout(ts)
        ts_layout.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.addTab(self._scrolled(self._build_polygons_tab()), "Polygons")
        self.tabs.addTab(self._scrolled(self._build_analysis_tab()), "Analysis")
        self.tabs.addTab(self._build_charts_tab(), "Charts")
        self.tabs.addTab(self._build_compare_tab(), "Compare")
        ts_layout.addWidget(self.tabs, 1)
        status_row = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setMaximumWidth(140)
        self.progress.hide()
        self.status_label = QLabel("Ready — add polygons on the Polygons tab.")
        self.status_label.setWordWrap(True)
        status_row.addWidget(self.status_label, 1)
        status_row.addWidget(self.progress)
        ts_layout.addLayout(status_row)
        self.sections.addWidget(ts)

        # Classification: Setup / Results / Class profiles / Training check / Crop rotation
        self.classify_tab = ClassificationTab(self)
        self.rotation_tab = RotationTab(self)
        self.classify_tab.inner.addTab(self.rotation_tab, "Crop rotation")
        self.sections.addWidget(self.classify_tab)

        # Export imagery
        self.sections.addWidget(self._scrolled(self._build_export_tab()))

        # long combo entries (e.g. data types) must not widen the dock; the
        # open list still shows the full text
        for combo in root.findChildren(QComboBox):
            combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(10)
        for box in root.findChildren(QGroupBox):       # hover help in every group title
            if box.title() in H.GROUPS:
                style.attach_info(box, H.GROUPS[box.title()])
        self.setWidget(root)
        self._show_section(0)

    def _build_header(self):
        header = QFrame()
        header.setObjectName("appHeader")
        lay = QVBoxLayout(header)
        lay.setContentsMargins(2, 2, 2, 6)
        lay.setSpacing(5)
        title_row = QHBoxLayout()
        icon = QLabel()
        icon.setPixmap(QIcon(os.path.join(os.path.dirname(__file__), "icons", "gee_icon.svg")).pixmap(22, 22))
        title = QLabel("GEE Time Series Analyzer & Classifier")
        title.setObjectName("appTitle")
        sub = QLabel("Sentinel-1 · Sentinel-2 · Landsat · Google Earth Engine")
        sub.setObjectName("appSub")
        title_row.addWidget(icon)
        title_row.addWidget(title)
        title_row.addStretch(1)
        title_row.addWidget(sub)
        lay.addLayout(title_row)
        seg_row = QHBoxLayout()
        seg_row.setSpacing(0)
        self.section_buttons = []
        group = QButtonGroup(header)
        group.setExclusive(True)
        for i, label in enumerate(SECTIONS):
            b = QToolButton()
            b.setObjectName("seg")
            b.setText(label)
            b.setCheckable(True)
            b.setProperty("pos", "first" if i == 0 else ("last" if i == len(SECTIONS) - 1 else "mid"))
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            b.clicked.connect(lambda _=False, i=i: self._show_section(i))
            group.addButton(b, i)
            self.section_buttons.append(b)
            seg_row.addWidget(b, 1)
        lay.addLayout(seg_row)
        self._section_group = group
        return header

    def _show_section(self, i):
        self.sections.setCurrentIndex(i)
        self.section_buttons[i].setChecked(True)

    @staticmethod
    def _scrolled(widget):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(widget)
        return scroll

    def _build_polygons_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)

        # -- Earth Engine connection ------------------------------------
        ee_box = QGroupBox("Earth Engine")
        ee_form = QFormLayout(ee_box)
        self.project_edit = QLineEdit()
        self.project_edit.setPlaceholderText("Cloud project (default: the Google Earth Engine plugin's)")
        ee_row = QHBoxLayout()
        ee_row.addWidget(self.project_edit, 1)
        self.btn_connect = QPushButton("Connect")
        self.btn_connect.clicked.connect(self.on_connect)
        ee_row.addWidget(self.btn_connect)
        ee_form.addRow("Project:", ee_row)
        self.ee_status = QLabel("Not connected yet (connects automatically on first use).")
        self.ee_status.setWordWrap(True)
        self.ee_status.setStyleSheet("color:#666")
        ee_form.addRow(self.ee_status)
        layout.addWidget(ee_box)

        # -- Draw ---------------------------------------------------------
        draw_box = QGroupBox("Draw polygons")
        draw_layout = QVBoxLayout(draw_box)
        self.btn_draw = QPushButton("Draw polygon on map")
        self.btn_draw.setCheckable(True)
        self.btn_draw.toggled.connect(self.on_toggle_draw)
        draw_layout.addWidget(self.btn_draw)
        hint = QLabel("Left click: add vertex · right/double click or Enter: finish · "
                      "Backspace: undo vertex · Esc: discard")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#666; font-size:11px")
        draw_layout.addWidget(hint)
        layout.addWidget(draw_box)

        # -- From a layer / file / click -----------------------------------
        src_box = QGroupBox("From a polygon layer or shapefile")
        src_layout = QVBoxLayout(src_box)
        form = QFormLayout()
        layer_row = QHBoxLayout()
        self.source_combo = QgsMapLayerComboBox()
        self.source_combo.setFilters(POLYGON_FILTER)
        self.source_combo.setAllowEmptyLayer(True)
        layer_row.addWidget(self.source_combo, 1)
        btn_open = QPushButton("Open file…")
        btn_open.clicked.connect(self.on_open_file)
        layer_row.addWidget(btn_open)
        form.addRow("Layer:", layer_row)
        self.label_expr = QgsFieldExpressionWidget()
        self.label_expr.setLayer(self.source_combo.currentLayer())
        self.source_combo.layerChanged.connect(self.on_source_layer_changed)
        form.addRow("Polygon name:", self.label_expr)
        src_layout.addLayout(form)

        self.chk_selected_only = QCheckBox("Selected features only")
        src_layout.addWidget(self.chk_selected_only)
        self.btn_import = QPushButton("Import features as analysis polygons")
        self.btn_import.clicked.connect(self.on_import_features)
        src_layout.addWidget(self.btn_import)

        self.btn_pick = QPushButton("Click a field on the map")
        self.btn_pick.setCheckable(True)
        self.btn_pick.toggled.connect(self.on_toggle_pick)
        src_layout.addWidget(self.btn_pick)
        self.chk_inspector = QCheckBox("Inspector mode: analyse a clicked field at once")
        self.chk_inspector.setChecked(True)
        src_layout.addWidget(self.chk_inspector)
        self.chk_keep_picked = QCheckBox("Keep previously clicked fields")
        self.chk_keep_picked.setToolTip("Switch between / compare clicked fields; only new fields are queried.")
        self.chk_keep_picked.setChecked(True)
        src_layout.addWidget(self.chk_keep_picked)
        layout.addWidget(src_box)

        # -- GEE asset ----------------------------------------------------
        asset_box = QGroupBox("Field layer from a GEE table asset")
        asset_form = QFormLayout(asset_box)
        self.asset_combo = QComboBox()
        self.asset_combo.setEditable(True)
        for asset, crop_col, year in ASSET_PRESETS:
            self.asset_combo.addItem(asset, (crop_col, year))
        self.asset_combo.lineEdit().setPlaceholderText("projects/<your-project>/assets/<table>")
        self.asset_combo.currentIndexChanged.connect(self.on_asset_preset_changed)
        asset_form.addRow("Asset:", self.asset_combo)
        self.class_field_edit = QLineEdit("crop")
        self.class_field_edit.setPlaceholderText("optional, e.g. crop")
        asset_form.addRow("Style by field:", self.class_field_edit)
        self.btn_load_asset = QPushButton("Download asset as layer")
        self.btn_load_asset.clicked.connect(self.on_load_asset)
        asset_form.addRow(self.btn_load_asset)
        layout.addWidget(asset_box)

        # -- Current polygons ------------------------------------------------
        list_box = QGroupBox("Analysis polygons")
        list_layout = QVBoxLayout(list_box)
        self.poly_list = QListWidget()
        self.poly_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.poly_list.setMinimumHeight(110)
        self.poly_list.setToolTip("Click: show this polygon's charts and select it on the map · "
                                  "double click: open the Charts tab")
        self.poly_list.itemClicked.connect(
            lambda it: self.show_polygon(it.data(Qt.UserRole), switch_tab=False))
        self.poly_list.itemDoubleClicked.connect(
            lambda it: self.show_polygon(it.data(Qt.UserRole), switch_tab=True))
        list_layout.addWidget(self.poly_list)
        btn_row = QHBoxLayout()
        btn_zoom = QPushButton("Zoom to")
        btn_zoom.clicked.connect(self.on_zoom_polygons)
        btn_remove = QPushButton("Remove selected")
        btn_remove.clicked.connect(self.on_remove_selected)
        btn_clear = QPushButton("Clear all")
        btn_clear.clicked.connect(self.on_clear_polygons)
        for b in (btn_zoom, btn_remove, btn_clear):
            btn_row.addWidget(b)
        list_layout.addLayout(btn_row)
        self.poly_status = QLabel("")
        self.poly_status.setWordWrap(True)
        list_layout.addWidget(self.poly_status)
        layout.addWidget(list_box)

        layout.addStretch(1)
        return w

    def _build_analysis_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)

        # -- Time window ----------------------------------------------------
        time_box = QGroupBox("Time window")
        time_form = QFormLayout(time_box)
        self.year_spin = QSpinBox()
        self.year_spin.setRange(c.MIN_YEAR, c.MAX_YEAR)
        self.year_spin.setValue(CURRENT_YEAR)
        self.year_spin.valueChanged.connect(self._sync_dates_to_year)
        time_form.addRow("Year:", self.year_spin)
        self.chk_custom_dates = QCheckBox("Custom date range")
        self.chk_custom_dates.toggled.connect(self._on_custom_dates_toggled)
        time_form.addRow(self.chk_custom_dates)
        date_row = QHBoxLayout()
        self.start_date = QDateEdit()
        self.end_date = QDateEdit()
        for d in (self.start_date, self.end_date):
            d.setCalendarPopup(True)
            d.setDisplayFormat("yyyy-MM-dd")
            d.setMinimumDate(QDate(2014, 10, 1))
            d.setEnabled(False)
            date_row.addWidget(d)
        time_form.addRow("From / to:", date_row)
        self._sync_dates_to_year(CURRENT_YEAR)
        layout.addWidget(time_box)

        # -- Products -------------------------------------------------------
        prod_box = QGroupBox("Time series")
        prod_layout = QVBoxLayout(prod_box)
        self.product_checks = {}
        for key, meta in c.PRODUCTS.items():
            cb = QCheckBox(meta["label"])
            cb.setChecked(True)
            self.product_checks[key] = cb
            prod_layout.addWidget(cb)
        form = QFormLayout()
        self.index_combo = QComboBox()
        for key, label in c.OPTICAL_INDICES.items():
            self.index_combo.addItem(f"{key} — {label}", key)
        form.addRow("Optical index:", self.index_combo)
        self.cloud_spin = QSpinBox()
        self.cloud_spin.setRange(1, 100)
        self.cloud_spin.setValue(70)
        self.cloud_spin.setSuffix(" %")
        self.cloud_spin.setToolTip("Scene-level pre-filter (CLOUDY_PIXEL_PERCENTAGE / CLOUD_COVER)")
        form.addRow("Max. scene cloud cover:", self.cloud_spin)
        self.valid_spin = QSpinBox()
        self.valid_spin.setRange(0, 100)
        self.valid_spin.setValue(50)
        self.valid_spin.setSuffix(" %")
        self.valid_spin.setToolTip("An image is only plotted for a polygon if at least this share of "
                                   "the polygon is cloud-free (optical) / free of border noise (S1). "
                                   "Changing it re-filters the charts without querying GEE again.")
        self.valid_spin.valueChanged.connect(lambda *_: self._refresh_charts())
        form.addRow("Min. valid pixels in polygon:", self.valid_spin)
        prod_layout.addLayout(form)
        layout.addWidget(prod_box)

        # -- Composites -----------------------------------------------------
        comp_box = QGroupBox("Sentinel-2 monthly composites (map layers)")
        comp_layout = QVBoxLayout(comp_box)
        comp_hint = QLabel("Quick look on the map: calendar-month median of Sentinel-2 SR, SCL cloud mask "
                           "(classes 4–7 kept), scenes with ≤ 60 % cloud. For other periods, aggregations, "
                           "cloud masks, variables or files use the Export imagery section.")
        comp_hint.setWordWrap(True)
        comp_hint.setStyleSheet("color:#666; font-size:11px")
        comp_layout.addWidget(comp_hint)
        self.chk_composites = QCheckBox("Add monthly median composites of the analysis year")
        self.chk_composites.setChecked(True)
        comp_layout.addWidget(self.chk_composites)
        month_grid = QGridLayout()
        self.month_checks = []
        for i, name in enumerate(c.MONTH_NAMES):
            cb = QCheckBox(name[:3])
            cb.setChecked(i + 1 in (5, 7, 9))
            self.month_checks.append(cb)
            month_grid.addWidget(cb, i // 6, i % 6)
        comp_layout.addLayout(month_grid)
        comp_form = QFormLayout()
        self.comp_mode_combo = QComboBox()
        for key, meta in c.COMPOSITE_MODES.items():
            self.comp_mode_combo.addItem(meta["label"], key)
        comp_form.addRow("Rendering:", self.comp_mode_combo)
        extent_row = QHBoxLayout()
        self.extent_group = QButtonGroup(w)
        self.radio_extent_polys = QRadioButton("Polygons +")
        self.radio_extent_view = QRadioButton("Current map view")
        self.radio_extent_polys.setChecked(True)
        self.extent_group.addButton(self.radio_extent_polys)
        self.extent_group.addButton(self.radio_extent_view)
        self.buffer_spin = QDoubleSpinBox()
        self.buffer_spin.setRange(0, 100)
        self.buffer_spin.setValue(5)
        self.buffer_spin.setSuffix(" km")
        extent_row.addWidget(self.radio_extent_polys)
        extent_row.addWidget(self.buffer_spin)
        extent_row.addWidget(self.radio_extent_view)
        comp_form.addRow("Extent:", extent_row)
        comp_layout.addLayout(comp_form)
        self.btn_composites_only = QPushButton("Add composites only")
        self.btn_composites_only.clicked.connect(self.on_composites_only)
        comp_layout.addWidget(self.btn_composites_only)
        layout.addWidget(comp_box)

        # -- Run --------------------------------------------------------------
        run_row = QHBoxLayout()
        self.btn_analyze = QPushButton("Analyze")
        style.primary(self.btn_analyze)
        self.btn_analyze.clicked.connect(self.on_analyze)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.on_cancel)
        run_row.addWidget(self.btn_analyze, 1)
        run_row.addWidget(self.btn_cancel)
        layout.addLayout(run_row)

        layout.addStretch(1)
        return w

    def _build_export_tab(self):
        """Composites / time-series stacks with the full recipe (the same
        engine as the Classify tab) for a chosen area, as GeoTIFF files with
        band descriptions, band metadata and a JSON sidecar."""
        w = QWidget()
        layout = QVBoxLayout(w)

        area = QGroupBox("Area")
        form = QFormLayout(area)
        self.exp_extent_kind = QComboBox()
        self.exp_extent_kind.addItems(["Polygon layer / shapefile", "Rectangle drawn on the map",
                                       "Current map view"])
        form.addRow("Area:", self.exp_extent_kind)
        stack = QStackedWidget()
        lw = QWidget()
        lrow = QHBoxLayout(lw)
        lrow.setContentsMargins(0, 0, 0, 0)
        self.exp_layer = QgsMapLayerComboBox()
        self.exp_layer.setFilters(POLYGON_FILTER)
        self.exp_layer.setAllowEmptyLayer(True)
        self.exp_layer.setLayer(None)
        btn_open = QPushButton("Open file…")
        btn_open.clicked.connect(self.on_export_open_file)
        self.chk_exp_selected = QCheckBox("Selected only")
        lrow.addWidget(self.exp_layer, 1)
        lrow.addWidget(btn_open)
        lrow.addWidget(self.chk_exp_selected)
        rw = QWidget()
        rrow = QHBoxLayout(rw)
        rrow.setContentsMargins(0, 0, 0, 0)
        self.btn_rect = QPushButton("Draw rectangle")
        self.btn_rect.setCheckable(True)
        self.btn_rect.setToolTip("Click and drag on the map")
        self.btn_rect.toggled.connect(self.on_toggle_rect)
        self.rect_label = QLabel("No rectangle yet.")
        rrow.addWidget(self.btn_rect)
        rrow.addWidget(self.rect_label, 1)
        stack.addWidget(lw)
        stack.addWidget(rw)
        stack.addWidget(QLabel("The map extent at the time of the export."))
        self.exp_extent_kind.currentIndexChanged.connect(stack.setCurrentIndex)
        self.exp_extent_kind.currentIndexChanged.connect(lambda *_: self._show_rect_band())
        form.addRow(stack)
        self.chk_exp_clip = QCheckBox("Clip to the polygons")
        self.chk_exp_clip.setToolTip("Outside the polygons = NoData; unticked: their bounding box")
        self.chk_exp_clip.setChecked(True)
        form.addRow(self.chk_exp_clip)
        layout.addWidget(area)

        comp = QGroupBox("Composites")
        clay = QVBoxLayout(comp)
        self.exp_recipe = FeatureRecipeWidget("Resolution:", year_range=True)
        clay.addWidget(self.exp_recipe)
        layout.addWidget(comp)

        out = QGroupBox("Output")
        oform = QFormLayout(out)
        self.exp_content = QComboBox()
        for key, label in k.EXPORT_CONTENT.items():
            self.exp_content.addItem(label, key)
        oform.addRow("Content:", self.exp_content)
        self.exp_layout = QComboBox()
        for key, label in k.EXPORT_LAYOUTS.items():
            self.exp_layout.addItem(label, key)
        oform.addRow("Files:", style.with_info(self.exp_layout, H.EXP_LAYOUT))
        self.exp_dtype = QComboBox()
        for key, label in k.EXPORT_DTYPES.items():
            self.exp_dtype.addItem(label, key)
        oform.addRow("Data type:", style.with_info(self.exp_dtype, H.EXP_DTYPE))
        self.chk_exp_wc = QCheckBox("Add the ESA WorldCover band")
        self.chk_exp_wc.setToolTip("Used as mask by the offline classification")
        oform.addRow(self.chk_exp_wc)
        self.exp_crs = QLineEdit()
        self.exp_crs.setPlaceholderText("UTM of the area (e.g. EPSG:32641)")
        oform.addRow("CRS:", self.exp_crs)
        self.exp_prefix = QLineEdit("s2")
        oform.addRow("File name prefix:", self.exp_prefix)
        layout.addWidget(out)

        self.exp_summary = QLabel()
        self.exp_summary.setWordWrap(True)
        self.exp_summary.setTextFormat(Qt.RichText)
        self.exp_summary.setStyleSheet("background:#f4f6f8; border:1px solid #d0d7de; padding:6px")
        layout.addWidget(self.exp_summary)

        brow = QHBoxLayout()
        self.btn_exp_download = QPushButton("Download GeoTIFFs…")
        style.primary(self.btn_exp_download)
        self.btn_exp_download.setToolTip("Computed tile by tile and saved locally with a JSON sidecar "
                                         "(areas up to a few hundred km² at 10 m); added to the project")
        self.btn_exp_download.clicked.connect(self.on_export_composites_download)
        self.btn_exp_cancel = QPushButton("Cancel")
        self.btn_exp_cancel.setEnabled(False)
        self.btn_exp_cancel.clicked.connect(lambda: self._export_task and self._export_task.cancel())
        brow.addWidget(self.btn_exp_download, 1)
        brow.addWidget(self.btn_exp_cancel)
        layout.addLayout(brow)
        drow = QHBoxLayout()
        self.exp_drive_folder = QLineEdit("GEE_Exports")
        btn_drive = QPushButton("Export to Drive")
        btn_drive.setToolTip("One Earth Engine batch task per file (large areas)")
        btn_drive.clicked.connect(self.on_export_composites_drive)
        drow.addWidget(QLabel("Drive folder:"))
        drow.addWidget(self.exp_drive_folder, 1)
        drow.addWidget(btn_drive)
        layout.addLayout(drow)
        layout.addStretch(1)

        self.exp_recipe.changed.connect(self._update_export_summary)
        for combo in (self.exp_content, self.exp_layout, self.exp_dtype, self.exp_extent_kind):
            combo.currentIndexChanged.connect(self._update_export_summary)
        self.exp_layer.layerChanged.connect(self._update_export_summary)
        for cb in (self.chk_exp_wc, self.chk_exp_selected, self.chk_exp_clip):
            cb.toggled.connect(self._update_export_summary)
        self.exp_prefix.textChanged.connect(self._update_export_summary)
        self.exp_content.currentIndexChanged.connect(self._on_export_content_changed)
        self._update_export_summary()
        return w

    def _on_export_content_changed(self, *_):
        data = self.exp_content.currentData() == "data"
        for wdg in (self.exp_layout, self.exp_dtype, self.chk_exp_wc):
            wdg.setEnabled(data)

    def _export_bbox(self):
        """WGS84 bounding box of the export area without Earth Engine, or None."""
        kind = self.exp_extent_kind.currentIndex()
        if kind == 0:
            layer = self.exp_layer.currentLayer()
            if not pl.is_polygon_layer(layer):
                return None
            if self.chk_exp_selected.isChecked() and layer.selectedFeatureCount():
                rect = layer.boundingBoxOfSelected()
            else:
                rect = layer.extent()
            xform = QgsCoordinateTransform(layer.crs(), pl.WGS84, QgsProject.instance())
            return xform.transformBoundingBox(rect)
        if kind == 1:
            return self._export_rect
        xform = QgsCoordinateTransform(self.canvas.mapSettings().destinationCrs(), pl.WGS84,
                                       QgsProject.instance())
        return xform.transformBoundingBox(self.canvas.extent())

    def _update_export_summary(self, *_):
        f = self.exp_recipe.features()
        content = self.exp_content.currentData()
        data = content == "data"
        problem = self.exp_recipe.problem(f, need_variables=data)
        if problem:
            self.exp_summary.setText(f"<b style='color:#b00'>{problem}</b>")
            return
        years = self.exp_recipe.years()
        prefix = self.exp_prefix.text().strip() or "s2"
        plan = k.export_plan(years, f, content, self.exp_layout.currentData(), prefix)
        pers = k.describe_periods(f, years[0])
        shown = ", ".join(pers[:12]) + (f", … ({len(pers)} in total)" if len(pers) > 12 else "")
        s2 = f["s2_features"] if data else plan[0]["recipe"]["s2_features"]
        lines = [f"<b>{len(pers)} period(s)</b> in {years[0]} ({k.PERIODS[f['period']]}): {shown}"]
        reds = f["reducers"] if data else f["reducers"][:1]
        lines.append("<b>Aggregation:</b> " + ", ".join(k.REDUCERS[r] for r in reds) + " of all clear "
                     "observations per period" + ("; gaps filled from the neighbouring periods"
                                                  if f["gap_fill"] == "window" else "; gaps stay NoData"))
        if s2:
            mask = (f"Cloud Score+ (cs_cdf ≥ {f['cs_threshold']:.2f})" if f["s2_mask"] == "csplus"
                    else "SCL (vegetation, bare soil, water, unclassified kept)")
            lines.append(f"<b>Sentinel-2:</b> {', '.join(s2)} · cloud mask {mask} · scenes with ≤ "
                         f"{f['s2_max_cloud']} % cloud")
        if data and f["s1_features"]:
            lines.append(f"<b>Sentinel-1:</b> {', '.join(f['s1_features'])} · {k.S1_ORBITS[f['s1_orbit']]}")
        nb = len(plan[0]["rows"]) + (1 if data and self.chk_exp_wc.isChecked() else 0)
        nbytes = 1 if not data else (2 if self.exp_dtype.currentData() == "int16" else 4)
        lines.append(f"<b>{len(plan)} file(s)</b> ({len(plan) // len(years)} per year × {len(years)} year(s)), "
                     f"{nb} band(s) each, e.g. <code>{plan[0]['name']}.tif</code> + <code>.json</code>")
        bbox = self._export_bbox()
        if bbox is not None and not bbox.isEmpty():
            lat = math.radians(bbox.center().y())
            w_km = (bbox.xMaximum() - bbox.xMinimum()) * 111.32 * math.cos(lat)
            h_km = (bbox.yMaximum() - bbox.yMinimum()) * 110.57
            res = f["scale"]
            px = (w_km * 1000 / res) * (h_km * 1000 / res)
            total = px * nbytes * sum(len(q["rows"]) + (1 if data and self.chk_exp_wc.isChecked() else 0)
                                      for q in plan)
            warn = (" <b style='color:#b00'>— too large for the direct download, use Drive or a coarser "
                    "resolution</b>" if px > c.MAX_DOWNLOAD_PIXELS else "")
            npx = f"{px / 1e6:.1f} Mpx" if px >= 1e6 else f"{px:,.0f} px"
            lines.append(f"<b>Area:</b> ≈ {w_km:.1f} × {h_km:.1f} km = {npx} at {res} m · "
                         f"≈ {total / 1e6:,.0f} MB uncompressed in total{warn}")
        else:
            lines.append("<b>Area:</b> choose a layer or draw a rectangle.")
        self.exp_summary.setText("<br>".join(lines))

    def _build_charts_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        view_row = QHBoxLayout()
        view_row.addWidget(QLabel("Show:"))
        self.view_combo = QComboBox()
        self.view_combo.currentIndexChanged.connect(self._on_view_changed)
        btn_prev = QPushButton("◀")
        btn_next = QPushButton("▶")
        for b, step in ((btn_prev, -1), (btn_next, 1)):
            b.setFixedWidth(30)
            b.setToolTip("Previous / next polygon")
            b.clicked.connect(lambda _=False, st=step: self._step_view(st))
        view_row.addWidget(btn_prev)
        view_row.addWidget(self.view_combo, 1)
        view_row.addWidget(btn_next)
        layout.addLayout(view_row)

        self.chart_info = QLabel("Run an analysis to see charts.")
        self.chart_info.setWordWrap(True)
        self.chart_info.setStyleSheet("color:#666")
        layout.addWidget(self.chart_info)

        self.plot = PlotPanel()
        p.plot_placeholder(self.plot.figure, "No analysis yet.")
        layout.addWidget(self.plot, 1)

        exp_row = QHBoxLayout()
        self.btn_csv = QPushButton("Export CSV…")
        self.btn_csv.clicked.connect(self.on_export_csv)
        self.btn_png = QPushButton("Export all charts as PNG…")
        self.btn_png.clicked.connect(self.on_export_png)
        exp_row.addWidget(self.btn_csv)
        exp_row.addWidget(self.btn_png)
        layout.addLayout(exp_row)
        return w

    def _build_compare_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        top = QHBoxLayout()
        top.addWidget(QLabel("Series:"))
        self.cmp_series_combo = QComboBox()
        self.cmp_series_combo.currentIndexChanged.connect(lambda *_: self._redraw_compare())
        top.addWidget(self.cmp_series_combo, 1)
        layout.addLayout(top)

        opts = QHBoxLayout()
        self.chk_cmp_uniform = QCheckBox("Uniform colour")
        self.chk_cmp_uniform.setChecked(True)
        self.chk_cmp_uniform.setToolTip("All lines grey, the highlighted one red — best for many polygons")
        self.chk_cmp_band = QCheckBox("Mean ± 1σ")
        self.chk_cmp_band.setChecked(True)
        self.chk_cmp_pan = QCheckBox("Pan map to highlighted")
        self.chk_cmp_pan.setChecked(True)
        for cb in (self.chk_cmp_uniform, self.chk_cmp_band):
            cb.toggled.connect(lambda *_: self._redraw_compare())
        for cb in (self.chk_cmp_uniform, self.chk_cmp_band, self.chk_cmp_pan):
            opts.addWidget(cb)
        layout.addLayout(opts)

        splitter = QSplitter(Qt.Vertical)
        list_w = QWidget()
        list_layout = QVBoxLayout(list_w)
        list_layout.setContentsMargins(0, 0, 0, 0)
        self.cmp_tree = QTreeWidget()
        self.cmp_tree.setColumnCount(2)
        self.cmp_tree.setHeaderLabels(["Polygon", "Deviation (σ)"])
        self.cmp_tree.setRootIsDecorated(False)
        self.cmp_tree.setSortingEnabled(True)
        self.cmp_tree.setToolTip("Tick the polygons to chart. Deviation = mean |z-score| against the "
                                 "ticked polygons on the same dates; high values are outlier candidates. "
                                 "Click a row to highlight it.")
        self.cmp_tree.itemChanged.connect(self._on_cmp_item_changed)
        self.cmp_tree.itemClicked.connect(
            lambda it, col: self._highlight_compare(it.data(0, Qt.UserRole)))
        list_layout.addWidget(self.cmp_tree)
        btn_row = QHBoxLayout()
        for label, slot in (("All", lambda: self._cmp_set_checked(None)),
                            ("None", lambda: self._cmp_set_checked(set())),
                            ("Map selection", self._cmp_from_map_selection),
                            ("Clear selection", self._cmp_clear_selection)):
            b = QPushButton(label)
            b.clicked.connect(lambda _=False, f=slot: f())
            btn_row.addWidget(b)
        list_layout.addLayout(btn_row)
        splitter.addWidget(list_w)

        self.cmp_plot = PlotPanel()
        p.plot_placeholder(self.cmp_plot.figure, "No analysis yet.")
        self.cmp_plot.canvas.mpl_connect("button_press_event", self._on_cmp_press)
        self.cmp_plot.canvas.mpl_connect("motion_notify_event", self._on_cmp_motion)
        splitter.addWidget(self.cmp_plot)
        splitter.setSizes([170, 520])
        layout.addWidget(splitter, 1)

        self.cmp_info = QLabel("Click a line to highlight it and its polygon on the map; "
                               "select a polygon on the map to highlight its line.")
        self.cmp_info.setWordWrap(True)
        self.cmp_info.setStyleSheet("color:#666")
        layout.addWidget(self.cmp_info)
        return w

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------
    def _run_busy(self, fn, *args, **kwargs):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            return fn(*args, **kwargs)
        finally:
            QApplication.restoreOverrideCursor()

    def _status(self, msg):
        self.status_label.setText(msg)
        QApplication.processEvents()

    def _warn(self, msg):
        self.iface.messageBar().pushWarning(TITLE, msg)

    def _error(self, msg):
        self.iface.messageBar().pushCritical(TITLE, msg)

    def _analysis_layer(self):
        layer = pl.get_or_create_analysis_layer()
        self.source_combo.setExceptedLayerList([layer])
        self._hook_analysis_layer(layer)
        return layer

    def _hook_analysis_layer(self, layer):
        """Map selection -> chart: listen to the analysis layer's selection."""
        if layer is not None and layer.id() != self._hooked_layer_id:
            layer.selectionChanged.connect(self._on_analysis_selection_changed)
            self._hooked_layer_id = layer.id()

    def _ensure_ee(self, force=False):
        project = self.project_edit.text().strip() or None
        try:
            self._status("Connecting to Earth Engine …")
            self._run_busy(c.ensure_initialised, project, force)
        except ImportError as e:
            self._error(str(e))
            self.ee_status.setText("earthengine-api not available.")
            return False
        except Exception as e:
            self._error(f"Earth Engine initialisation failed: {e}. Sign in via the "
                        "Google Earth Engine plugin and check the Cloud project.")
            self.ee_status.setText("Not connected.")
            self._status("Earth Engine not connected.")
            return False
        self.ee_status.setText(f"Connected ✓ (project: {project or c.ee_plugin_project() or 'default'})")
        self._status("Earth Engine connected.")
        return True

    def _date_window(self):
        """-> (start, end_exclusive) as 'YYYY-MM-DD' strings."""
        if self.chk_custom_dates.isChecked():
            start = self.start_date.date()
            end = self.end_date.date().addDays(1)
        else:
            y = self.year_spin.value()
            start, end = QDate(y, 1, 1), QDate(y + 1, 1, 1)
        return start.toString("yyyy-MM-dd"), end.toString("yyyy-MM-dd")

    def _sync_dates_to_year(self, year):
        if not self.chk_custom_dates.isChecked():
            self.start_date.setDate(QDate(year, 1, 1))
            self.end_date.setDate(QDate(year, 12, 31))

    def _on_custom_dates_toggled(self, on):
        self.start_date.setEnabled(on)
        self.end_date.setEnabled(on)
        self.year_spin.setEnabled(not on)
        if not on:
            self._sync_dates_to_year(self.year_spin.value())

    def _set_map_tool(self, tool, checked, other_button):
        if checked:
            if other_button.isChecked():
                other_button.setChecked(False)
            if self.btn_rect.isChecked():
                self.btn_rect.setChecked(False)
            if self.canvas.mapTool() not in (self.draw_tool, self.pick_tool, self.rect_tool):
                self._prev_map_tool = self.canvas.mapTool()
            self.canvas.setMapTool(tool)
        elif self.canvas.mapTool() is tool:
            if self._prev_map_tool is not None:
                self.canvas.setMapTool(self._prev_map_tool)
            else:
                self.canvas.unsetMapTool(tool)

    def _on_map_tool_set(self, new_tool, *_):
        # another QGIS tool (pan, identify, ...) took over -> pop our buttons
        if new_tool is not self.draw_tool and self.btn_draw.isChecked():
            self.btn_draw.setChecked(False)
        if new_tool is not self.pick_tool and self.btn_pick.isChecked():
            self.btn_pick.setChecked(False)
        if new_tool is not self.rect_tool and self.btn_rect.isChecked():
            self.btn_rect.setChecked(False)

    # ------------------------------------------------------------------
    # Polygons tab
    # ------------------------------------------------------------------
    def on_connect(self):
        self._ensure_ee(force=bool(self.project_edit.text().strip()))

    def on_toggle_draw(self, checked):
        if checked:
            layer = self._analysis_layer()
            self.draw_tool.set_colour(c.POLYGON_COLOURS[layer.featureCount() % len(c.POLYGON_COLOURS)])
        self._set_map_tool(self.draw_tool, checked, self.btn_pick)

    def on_toggle_pick(self, checked):
        if checked and not pl.is_polygon_layer(self.source_combo.currentLayer()):
            self._warn("Choose (or open / download) a polygon layer first.")
            self.btn_pick.setChecked(False)
            return
        self._set_map_tool(self.pick_tool, checked, self.btn_draw)
        if checked:
            self.poly_status.setText(f"Click a polygon of '{self.source_combo.currentLayer().name()}'.")

    def on_polygon_drawn(self, geom):
        layer = self._analysis_layer()
        if layer.featureCount() >= MAX_POLYGONS:
            self._warn(f"Maximum {MAX_POLYGONS} polygons. Clear some first.")
            return
        g = pl.to_wgs84(geom, self.canvas.mapSettings().destinationCrs())
        pid = pl.add_polygon(layer, g, source="drawn")
        self.draw_tool.set_colour(c.POLYGON_COLOURS[layer.featureCount() % len(c.POLYGON_COLOURS)])
        self.refresh_polygon_list()
        self.poly_status.setText(f"Polygon {pid} added. Draw another or go to the Analysis tab.")

    def on_source_layer_changed(self, layer):
        self.label_expr.setLayer(layer)
        if layer is None or not hasattr(self, "class_field_edit"):
            return
        names = layer.fields().names()
        crop = self.class_field_edit.text().strip()
        if "fid" in names and crop in names:
            self.label_expr.setExpression(
                f"'Field ' || to_string(\"fid\") || ' (' || coalesce(\"{crop}\", 'NA') || ')'")
        elif "name" in names:
            self.label_expr.setExpression("\"name\"")
        elif "fid" in names:
            self.label_expr.setExpression("'Field ' || to_string(\"fid\")")
        else:
            self.label_expr.setExpression("'" + layer.name().replace("'", "") + " #' || to_string($id)")

    def on_open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open polygon file", "",
            "Vector files (*.shp *.gpkg *.geojson *.json *.kml *.gml *.fgb);;All files (*)")
        if not path:
            return
        layer = QgsVectorLayer(path, os.path.splitext(os.path.basename(path))[0], "ogr")
        if not layer.isValid():
            self._error(f"Could not open {path}")
            return
        if not pl.is_polygon_layer(layer):
            self._error("The file does not contain polygons.")
            return
        QgsProject.instance().addMapLayer(layer)
        self.source_combo.setLayer(layer)
        self.canvas.setExtent(pl.extent_in_canvas_crs(layer, self.canvas))
        self.canvas.refresh()
        if layer.featureCount() <= MAX_POLYGONS:
            self.on_import_features()
        else:
            self.poly_status.setText(
                f"Layer has {layer.featureCount()} polygons (> {MAX_POLYGONS}). Select some and import "
                "'Selected features only', or use 'Click a field on the map'.")

    def on_import_features(self):
        src = self.source_combo.currentLayer()
        if not pl.is_polygon_layer(src):
            self._warn("Choose a polygon layer first.")
            return
        feats = list(src.getSelectedFeatures() if self.chk_selected_only.isChecked() else src.getFeatures())
        if not feats:
            self._warn("No features to import (nothing selected?).")
            return
        layer = self._analysis_layer()
        if layer.featureCount() + len(feats) > MAX_POLYGONS:
            self._warn(f"That would exceed {MAX_POLYGONS} polygons. Select fewer features.")
            return
        pids = pl.import_features(layer, src, feats, self.label_expr.expression())
        self.refresh_polygon_list()
        self.on_zoom_polygons()
        self.poly_status.setText(f"Imported {len(pids)} polygon(s) from '{src.name()}'.")

    def on_feature_picked(self, feat):
        src = self.source_combo.currentLayer()
        if src is None:
            return
        layer = self._analysis_layer()
        expr = self.label_expr.expression()
        name = pl.label_for(src, feat, expr)
        self.canvas.flashFeatureIds(src, [feat.id()])
        keep = self.chk_keep_picked.isChecked()

        existing = pl.find_pid_by_source(layer, pl.source_key(src, feat)) if keep else None
        if existing is not None:
            # clicked again: just switch to it -- no new GEE request
            self.poly_status.setText(f"{name} — already analysed, switched to it.")
            if self._is_analysed(existing):
                self.show_polygon(existing)
                return
            self._focus_pid = existing
        else:
            if not keep:
                pl.clear_polygons(layer)
            elif layer.featureCount() >= MAX_POLYGONS:
                self._warn(f"Maximum {MAX_POLYGONS} polygons.")
                return
            pids = pl.import_features(layer, src, [feat], expr, source="picked")
            self._focus_pid = pids[0] if pids else None
            self.refresh_polygon_list()
            self.poly_status.setText(f"Picked {name}.")
        if self.chk_inspector.isChecked():
            self.on_analyze(from_pick=True)

    def on_asset_preset_changed(self, idx):
        data = self.asset_combo.itemData(idx)
        if data:
            crop_col, year = data
            self.class_field_edit.setText(crop_col)
            if not self.chk_custom_dates.isChecked():
                self.year_spin.setValue(year)

    def on_load_asset(self):
        asset_id = self.asset_combo.currentText().strip()
        if not asset_id:
            return
        if not self._ensure_ee():
            return
        self.btn_load_asset.setEnabled(False)
        try:
            feats = self._run_busy(c.fetch_asset_features, asset_id,
                                   progress_cb=lambda i, n, msg: self._status(msg))
            layer = self._run_busy(pl.add_asset_layer, asset_id, feats,
                                   self.class_field_edit.text().strip() or None)
        except Exception as e:
            self._error(f"Could not load asset {asset_id}: {e}")
            return
        finally:
            self.btn_load_asset.setEnabled(True)
        self.source_combo.setLayer(layer)
        self.canvas.setExtent(pl.extent_in_canvas_crs(layer, self.canvas))
        self.canvas.refresh()
        self._status(f"Loaded {layer.featureCount()} features from {asset_id}. "
                     "Use 'Click a field on the map' to inspect a field.")

    def refresh_polygon_list(self):
        self.poly_list.clear()
        layer = pl.find_analysis_layer()
        self._hook_analysis_layer(layer)
        polys = pl.read_polygons(layer)
        for poly in polys:
            item = QListWidgetItem(QIcon(_swatch(poly["color"])), f"{poly['name']}")
            item.setData(Qt.UserRole, poly["pid"])
            self.poly_list.addItem(item)
        self.poly_list.setToolTip(f"{len(polys)} polygon(s)")

    def on_zoom_polygons(self):
        layer = pl.find_analysis_layer()
        if layer is None or layer.featureCount() == 0:
            return
        layer.updateExtents()
        extent = pl.extent_in_canvas_crs(layer, self.canvas)
        extent.scale(1.2)
        self.canvas.setExtent(extent)
        self.canvas.refresh()

    def on_remove_selected(self):
        layer = pl.find_analysis_layer()
        pids = {it.data(Qt.UserRole) for it in self.poly_list.selectedItems()}
        if layer is not None and pids:
            pl.remove_polygons(layer, pids)
            self.refresh_polygon_list()

    def on_clear_polygons(self):
        layer = pl.find_analysis_layer()
        if layer is not None:
            pl.clear_polygons(layer)
        self.refresh_polygon_list()
        self._cmp_unchecked = set()
        self.draw_tool.set_colour(c.POLYGON_COLOURS[0])
        self.poly_status.setText("All polygons cleared.")

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------
    def _selected_products(self):
        return [k for k, cb in self.product_checks.items() if cb.isChecked()]

    def _current_key(self):
        """Everything that, when changed, invalidates the cached results."""
        start, end = self._date_window()
        return (tuple(self._selected_products()), self.index_combo.currentData(),
                start, end, self.cloud_spin.value())

    def _is_analysed(self, pid):
        return (self.raw_df is not None and self.analysis_meta.get("key") == self._current_key()
                and any(q["pid"] == pid for q in self.analysis_polygons))

    def on_analyze(self, *_, from_pick=False):
        if self._task is not None:
            # e.g. a new field clicked while the previous one is still loading:
            # queue it, the running extraction's results are kept
            self._pending_analyze = True
            self._status("Queued — runs when the current extraction has finished …")
            return
        layer = pl.find_analysis_layer()
        polygons = pl.read_polygons(layer)
        self.refresh_polygon_list()
        if not polygons:
            self._warn("Draw, import or pick at least one polygon first.")
            return
        products = self._selected_products()
        if not products and not self.chk_composites.isChecked():
            self._warn("Nothing to do: select at least one time series or the composites.")
            return
        if not self._ensure_ee():
            return

        start, end = self._date_window()
        if self.chk_composites.isChecked():
            # Inspector clicks keep an existing composite set of the same year
            # instead of rebuilding it around every clicked field.
            self._add_composites(polygons, int(start[:4]), keep_existing=from_pick)
        if not products:
            return

        index = self.index_combo.currentData()
        key = self._current_key()
        # Reuse the cached rows of every polygon that was already extracted
        # with the same settings and an unchanged geometry.
        keep = set()
        if self.raw_df is not None and self.analysis_meta.get("key") == key:
            known = {q["pid"]: q["geojson"] for q in self.analysis_polygons}
            keep = {q["pid"] for q in polygons if known.get(q["pid"]) == q["geojson"]}
        new = [q for q in polygons if q["pid"] not in keep]
        self._pending_meta = {"products": products, "index": index, "start": start, "end": end, "key": key}
        self._pending_polygons = polygons
        self._pending_keep = keep
        if not new:
            self._apply_results(self.raw_df[self.raw_df["pid"].isin(keep)], {})
            return
        params = {
            "polygons": [{"pid": q["pid"], "geojson": q["geojson"]} for q in new],
            "start": start, "end": end, "products": products,
            "index": index, "max_cloud": self.cloud_spin.value(),
        }
        self._task = ExtractTask(params, self._on_task_done)
        self._task.progressText.connect(self._on_task_progress)
        self._task.progressChanged.connect(lambda v: self.progress.setValue(int(v)))
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.show()
        self.btn_cancel.setEnabled(True)
        self.btn_analyze.setText("Analyzing …")
        cached = f" ({len(keep)} cached)" if keep else ""
        self._status(f"Extracting time series for {len(new)} polygon(s){cached}, {start} – {end} …")
        QgsApplication.taskManager().addTask(self._task)

    def _on_task_progress(self, msg):
        self.status_label.setText(msg)

    def on_cancel(self):
        if self._task is not None:
            self._pending_analyze = False
            self._task.cancel()

    def _on_task_done(self, task, ok):
        self._task = None
        self.progress.hide()
        self.btn_cancel.setEnabled(False)
        self.btn_analyze.setText("Analyze")
        if task.exception is not None:
            self._error(f"Extraction failed: {task.exception}")
            self._status("Extraction failed.")
        elif not ok:
            self._status("Extraction cancelled.")
        else:
            old = None
            if self.raw_df is not None and self._pending_keep:
                old = self.raw_df[self.raw_df["pid"].isin(self._pending_keep)]
            frames = [f for f in (old, task.df) if f is not None and len(f)]
            self._apply_results(pd.concat(frames, ignore_index=True) if frames else task.df, task.errors)
        if self._pending_analyze:
            self._pending_analyze = False
            self.on_analyze(from_pick=True)

    def _apply_results(self, df, errors):
        self.raw_df = df
        self.analysis_polygons = self._pending_polygons
        self.analysis_meta = self._pending_meta
        for product, err in errors.items():
            self._warn(f"{c.PRODUCTS[product]['label']}: {err}")

        focus = self._focus_pid if self._focus_pid is not None else self.view_combo.currentData()
        self._focus_pid = None
        self.view_combo.blockSignals(True)
        self.view_combo.clear()
        if len(self.analysis_polygons) > 1:
            self.view_combo.addItem("Combined — all polygons", None)
        for poly in self.analysis_polygons:
            self.view_combo.addItem(QIcon(_swatch(poly["color"])), poly["name"], poly["pid"])
        idx = self.view_combo.findData(focus) if focus is not None else 0
        self.view_combo.setCurrentIndex(max(idx, 0))
        self.view_combo.blockSignals(False)
        self._populate_compare_list()
        self._refresh_charts()
        if self.tabs.currentIndex() not in (2, 3):
            self.tabs.setCurrentIndex(2)
        self._show_section(0)
        if focus is not None:
            self._select_on_map([focus])
        n_img = 0 if self.raw_df is None or len(self.raw_df) == 0 else self.raw_df["date"].nunique()
        self._status(f"Analysis complete: {n_img} acquisition dates "
                     f"{self.analysis_meta['start']} – {self.analysis_meta['end']}.")

    def _series_keys(self):
        keys = []
        for product in self.analysis_meta.get("products", []):
            keys.extend(c.PRODUCTS[product]["series"])
        return [k for k in c.SERIES if k in keys]

    def _filtered(self):
        if self.raw_df is None:
            return None
        return c.filter_series(self.raw_df, self.valid_spin.value() / 100.0)

    def _date_range(self):
        start = pd.Timestamp(self.analysis_meta["start"])
        end = pd.Timestamp(self.analysis_meta["end"]) - pd.Timedelta(days=1)
        return start, end

    def _refresh_charts(self):
        if self.raw_df is None:
            return
        self._filtered_df = self._filtered()
        n = 0 if self._filtered_df is None else len(self._filtered_df)
        self.chart_info.setText(
            f"{len(self.analysis_polygons)} polygon(s) · {self.analysis_meta['start']} – "
            f"{self.analysis_meta['end']} · {n} observations with ≥ {self.valid_spin.value()} % "
            f"valid pixels (of {len(self.raw_df)} raw).")
        self._redraw_chart()
        self._redraw_compare()

    def _draw_view(self, fig, pid):
        keys = self._series_keys()
        index = self.analysis_meta["index"]
        if pid is None:
            p.plot_combined(fig, self._filtered_df, keys, self.analysis_polygons, index, self._date_range())
        else:
            name = next((q["name"] for q in self.analysis_polygons if q["pid"] == pid), f"Polygon {pid}")
            p.plot_single_polygon(fig, self._filtered_df, keys, pid, name, index, self._date_range())
        return len(keys)

    def _redraw_chart(self):
        if self.raw_df is None or self.view_combo.count() == 0:
            return
        n = self._draw_view(self.plot.figure, self.view_combo.currentData())
        self.plot.set_panels(n)
        self.plot.redraw()

    def _on_view_changed(self, *_):
        self._redraw_chart()
        pid = self.view_combo.currentData()
        if pid is not None:
            self._select_on_map([pid])

    def _step_view(self, step):
        n = self.view_combo.count()
        if n:
            self.view_combo.setCurrentIndex((self.view_combo.currentIndex() + step) % n)

    def show_polygon(self, pid, switch_tab=True):
        """Switch every view to one polygon: Charts combo, Compare highlight, map."""
        if pid is None:
            return
        idx = self.view_combo.findData(pid)
        if idx >= 0 and self.view_combo.currentIndex() != idx:
            self.view_combo.setCurrentIndex(idx)          # redraws + selects on map
        else:
            self._select_on_map([pid])
        self._highlight_compare(pid, select_on_map=False)
        if switch_tab and idx >= 0 and self.tabs.currentIndex() not in (2, 3):
            self.tabs.setCurrentIndex(2)
            self._show_section(0)

    def _poly_by_pid(self, pid):
        return next((q for q in self.analysis_polygons if q["pid"] == pid), None)

    def _select_on_map(self, pids, flash=True):
        layer = pl.find_analysis_layer()
        if layer is None:
            return
        ids = pl.feature_ids_for_pids(layer, set(pids))
        self._syncing = True
        try:
            layer.selectByIds(ids)
        finally:
            self._syncing = False
        if ids and flash:
            self.canvas.flashFeatureIds(layer, ids)

    def _pan_to(self, pid):
        poly = self._poly_by_pid(pid)
        if poly is None:
            return
        xform = QgsCoordinateTransform(pl.WGS84, self.canvas.mapSettings().destinationCrs(),
                                       QgsProject.instance())
        r = xform.transformBoundingBox(poly["bbox"])
        if not self.canvas.extent().contains(r):
            self.canvas.setCenter(r.center())
            self.canvas.refresh()

    def _on_analysis_selection_changed(self, *_):
        """Polygon selected on the map (select tool, attribute table, ...)
        -> show it in the Charts tab and highlight it on the Compare tab."""
        if self._syncing or self.raw_df is None:
            return
        layer = pl.find_analysis_layer()
        if layer is None:
            return
        pids = [f["pid"] for f in layer.getSelectedFeatures()]
        if len(pids) != 1 or self._poly_by_pid(pids[0]) is None:
            return
        pid = pids[0]
        self._syncing = True
        try:
            idx = self.view_combo.findData(pid)
            if idx >= 0:
                self.view_combo.setCurrentIndex(idx)
            self._highlight_compare(pid, select_on_map=False)
        finally:
            self._syncing = False

    # -- Compare tab ----------------------------------------------------------
    def _populate_compare_list(self):
        self.cmp_tree.blockSignals(True)
        self.cmp_tree.setSortingEnabled(False)
        self.cmp_tree.clear()
        for q in self.analysis_polygons:
            it = _ScoreItem([q["name"], "–"])
            it.setIcon(0, QIcon(_swatch(q["color"])))
            it.setData(0, Qt.UserRole, q["pid"])
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(0, Qt.Unchecked if q["pid"] in self._cmp_unchecked else Qt.Checked)
            self.cmp_tree.addTopLevelItem(it)
        self.cmp_tree.setSortingEnabled(True)
        self.cmp_tree.sortByColumn(1, Qt.DescendingOrder)
        self.cmp_tree.resizeColumnToContents(1)
        self.cmp_tree.blockSignals(False)

        current = self.cmp_series_combo.currentData()
        index = self.analysis_meta.get("index", "")
        self.cmp_series_combo.blockSignals(True)
        self.cmp_series_combo.clear()
        for key in self._series_keys():
            meta = c.SERIES[key]
            self.cmp_series_combo.addItem(
                f"{meta['label']} {index}" if meta["unit"] is None else meta["label"], key)
        self.cmp_series_combo.addItem("All series (stacked)", "ALL")
        i = self.cmp_series_combo.findData(current)
        self.cmp_series_combo.setCurrentIndex(i if i >= 0 else 0)
        self.cmp_series_combo.blockSignals(False)

    def _cmp_series(self):
        key = self.cmp_series_combo.currentData()
        keys = self._series_keys()
        if key == "ALL" or key not in keys:
            return keys
        return [key]

    def _cmp_polygons(self):
        return [q for q in self.analysis_polygons if q["pid"] not in self._cmp_unchecked]

    def _on_cmp_item_changed(self, item, column):
        if column != 0:
            return
        pid = item.data(0, Qt.UserRole)
        if item.checkState(0) == Qt.Checked:
            self._cmp_unchecked.discard(pid)
        else:
            self._cmp_unchecked.add(pid)
        self._redraw_compare()

    def _cmp_set_checked(self, pids):
        """pids=None -> all."""
        all_pids = {q["pid"] for q in self.analysis_polygons}
        self._cmp_unchecked = set() if pids is None else all_pids - set(pids)
        self._populate_compare_list()
        self._redraw_compare()

    def _cmp_from_map_selection(self):
        layer = pl.find_analysis_layer()
        pids = {f["pid"] for f in layer.getSelectedFeatures()} if layer is not None else set()
        if not pids:
            self._warn("Select polygons of 'GEE Analysis Polygons' on the map first "
                       "(select tool or attribute table).")
            return
        self._cmp_set_checked(pids)

    def _cmp_clear_selection(self):
        """Removes the red highlight, the list selection and the polygon
        selection on the map; the ticked polygons stay charted."""
        self._highlight_compare(None, select_on_map=False, pan=False)
        layer = pl.find_analysis_layer()
        if layer is not None:
            self._syncing = True
            try:
                layer.removeSelection()
            finally:
                self._syncing = False

    def _update_compare_scores(self, keys, pids):
        scores = c.deviation_scores(self._filtered_df, keys, pids)
        self.cmp_tree.blockSignals(True)
        self.cmp_tree.setSortingEnabled(False)
        for i in range(self.cmp_tree.topLevelItemCount()):
            it = self.cmp_tree.topLevelItem(i)
            s = scores.get(it.data(0, Qt.UserRole))
            if s is None or s != s:
                it.setText(1, "–")
                it.setData(1, Qt.UserRole, -1.0)
            else:
                it.setText(1, f"{s:.2f}")
                it.setData(1, Qt.UserRole, float(s))
        self.cmp_tree.setSortingEnabled(True)
        self.cmp_tree.blockSignals(False)

    def _redraw_compare(self):
        if self.raw_df is None or not self.analysis_meta:
            return
        keys = self._cmp_series()
        polys = self._cmp_polygons()
        self._update_compare_scores(keys, [q["pid"] for q in polys])
        self._cmp_artists = p.plot_compare(
            self.cmp_plot.figure, self._filtered_df, keys, polys, self.analysis_meta["index"],
            self._date_range(), uniform=self.chk_cmp_uniform.isChecked(),
            show_band=self.chk_cmp_band.isChecked())
        self.cmp_plot.set_panels(max(len(keys), 2))
        if self._cmp_highlight in self._cmp_artists.get("lines", {}):
            self._highlight_compare(self._cmp_highlight, select_on_map=False, pan=False)
        else:
            self._cmp_highlight = None
        self.cmp_plot.redraw()

    def _highlight_compare(self, pid, select_on_map=True, pan=True):
        lines = self._cmp_artists.get("lines", {})
        if pid is not None and pid not in lines:
            pid = None
        poly = self._poly_by_pid(pid) if pid is not None else None
        p.highlight_compare(self._cmp_artists, pid, f"▶ {poly['name']}" if poly else "")
        self._cmp_highlight = pid
        self.cmp_plot.redraw()

        self.cmp_tree.blockSignals(True)
        for i in range(self.cmp_tree.topLevelItemCount()):
            it = self.cmp_tree.topLevelItem(i)
            if it.data(0, Qt.UserRole) == pid:
                self.cmp_tree.setCurrentItem(it)
                self.cmp_tree.scrollToItem(it)
                break
        else:
            self.cmp_tree.clearSelection()
        self.cmp_tree.blockSignals(False)

        if poly is None:
            self.cmp_info.setText("Click a line to highlight it and its polygon on the map.")
            return
        score = next((self.cmp_tree.topLevelItem(i).text(1) for i in range(self.cmp_tree.topLevelItemCount())
                      if self.cmp_tree.topLevelItem(i).data(0, Qt.UserRole) == pid), "–")
        self.cmp_info.setText(f"Highlighted: {poly['name']} · deviation {score} σ")
        if select_on_map:
            self._select_on_map([pid])
        if pan and self.chk_cmp_pan.isChecked():
            self._pan_to(pid)

    def _cmp_line_at(self, event):
        if event.inaxes is None:
            return None
        candidates = [ln for ls in self._cmp_artists.get("lines", {}).values() for ln in ls
                      if ln.axes is event.inaxes]
        for ln in sorted(candidates, key=lambda l: l.get_zorder(), reverse=True):
            if ln.contains(event)[0]:
                return int(ln.get_gid())
        return None

    def _on_cmp_press(self, event):
        if event.button != 1 or event.inaxes is None or self.cmp_plot.toolbar.mode:
            return
        self._highlight_compare(self._cmp_line_at(event))

    def _on_cmp_motion(self, event):
        pid = self._cmp_line_at(event)
        if pid == self._cmp_hover:
            return
        self._cmp_hover = pid
        poly = self._poly_by_pid(pid) if pid is not None else None
        self.cmp_plot.canvas.setToolTip(poly["name"] if poly else "")

    # ------------------------------------------------------------------
    # Sentinel-2 composites
    # ------------------------------------------------------------------
    def on_composites_only(self):
        polygons = pl.read_polygons(pl.find_analysis_layer())
        if self.radio_extent_polys.isChecked() and not polygons:
            self._warn("No polygons — draw some or use 'Current map view' as extent.")
            return
        if not self._ensure_ee():
            return
        start, _ = self._date_window()
        self._add_composites(polygons, int(start[:4]))

    def _composite_bbox(self, polygons):
        if self.radio_extent_view.isChecked() or not polygons:
            ext = self.canvas.extent()
            xform = QgsCoordinateTransform(self.canvas.mapSettings().destinationCrs(), pl.WGS84,
                                           QgsProject.instance())
            r = xform.transformBoundingBox(ext)
            return (r.xMinimum(), r.yMinimum(), r.xMaximum(), r.yMaximum()), 0
        r = QgsRectangle(polygons[0]["bbox"])
        for q in polygons[1:]:
            r.combineExtentWith(q["bbox"])
        return (r.xMinimum(), r.yMinimum(), r.xMaximum(), r.yMaximum()), self.buffer_spin.value() * 1000

    def _add_composites(self, polygons, year, keep_existing=False):
        """The 'May, July & September RGB layers' of the original app, for any
        months and rendering. Like the original, nothing is reloaded if the
        same set is already on the map; otherwise the old set is replaced.
        Only the first month is switched on; the others are in the group but
        not rendered until they are ticked."""
        months = [i + 1 for i, cb in enumerate(self.month_checks) if cb.isChecked()]
        if not months:
            return
        mode = self.comp_mode_combo.currentData()
        bbox, buffer_m = self._composite_bbox(polygons)
        key = (mode, year, tuple(months), tuple(round(v, 4) for v in bbox), buffer_m)
        names = [c.composite_layer_name(mode, year, m) for m in months]
        project = QgsProject.instance()
        if all(project.mapLayersByName(n) for n in names) and (keep_existing or key == self._composite_key):
            return

        root = project.layerTreeRoot()
        group = root.findGroup(COMPOSITE_GROUP)
        if group is None:
            group = root.insertGroup(0, COMPOSITE_GROUP)
        for node in group.findLayers():
            project.removeMapLayer(node.layerId())

        region = c.bbox_geometry(bbox, buffer_m)
        for i, (month, name) in enumerate(zip(months, names)):
            self._status(f"Adding {name} ({i + 1}/{len(months)}) …")
            try:
                img, vis = c.s2_monthly_composite(region, year, month, mode)
                layer = self._run_busy(self._add_ee_layer, img, vis, name, i == 0)
            except Exception as e:
                self._warn(f"{name}: {e}")
                continue
            self._move_into_group(layer, group, i)
            node = root.findLayer(layer.id()) if layer is not None else None
            if node is not None:
                node.setItemVisibilityChecked(i == 0)
        self._composite_key = key
        self._status(f"Added {len(months)} Sentinel-2 composite layer(s) for {year} "
                     "(only the first is shown; tick the others in the layer panel).")

    # -- Composite export ---------------------------------------------------
    def on_toggle_rect(self, checked):
        if checked:
            for b in (self.btn_draw, self.btn_pick):
                b.setChecked(False)
            if self.canvas.mapTool() not in (self.draw_tool, self.pick_tool, self.rect_tool):
                self._prev_map_tool = self.canvas.mapTool()
            self.canvas.setMapTool(self.rect_tool)
        elif self.canvas.mapTool() is self.rect_tool:
            if self._prev_map_tool is not None:
                self.canvas.setMapTool(self._prev_map_tool)
            else:
                self.canvas.unsetMapTool(self.rect_tool)

    def on_export_rect_drawn(self, rect):
        if rect.isEmpty():
            return
        xform = QgsCoordinateTransform(self.canvas.mapSettings().destinationCrs(), pl.WGS84,
                                       QgsProject.instance())
        self._export_rect = xform.transformBoundingBox(rect)
        r = self._export_rect
        km_w = (r.xMaximum() - r.xMinimum()) * 111.32 * abs(math.cos(math.radians(r.center().y())))
        km_h = (r.yMaximum() - r.yMinimum()) * 110.57
        self.rect_label.setText(f"{km_w:.1f} × {km_h:.1f} km")
        self.rect_tool.clearRubberBand()
        self.btn_rect.setChecked(False)
        self._show_rect_band()
        self._update_export_summary()

    def _show_rect_band(self):
        """Keeps the drawn rectangle visible while it is the export area."""
        if self._rect_band is None:
            self._rect_band = QgsRubberBand(self.canvas, QgsWkbTypes.PolygonGeometry)
            self._rect_band.setStrokeColor(QColor("#e31a1c"))
            self._rect_band.setFillColor(QColor(227, 26, 28, 30))
            self._rect_band.setWidth(2)
        self._rect_band.reset(QgsWkbTypes.PolygonGeometry)
        if self._export_rect is not None and self.exp_extent_kind.currentIndex() == 1:
            self._rect_band.setToGeometry(QgsGeometry.fromRect(self._export_rect), pl.WGS84)

    def on_export_open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open polygon file", "",
            "Vector files (*.shp *.gpkg *.geojson *.json *.kml *.gml *.fgb);;All files (*)")
        if not path:
            return
        layer = QgsVectorLayer(path, os.path.splitext(os.path.basename(path))[0], "ogr")
        if not layer.isValid() or not pl.is_polygon_layer(layer):
            self._error(f"{path} is not a polygon layer.")
            return
        QgsProject.instance().addMapLayer(layer)
        self.exp_layer.setLayer(layer)

    def _export_region(self):
        kind = self.exp_extent_kind.currentIndex()
        if kind == 0:
            layer = self.exp_layer.currentLayer()
            if not pl.is_polygon_layer(layer):
                raise ValueError("Choose (or open) a polygon layer for the export area.")
            ee = c.import_ee()
            return ee.Geometry(pl.layer_union_geojson(layer, self.chk_exp_selected.isChecked()), None, False)
        if kind == 1:
            if self._export_rect is None:
                raise ValueError("Draw a rectangle on the map first.")
            r = self._export_rect
            return c.bbox_geometry((r.xMinimum(), r.yMinimum(), r.xMaximum(), r.yMaximum()))
        xform = QgsCoordinateTransform(self.canvas.mapSettings().destinationCrs(), pl.WGS84,
                                       QgsProject.instance())
        r = xform.transformBoundingBox(self.canvas.extent())
        return c.bbox_geometry((r.xMinimum(), r.yMinimum(), r.xMaximum(), r.yMaximum()))

    def _export_args(self):
        f = self.exp_recipe.features()
        content = self.exp_content.currentData()
        problem = self.exp_recipe.problem(f, need_variables=content == "data")
        if problem:
            raise ValueError(problem)
        prefix = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in self.exp_prefix.text().strip()) or "s2"
        return {"region": self._export_region(), "years": self.exp_recipe.years(), "features": f,
                "content": content, "layout": self.exp_layout.currentData(),
                "dtype": self.exp_dtype.currentData(), "prefix": prefix,
                "worldcover": self.chk_exp_wc.isChecked(),
                "clip": self.chk_exp_clip.isChecked() and self.exp_extent_kind.currentIndex() == 0,
                "scale": f["scale"], "crs": self.exp_crs.text().strip() or None}

    def on_export_composites_download(self):
        if self._export_task is not None:
            return
        if not self._ensure_ee():
            return
        try:
            kwargs = self._export_args()
        except ValueError as e:
            self._warn(str(e))
            return
        folder = QFileDialog.getExistingDirectory(self, "Folder for the exported GeoTIFFs")
        if not folder:
            return
        kwargs["folder"] = folder
        self._export_task = CallTask("Downloading composites", k.download_export, kwargs,
                                     self._on_export_done)
        self._export_task.progressText.connect(self.status_label.setText)
        self._export_task.progressChanged.connect(lambda v: self.progress.setValue(int(v)))
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.show()
        self.btn_exp_download.setEnabled(False)
        self.btn_exp_cancel.setEnabled(True)
        self._status(f"Downloading {len(k.export_plan(kwargs['years'], kwargs['features'], kwargs['content'], kwargs['layout'], kwargs['prefix']))} file(s) …")
        QgsApplication.taskManager().addTask(self._export_task)

    def _on_export_done(self, task, ok):
        self._export_task = None
        if self._task is None:
            self.progress.hide()
        self.btn_exp_download.setEnabled(True)
        self.btn_exp_cancel.setEnabled(False)
        if task.exception is not None:
            self._error(f"Composite download failed: {task.exception}")
            self._status("Composite download failed.")
            return
        if not ok:
            self._status("Composite download cancelled.")
            return
        project = QgsProject.instance()
        root = project.layerTreeRoot()
        group = root.findGroup(COMPOSITE_EXPORT_GROUP) or root.insertGroup(0, COMPOSITE_EXPORT_GROUP)
        for i, path in enumerate(task.result):
            layer = QgsRasterLayer(path, os.path.splitext(os.path.basename(path))[0])
            if not layer.isValid():
                continue
            project.addMapLayer(layer, False)
            node = group.insertLayer(i, layer)
            node.setItemVisibilityChecked(i == 0)
        folder = os.path.dirname(task.result[0]) if task.result else ""
        self._status(f"Saved {len(task.result)} GeoTIFF(s) + JSON sidecar(s) to {folder}.")
        self.iface.messageBar().pushSuccess(TITLE, f"Saved {len(task.result)} file(s) to {folder}")

    def on_export_composites_drive(self):
        if not self._ensure_ee():
            return
        try:
            kwargs = self._export_args()
        except ValueError as e:
            self._warn(str(e))
            return
        try:
            ids = self._run_busy(k.drive_export, folder=self.exp_drive_folder.text().strip(), **kwargs)
        except Exception as e:
            self._error(f"Export failed: {e}")
            return
        self.iface.messageBar().pushSuccess(
            TITLE, f"Started {len(ids)} export task(s) → Drive folder '{self.exp_drive_folder.text().strip()}'. "
            "Follow them at https://code.earthengine.google.com/tasks")

    def _add_ee_layer(self, image, vis, name, shown=True):
        """Prefer the Google Earth Engine plugin's Map.addLayer (its layers are
        re-authenticated when a project is reopened); fall back to a plain
        XYZ layer from getMapId."""
        try:
            from ee_plugin import Map as ee_map
        except ImportError:
            ee_map = None
        if ee_map is not None:
            layer = ee_map.addLayer(image, vis, name, shown)
            if layer is None:
                found = QgsProject.instance().mapLayersByName(name)
                layer = found[0] if found else None
            return layer
        mapid = image.getMapId(vis)
        url = mapid["tile_fetcher"].url_format
        layer = QgsRasterLayer(f"type=xyz&url={quote(url, safe=':/{}')}&zmin=0&zmax=20", name, "wms")
        QgsProject.instance().addMapLayer(layer)
        node = QgsProject.instance().layerTreeRoot().findLayer(layer.id())
        if node is not None:
            node.setItemVisibilityChecked(shown)
        return layer

    @staticmethod
    def _move_into_group(layer, group, position):
        if layer is None:
            return
        root = QgsProject.instance().layerTreeRoot()
        node = root.findLayer(layer.id())
        if node is None:
            group.insertChildNode(position, QgsLayerTreeLayer(layer))
            return
        if node.parent() is group:
            return
        clone = node.clone()
        group.insertChildNode(position, clone)
        node.parent().removeChildNode(node)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def _export_table(self):
        df = self._filtered_df.copy()
        names = {q["pid"]: q["name"] for q in self.analysis_polygons}
        index = self.analysis_meta["index"]
        df.insert(0, "polygon", df["pid"].map(names))
        df["series"] = df["series"].map(
            lambda k: f"{c.SERIES[k]['label']} {index}" if c.SERIES[k]["unit"] is None else c.SERIES[k]["label"])
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        return df.rename(columns={"valid": "valid_fraction"})

    def on_export_csv(self):
        if self.raw_df is None or self._filtered_df is None or len(self._filtered_df) == 0:
            self._warn("No time series to export yet.")
            return
        default = f"gee_timeseries_{self.analysis_meta['start']}_{self.analysis_meta['end']}.csv"
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", default, "CSV files (*.csv)")
        if not path:
            return
        self._export_table().to_csv(path, index=False)
        self.iface.messageBar().pushSuccess(TITLE, f"Saved {path}")

    def on_export_png(self):
        if self.raw_df is None:
            self._warn("No charts to export yet.")
            return
        folder = QFileDialog.getExistingDirectory(self, "Folder for chart PNGs")
        if not folder:
            return
        views = [None] if len(self.analysis_polygons) > 1 else []
        views += [q["pid"] for q in self.analysis_polygons]
        names = {q["pid"]: q["name"] for q in self.analysis_polygons}
        for pid in views:
            fig = Figure(figsize=(8, 2.0 * max(1, len(self._series_keys())) + 0.4), dpi=150)
            self._draw_view(fig, pid)
            label = "combined" if pid is None else f"{pid:03d}_{names[pid]}"
            safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in label)
            fig.savefig(os.path.join(folder, f"{safe}.png"))
        self.iface.messageBar().pushSuccess(TITLE, f"Saved {len(views)} chart(s) to {folder}")

    # ------------------------------------------------------------------
    def closeEvent(self, event):
        for tool in (self.draw_tool, self.pick_tool, self.rect_tool):
            if self.canvas.mapTool() is tool:
                self.canvas.unsetMapTool(tool)
        self.btn_draw.setChecked(False)
        self.btn_pick.setChecked(False)
        self.btn_rect.setChecked(False)
        super().closeEvent(event)

    def cleanup(self):
        """Called on plugin unload."""
        try:
            self.canvas.mapToolSet.disconnect(self._on_map_tool_set)
        except TypeError:
            pass
        if self._task is not None:
            self._task.cancel()
        self.classify_tab.cleanup()
        self.rotation_tab.cleanup()
        if self._export_task is not None:
            self._export_task.cancel()
        if self._rect_band is not None:
            self.canvas.scene().removeItem(self._rect_band)
            self._rect_band = None
        for tool in (self.draw_tool, self.pick_tool, self.rect_tool):
            if self.canvas.mapTool() is tool:
                self.canvas.unsetMapTool(tool)
        self.draw_tool.reset()
