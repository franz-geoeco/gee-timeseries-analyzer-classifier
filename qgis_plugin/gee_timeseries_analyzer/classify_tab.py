"""The Classification tab: supervised crop classification in Earth Engine
(port of a Code Editor crop-classification script) driven from
QGIS. Training polygons and the classification area come from QGIS layers
or GEE table assets; features, aggregation and classifier are selectable;
the trained model can be saved as JSON and applied to other years.

Two engines: Earth Engine (online, classification.py) or a local raster
stack with scikit-learn (offline, local_classifier.py). Feature stacks for
the offline mode are downloaded here once; models of both engines share one
file format.

  Setup    -- training data + classes, classification area + mask, feature
              recipe, classifier, train / save / load / apply
  Results  -- accuracy summary, confusion matrices, per-class accuracy,
              variable importance, map / CSV exports

Earth Engine work runs in a QgsTask (tasks.CallTask); classification.py
holds the Earth Engine side.
"""
import json
import os
from collections import Counter

from qgis.core import (
    QgsApplication, QgsProject, QgsCoordinateTransform, QgsRasterLayer, QgsVectorLayer,
    QgsPalettedRasterRenderer, QgsField, QgsFields, QgsFeature, QgsWkbTypes,
    QgsCategorizedSymbolRenderer, QgsRendererCategory, NULL,
)
from qgis.PyQt.QtCore import QVariant
from qgis.gui import QgsMapLayerComboBox, QgsCollapsibleGroupBox
from qgis.PyQt.QtCore import Qt, QUrl
from qgis.PyQt.QtGui import QColor, QBrush, QDesktopServices
from qgis.PyQt.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout, QTabWidget,
    QPushButton, QLabel, QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox,
    QGroupBox, QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QFileDialog, QColorDialog, QProgressBar, QStackedWidget, QListWidget, QListWidgetItem,
    QSplitter,
)

from . import classification as k
from . import gee_client as c
from . import local_classifier as lc
from . import analysis_tools as at
from . import plotting as p
from . import polygon_layer as pl
from . import report
from . import style
from . import help_texts as H
from . import tree_export
from .recipe_widget import FeatureRecipeWidget
from .tasks import CallTask

CLASS_GROUP = "GEE classification"
MAX_AOI_VERTICES = 50000

# (asset id, class field, name field, split field)
# Empty on purpose -- users type their own asset ids.
TRAINING_PRESETS = []
AOI_PRESETS = []
ASSET_PLACEHOLDER = "projects/<your-project>/assets/<table>"

COL_VALUE, COL_CODE, COL_NAME, COL_COLOUR, COL_N = range(5)


def _attr(v):
    return None if v is None or v == NULL else v


def _field_majority_local(raster_path, polygons, progress_cb=None, cancel_cb=None):
    return at.field_majority(raster_path, polygons, cancel_cb=cancel_cb)


class ClassificationTab(QWidget):
    def __init__(self, dock):
        super().__init__()
        self.dock = dock
        self.iface = dock.iface
        self.canvas = dock.canvas
        self.classes = []          # class table of the training data
        self._classes_key = None   # training source + fields the class table was loaded for
        self._value_counts = {}    # last "Load classes": {value: n}, {value key: name}
        self._value_names = {}
        self.model = None          # last trained or loaded model
        self.pool_models = []      # (file name, model) whose samples are pooled into training
        self.excluded = set()      # training polygon ids excluded after the training check
        self._excluded_key = None  # training source the exclusions belong to
        self._train_layer_used = None   # QGIS layer of the last training run (training check)
        self._suspicious = None
        self.result = None         # last train / apply result
        self._task = None
        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.inner = QTabWidget()
        self.inner.setDocumentMode(True)
        self.inner.addTab(self.dock._scrolled(self._build_setup()), "Setup")
        self.inner.addTab(self._build_results(), "Results")
        self.inner.addTab(self._build_profiles_tab(), "Class profiles")
        self.inner.addTab(self._build_check_tab(), "Training check")
        layout.addWidget(self.inner, 1)
        self._on_engine_changed()

        row = QHBoxLayout()
        self.status = QLabel("Choose training polygons and a classification area.")
        self.status.setWordWrap(True)
        self.progress = QProgressBar()
        self.progress.setMaximumHeight(14)
        self.progress.setTextVisible(False)
        self.progress.hide()
        row.addWidget(self.status, 1)
        row.addWidget(self.progress)
        layout.addLayout(row)

    def _build_setup(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.addWidget(self._build_engine_box())
        layout.addWidget(self._build_training_box())
        layout.addWidget(self._build_aoi_box())
        layout.addWidget(self._build_features_box())
        layout.addWidget(self._build_classifier_box())

        run_row = QHBoxLayout()
        self.btn_train = QPushButton("Train && classify")
        style.primary(self.btn_train)
        self.btn_train.clicked.connect(self.on_train)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.on_cancel)
        run_row.addWidget(self.btn_train, 1)
        run_row.addWidget(self.btn_cancel)
        layout.addLayout(run_row)

        layout.addWidget(self._build_model_box())
        layout.addStretch(1)
        return w

    def _build_engine_box(self):
        box = QGroupBox("Engine")
        layout = QVBoxLayout(box)
        self.engine = QComboBox()
        self.engine.addItem("Earth Engine (online)", "ee")
        self.engine.addItem("Local raster stack (offline, scikit-learn)", "local")
        layout.addWidget(self.engine)

        self.local_box = QWidget()
        form = QFormLayout(self.local_box)
        form.setContentsMargins(0, 4, 0, 0)
        hint = QLabel("Offline: training, accuracy and classification run on local GeoTIFF stacks "
                      "(band names = features). Download a feature stack once (uses Earth Engine, the "
                      "year and features below, the classification area and the training polygons) "
                      "or use any stack, e.g. one written by Export imagery.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#666; font-size:11px")
        form.addRow(hint)
        drow = QHBoxLayout()
        self.stack_extent = QComboBox()
        for key, label in k.STACK_EXTENTS.items():
            self.stack_extent.addItem(label, key)
        self.stack_extent.setCurrentIndex(2)
        self.btn_stack_download = QPushButton("Download feature stack…")
        self.btn_stack_download.clicked.connect(self.on_download_stack)
        drow.addWidget(self.stack_extent, 1)
        drow.addWidget(self.btn_stack_download)
        form.addRow("Stack area:", drow)

        self.train_raster = self._raster_combo(allow_empty=False)
        form.addRow("Training stack:", self._with_open(self.train_raster))
        self.pred_raster = self._raster_combo(allow_empty=True)
        self.pred_raster.setToolTip("Raster to classify; empty = the training stack")
        form.addRow("Stack to classify:", self._with_open(self.pred_raster))
        self.stack_info_label = QLabel()
        self.stack_info_label.setWordWrap(True)
        self.stack_info_label.setStyleSheet("color:#666")
        form.addRow(self.stack_info_label)
        for combo in (self.train_raster, self.pred_raster):
            combo.layerChanged.connect(lambda *_: self._update_stack_info())
        layout.addWidget(self.local_box)
        self.engine.currentIndexChanged.connect(self._on_engine_changed)
        return box

    def _raster_combo(self, allow_empty):
        from .dock_widget import RASTER_FILTER
        combo = QgsMapLayerComboBox()
        combo.setFilters(RASTER_FILTER)
        combo.setAllowEmptyLayer(allow_empty)
        if allow_empty:
            combo.setLayer(None)
        return combo

    def _with_open(self, combo):
        w = QWidget()
        row = QHBoxLayout(w)
        row.setContentsMargins(0, 0, 0, 0)
        btn = QPushButton("Open…")
        btn.clicked.connect(lambda: self._open_raster(combo))
        row.addWidget(combo, 1)
        row.addWidget(btn)
        return w

    def _open_raster(self, combo):
        path, _ = QFileDialog.getOpenFileName(self, "Open raster stack", "",
                                              "GeoTIFF (*.tif *.tiff);;All files (*)")
        if path:
            combo.setLayer(self._add_raster(path))

    @staticmethod
    def _add_raster(path, shown=True):
        project = QgsProject.instance()
        for lyr in project.mapLayers().values():
            if isinstance(lyr, QgsRasterLayer) and os.path.normcase(lyr.source()) == os.path.normcase(path):
                return lyr
        layer = QgsRasterLayer(path, os.path.splitext(os.path.basename(path))[0])
        if not layer.isValid():
            raise RuntimeError(f"QGIS could not open {path}")
        project.addMapLayer(layer)
        node = project.layerTreeRoot().findLayer(layer.id())
        if node is not None:
            node.setItemVisibilityChecked(shown)
        return layer

    def _local(self):
        return self.engine.currentData() == "local"

    def _on_engine_changed(self, *_):
        local = self._local()
        self.local_box.setVisible(local)
        self.btn_train.setText("Train && classify stack" if local else "Train && classify")
        self.apply_year.setEnabled(not local)
        # offline, the features come from the stack's bands; the recipe and
        # the area only matter for "Download feature stack"
        self.features_box.setTitle("Features — only for 'Download feature stack'" if local
                                   else "Features (time-series composites)")
        self.features_box.setCollapsed(local)
        self.aoi_box.setTitle("Stack download area & mask" if local else "Classification area")
        self._update_stack_info()

    @staticmethod
    def _stack_path(combo):
        layer = combo.currentLayer()
        if layer is None:
            return None
        return layer.source().split("|")[0]

    def _update_stack_info(self):
        lines = []
        for label, combo in (("Training", self.train_raster), ("Classify", self.pred_raster)):
            path = self._stack_path(combo)
            if path is None:
                continue
            try:
                lines.append(f"{label}: {lc.describe_stack(lc.stack_info(path))}")
            except Exception as e:
                lines.append(f"{label}: {e}")
        self.stack_info_label.setText("<br>".join(lines))

    @staticmethod
    def _source_combo(items):
        combo = QComboBox()
        for label in items:
            combo.addItem(label)
        return combo

    def _build_training_box(self):
        box = QGroupBox("Training data")
        form = QFormLayout(box)

        self.train_kind = self._source_combo(["QGIS layer", "GEE table asset"])
        form.addRow("Source:", self.train_kind)
        self.train_stack = QStackedWidget()
        lw = QWidget()
        lrow = QHBoxLayout(lw)
        lrow.setContentsMargins(0, 0, 0, 0)
        self.train_layer = QgsMapLayerComboBox()
        self.train_layer.setFilters(self._polygon_filter())
        self.train_layer.setAllowEmptyLayer(True)
        self.train_layer.setLayer(None)
        self.train_layer.layerChanged.connect(self._on_train_layer_changed)
        self.chk_train_selected = QCheckBox("Selected only")
        lrow.addWidget(self.train_layer, 1)
        lrow.addWidget(self.chk_train_selected)
        aw = QWidget()
        arow = QHBoxLayout(aw)
        arow.setContentsMargins(0, 0, 0, 0)
        self.train_asset = QComboBox()
        self.train_asset.setEditable(True)
        for asset, cls, name, split in TRAINING_PRESETS:
            self.train_asset.addItem(asset, (cls, name, split))
        self.train_asset.lineEdit().setPlaceholderText(ASSET_PLACEHOLDER)
        self.train_asset.currentIndexChanged.connect(self._on_train_asset_changed)
        btn_read = QPushButton("Read fields")
        btn_read.setToolTip("Fetch the asset's property names")
        btn_read.clicked.connect(self.on_read_asset_fields)
        arow.addWidget(self.train_asset, 1)
        arow.addWidget(btn_read)
        self.train_stack.addWidget(lw)
        self.train_stack.addWidget(aw)
        self.train_kind.currentIndexChanged.connect(self._on_train_kind_changed)
        form.addRow(self.train_stack)

        self.class_field = QComboBox()
        self.class_field.setEditable(True)
        self.class_field.setToolTip("Attribute with the class: numeric codes (kept as map values "
                                    "if 0-254) or names (coded 0..n-1)")
        form.addRow("Class field:", self.class_field)
        self.name_field = QComboBox()
        self.name_field.setEditable(True)
        self.name_field.setToolTip("Optional attribute with readable class names (e.g. crop)")
        form.addRow("Class name field:", self.name_field)

        btn_classes = QPushButton("Load classes")
        btn_classes.clicked.connect(self.on_load_classes)
        form.addRow(btn_classes)
        self.class_table = QTableWidget(0, 5)
        self.class_table.setHorizontalHeaderLabels(["Value", "Code", "Name", "Colour", "Polygons"])
        self.class_table.verticalHeader().setVisible(False)
        self.class_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.class_table.horizontalHeader().setSectionResizeMode(COL_NAME, QHeaderView.Stretch)
        self.class_table.setMinimumHeight(150)
        self.class_table.setToolTip("Edit a name; double click a colour to change it")
        self.class_table.cellDoubleClicked.connect(self._on_class_cell_double_clicked)
        self.class_table.itemChanged.connect(self._on_class_item_changed)
        form.addRow(self.class_table)

        self.split_mode = QComboBox()
        self.split_mode.addItem("By attribute (test value)", "attribute")
        self.split_mode.addItem("Random by polygon", "random")
        self.split_mode.addItem("No test set (train on everything)", "none")
        self.split_mode.currentIndexChanged.connect(self._on_split_changed)
        form.addRow("Train / test split:", style.with_info(self.split_mode, H.SPLIT))
        srow = QHBoxLayout()
        self.split_field = QComboBox()
        self.split_field.setEditable(True)
        self.split_value = QLineEdit("test")
        self.split_value.setMaximumWidth(80)
        self.split_frac = QSpinBox()
        self.split_frac.setRange(5, 80)
        self.split_frac.setValue(30)
        self.split_frac.setSuffix(" % test")
        self.split_field_label = QLabel("field")
        self.split_value_label = QLabel("= test value")
        for wdg in (self.split_field_label, self.split_field, self.split_value_label,
                    self.split_value, self.split_frac):
            srow.addWidget(wdg)
        form.addRow("", srow)

        self.sampling = QComboBox()
        for key, label in k.SAMPLING.items():
            self.sampling.addItem(label, key)
        self.sampling.currentIndexChanged.connect(self._on_sampling_changed)
        form.addRow("Samples:", style.with_info(self.sampling, H.SAMPLING))
        self.max_px = QSpinBox()
        self.max_px.setRange(10, 5000)
        self.max_px.setValue(200)
        self.max_px.setToolTip("Pixel samples: at most this many random pixels per class. "
                               "They are stored in the model file.")
        form.addRow("Max. pixels per class:", self.max_px)
        self.chk_pixel_acc = QCheckBox("Also assess the map pixels inside the test polygons")
        self.chk_pixel_acc.setChecked(True)
        form.addRow(self.chk_pixel_acc)

        self._on_train_kind_changed(0)
        self._on_split_changed()
        self._on_sampling_changed()
        return box

    def _build_aoi_box(self):
        box = self.aoi_box = QGroupBox("Classification area")
        form = QFormLayout(box)
        self.aoi_kind = self._source_combo(["QGIS layer", "GEE table asset", "Current map view"])
        form.addRow("Area:", self.aoi_kind)
        self.aoi_stack = QStackedWidget()
        lw = QWidget()
        lrow = QHBoxLayout(lw)
        lrow.setContentsMargins(0, 0, 0, 0)
        self.aoi_layer = QgsMapLayerComboBox()
        self.aoi_layer.setFilters(self._polygon_filter())
        self.aoi_layer.setAllowEmptyLayer(True)
        self.aoi_layer.setLayer(None)
        self.chk_aoi_selected = QCheckBox("Selected only")
        lrow.addWidget(self.aoi_layer, 1)
        lrow.addWidget(self.chk_aoi_selected)
        self.aoi_asset = QComboBox()
        self.aoi_asset.setEditable(True)
        self.aoi_asset.addItems(AOI_PRESETS)
        self.aoi_asset.lineEdit().setPlaceholderText(ASSET_PLACEHOLDER)
        self.aoi_stack.addWidget(lw)
        self.aoi_stack.addWidget(self.aoi_asset)
        self.aoi_stack.addWidget(QLabel("The map extent at the time of the run."))
        self.aoi_kind.currentIndexChanged.connect(self.aoi_stack.setCurrentIndex)
        form.addRow(self.aoi_stack)

        mrow = QHBoxLayout()
        self.chk_mask = QCheckBox("Mask to ESA WorldCover classes")
        self.chk_mask.setChecked(True)
        self.chk_mask.setToolTip("The training data contain only farmland. Without a mask, desert, "
                                 "settlements and water get a crop class too. "
                                 "40 = cropland, 10 = tree cover (orchards), 30 = grassland.")
        self.mask_classes = QLineEdit("40, 10")
        self.mask_classes.setMaximumWidth(90)
        mrow.addWidget(self.chk_mask)
        mrow.addWidget(self.mask_classes)
        mrow.addStretch(1)
        form.addRow(mrow)
        return box

    def _build_features_box(self):
        box = self.features_box = QgsCollapsibleGroupBox("Features (time-series composites)")
        box.setSaveCollapsedState(False)
        lay = QVBoxLayout(box)
        self.recipe = FeatureRecipeWidget("Sampling scale:")
        self.year = self.recipe.year
        lay.addWidget(self.recipe)
        return box

    def _build_classifier_box(self):
        box = QGroupBox("Classifier")
        form = QFormLayout(box)
        self.clf_type = QComboBox()
        for key, label in k.CLASSIFIERS.items():
            self.clf_type.addItem(label, key)
        self.clf_type.currentIndexChanged.connect(self._on_clf_changed)
        form.addRow("Algorithm:", self.clf_type)
        self.trees = QSpinBox()
        self.trees.setRange(1, 2000)
        form.addRow("Number of trees:", self.trees)
        self.vps = QSpinBox()
        self.vps.setRange(0, 500)
        self.vps.setSpecialValueText("auto (√ features)")
        form.addRow("Variables per split:", self.vps)
        self.min_leaf = QSpinBox()
        self.min_leaf.setRange(1, 100)
        form.addRow("Min. leaf population:", self.min_leaf)
        self.bag = QDoubleSpinBox()
        self.bag.setRange(0.1, 1.0)
        self.bag.setSingleStep(0.05)
        self.bag.setToolTip("RF: bag fraction · GTB: sampling rate")
        form.addRow("Bag fraction / sampling rate:", self.bag)
        self.shrinkage = QDoubleSpinBox()
        self.shrinkage.setRange(0.001, 1.0)
        self.shrinkage.setDecimals(3)
        self.shrinkage.setSingleStep(0.01)
        form.addRow("Shrinkage (GTB):", self.shrinkage)
        self.max_nodes = QSpinBox()
        self.max_nodes.setRange(0, 100000)
        self.max_nodes.setSpecialValueText("unlimited")
        form.addRow("Max. nodes per tree:", self.max_nodes)
        self.seed = QSpinBox()
        self.seed.setRange(0, 999999)
        form.addRow("Seed:", self.seed)
        self._set_classifier(k.DEFAULT_CLASSIFIER)
        cv_row = QHBoxLayout()
        self.chk_cv = QCheckBox("Spatial k-fold cross-validation")
        self.chk_cv.setToolTip("Folds of spatial blocks (polygon centroids): neighbouring fields and the pixels "
                               "of one field never end up in training and test at once. Uses train + test "
                               "samples; reported as mean ± sd over the folds.")
        self.cv_k = QSpinBox()
        self.cv_k.setRange(2, 20)
        self.cv_k.setValue(5)
        self.cv_k.setSuffix(" folds")
        self.cv_block = QDoubleSpinBox()
        self.cv_block.setRange(0.1, 100)
        self.cv_block.setValue(5)
        self.cv_block.setSuffix(" km blocks")
        cv_row.addWidget(self.chk_cv)
        cv_row.addWidget(self.cv_k)
        cv_row.addWidget(self.cv_block)
        cv_row.addWidget(style.InfoButton(H.CV))
        form.addRow(cv_row)
        return box

    def _build_model_box(self):
        box = QGroupBox("Model")
        layout = QVBoxLayout(box)
        self.model_info = QLabel("No model yet — train one or load a model file.")
        self.model_info.setWordWrap(True)
        self.model_info.setStyleSheet("color:#666")
        layout.addWidget(self.model_info)
        row = QHBoxLayout()
        self.btn_save_model = QPushButton("Save model…")
        self.btn_save_model.clicked.connect(self.on_save_model)
        btn_load = QPushButton("Load model…")
        btn_load.clicked.connect(self.on_load_model)
        row.addWidget(self.btn_save_model)
        row.addWidget(btn_load)
        layout.addLayout(row)
        arow = QHBoxLayout()
        arow.addWidget(QLabel("Apply to year:"))
        self.apply_year = QSpinBox()
        self.apply_year.setRange(2017, c.MAX_YEAR)
        self.apply_year.setValue(c.MAX_YEAR - 1)
        arow.addWidget(self.apply_year)
        self.btn_apply = QPushButton("Apply model")
        self.btn_apply.clicked.connect(self.on_apply)
        arow.addWidget(self.btn_apply, 1)
        layout.addLayout(arow)
        prow = QHBoxLayout()
        btn_pool = QPushButton("Pool samples from models…")
        btn_pool.setToolTip("Multi-year training: the samples stored in other model files (e.g. of 2025) are "
                            "added to the training samples of the next run. They need the same feature bands; "
                            "classes are matched by value or name.")
        btn_pool.clicked.connect(self.on_add_pool)
        btn_pool_clear = QPushButton("Clear")
        btn_pool_clear.clicked.connect(self.on_clear_pool)
        prow.addWidget(btn_pool, 1)
        prow.addWidget(btn_pool_clear)
        layout.addLayout(prow)
        self.pool_label = QLabel("No pooled samples.")
        self.pool_label.setWordWrap(True)
        self.pool_label.setStyleSheet("color:#666; font-size:11px")
        layout.addWidget(self.pool_label)
        self.chk_apply_validate = QCheckBox("Validate with the training data above")
        self.chk_apply_validate.setToolTip("Uses the training polygons above as reference polygons of the "
                                           "applied year; classes are matched by value or name.")
        layout.addWidget(self.chk_apply_validate)
        hint = QLabel("Applying rebuilds the model's features for the chosen year over the "
                      "classification area and retrains the identical classifier from the samples "
                      "stored in the model file.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#666; font-size:11px")
        layout.addWidget(hint)
        self._update_model_buttons()
        return box

    def _build_results(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        self.res_info = QLabel("No classification yet.")
        self.res_info.setWordWrap(True)
        self.res_info.setTextFormat(Qt.RichText)
        layout.addWidget(self.res_info)
        from .dock_widget import PlotPanel
        self.res_plot = PlotPanel()
        p.plot_placeholder(self.res_plot.figure, "No classification yet.")
        layout.addWidget(self.res_plot, 1)

        fld = QGroupBox("Field-level results")
        fform = QFormLayout(fld)
        frow = QHBoxLayout()
        self.fld_layer = QgsMapLayerComboBox()
        self.fld_layer.setFilters(self._polygon_filter())
        self.fld_layer.setAllowEmptyLayer(True)
        self.fld_layer.setLayer(None)
        self.chk_fld_selected = QCheckBox("Selected only")
        frow.addWidget(self.fld_layer, 1)
        frow.addWidget(self.chk_fld_selected)
        fform.addRow("Fields:", frow)
        btn_fld = QPushButton("Aggregate the map to the fields")
        btn_fld.setToolTip("Majority class per field, its share of the field's pixels (confidence), the "
                           "runner-up class and the pixel count -> a new, class-styled layer")
        btn_fld.clicked.connect(self.on_fields)
        fform.addRow(btn_fld)
        layout.addWidget(fld)

        exp = QGroupBox("Save the map")
        form = QFormLayout(exp)
        note = QLabel("The map layer in QGIS is an Earth Engine tile service and can't be saved "
                      "with 'Export → Save As'. Download it as GeoTIFF (class codes, 255 = NoData) "
                      "or export it as an Earth Engine batch task (large areas).")
        note.setWordWrap(True)
        note.setStyleSheet("color:#666; font-size:11px")
        form.addRow(note)
        self.exp_name = QLineEdit()
        self.exp_name.setPlaceholderText("e.g. cropmap_2026")
        form.addRow("Name:", self.exp_name)
        erow = QHBoxLayout()
        self.exp_scale = QSpinBox()
        self.exp_scale.setRange(10, 1000)
        self.exp_scale.setValue(10)
        self.exp_scale.setSuffix(" m")
        self.exp_crs = QLineEdit()
        self.exp_crs.setPlaceholderText("UTM of the area")
        erow.addWidget(self.exp_scale)
        erow.addWidget(QLabel("CRS:"))
        erow.addWidget(self.exp_crs, 1)
        form.addRow("Scale:", erow)
        self.btn_download = QPushButton("Download GeoTIFF…")
        self.btn_download.setToolTip("Computes the map tile by tile and saves it locally; added to the "
                                     "project as a normal raster layer with the class colours")
        self.btn_download.clicked.connect(self.on_download)
        form.addRow(self.btn_download)
        drow = QHBoxLayout()
        self.exp_folder = QLineEdit("GEE_Exports")
        btn_drive = QPushButton("To Drive (GeoTIFF)")
        btn_drive.setToolTip("Cloud-optimised GeoTIFF, background = 255 = NoData")
        btn_drive.clicked.connect(lambda: self.on_export("drive"))
        drow.addWidget(self.exp_folder, 1)
        drow.addWidget(btn_drive)
        form.addRow("Drive folder:", drow)
        arow = QHBoxLayout()
        self.exp_asset = QLineEdit()
        self.exp_asset.setPlaceholderText("projects/<project>/assets/<name>")
        btn_asset = QPushButton("To asset")
        btn_asset.setToolTip("Image asset with masked background (input for follow-up scripts)")
        btn_asset.clicked.connect(lambda: self.on_export("asset"))
        arow.addWidget(self.exp_asset, 1)
        arow.addWidget(btn_asset)
        form.addRow("Asset id:", arow)
        layout.addWidget(exp)

        vrow = QHBoxLayout()
        btn_html = QPushButton("Export HTML report…")
        btn_html.setToolTip("Self-contained accuracy report (summary, kappa scale, confusion matrices, "
                            "most frequent confusions, sortable per-class table, charts, settings); "
                            "opens in the browser")
        btn_html.clicked.connect(self.on_save_html)
        btn_csv = QPushButton("Save validation tables (CSV)…")
        btn_csv.clicked.connect(self.on_save_csv)
        btn_trees = QPushButton("Export trees (HTML)…")
        btn_trees.setToolTip("Interactive viewer of all Random Forest / CART trees of the model: forest "
                             "overview with variable usage, zoomable tree view, click nodes to expand")
        btn_trees.clicked.connect(self.on_save_trees)
        vrow.addWidget(btn_html)
        vrow.addWidget(btn_trees)
        vrow.addWidget(btn_csv)
        layout.addLayout(vrow)
        return w

    @staticmethod
    def _polygon_filter():
        from .dock_widget import POLYGON_FILTER
        return POLYGON_FILTER

    # ------------------------------------------------------------------
    # Widget <-> settings
    # ------------------------------------------------------------------
    @staticmethod
    def _set_combo(combo, data):
        i = combo.findData(data)
        if i >= 0:
            combo.setCurrentIndex(i)

    def _features(self):
        return self.recipe.features()

    def _set_features(self, f):
        self.recipe.set_features(f)

    def _classifier(self):
        return {"type": self.clf_type.currentData(), "trees": self.trees.value(),
                "variables_per_split": self.vps.value(), "min_leaf": self.min_leaf.value(),
                "bag_fraction": round(self.bag.value(), 3), "max_nodes": self.max_nodes.value(),
                "shrinkage": round(self.shrinkage.value(), 4), "seed": self.seed.value()}

    def _set_classifier(self, cfg):
        cfg = dict(k.DEFAULT_CLASSIFIER, **cfg)
        self._set_combo(self.clf_type, cfg["type"])
        self.trees.setValue(cfg["trees"])
        self.vps.setValue(cfg["variables_per_split"])
        self.min_leaf.setValue(cfg["min_leaf"])
        self.bag.setValue(cfg["bag_fraction"])
        self.max_nodes.setValue(cfg["max_nodes"])
        self.shrinkage.setValue(cfg["shrinkage"])
        self.seed.setValue(cfg["seed"])
        self._on_clf_changed()

    def _update_feature_info(self):
        self.recipe.update_info()

    def _on_clf_changed(self, *_):
        kind = self.clf_type.currentData()
        self.trees.setEnabled(kind in ("rf", "gtb"))
        self.vps.setEnabled(kind == "rf")
        self.bag.setEnabled(kind in ("rf", "gtb"))
        self.shrinkage.setEnabled(kind == "gtb")
        self.min_leaf.setEnabled(kind in ("rf", "cart"))

    def _on_split_changed(self, *_):
        mode = self.split_mode.currentData()
        for wdg in (self.split_field_label, self.split_field, self.split_value_label, self.split_value):
            wdg.setVisible(mode == "attribute")
        self.split_frac.setVisible(mode == "random")
        self.chk_pixel_acc.setEnabled(mode != "none" and self.sampling.currentData() == "mean")

    def _on_sampling_changed(self, *_):
        self.max_px.setEnabled(self.sampling.currentData() == "pixels")
        self._on_split_changed()

    def _on_train_kind_changed(self, idx):
        self.train_stack.setCurrentIndex(idx)
        if idx == 0:
            self._on_train_layer_changed(self.train_layer.currentLayer())
        else:
            self._on_train_asset_changed(self.train_asset.currentIndex())

    def _fill_field_combos(self, names, cls="", name="", split=""):
        for combo, default in ((self.class_field, cls), (self.name_field, name), (self.split_field, split)):
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            if combo is not self.class_field:
                combo.addItem("")
            combo.addItems(names)
            pick = default if default in names or not names else (current if current in names else "")
            combo.setCurrentText(pick)
            combo.blockSignals(False)

    def _on_train_layer_changed(self, layer):
        if layer is None:
            self._fill_field_combos([])
            return
        names = layer.fields().names()

        def guess(*cands):
            return next((n for n in cands if n in names), "")
        self._fill_field_combos(names, guess("class", "crop", "Class", "CLASS", "label"),
                                guess("crop", "name", "class_name"), guess("split", "Split"))

    def _on_train_asset_changed(self, idx):
        data = self.train_asset.itemData(idx)
        if data:
            cls, name, split = data
            self._fill_field_combos([n for n in (cls, name, split) if n], cls, name, split)

    def on_read_asset_fields(self):
        asset = self.train_asset.currentText().strip()
        if not asset or not self.dock._ensure_ee():
            return
        try:
            names = self.dock._run_busy(k.asset_property_names, asset)
        except Exception as e:
            self.dock._error(f"Could not read {asset}: {e}")
            return
        names = [n for n in names if not n.startswith("system:")]
        self._fill_field_combos(names, self.class_field.currentText(), self.name_field.currentText(),
                                self.split_field.currentText())
        self._set_status(f"{asset}: {len(names)} properties.")

    # ------------------------------------------------------------------
    # Classes
    # ------------------------------------------------------------------
    def _train_features_iter(self):
        layer = self.train_layer.currentLayer()
        if not pl.is_polygon_layer(layer):
            raise ValueError("Choose a training polygon layer.")
        if self.chk_train_selected.isChecked():
            return layer, list(layer.getSelectedFeatures())
        return layer, list(layer.getFeatures())

    def _source_key(self):
        if self.train_kind.currentIndex() == 0:
            layer = self.train_layer.currentLayer()
            src = (layer.id() if layer else None, self.chk_train_selected.isChecked(),
                   layer.selectedFeatureCount() if layer and self.chk_train_selected.isChecked() else 0)
        else:
            src = (self.train_asset.currentText().strip(),)
        return (self.train_kind.currentIndex(), src, self.class_field.currentText().strip(),
                self.name_field.currentText().strip())

    def _classes_current(self):
        """Loads the classes if the table is empty or belongs to other data."""
        return (self.classes and self._classes_key == self._source_key()) or self.on_load_classes()

    def on_load_classes(self):
        cls = self.class_field.currentText().strip()
        name = self.name_field.currentText().strip() or None
        if not cls:
            self.dock._warn("Choose the class field first.")
            return False
        try:
            if self.train_kind.currentIndex() == 0:
                layer, feats = self._train_features_iter()
                if layer.fields().indexOf(cls) < 0:
                    raise ValueError(f"'{cls}' is not a field of {layer.name()}.")
                counts, names = Counter(), {}
                for f in feats:
                    v = _attr(f[cls])
                    if v is None or str(v).strip() == "":
                        continue
                    counts[v] += 1
                    if name and layer.fields().indexOf(name) >= 0:
                        nm = _attr(f[name])
                        if nm is not None:
                            names.setdefault(k._key(v), str(nm))
                if not counts:
                    raise ValueError(f"No values in '{cls}'.")
                self._value_counts, self._value_names = dict(counts), names
                classes = k.build_classes(dict(counts), names)
            else:
                asset = self.train_asset.currentText().strip()
                if not self.dock._ensure_ee():
                    return False
                classes = self.dock._run_busy(k.asset_classes, asset, cls, name)
                self._value_counts = {q["value"]: q["n"] for q in classes}
                self._value_names = {k._key(q["value"]): q["name"] for q in classes}
        except Exception as e:
            self.dock._error(f"Could not load the classes: {e}")
            return False
        old = {k._key(q["value"]): q for q in self.classes}
        for q in classes:     # keep edited names / colours of a reload
            prev = old.get(k._key(q["value"]))
            if prev is not None:
                q["name"], q["color"] = prev["name"], prev["color"]
        self.classes = classes
        self._classes_key = self._source_key()
        self._fill_class_table()
        self._set_status(f"{len(classes)} classes, {sum(q['n'] for q in classes)} polygons.")
        return True

    def _fill_class_table(self):
        t = self.class_table
        t.blockSignals(True)
        t.setRowCount(len(self.classes))
        for r, q in enumerate(self.classes):
            cells = [str(q["value"]), str(q["code"]), q["name"], "", str(q.get("n", ""))]
            for col, text in enumerate(cells):
                it = QTableWidgetItem(text)
                if col != COL_NAME:
                    it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                if col == COL_COLOUR:
                    it.setBackground(QBrush(QColor(q["color"])))
                t.setItem(r, col, it)
        t.resizeColumnsToContents()
        t.horizontalHeader().setSectionResizeMode(COL_NAME, QHeaderView.Stretch)
        t.blockSignals(False)

    def _on_class_item_changed(self, item):
        if item.column() == COL_NAME and item.row() < len(self.classes):
            self.classes[item.row()]["name"] = item.text().strip() or str(self.classes[item.row()]["value"])

    def _on_class_cell_double_clicked(self, row, col):
        if col != COL_COLOUR or row >= len(self.classes):
            return
        colour = QColorDialog.getColor(QColor(self.classes[row]["color"]), self, "Class colour")
        if colour.isValid():
            self.classes[row]["color"] = colour.name()
            self.class_table.item(row, COL_COLOUR).setBackground(QBrush(colour))

    # ------------------------------------------------------------------
    # Inputs for a run
    # ------------------------------------------------------------------
    def _training_source(self):
        cls = self.class_field.currentText().strip()
        split_field = self.split_field.currentText().strip()
        by_attr = self.split_mode.currentData() == "attribute"
        if by_attr and not split_field:
            raise ValueError("Choose the split field (or another split mode).")
        if self.train_kind.currentIndex() == 0:
            layer, feats = self._train_features_iter()
            if by_attr and layer.fields().indexOf(split_field) < 0:
                raise ValueError(f"'{split_field}' is not a field of {layer.name()}.")
            out = []
            excluded = self._active_exclusions()
            for f in feats:
                g = f.geometry()
                v = _attr(f[cls])
                if g is None or g.isEmpty() or v is None or str(f.id()) in excluded:
                    continue
                g = pl.to_wgs84(g, layer.crs())
                if not g.isGeosValid():
                    g = g.makeValid()
                out.append({"geojson": json.loads(g.asJson(7)), "value": v,
                            "split": _attr(f[split_field]) if by_attr else None, "id": f.id()})
            if not out:
                raise ValueError("No training polygons with a class value.")
            return {"type": "features", "features": out, "layer": layer.name(), "class_prop": cls}
        asset = self.train_asset.currentText().strip()
        if not asset:
            raise ValueError("Enter the training asset id.")
        return {"type": "asset", "asset_id": asset, "class_prop": cls,
                "split_prop": split_field if by_attr else None,
                "exclude": sorted(self._active_exclusions())}

    def _split(self):
        return {"mode": self.split_mode.currentData(), "test_fraction": self.split_frac.value() / 100.0,
                "test_value": self.split_value.text().strip(), "seed": self.seed.value()}

    def _aoi(self):
        kind = self.aoi_kind.currentIndex()
        if kind == 0:
            layer = self.aoi_layer.currentLayer()
            if not pl.is_polygon_layer(layer):
                raise ValueError("Choose a polygon layer for the classification area.")
            return {"type": "geojson", "geojson": pl.layer_union_geojson(
                layer, self.chk_aoi_selected.isChecked(), MAX_AOI_VERTICES)}
        if kind == 1:
            asset = self.aoi_asset.currentText().strip()
            if not asset:
                raise ValueError("Enter the classification area asset id.")
            return {"type": "asset", "asset_id": asset}
        xform = QgsCoordinateTransform(self.canvas.mapSettings().destinationCrs(), pl.WGS84,
                                       QgsProject.instance())
        r = xform.transformBoundingBox(self.canvas.extent())
        return {"type": "bbox", "bbox": (r.xMinimum(), r.yMinimum(), r.xMaximum(), r.yMaximum())}

    def _mask(self):
        if not self.chk_mask.isChecked():
            return None
        try:
            vals = [int(v) for v in self.mask_classes.text().replace(";", ",").split(",") if v.strip()]
        except ValueError:
            raise ValueError("WorldCover classes must be numbers, e.g. 40, 10.")
        return vals or None

    # ------------------------------------------------------------------
    # Runs
    # ------------------------------------------------------------------
    def _start(self, description, fn, kwargs, on_done=None):
        self._task = CallTask(description, fn, kwargs, on_done or self._on_done)
        self._task.progressText.connect(self.status.setText)
        self._task.progressChanged.connect(lambda v: self.progress.setValue(int(v)))
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.show()
        self.btn_cancel.setEnabled(True)
        self.btn_train.setEnabled(False)
        self.btn_apply.setEnabled(False)
        self._set_status(f"{description} …")
        QgsApplication.taskManager().addTask(self._task)

    def _output_path(self, default):
        path, _ = QFileDialog.getSaveFileName(self, "Save the classified map", default, "GeoTIFF (*.tif *.tiff)")
        return path

    def on_download_stack(self):
        if self._task is not None:
            return
        f = self._features()
        if self.recipe.problem(f):
            self.dock._warn(self.recipe.problem(f))
            return
        extent = self.stack_extent.currentData()
        if not self.dock._ensure_ee():
            return
        try:
            aoi = self._aoi() if extent in ("aoi", "both") else None
            training = classes = None
            if extent in ("training", "both"):
                if not self._classes_current():
                    return
                training, classes = self._training_source(), [dict(q) for q in self.classes]
        except ValueError as e:
            self.dock._warn(str(e))
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save feature stack", f"feature_stack_{self.year.value()}.tif",
                                              "GeoTIFF (*.tif *.tiff)")
        if not path:
            return
        self._start(f"Downloading the {self.year.value()} feature stack", k.download_feature_stack,
                    {"year": self.year.value(), "features": f, "path": path, "aoi": aoi, "training": training,
                     "classes": classes, "extent": extent, "scale": f["scale"]},
                    on_done=self._on_stack_done)

    def _on_stack_done(self, task, ok):
        if not self._finish(task, ok, "Stack download"):
            return
        try:
            layer = self._add_raster(task.result, shown=False)
        except RuntimeError as e:
            self.dock._error(str(e))
            return
        self.train_raster.setLayer(layer)
        self._update_stack_info()
        self._set_status(f"Saved {task.result} — selected as training stack.")
        self.iface.messageBar().pushSuccess("GEE classification", f"Saved {task.result}")

    def _local_reference(self, classes):
        """(training source, split, matched classes) of the training layer, for validation."""
        if self.train_kind.currentIndex() != 0:
            raise ValueError("Offline, the training / reference polygons must be a QGIS layer.")
        matched, unmatched = k.match_classes(classes, self._value_counts, self._value_names)
        if not matched:
            raise ValueError("None of the reference classes matches a class of the model (by value or name).")
        if unmatched:
            self.dock._warn(f"Reference classes not in the model (ignored): {', '.join(map(str, unmatched[:8]))}")
        return self._training_source(), self._split(), matched

    def on_train_local(self):
        try:
            lc.require_sklearn()
        except RuntimeError as e:
            self.dock._error(str(e))
            return
        if self.train_kind.currentIndex() != 0:
            self.dock._warn("Offline, the training polygons must be a QGIS layer (download the asset as a "
                            "layer on the Polygons tab first).")
            return
        train = self._stack_path(self.train_raster)
        if train is None:
            self.dock._warn("Choose (or download) a training stack.")
            return
        pred = self._stack_path(self.pred_raster) or train
        if not self._classes_current():
            return
        try:
            training = self._training_source()
            mask = self._mask()
            extra = self._pooled_extra(lc.stack_info(train)["bands"])
            self._remember_train_layer()
        except (ValueError, RuntimeError) as e:
            self.dock._warn(str(e))
            return
        out = self._output_path(os.path.splitext(pred)[0] + "_classified.tif")
        if not out:
            return
        self._start("Training on the local stack", lc.run_local_training,
                    {"train_stack": train, "predict_stack_path": pred, "out_path": out,
                     "classifier_cfg": self._classifier(), "classes": [dict(q) for q in self.classes],
                     "training": training, "split": self._split(), "mask_classes": mask,
                     "sampling": self.sampling.currentData(), "max_per_class": self.max_px.value(),
                     "pixel_accuracy": self.chk_pixel_acc.isChecked(), "extra_samples": extra, "cv": self._cv()})

    def on_apply_local(self):
        try:
            lc.require_sklearn()
        except RuntimeError as e:
            self.dock._error(str(e))
            return
        stack = self._stack_path(self.pred_raster) or self._stack_path(self.train_raster)
        if stack is None:
            self.dock._warn("Choose the stack to classify.")
            return
        try:
            missing = [b for b in self.model["bands"] if b not in lc.stack_info(stack)["index"]]
            if missing:
                raise ValueError(f"The stack lacks {len(missing)} band(s) of the model, e.g. "
                                 f"{', '.join(missing[:4])}. Download a stack with the model's features "
                                 "(load the model, then 'Download feature stack').")
            reference = None
            if self.chk_apply_validate.isChecked():
                if not self._classes_current():
                    return
                reference = self._local_reference(self.model["classes"])
            mask = self._mask()
        except (ValueError, RuntimeError) as e:
            self.dock._warn(str(e))
            return
        out = self._output_path(os.path.splitext(stack)[0] + "_classified.tif")
        if not out:
            return
        self._start("Applying the model to the local stack", lc.run_local_apply,
                    {"model": self.model, "stack_path": stack, "out_path": out, "mask_classes": mask,
                     "reference": reference})

    def on_train(self):
        if self._task is not None:
            return
        if self._local():
            self.on_train_local()
            return
        f = self._features()
        if self.recipe.problem(f):
            self.dock._warn(self.recipe.problem(f))
            return
        if not self.dock._ensure_ee():
            return
        if not self._classes_current():
            return
        try:
            extra = self._pooled_extra(k.band_names(f, self.year.value()))
            self._remember_train_layer()
            kwargs = {"extra_samples": extra, "cv": self._cv(),
                      "year": self.year.value(), "features": f, "classifier_cfg": self._classifier(),
                      "classes": [dict(q) for q in self.classes], "training": self._training_source(),
                      "split": self._split(), "aoi": self._aoi(), "mask_classes": self._mask(),
                      "sampling": self.sampling.currentData(), "max_per_class": self.max_px.value(),
                      "pixel_accuracy": self.chk_pixel_acc.isChecked()}
        except ValueError as e:
            self.dock._warn(str(e))
            return
        self._start(f"Training the {self.year.value()} classification", k.run_training, kwargs)

    def on_apply(self):
        if self._task is not None or self.model is None:
            return
        if self._local():
            self.on_apply_local()
            return
        if not self.model.get("features"):
            self.dock._warn("This model was trained on a stack without a feature recipe (e.g. a composite "
                            "stack); apply it offline to a stack with the same bands.")
            return
        if not self.dock._ensure_ee():
            return
        year = self.apply_year.value()
        reference = None
        try:
            aoi = self._aoi()
            if self.chk_apply_validate.isChecked():
                if not self._classes_current():
                    return
                matched, unmatched = k.match_classes(self.model["classes"], self._value_counts,
                                                     self._value_names)
                if not matched:
                    raise ValueError("None of the reference classes matches a class of the model "
                                     "(by value or name).")
                if unmatched:
                    self.dock._warn(f"Reference classes not in the model (ignored): "
                                    f"{', '.join(map(str, unmatched[:8]))}")
                reference = (self._training_source(), self._split(), matched)
            mask = self._mask()
        except ValueError as e:
            self.dock._warn(str(e))
            return
        self._start(f"Applying the model to {year}", k.run_apply,
                    {"model": self.model, "year": year, "aoi": aoi, "mask_classes": mask,
                     "reference": reference})

    def on_cancel(self):
        if self._task is not None:
            self._task.cancel()

    def _finish(self, task, ok, what):
        """Common end of a task; True if it succeeded."""
        self._task = None
        self.progress.hide()
        self.btn_cancel.setEnabled(False)
        self.btn_train.setEnabled(True)
        self._update_model_buttons()
        if task.exception is not None:
            self.dock._error(f"{what} failed: {task.exception}")
            self._set_status(f"{what} failed.")
            return False
        if not ok:
            self._set_status(f"{what} cancelled.")
            return False
        return True

    def _on_done(self, task, ok):
        if not self._finish(task, ok, "Classification"):
            return
        res = task.result
        self.result = res
        self.model = res["model"]
        self.model_info.setText(k.model_summary(self.model))
        self._update_model_buttons()
        if res.get("local"):
            self._add_local_class_layer(res["map_path"], res["classes"])
        else:
            self._add_map_layer(res)
            if not self.exp_crs.text().strip():
                self.exp_crs.setPlaceholderText(res.get("utm") or "UTM of the area")
        self._show_results(res)
        self.exp_name.setText(f"classification_{res['year']}")
        self._refresh_profiles()
        self._refresh_check()
        self.inner.setCurrentIndex(1)

    def _add_local_class_layer(self, path, classes):
        project = QgsProject.instance()
        name = os.path.splitext(os.path.basename(path))[0]
        for lyr in project.mapLayersByName(name):
            if os.path.normcase(lyr.source()) == os.path.normcase(path):
                project.removeMapLayer(lyr.id())
        layer = QgsRasterLayer(path, name)
        if not layer.isValid():
            self.dock._error(f"Saved {path}, but QGIS could not open it.")
            return None
        cats = [QgsPalettedRasterRenderer.Class(int(q["code"]), QColor(q["color"]), f"{q['code']} {q['name']}")
                for q in sorted(classes, key=lambda q: int(q["code"]))]
        layer.setRenderer(QgsPalettedRasterRenderer(layer.dataProvider(), 1, cats))
        project.addMapLayer(layer)
        self._set_status(f"Saved {path}")
        return layer

    def _add_map_layer(self, res):
        name = f"Classification {res['year']}"
        project = QgsProject.instance()
        for lyr in project.mapLayersByName(name):
            project.removeMapLayer(lyr.id())
        root = project.layerTreeRoot()
        group = root.findGroup(CLASS_GROUP) or root.insertGroup(0, CLASS_GROUP)
        try:
            layer = self.dock._run_busy(self.dock._add_ee_layer, res["map"], k.map_vis(res["classes"]), name)
        except Exception as e:
            self.dock._error(f"Could not add the classification layer: {e}")
            return
        self.dock._move_into_group(layer, group, 0)
        self._set_status(f"Added '{name}' (tiles are computed on the fly; zoom in if they are slow).")

    def _show_results(self, res):
        classes = {int(q["code"]): q for q in res["classes"]}
        applied = res.get("applied", False)
        ref = "reference" if applied else "test"
        trained = res["model"].get("trained_year") or "a local stack"
        title = f"Model of {trained} applied to" if applied else "Classification"
        offline = " (offline)" if res.get("local") else ""
        lines = [f"<b>{title} {res['year']}{offline}</b> · {res['n_train']} training samples"
                 + (f" · {res['n_test']} {ref} samples" if res["n_test"] else "")]
        if res.get("n_pooled"):
            lines.append(f"incl. {res['n_pooled']} pooled samples from "
                         f"{', '.join(res['model'].get('pooled_from', [])) or 'other models'}")
        if res.get("oob_error") is not None:
            lines.append(f"Out-of-bag error: {res['oob_error']:.1%}")
        cv = res.get("cv")
        if cv:
            lines.append(f"Spatial {cv['k']}-fold CV ({cv['block_km']:g} km blocks, {cv['n_blocks']} blocks): "
                         f"<b>OA {cv['oa_mean']:.1%} ± {cv['oa_sd']:.1%}</b>, κ {cv['kappa_mean']:.2f} ± "
                         f"{cv['kappa_sd']:.2f}")
        for key, label in (("metrics", f"{ref.capitalize()} samples"),
                           ("pixel_metrics", f"Map pixels in {ref} polygons")):
            m = res.get(key)
            if m and m["n"]:
                lines.append(f"{label}: <b>OA {m['overall_accuracy']:.1%}</b>, κ {m['kappa']:.2f}, n = {m['n']}")
        if not res.get("metrics"):
            lines.append("No reference data — no accuracy." if applied else
                         "No test set — no independent accuracy.")
        legend = " ".join(
            f"<span style='background:{q['color']}'>&nbsp;&nbsp;&nbsp;</span>&nbsp;{q['code']} {q['name']}"
            f" ({res['train_counts'].get(code, 0)})&nbsp;&nbsp;"
            for code, q in sorted(classes.items()))
        lines.append(f"<span style='font-size:11px'>{legend}</span>")
        self.res_info.setText("<br>".join(lines))
        self.res_plot.canvas.setMinimumHeight(sum(p.classification_heights(res)) + 40)
        n = len((res.get("metrics") or {}).get("codes", []))
        self.res_plot.canvas.setMinimumWidth(max(380, 170 + 24 * n))   # scrolls sideways if needed
        p.plot_classification(self.res_plot.figure, res, ref_label=ref)
        self.res_plot.redraw()

    def _set_status(self, msg):
        self.status.setText(msg)

    # ------------------------------------------------------------------
    # Model files
    # ------------------------------------------------------------------
    def _update_model_buttons(self):
        has = self.model is not None
        self.btn_save_model.setEnabled(has)
        self.btn_apply.setEnabled(has and self._task is None)

    def on_save_model(self):
        if self.model is None:
            return
        default = f"classification_model_{self.model.get('trained_year') or 'local'}.json"
        path, _ = QFileDialog.getSaveFileName(self, "Save model", default, "Model files (*.json)")
        if not path:
            return
        try:
            k.save_model(self.model, path)
        except Exception as e:
            self.dock._error(f"Could not save the model: {e}")
            return
        self.iface.messageBar().pushSuccess("GEE classification", f"Saved {path}")

    def on_load_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load model", "", "Model files (*.json);;All files (*)")
        if not path:
            return
        try:
            model = k.load_model(path)
        except Exception as e:
            self.dock._error(f"Could not load {os.path.basename(path)}: {e}")
            return
        self.model = model
        self.result = None
        # show the model's recipe; "Train" with it retrains on new data
        if model.get("features"):
            self._set_features(model["features"])
        self._set_classifier(model["classifier"])
        self._set_combo(self.sampling, model.get("sampling", "mean"))
        if model.get("trained_year"):
            self.year.setValue(model["trained_year"])
        self._update_feature_info()
        self.model_info.setText(f"{os.path.basename(path)}: {k.model_summary(model)}")
        self._refresh_profiles()
        self._update_model_buttons()
        self._set_status("Model loaded — choose a year and the classification area, then 'Apply model'.")

    # ------------------------------------------------------------------
    # Exports
    # ------------------------------------------------------------------
    def _local_result_note(self):
        if self.result is not None and self.result.get("local"):
            self.dock._warn(f"The offline map is already a local GeoTIFF: {self.result['map_path']}")
            return True
        return False

    def on_export(self, kind):
        if self.result is None:
            self.dock._warn("Run a classification first.")
            return
        if self._local_result_note():
            return
        name = self.exp_name.text().strip() or f"classification_{self.result['year']}"
        name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)[:100]
        asset_id = self.exp_asset.text().strip()
        if kind == "asset" and not asset_id:
            self.dock._warn("Enter the asset id to export to.")
            return
        crs = self.exp_crs.text().strip() or self.result.get("utm")
        try:
            task_id = self.dock._run_busy(k.export_map, self.result["map"], self.result["region"], kind, name,
                                          self.exp_scale.value(), crs, self.exp_folder.text().strip(), asset_id)
        except Exception as e:
            self.dock._error(f"Export failed: {e}")
            return
        where = f"Drive folder '{self.exp_folder.text().strip()}'" if kind == "drive" else asset_id
        self.iface.messageBar().pushSuccess(
            "GEE classification", f"Export task {task_id} started → {where}. "
            "Follow it at https://code.earthengine.google.com/tasks")

    def on_download(self):
        if self.result is None:
            self.dock._warn("Run a classification first.")
            return
        if self._task is not None or self._local_result_note():
            return
        name = self.exp_name.text().strip() or f"classification_{self.result['year']}"
        path, _ = QFileDialog.getSaveFileName(self, "Save classified map", f"{name}.tif",
                                              "GeoTIFF (*.tif *.tiff)")
        if not path:
            return
        crs = self.exp_crs.text().strip() or self.result.get("utm")
        self._start(f"Downloading {os.path.basename(path)}", k.download_map,
                    {"class_map": self.result["map"], "region": self.result["region"],
                     "classes": self.result["classes"], "path": path,
                     "scale": self.exp_scale.value(), "crs": crs},
                    on_done=self._on_download_done)

    def _on_download_done(self, task, ok):
        if not self._finish(task, ok, "Download"):
            return
        if self._add_local_class_layer(task.result, self.result["classes"]) is not None:
            self.iface.messageBar().pushSuccess("GEE classification", f"Saved {task.result}")

    def on_save_html(self):
        if self.result is None:
            self.dock._warn("Run a classification first.")
            return
        default = f"classification_report_{self.result['year']}.html"
        path, _ = QFileDialog.getSaveFileName(self, "Save HTML report", default, "HTML (*.html)")
        if not path:
            return
        try:
            self.dock._run_busy(report.save_report, self.result, path)
        except Exception as e:
            self.dock._error(f"Could not write the report: {e}")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))
        self.iface.messageBar().pushSuccess("GEE classification", f"Saved {path}")

    def on_save_trees(self):
        if self.result is None:
            self.dock._warn("Run a classification first.")
            return
        if self._task is not None:
            return
        if self.result["model"]["classifier"].get("type") not in ("rf", "cart"):
            self.dock._warn("Tree export is available for Random Forest and CART models.")
            return
        if not self.result.get("local") and not self.dock._ensure_ee():
            return
        default = f"classification_trees_{self.result['year']}.html"
        path, _ = QFileDialog.getSaveFileName(self, "Save tree viewer", default, "HTML (*.html)")
        if path:
            self._start("Exporting the trees", tree_export.save_trees_html,
                        {"result": self.result, "path": path}, on_done=self._on_trees_done)

    def _on_trees_done(self, task, ok):
        if not self._finish(task, ok, "Tree export"):
            return
        path, n = task.result
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))
        self.iface.messageBar().pushSuccess("GEE classification", f"Saved {n} tree(s) to {path}")

    def on_save_csv(self):
        res = self.result
        if res is None or not res.get("metrics"):
            self.dock._warn("No accuracy assessment to save (run with a test set).")
            return
        folder = QFileDialog.getExistingDirectory(self, "Folder for the validation tables")
        if not folder:
            return
        y = res["year"]
        names = {int(q["code"]): q["name"] for q in res["classes"]}
        k.metrics_table(res["metrics"], res["classes"]).to_csv(
            os.path.join(folder, f"validation_metrics_{y}.csv"), index=False)
        k.confusion_table(res["metrics"], res["classes"]).to_csv(
            os.path.join(folder, f"validation_confusion_matrix_samples_{y}.csv"), index=False)
        files = 2
        if res.get("pixel_metrics") and res["pixel_metrics"]["n"]:
            k.confusion_table(res["pixel_metrics"], res["classes"]).to_csv(
                os.path.join(folder, f"validation_confusion_matrix_pixels_{y}.csv"), index=False)
            files += 1
        ps = res.get("per_sample")
        if ps is not None and len(ps):
            ps = ps.rename(columns={k.ID_PROP: "polygon_id", k.CLASS_PROP: "class"})
            ps["ref_name"] = ps["class"].map(names)
            ps["pred_name"] = ps["classification"].map(names)
            ps["correct"] = ps["class"] == ps["classification"]
            ps.to_csv(os.path.join(folder, f"validation_per_sample_{y}.csv"), index=False)
            files += 1
        if res.get("importance"):
            import pandas as pd
            pd.DataFrame(list(res["importance"].items()), columns=["feature", "importance"]).to_csv(
                os.path.join(folder, f"variable_importance_{y}.csv"), index=False)
            files += 1
        self.iface.messageBar().pushSuccess("GEE classification", f"Saved {files} table(s) to {folder}")

    # ------------------------------------------------------------------
    # Cross-validation, pooled samples, exclusions
    # ------------------------------------------------------------------
    def _cv(self):
        if not self.chk_cv.isChecked():
            return None
        return {"k": self.cv_k.value(), "block_km": round(self.cv_block.value(), 3), "seed": self.seed.value()}

    def on_add_pool(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Models whose samples are pooled", "",
                                                "Model files (*.json)")
        for path in paths:
            try:
                self.pool_models.append((os.path.basename(path), k.load_model(path)))
            except Exception as e:
                self.dock._error(f"{os.path.basename(path)}: {e}")
        self._update_pool_label()

    def on_clear_pool(self):
        self.pool_models = []
        self._update_pool_label()

    def _update_pool_label(self):
        if not self.pool_models:
            self.pool_label.setText("No pooled samples.")
            return
        self.pool_label.setText("Pooled into the next training: " + "; ".join(
            f"{name} ({m.get('trained_year') or 'stack'}, {len(m['samples']['rows'])} samples)"
            for name, m in self.pool_models))

    def _pooled_extra(self, bands):
        """Samples of the pooled models, columns = the run's bands, classes
        mapped onto the current class table (by value or name)."""
        if not self.pool_models:
            return None
        rows, sources = [], []
        for name, m in self.pool_models:
            cols = m["samples"]["columns"]
            missing = [b for b in bands if b not in cols]
            if missing:
                raise ValueError(f"{name} lacks {len(missing)} band(s) of this run (e.g. {missing[0]}); "
                                 "pooled models need the same features.")
            counts = {q["value"]: 1 for q in m["classes"]}
            names = {k._key(q["value"]): q["name"] for q in m["classes"]}
            matched, unmatched = k.match_classes(self.classes, counts, names)
            code_map = {int(q["value"]) if k._is_int(q["value"]) else q["value"]: int(q["code"]) for q in matched}
            by_code = {int(q["code"]): q["value"] for q in m["classes"]}
            idx = [cols.index(b) for b in bands]
            n0 = len(rows)
            for r in m["samples"]["rows"]:
                v = by_code.get(int(r[0]))
                key = int(v) if k._is_int(v) else v
                if key in code_map:
                    rows.append([code_map[key]] + [r[i] for i in idx])
            sources.append(f"{name} ({m.get('trained_year') or 'stack'}, {len(rows) - n0} samples)")
            if unmatched:
                self.dock._warn(f"{name}: classes not in the current class table are skipped: "
                                f"{', '.join(map(str, unmatched[:6]))}")
        return {"columns": [k.CLASS_PROP] + list(bands), "rows": rows, "sources": sources}

    def _active_exclusions(self):
        key = self._source_key()[:2]
        return self.excluded if self.excluded and self._excluded_key == key else set()

    def _remember_train_layer(self):
        self._train_layer_used = self.train_layer.currentLayer() if self.train_kind.currentIndex() == 0 else None

    # ------------------------------------------------------------------
    # Class profiles
    # ------------------------------------------------------------------
    def _build_profiles_tab(self):
        from .dock_widget import PlotPanel
        w = QWidget()
        lay = QVBoxLayout(w)
        row = QHBoxLayout()
        row.addWidget(QLabel("Variable:"))
        self.prof_var = QComboBox()
        self.prof_var.currentIndexChanged.connect(lambda *_: self._draw_profiles())
        row.addWidget(self.prof_var, 1)
        self.chk_prof_band = QCheckBox("± 1σ")
        self.chk_prof_band.setChecked(True)
        self.chk_prof_band.toggled.connect(lambda *_: self._draw_profiles())
        row.addWidget(self.chk_prof_band)
        row.addWidget(style.InfoButton(H.PROFILES))
        for label, state in (("All", True), ("None", False)):
            b = QPushButton(label)
            b.setFixedWidth(46)
            b.clicked.connect(lambda _=False, s=state: self._prof_check_all(s))
            row.addWidget(b)
        lay.addLayout(row)
        split = QSplitter(Qt.Horizontal)
        self.prof_classes = QListWidget()
        self.prof_classes.itemChanged.connect(lambda *_: self._draw_profiles())
        self.prof_plot = PlotPanel()
        self.prof_plot.canvas.setMinimumHeight(380)
        p.plot_placeholder(self.prof_plot.figure, "Train or load a model to see the class profiles.")
        split.addWidget(self.prof_classes)
        split.addWidget(self.prof_plot)
        split.setSizes([150, 520])
        lay.addWidget(split, 1)
        hint = QLabel("Mean ± 1 standard deviation of the model's training samples per class and period "
                      "(pooled samples included). Overlapping bands explain confusions in the matrix.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#666; font-size:11px")
        lay.addWidget(hint)
        self._profiles = {}
        return w

    def _refresh_profiles(self):
        m = self.model
        self._profiles = {}
        self.prof_var.blockSignals(True)
        self.prof_var.clear()
        self.prof_classes.blockSignals(True)
        self.prof_classes.clear()
        if m is not None:
            try:
                self._profiles = at.class_profiles(k.model_samples(m), m["bands"], m.get("features"),
                                                   m.get("trained_year"))
            except Exception as e:
                self._set_status(f"Class profiles: {e}")
            for (var, red) in self._profiles:
                self.prof_var.addItem(f"{var} · {red}", (var, red))
            for q in sorted(m["classes"], key=lambda q: int(q["code"])):
                it = QListWidgetItem(f"{q['code']} {q['name']}")
                it.setData(Qt.UserRole, int(q["code"]))
                it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                it.setCheckState(Qt.Checked)
                it.setForeground(QBrush(QColor(q["color"]).darker(130)))
                self.prof_classes.addItem(it)
        self.prof_var.blockSignals(False)
        self.prof_classes.blockSignals(False)
        self._draw_profiles()

    def _prof_check_all(self, state):
        self.prof_classes.blockSignals(True)
        for i in range(self.prof_classes.count()):
            self.prof_classes.item(i).setCheckState(Qt.Checked if state else Qt.Unchecked)
        self.prof_classes.blockSignals(False)
        self._draw_profiles()

    def _draw_profiles(self):
        key = self.prof_var.currentData()
        if key is None or key not in self._profiles or self.model is None:
            p.plot_placeholder(self.prof_plot.figure, "Train or load a model to see the class profiles.")
            self.prof_plot.redraw()
            return
        codes = [self.prof_classes.item(i).data(Qt.UserRole) for i in range(self.prof_classes.count())
                 if self.prof_classes.item(i).checkState() == Qt.Checked]
        p.plot_class_profiles(self.prof_plot.figure, self._profiles[key], self.model["classes"], codes,
                              self.chk_prof_band.isChecked(), f"{key[0]} ({key[1]}) per class")
        self.prof_plot.redraw()

    # ------------------------------------------------------------------
    # Training check
    # ------------------------------------------------------------------
    def _build_check_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        row = QHBoxLayout()
        row.addWidget(QLabel("Flag if mean |z| from the own class >"))
        self.chk_thr = QDoubleSpinBox()
        self.chk_thr.setRange(1.0, 10.0)
        self.chk_thr.setSingleStep(0.5)
        self.chk_thr.setValue(2.5)
        self.chk_thr.valueChanged.connect(lambda *_: self._refresh_check())
        row.addWidget(self.chk_thr)
        row.addWidget(QLabel("or closer to another class"))
        row.addStretch(1)
        row.addWidget(style.InfoButton(H.CHECK))
        lay.addLayout(row)
        self.chk_table = QTableWidget(0, 6)
        self.chk_table.setHorizontalHeaderLabels(["Polygon id", "Class", "Nearest class", "Score (|z|)",
                                                  "Pixels", "Reason"])
        self.chk_table.verticalHeader().setVisible(False)
        self.chk_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.chk_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.chk_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)
        self.chk_table.itemSelectionChanged.connect(self._on_check_row)
        lay.addWidget(self.chk_table, 1)
        brow = QHBoxLayout()
        b_sel = QPushButton("Select flagged on map")
        b_sel.clicked.connect(self._select_flagged)
        b_ex = QPushButton("Exclude flagged && retrain")
        b_ex.clicked.connect(self._exclude_flagged)
        b_clear = QPushButton("Clear exclusions")
        b_clear.clicked.connect(self._clear_exclusions)
        for b in (b_sel, b_ex, b_clear):
            brow.addWidget(b)
        lay.addLayout(brow)
        self.chk_info = QLabel("Run a training to check the training samples.")
        self.chk_info.setWordWrap(True)
        self.chk_info.setStyleSheet("color:#666")
        lay.addWidget(self.chk_info)
        return w

    def _refresh_check(self):
        res = self.result
        self.chk_table.setRowCount(0)
        self._suspicious = None
        if res is None or res.get("train_df") is None:
            self.chk_info.setText("Run a training to check the training samples (applied models have none).")
            return
        df = res["train_df"]
        df = df[~df[k.ID_PROP].astype(str).str.startswith("pooled")]
        sus = at.suspicious_samples(df, res["model"]["bands"], self.chk_thr.value())
        self._suspicious = sus
        names = {int(q["code"]): q["name"] for q in res["classes"]}
        self.chk_table.setRowCount(len(sus))
        flag_bg = QBrush(QColor("#fde2e2"))
        for r, row in enumerate(sus.itertuples(index=False)):
            d = row._asdict()
            vals = [str(d[k.ID_PROP]), names.get(int(d[k.CLASS_PROP]), d[k.CLASS_PROP]),
                    names.get(int(d["nearest"]), d["nearest"]),
                    "–" if d["score"] != d["score"] else f"{d['score']:.2f}", str(int(d["n"])), d["reason"]]
            for col, v in enumerate(vals):
                it = QTableWidgetItem(str(v))
                if d["flag"]:
                    it.setBackground(flag_bg)
                self.chk_table.setItem(r, col, it)
        self.chk_table.resizeColumnsToContents()
        n_flag = int(sus["flag"].sum()) if len(sus) else 0
        excl = len(self._active_exclusions())
        where = (f"layer '{self._train_layer_used.name()}'" if self._train_layer_used is not None
                 else "the asset (ids = system:index)")
        self.chk_info.setText(f"{n_flag} of {len(sus)} training polygons flagged (training samples only; "
                              f"test samples are not checked). Ids refer to {where}. "
                              f"{excl} polygon(s) currently excluded." + (" Click a row to show it on the map."
                                                                          if self._train_layer_used else ""))

    def _flagged_ids(self):
        if self._suspicious is None or not len(self._suspicious):
            return []
        return [str(v) for v in self._suspicious.loc[self._suspicious["flag"], k.ID_PROP]]

    def _on_check_row(self):
        layer = self._train_layer_used
        rows = {i.row() for i in self.chk_table.selectedItems()}
        if layer is None or not rows or self._suspicious is None:
            return
        ids = [int(self._suspicious.iloc[r][k.ID_PROP]) for r in rows]
        layer.selectByIds(ids)
        self.canvas.zoomToSelected(layer)
        self.canvas.flashFeatureIds(layer, ids)

    def _select_flagged(self):
        layer = self._train_layer_used
        if layer is None:
            self.dock._warn("Selecting works for training polygons from a QGIS layer.")
            return
        layer.selectByIds([int(i) for i in self._flagged_ids()])
        self.canvas.zoomToSelected(layer)

    def _exclude_flagged(self):
        ids = self._flagged_ids()
        if not ids:
            self.dock._warn("Nothing flagged.")
            return
        key = self._source_key()[:2]
        if self._excluded_key != key:
            self.excluded = set()
        self._excluded_key = key
        self.excluded |= set(ids)
        self._set_status(f"{len(self.excluded)} training polygon(s) excluded — retraining …")
        self.on_train()

    def _clear_exclusions(self):
        self.excluded = set()
        self._excluded_key = None
        self._refresh_check()
        self._set_status("Exclusions cleared — they apply from the next training run.")

    # ------------------------------------------------------------------
    # Field-level results
    # ------------------------------------------------------------------
    def on_fields(self):
        res = self.result
        if res is None:
            self.dock._warn("Run a classification first.")
            return
        layer = self.fld_layer.currentLayer()
        if not pl.is_polygon_layer(layer):
            self.dock._warn("Choose the field polygon layer.")
            return
        if self._task is not None:
            return
        feats = list(layer.getSelectedFeatures() if self.chk_fld_selected.isChecked() else layer.getFeatures())
        polys = []
        for f in feats:
            g = f.geometry()
            if g is None or g.isEmpty():
                continue
            polys.append({"geojson": json.loads(pl.to_wgs84(g, layer.crs()).asJson(7)), "id": f.id()})
        if not polys:
            self.dock._warn("No (selected) polygons.")
            return
        self._fld_source = layer
        if res.get("local"):
            self._start(f"Field statistics of {len(polys)} polygon(s)", _field_majority_local,
                        {"raster_path": res["map_path"], "polygons": polys}, on_done=self._on_fields_done)
        else:
            if not self.dock._ensure_ee():
                return
            self._start(f"Field statistics of {len(polys)} polygon(s)", k.field_majority_ee,
                        {"class_map": res["map"], "polygons": polys, "scale": self.exp_scale.value()},
                        on_done=self._on_fields_done)

    def _on_fields_done(self, task, ok):
        if not self._finish(task, ok, "Field statistics"):
            return
        df = task.result.set_index("id")
        src = self._fld_source
        classes = {int(q["code"]): q for q in self.result["classes"]}
        fields = QgsFields(src.fields())
        new = [("cls_code", QVariant.Int), ("cls_name", QVariant.String), ("cls_share", QVariant.Double),
               ("cls2_code", QVariant.Int), ("cls2_name", QVariant.String), ("cls2_share", QVariant.Double),
               ("n_pixels", QVariant.Int)]
        for name, typ in new:
            if fields.indexOf(name) < 0:
                fields.append(QgsField(name, typ))
        geom = QgsWkbTypes.displayString(QgsWkbTypes.multiType(src.wkbType()))
        out = QgsVectorLayer(f"{geom}?crs={src.crs().authid() or 'EPSG:4326'}",
                             f"{src.name()} – classes {self.result['year']}", "memory")
        out.setCrs(src.crs())
        out.dataProvider().addAttributes(fields.toList())
        out.updateFields()
        feats = []
        n_ok = 0
        for f in src.getFeatures():
            if f.id() not in df.index:
                continue
            r = df.loc[f.id()]
            nf = QgsFeature(out.fields())
            g = f.geometry()
            g.convertToMultiType()
            nf.setGeometry(g)
            for name in src.fields().names():
                nf[name] = f[name]
            code = None if r["class_code"] != r["class_code"] or r["class_code"] is None else int(r["class_code"])
            code2 = None if r["second_code"] != r["second_code"] or r["second_code"] is None else int(r["second_code"])
            nf["cls_code"] = code
            nf["cls_name"] = classes.get(code, {}).get("name") if code is not None else None
            nf["cls_share"] = None if code is None else round(float(r["share"]), 4)
            nf["cls2_code"] = code2
            nf["cls2_name"] = classes.get(code2, {}).get("name") if code2 is not None else None
            nf["cls2_share"] = None if code2 is None else round(float(r["second_share"]), 4)
            nf["n_pixels"] = int(r["n_pixels"])
            n_ok += code is not None
            feats.append(nf)
        out.dataProvider().addFeatures(feats)
        cats = [QgsRendererCategory(q["name"], pl._class_symbol(q["color"]), f"{code} {q['name']}")
                for code, q in sorted(classes.items())]
        cats.append(QgsRendererCategory("", pl._class_symbol(pl.NA_COLOUR), "no pixels"))
        out.setRenderer(QgsCategorizedSymbolRenderer("cls_name", cats))
        QgsProject.instance().addMapLayer(out)
        low = int((df["share"].dropna() < 0.6).sum())
        self._set_status(f"{n_ok} of {len(feats)} fields classified; {low} with < 60 % majority share "
                         "(filter on cls_share). The layer is temporary — right click → Make Permanent.")
        self.iface.messageBar().pushSuccess("GEE classification", f"Field layer '{out.name()}' added")

    def cleanup(self):
        if self._task is not None:
            self._task.cancel()
