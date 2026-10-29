"""The feature-recipe panel shared by the Classify and Export tabs: year(s),
season, aggregation period, reducers, gap filling, Sentinel-2 variables and
cloud mask, Sentinel-1 variables and orbit, and a live description of the
resulting bands. features() returns the recipe dict of classification.py."""
from qgis.PyQt.QtCore import pyqtSignal
from qgis.PyQt.QtWidgets import (
    QWidget, QFormLayout, QHBoxLayout, QGridLayout, QGroupBox, QLabel, QSpinBox,
    QDoubleSpinBox, QCheckBox, QComboBox,
)

from . import classification as k
from . import gee_client as c
from . import help_texts as H
from . import style


class FeatureRecipeWidget(QWidget):
    changed = pyqtSignal()

    def __init__(self, scale_label="Sampling scale:", year_range=False, parent=None):
        super().__init__(parent)
        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)

        yrow = QHBoxLayout()
        self.year = QSpinBox()
        self.year.setRange(2017, c.MAX_YEAR)
        self.year.setValue(c.MAX_YEAR)
        yrow.addWidget(self.year)
        self.year_to = None
        if year_range:
            self.year_to = QSpinBox()
            self.year_to.setRange(2017, c.MAX_YEAR)
            self.year_to.setValue(c.MAX_YEAR)
            self.year_to.setToolTip("Last year (one set of files per year)")
            yrow.addWidget(QLabel("to"))
            yrow.addWidget(self.year_to)
            yrow.addStretch(1)
        form.addRow("Years:" if year_range else "Year:", yrow)

        srow = QHBoxLayout()
        self.start_month = QComboBox()
        self.end_month = QComboBox()
        for combo in (self.start_month, self.end_month):
            for i, name in enumerate(c.MONTH_NAMES):
                combo.addItem(name, i + 1)
            srow.addWidget(combo)
        self.start_month.setToolTip("If the start month is after the end month, the season starts "
                                    "in the previous year (e.g. October-June for winter crops).")
        form.addRow("Season from / to:", srow)
        self.period = QComboBox()
        for key, label in k.PERIODS.items():
            self.period.addItem(label, key)
        form.addRow("Aggregation period:", style.with_info(self.period, H.PERIOD))

        red_grid = QGridLayout()
        self.reducer_checks = {}
        short = {"median": "Median", "mean": "Mean", "min": "Min", "max": "Max", "p10": "P10", "p25": "P25",
                 "p75": "P75", "p90": "P90", "stdDev": "Std. dev."}
        for i, (key, label) in enumerate(k.REDUCERS.items()):
            cb = QCheckBox(short.get(key, label))
            cb.setToolTip(label)
            self.reducer_checks[key] = cb
            red_grid.addWidget(cb, i // 5, i % 5)
        red_w = QWidget()
        red_w.setLayout(red_grid)
        red_grid.setContentsMargins(0, 0, 0, 0)
        form.addRow("Aggregation:", style.with_info(red_w, H.REDUCERS))
        self.gap_fill = QComboBox()
        for key, label in k.GAP_FILL.items():
            self.gap_fill.addItem(label, key)
        form.addRow("Gap filling:", style.with_info(self.gap_fill, H.GAP_FILL))

        s2_box = QGroupBox("Sentinel-2 SR (harmonized)")
        s2_form = QFormLayout(s2_box)
        s2_grid = QGridLayout()
        self.s2_checks = {}
        for i, key in enumerate(list(k.S2_INDICES) + k.S2_BANDS):
            cb = QCheckBox(key)
            cb.setToolTip(k.S2_INDICES.get(key, f"Surface reflectance {key}"))
            self.s2_checks[key] = cb
            s2_grid.addWidget(cb, i // 6, i % 6)
        s2_form.addRow(s2_grid)
        mrow = QHBoxLayout()
        self.s2_mask = QComboBox()
        for key, label in k.S2_MASKS.items():
            self.s2_mask.addItem(label, key)
        self.cs_thr = QDoubleSpinBox()
        self.cs_thr.setRange(0.3, 0.95)
        self.cs_thr.setSingleStep(0.05)
        self.cs_thr.setToolTip("Cloud Score+: a pixel is clear if cs_cdf ≥ this value "
                               "(0.6 = default, higher = stricter)")
        self.s2_mask.currentIndexChanged.connect(
            lambda *_: self.cs_thr.setEnabled(self.s2_mask.currentData() == "csplus"))
        mrow.addWidget(self.s2_mask, 1)
        mrow.addWidget(QLabel("clear ≥"))
        mrow.addWidget(self.cs_thr)
        mrow.addWidget(style.InfoButton(H.CLOUD_MASK))
        s2_form.addRow("Cloud mask:", mrow)
        self.s2_cloud = QSpinBox()
        self.s2_cloud.setRange(1, 100)
        self.s2_cloud.setSuffix(" %")
        self.s2_cloud.setToolTip("Scenes with a higher CLOUDY_PIXEL_PERCENTAGE are skipped")
        s2_form.addRow("Max. scene cloud cover:", style.with_info(self.s2_cloud, H.SCENE_CLOUD))
        form.addRow(s2_box)

        s1_box = QGroupBox("Sentinel-1 GRD (IW, dB, border noise < −30 dB masked)")
        s1_form = QFormLayout(s1_box)
        s1_row = QHBoxLayout()
        self.s1_checks = {}
        for key, label in k.S1_FEATURES.items():
            cb = QCheckBox(label)
            self.s1_checks[key] = cb
            s1_row.addWidget(cb)
        s1_form.addRow(s1_row)
        self.s1_orbit = QComboBox()
        for key, label in k.S1_ORBITS.items():
            self.s1_orbit.addItem(label, key)
        s1_form.addRow("Orbit:", style.with_info(self.s1_orbit, H.S1_ORBIT))
        form.addRow(s1_box)

        self.scale = QSpinBox()
        self.scale.setRange(10, 1000)
        self.scale.setSuffix(" m")
        form.addRow(scale_label, style.with_info(self.scale, H.SCALE))
        self.info = QLabel()
        self.info.setWordWrap(True)
        self.info.setStyleSheet("color:#666")
        form.addRow(self.info)

        self.set_features(k.DEFAULT_FEATURES)
        for wdg in (self.start_month, self.end_month, self.period, self.gap_fill, self.s2_mask, self.s1_orbit):
            wdg.currentIndexChanged.connect(self._emit)
        for wdg in (self.year, self.s2_cloud, self.scale) + ((self.year_to,) if self.year_to else ()):
            wdg.valueChanged.connect(self._emit)
        self.cs_thr.valueChanged.connect(self._emit)
        for cb in (list(self.reducer_checks.values()) + list(self.s2_checks.values())
                   + list(self.s1_checks.values())):
            cb.toggled.connect(self._emit)
        self.update_info()

    def _emit(self, *_):
        self.update_info()
        self.changed.emit()

    @staticmethod
    def _set_combo(combo, data):
        i = combo.findData(data)
        if i >= 0:
            combo.setCurrentIndex(i)

    def years(self):
        y0 = self.year.value()
        y1 = self.year_to.value() if self.year_to is not None else y0
        return list(range(min(y0, y1), max(y0, y1) + 1))

    def features(self):
        return {
            "start_month": self.start_month.currentData(), "end_month": self.end_month.currentData(),
            "period": self.period.currentData(),
            "reducers": [key for key, cb in self.reducer_checks.items() if cb.isChecked()],
            "gap_fill": self.gap_fill.currentData(),
            "s2_features": [key for key, cb in self.s2_checks.items() if cb.isChecked()],
            "s2_mask": self.s2_mask.currentData(), "cs_threshold": round(self.cs_thr.value(), 3),
            "s2_max_cloud": self.s2_cloud.value(),
            "s1_features": [key for key, cb in self.s1_checks.items() if cb.isChecked()],
            "s1_orbit": self.s1_orbit.currentData(),
            "scale": self.scale.value(),
        }

    def set_features(self, f):
        f = dict(k.DEFAULT_FEATURES, **f)
        self._set_combo(self.start_month, f["start_month"])
        self._set_combo(self.end_month, f["end_month"])
        self._set_combo(self.period, f["period"])
        for key, cb in self.reducer_checks.items():
            cb.setChecked(key in f["reducers"])
        self._set_combo(self.gap_fill, f["gap_fill"])
        for key, cb in self.s2_checks.items():
            cb.setChecked(key in f["s2_features"])
        self._set_combo(self.s2_mask, f["s2_mask"])
        self.cs_thr.setValue(f["cs_threshold"])
        self.cs_thr.setEnabled(f["s2_mask"] == "csplus")
        self.s2_cloud.setValue(f["s2_max_cloud"])
        for key, cb in self.s1_checks.items():
            cb.setChecked(key in f["s1_features"])
        self._set_combo(self.s1_orbit, f["s1_orbit"])
        self.scale.setValue(f["scale"])

    def problem(self, f=None, need_variables=True):
        """Why the recipe can't be used (or None)."""
        f = f or self.features()
        if not f["reducers"]:
            return "Select at least one aggregation."
        if need_variables and not (f["s2_features"] or f["s1_features"]):
            return "Select at least one Sentinel-2 or Sentinel-1 variable."
        return None

    def update_info(self):
        f = self.features()
        msg = self.problem(f)
        self.info.setText(msg or k.describe_features(f))
