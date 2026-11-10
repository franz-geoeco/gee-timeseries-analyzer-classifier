import os

from qgis.core import Qgis, QgsApplication, QgsMessageLog
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction, QMessageBox, QPushButton

from . import env_check

TITLE = "GEE Time Series Analyzer & Classifier"


class GEETimeSeriesAnalyzerPlugin:
    """The dock and the Processing provider import numpy / pandas /
    matplotlib, so they are loaded only after env_check: a broken Python
    environment then gives an explanation instead of a cryptic traceback."""

    def __init__(self, iface):
        self.iface = iface
        self.action = None
        self.dock_widget = None
        self.processing_provider = None
        self._env_problem = None

    def initGui(self):
        icon_path = os.path.join(os.path.dirname(__file__), "icons", "gee_icon.svg")
        self.action = QAction(QIcon(icon_path), TITLE, self.iface.mainWindow())
        self.action.setCheckable(True)
        self.action.triggered.connect(self.toggle_dock)

        self.iface.addToolBarIcon(self.action)
        self.iface.addPluginToMenu(f"&{TITLE}", self.action)

        conflict = env_check.numpy_conflict()
        if conflict is not None:
            self._env_problem = env_check.numpy_conflict_text(conflict)
            self._report_env_problem(conflict)
        try:
            from .processing_provider import GEEProcessingProvider
            self.processing_provider = GEEProcessingProvider()
            QgsApplication.processingRegistry().addProvider(self.processing_provider)
        except Exception as e:
            self.processing_provider = None
            if self._env_problem is None:
                raise
            QgsMessageLog.logMessage(f"Processing algorithms not loaded: {e}", TITLE, Qgis.Warning)

    def _report_env_problem(self, conflict):
        QgsMessageLog.logMessage(self._env_problem, TITLE, Qgis.Critical)
        bar = self.iface.messageBar()
        msg = bar.createMessage(TITLE, f"NumPy {conflict['loaded']} from your personal Python folder replaces "
                                       f"QGIS's NumPy {conflict['bundled']} — parts of QGIS and of this plugin "
                                       "will not work.")
        btn = QPushButton("Details / fix")
        btn.clicked.connect(self._show_env_problem)
        msg.layout().addWidget(btn)
        bar.pushWidget(msg, Qgis.Critical, 0)

    def _show_env_problem(self):
        QMessageBox.warning(self.iface.mainWindow(), f"{TITLE}: Python environment", self._env_problem)

    def toggle_dock(self, checked):
        if self.dock_widget is None:
            try:
                from .dock_widget import GEEDockWidget
                self.dock_widget = GEEDockWidget(self.iface, self.iface.mainWindow())
            except Exception:
                self.dock_widget = None
                if self.action is not None:
                    self.action.setChecked(False)
                if self._env_problem is None:
                    raise
                self._show_env_problem()
                return
            self.iface.addDockWidget(Qt.RightDockWidgetArea, self.dock_widget)
            self.dock_widget.visibilityChanged.connect(self._sync_action_state)
        self.dock_widget.setVisible(checked)

    def _sync_action_state(self, visible):
        if self.action is not None:
            self.action.setChecked(visible)

    def unload(self):
        if self.dock_widget is not None:
            self.dock_widget.cleanup()
            self.iface.removeDockWidget(self.dock_widget)
            self.dock_widget.deleteLater()
            self.dock_widget = None
        if self.action is not None:
            self.iface.removePluginMenu(f"&{TITLE}", self.action)
            self.iface.removeToolBarIcon(self.action)
            self.action = None
        if self.processing_provider is not None:
            QgsApplication.processingRegistry().removeProvider(self.processing_provider)
            self.processing_provider = None
