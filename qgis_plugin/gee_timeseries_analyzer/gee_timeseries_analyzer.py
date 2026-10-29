import os

from qgis.core import QgsApplication
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction

from .dock_widget import GEEDockWidget, TITLE
from .processing_provider import GEEProcessingProvider


class GEETimeSeriesAnalyzerPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.action = None
        self.dock_widget = None
        self.processing_provider = None

    def initGui(self):
        icon_path = os.path.join(os.path.dirname(__file__), "icons", "gee_icon.svg")
        self.action = QAction(QIcon(icon_path), TITLE, self.iface.mainWindow())
        self.action.setCheckable(True)
        self.action.triggered.connect(self.toggle_dock)

        self.iface.addToolBarIcon(self.action)
        self.iface.addPluginToMenu(f"&{TITLE}", self.action)

        self.processing_provider = GEEProcessingProvider()
        QgsApplication.processingRegistry().addProvider(self.processing_provider)

    def toggle_dock(self, checked):
        if self.dock_widget is None:
            self.dock_widget = GEEDockWidget(self.iface, self.iface.mainWindow())
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
