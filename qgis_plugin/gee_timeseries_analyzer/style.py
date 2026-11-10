"""Look of the dock: a calm, flat style on top of the QGIS theme. Colours
come from the palette (so dark themes keep working); one accent colour
marks the section switch, the active tab and the main action buttons
(objectName "primary"); InfoButton = the hover help."""
from qgis.PyQt.QtCore import Qt, QObject, QEvent, QPointF, QTimer
from qgis.PyQt.QtGui import QCursor, QPainter, QPalette
import os

from qgis.PyQt.QtWidgets import (QToolButton, QToolTip, QWidget, QHBoxLayout, QSpinBox, QSplitter, QSplitterHandle,
                                 QScrollArea, QFrame, QLabel)

ACCENT = "#1f6f8b"
ACCENT_DARK = "#185a71"

QSS = f"""
QLabel#appTitle {{ font-size: 12pt; font-weight: 600; }}
QLabel#appSub {{ color: palette(dark); font-size: 8pt; }}
QFrame#appHeader {{ border-bottom: 1px solid palette(midlight); }}

QToolButton#seg {{
    padding: 5px 12px; border: 1px solid palette(mid); background: palette(button);
    border-radius: 0; font-weight: 500;
}}
QToolButton#seg[pos="first"] {{ border-top-left-radius: 4px; border-bottom-left-radius: 4px; }}
QToolButton#seg[pos="last"] {{ border-top-right-radius: 4px; border-bottom-right-radius: 4px; }}
QToolButton#seg:hover {{ background: palette(midlight); }}
QToolButton#seg:checked {{ background: {ACCENT}; border-color: {ACCENT}; color: white; }}

QTabWidget::pane {{ border: 1px solid palette(midlight); border-radius: 4px; top: -1px; }}
QTabBar::tab {{
    background: transparent; border: none; border-bottom: 2px solid transparent;
    padding: 5px 9px; margin-right: 2px; color: palette(text);
}}
QTabBar::tab:hover {{ border-bottom: 2px solid palette(mid); }}
QTabBar::tab:selected {{ border-bottom: 2px solid {ACCENT}; color: {ACCENT}; }}

QGroupBox {{
    border: 1px solid palette(midlight); border-radius: 5px;
    margin-top: 12px; padding: 10px 6px 6px 6px;
}}
QGroupBox::title {{
    subcontrol-origin: margin; subcontrol-position: top left; left: 8px; padding: 0 4px;
    color: {ACCENT}; font-weight: 600;
}}

QPushButton#primary {{
    background: {ACCENT}; color: white; border: none; border-radius: 4px;
    padding: 6px 14px; font-weight: 600;
}}
QPushButton#primary:hover {{ background: {ACCENT_DARK}; }}
QPushButton#primary:disabled {{ background: palette(mid); color: palette(light); }}

QProgressBar {{ border: none; background: palette(midlight); border-radius: 3px; max-height: 6px; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 3px; }}
"""


CORES_KEY = "gee_timeseries_analyzer/cpu_cores"
REQUESTS_KEY = "gee_timeseries_analyzer/ee_requests"
SPLITTER_KEY = "gee_timeseries_analyzer/splitter/"


class _Grip(QSplitterHandle):
    """Splitter handle with a visible grip (three dots); double click =
    default sizes."""

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(self.palette().color(QPalette.Mid if self.underMouse() else QPalette.Midlight).darker(115))
        c = self.rect().center()
        vertical = self.orientation() == Qt.Vertical
        for k in (-8, 0, 8):
            x, y = (c.x() + k, c.y()) if vertical else (c.x(), c.y() + k)
            p.drawEllipse(QPointF(x, y + 0.5), 1.8, 1.8)

    def enterEvent(self, event):
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.update()
        super().leaveEvent(event)

    def mouseDoubleClickEvent(self, _event):
        self.splitter().reset()


class Splitter(QSplitter):
    """Sections the user can resize by dragging the grip between them
    (vertical by default). Sizes are remembered per `key` across sessions;
    a double click on a grip restores `defaults`. Sections marked
    collapsible can be dragged closed (drag the grip back to reopen)."""

    def __init__(self, key, orientation=Qt.Vertical, parent=None):
        super().__init__(orientation, parent)
        self._key = SPLITTER_KEY + key
        self._defaults = None
        self._restored = False
        self.setHandleWidth(9)
        self.setChildrenCollapsible(False)
        self.splitterMoved.connect(self._save)

    def createHandle(self):
        return _Grip(self.orientation(), self)

    def add(self, widget, stretch=0, collapsible=False):
        self.addWidget(widget)
        i = self.count() - 1
        self.setStretchFactor(i, stretch)
        self.setCollapsible(i, collapsible)
        return widget

    def set_defaults(self, sizes):
        """Default size (px) per section; None = the rest of the space."""
        self._defaults = list(sizes)

    def _default_sizes(self):
        if not self._defaults:
            return None
        total = (self.height() if self.orientation() == Qt.Vertical else self.width()) \
            - self.handleWidth() * (self.count() - 1)
        fixed = sum(int(s) for s in self._defaults if s is not None)
        rest = max(120, total - fixed) // max(1, sum(s is None for s in self._defaults))
        return [rest if s is None else int(s) for s in self._defaults]

    def showEvent(self, event):
        super().showEvent(event)
        if not self._restored:
            self._restored = True
            QTimer.singleShot(0, self._restore)          # after the first layout: real sizes known

    def _restore(self):
        from qgis.core import QgsSettings
        saved = QgsSettings().value(self._key)
        try:
            sizes = [int(v) for v in saved] if isinstance(saved, (list, tuple)) else None
        except (TypeError, ValueError):
            sizes = None
        sizes = sizes if sizes and len(sizes) == self.count() and sum(sizes) > 0 else self._default_sizes()
        if sizes:
            self.setSizes(sizes)

    def _save(self, *_):
        from qgis.core import QgsSettings
        QgsSettings().setValue(self._key, [int(s) for s in self.sizes()])

    def reset(self):
        from qgis.core import QgsSettings
        QgsSettings().remove(self._key)
        sizes = self._default_sizes()
        if sizes:
            self.setSizes(sizes)


class WrapLabel(QLabel):
    """Word-wrapping label that does not report height-for-width: inside a
    scroll area (the dock sections) such a label makes Qt size the whole
    page by its preferred height -- an outer scroll bar and splitter
    sections taller than the view -- instead of fitting the page in."""

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setWordWrap(True)

    def hasHeightForWidth(self):
        return False


def scroll_pane(widget):
    """A splitter section that scrolls when it is made smaller than its content."""
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QFrame.NoFrame)
    area.setWidget(widget)
    return area


def cpu_count():
    return max(1, os.cpu_count() or 1)


def cores_spin():
    """CPU cores for the offline engine; default all but one (QGIS stays responsive)."""
    from qgis.core import QgsSettings
    spin = QSpinBox()
    spin.setRange(1, cpu_count())
    spin.setSuffix(f" of {cpu_count()}")
    spin.setValue(min(cpu_count(), int(QgsSettings().value(CORES_KEY, max(1, cpu_count() - 1)))))
    spin.valueChanged.connect(lambda v: QgsSettings().setValue(CORES_KEY, int(v)))
    return spin


def requests_spin():
    """Parallel Earth Engine requests (tile downloads, cross-validation folds)."""
    from qgis.core import QgsSettings
    spin = QSpinBox()
    spin.setRange(1, 16)
    spin.setValue(int(QgsSettings().value(REQUESTS_KEY, 4)))
    spin.valueChanged.connect(lambda v: QgsSettings().setValue(REQUESTS_KEY, int(v)))
    return spin


def requests_value():
    from qgis.core import QgsSettings
    return max(1, min(16, int(QgsSettings().value(REQUESTS_KEY, 4))))


def primary(button):
    """Marks a button as the main action of its panel."""
    button.setObjectName("primary")
    button.setStyleSheet("")
    return button


# ---------------------------------------------------------------------------
# Info buttons: a small "i" that shows its explanation as soon as the mouse
# is over it (no tooltip delay).
# ---------------------------------------------------------------------------

INFO_QSS = f"""
QToolButton#info {{
    min-width: 15px; max-width: 15px; min-height: 15px; max-height: 15px; padding: 0;
    border: 1px solid {ACCENT}; border-radius: 8px; background: palette(base); color: {ACCENT};
    font-family: Georgia, serif; font-style: italic; font-weight: bold; font-size: 8pt;
}}
QToolButton#info:hover {{ background: {ACCENT}; color: white; }}
"""
QSS += INFO_QSS


class InfoButton(QToolButton):
    def __init__(self, text, parent=None):
        super().__init__(parent)
        self.setObjectName("info")
        self.setText("i")
        self.setFocusPolicy(Qt.NoFocus)
        self.setCursor(Qt.WhatsThisCursor)
        self.setToolTip(f"<div style='max-width:360px'>{text}</div>")

    def enterEvent(self, event):
        QToolTip.showText(QCursor.pos(), self.toolTip(), self)
        super().enterEvent(event)

    def mousePressEvent(self, event):      # click = show again (touchpads)
        QToolTip.showText(QCursor.pos(), self.toolTip(), self)


class _Corner(QObject):
    """Keeps an info button in the top-right corner of its group box."""

    def __init__(self, box, button):
        super().__init__(box)
        self.box, self.button = box, button
        box.installEventFilter(self)
        self._place()

    def eventFilter(self, obj, event):
        if event.type() in (QEvent.Resize, QEvent.Show):
            self._place()
        return False

    def _place(self):
        self.button.move(self.box.width() - self.button.width() - 8, 3)
        self.button.raise_()


def attach_info(box, text):
    """Puts an info button into the title line of a QGroupBox."""
    btn = InfoButton(text, box)
    btn.resize(17, 17)
    _Corner(box, btn)
    return btn


def with_info(widget, text):
    """widget + info button in one row (for form rows)."""
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(6)
    lay.addWidget(widget, 1)
    lay.addWidget(InfoButton(text))
    return w
