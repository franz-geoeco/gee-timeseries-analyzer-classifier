"""Look of the dock: a calm, flat style on top of the QGIS theme. Colours
come from the palette (so dark themes keep working); one accent colour
marks the section switch, the active tab and the main action buttons
(objectName "primary"); InfoButton = the hover help."""
from qgis.PyQt.QtCore import Qt, QObject, QEvent
from qgis.PyQt.QtGui import QCursor
from qgis.PyQt.QtWidgets import QToolButton, QToolTip, QWidget, QHBoxLayout

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
