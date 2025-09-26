from PyQt5.QtWidgets import QPushButton, QLabel, QLabel
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt, QPoint

import pkg_resources

# ---------------------
# ------ Buttons ------
# ---------------------

def create_info_button(self, tooltip_html: str) -> QPushButton:
    """Creates an info button that displays a floating tooltip on hover or click."""
    button = QPushButton("ℹ️")
    button.setCursor(Qt.PointingHandCursor)
    button.setFixedSize(24, 24)
    button.setProperty("pinned", False)
    button.setStyleSheet("""
        QPushButton {
            border: none;
            background-color: transparent;
            font-size: 16px;
            color: black;
        }
        QPushButton:hover {
            color: #007ACC;
        }
        QPushButton:pressed {
            color: #004b8d;
        }
        QPushButton[pinned="true"] {
            color: #007ACC;
            font-weight: bold;
        }
    """)

    # Tooltip QLabel
    tooltip_label = QLabel(self)
    tooltip_label.setText(tooltip_html)
    tooltip_label.setWindowFlags(Qt.ToolTip)
    tooltip_label.setStyleSheet("""
        QLabel {
            background-color: #ffffe0;
            border: 1px solid gray;
            padding: 8px;
            font-size: 10pt;
        }
    """)
    tooltip_label.hide()

    # State to track if tooltip is pinned
    tooltip_pinned = {"value": False} 

    def show_tooltip():
        pos = button.mapToGlobal(button.rect().bottomRight())
        tooltip_label.move(pos + QPoint(10, 10))
        tooltip_label.adjustSize()
        tooltip_label.show()

    def toggle_tooltip():
        if tooltip_pinned["value"]:
            tooltip_label.hide()
            tooltip_pinned["value"] = False
            button.setProperty("pinned", False)
        else:
            show_tooltip()
            tooltip_pinned["value"] = True
            button.setProperty("pinned", True)
        button.setStyle(button.style())  # force refresh

    # Connect events
    button.clicked.connect(toggle_tooltip)
    button.installEventFilter(self)

    if not hasattr(self, "_info_tooltips"):
        self._info_tooltips = {}
    self._info_tooltips[button] = (tooltip_label, tooltip_pinned)

    return button


def create_icon_button(tip_text: str, icon_path: str, on_click: callable) -> QPushButton:
    """
    Create a button with an icon and tooltip.
    Args:
        tip_text (str): The tooltip text to display on hover.
        icon_path (str): The path to the icon image.
        on_click (callable): The function to call when the button is clicked.
    Returns:
        QPushButton: The created button with the specified icon and tooltip.
    """
    button = QPushButton()
    button.setIcon(QIcon(pkg_resources.resource_filename("rl_coppelia", icon_path)))
    button.setToolTip(tip_text)
    button.setFixedSize(24, 24)
    button.setVisible(False)
    button.clicked.connect(on_click)

    return button


def create_styled_button(self, text: str, on_click: callable) -> QPushButton:
    """Create a consistently styled action button."""
    button = QPushButton(text)
    button.setFixedSize(220, 50)
    button.setStyleSheet("""
        QPushButton {
            background-color: #007ACC;
            color: white;
            font-size: 16px;
            font-weight: bold;
            border-radius: 10px;
            padding: 10px 20px;
        }
        QPushButton:hover {
            background-color: #005fa3;
        }
        QPushButton:pressed {
            background-color: #00487a;
        }
    """)
    button.clicked.connect(on_click)
    return button
    

