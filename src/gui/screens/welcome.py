# -*- coding: utf-8 -*-
"""Welcome screen widget."""

from __future__ import annotations
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton
from PyQt5.QtGui import QPixmap
import pkg_resources

class WelcomeScreen(QWidget):
    """Simple welcome screen with a 'continue' CTA."""
    continue_clicked = pyqtSignal()

    def __init__(self, parent=None):
        """Build the welcome UI."""
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        """Create the UI layout."""
        main_layout = QHBoxLayout(self)

        # Left: logo
        logo_label = QLabel()
        logo_path = pkg_resources.resource_filename("rl_coppelia", "../gui/assets/uncore.png")
        logo_pixmap = QPixmap(logo_path).scaledToHeight(180, Qt.SmoothTransformation)
        logo_label.setPixmap(logo_pixmap)
        logo_label.setAlignment(Qt.AlignCenter)

        left = QVBoxLayout()
        left.addStretch()
        left.addWidget(logo_label)
        left.addStretch()
        left_w = QWidget(); left_w.setLayout(left)
        main_layout.addWidget(left_w, stretch=1)

        # Right: text + button
        right = QVBoxLayout()
        title = QLabel("<h1>Welcome to RL Coppelia GUI</h1>")
        title.setAlignment(Qt.AlignLeft)
        right.addWidget(title)

        team_text = QLabel("""
            <p>
            Created by <b>UnCoRE: UNexpected COgnitive, Robotics & Education Team</b>, a enthusiastic bunch of 
            researchers and teachers working at Universidad de Málaga.
            </p>
        """)
        team_text.setWordWrap(True)
        right.addWidget(team_text)

        desc = QLabel("""
            <p>
            This application helps you manage, test, train and analyze reinforcement learning experiments
            in robotic environments simulated in CoppeliaSim.
            </p>
            <p>Have fun!</p>
        """)
        desc.setWordWrap(True)
        right.addWidget(desc)

        license_text = QLabel("""
            <p style='font-size:10pt; color:gray;'>
            Licensed under the <b>GNU General Public License v3.0</b>
            </p>
        """)
        license_text.setWordWrap(True)
        right.addWidget(license_text)

        # CTA button
        btn = QPushButton("Let's go")
        btn.setFixedSize(150, 50)
        btn.clicked.connect(self.continue_clicked.emit)
        btn.setStyleSheet("""
            QPushButton {
                background-color: #007ACC;
                color: white;
                font-size: 16px;
                font-weight: bold;
                border-radius: 10px;
                padding: 10px 20px;
            }
            QPushButton:hover { background-color: #005fa3; }
            QPushButton:pressed { background-color: #00487a; }
        """)

        right.addSpacing(30)
        btn_row = QHBoxLayout(); btn_row.addStretch(); btn_row.addWidget(btn); btn_row.addStretch()
        right.addLayout(btn_row)
        right.addStretch()

        right_w = QWidget(); right_w.setLayout(right)
        main_layout.addWidget(right_w, stretch=2)
