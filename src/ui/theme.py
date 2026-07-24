"""Tema visual centralizado do VisionX Neural."""

BACKGROUND = "#050505"
SURFACE = "#0d0d0d"
SURFACE_RAISED = "#151515"
SURFACE_HOVER = "#202020"
BORDER = "#303030"
ACCENT = "#f5c518"
ACCENT_HOVER = "#ffd84d"
ACCENT_SOFT = "#2b2406"
TEXT = "#f5f5f5"
TEXT_MUTED = "#a6a6a6"
TEXT_DIM = "#737373"
SUCCESS = "#4ade80"
DANGER = "#ff6262"
WARNING = "#f5c518"

APP_STYLESHEET = f"""
QWidget {{
    background-color: transparent;
    color: {TEXT};
    font-family: "Segoe UI", Arial, sans-serif;
}}

QWidget#rootWindow,
QWidget#rootContent,
QWidget#rootViewport {{
    background-color: {BACKGROUND};
}}

QFrame#headerFrame,
QFrame#sectionPanel,
QFrame#infoSection,
QFrame#confidenceFrame,
QFrame#controlsSection {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 10px;
}}

QFrame#infoCard,
QFrame#footerCard,
QFrame#imageCard,
QFrame#debugCard,
QFrame#modeCard,
QFrame#latencyCard {{
    background-color: {SURFACE_RAISED};
    border: 1px solid {BORDER};
    border-radius: 8px;
}}

QFrame#infoCard:hover,
QFrame#imageCard:hover,
QFrame#debugCard:hover {{
    border: 1px solid {ACCENT};
}}

QLabel#pageTitle {{
    color: {TEXT};
    font-size: 22px;
    font-weight: 800;
}}

QLabel#pageSubtitle,
QLabel#eyebrowLabel {{
    color: {ACCENT};
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1px;
}}

QLabel#sectionTitle {{
    color: {TEXT};
    font-size: 13px;
    font-weight: 800;
}}

QLabel#sectionHint,
QLabel#fieldLabel,
QLabel#metricName,
QLabel#imageTitle {{
    color: {TEXT_MUTED};
    font-size: 10px;
    font-weight: 600;
}}

QLabel#fieldValue {{
    color: {TEXT};
    font-size: 13px;
    font-weight: 700;
}}

QLabel#accentValue,
QLabel#latencyLabel {{
    color: {ACCENT};
    font-size: 13px;
    font-weight: 800;
}}

QComboBox#modeSelector {{
    background-color: {ACCENT_SOFT};
    color: {ACCENT};
    border: 1px solid {ACCENT};
    border-radius: 7px;
    padding: 7px 30px 7px 12px;
    font-size: 13px;
    font-weight: 800;
    min-width: 145px;
}}
QComboBox#modeSelector:hover {{ background-color: #352d08; }}
QComboBox#modeSelector::drop-down {{ border: none; width: 24px; }}
QComboBox#modeSelector QAbstractItemView {{
    background-color: {SURFACE_RAISED};
    color: {TEXT};
    border: 1px solid {ACCENT};
    selection-background-color: {ACCENT};
    selection-color: #000000;
}}

QLabel#imageViewport,
QLabel#focusViewport {{
    background-color: #090909;
    border: 1px solid {BORDER};
    border-radius: 6px;
    color: {TEXT_DIM};
}}
QLabel#focusViewport {{ border: 1px solid {ACCENT}; }}

QPushButton {{
    min-height: 38px;
    border-radius: 7px;
    padding: 7px 12px;
    font-size: 12px;
    font-weight: 800;
}}
QPushButton#primaryButton {{
    background-color: {ACCENT};
    color: #080808;
    border: 1px solid {ACCENT};
}}
QPushButton#primaryButton:hover {{ background-color: {ACCENT_HOVER}; border-color: {ACCENT_HOVER}; }}
QPushButton#secondaryButton,
QPushButton#lightButton {{
    background-color: {SURFACE_RAISED};
    color: {TEXT};
    border: 1px solid {BORDER};
}}
QPushButton#secondaryButton:hover,
QPushButton#lightButton:hover {{
    color: {ACCENT};
    border-color: {ACCENT};
    background-color: {SURFACE_HOVER};
}}
QPushButton#outlineAccentButton {{
    background-color: {SURFACE_RAISED};
    color: {ACCENT};
    border: 1px solid {ACCENT};
}}
QPushButton#outlineAccentButton:hover {{ background-color: {ACCENT_SOFT}; }}
QPushButton#deleteDatasetButton {{
    background-color: {ACCENT_SOFT};
    color: {ACCENT};
    border: 1px solid {ACCENT};
}}
QPushButton#deleteDatasetButton:hover {{
    background-color: {ACCENT};
    color: #080808;
}}
QPushButton:disabled {{
    background-color: #0a0a0a;
    color: #505050;
    border: 1px solid #222222;
}}

QScrollArea {{ border: none; background-color: transparent; }}
QScrollBar:vertical {{ background: #0a0a0a; width: 11px; margin: 0; }}
QScrollBar::handle:vertical {{ background: #3a3a3a; min-height: 28px; border-radius: 5px; }}
QScrollBar::handle:vertical:hover {{ background: {ACCENT}; }}
QScrollBar:horizontal {{ background: #0a0a0a; height: 11px; margin: 0; }}
QScrollBar::handle:horizontal {{ background: #3a3a3a; min-width: 28px; border-radius: 5px; }}
QScrollBar::handle:horizontal:hover {{ background: {ACCENT}; }}

QFrame#statusBar {{
    background-color: #090909;
    border-top: 1px solid {ACCENT};
    border-radius: 0;
}}
QFrame#statusBar QLabel {{
    background: transparent;
    border: none;
    color: {TEXT_MUTED};
    font-size: 10px;
    font-weight: 700;
}}

QToolTip {{
    background-color: {ACCENT};
    color: #000000;
    border: 1px solid {ACCENT_HOVER};
    padding: 5px;
}}
"""
