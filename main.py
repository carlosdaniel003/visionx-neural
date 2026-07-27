# C:\Users\cdaniel\visionx-neural\main.py
"""
Ponto de entrada principal do VisionX Neural.
Inicializa o Painel de Controle.
"""
import sys
from PyQt6.QtWidgets import QApplication
from src.ui.control_panel import ControlPanel
from src.ui.decision_panel import install_decision_panel
from src.ui.test_mode_dataset_controls import install_test_mode_dataset_controls


def main():
    app = QApplication(sys.argv)

    panel = ControlPanel()
    install_decision_panel(panel)
    install_test_mode_dataset_controls(panel)
    panel.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
