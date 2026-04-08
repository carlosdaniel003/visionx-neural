# C:\Users\cdaniel\visionx-neural\main.py
"""
Ponto de entrada principal do VisionX Neural.
Inicializa o Painel de Controle.
"""
import sys
from PyQt6.QtWidgets import QApplication
from src.ui.control_panel import ControlPanel

def main():
    app = QApplication(sys.argv)
    
    # Inicia direto no painel de controle
    panel = ControlPanel()
    panel.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()