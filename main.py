# C:\Users\cdaniel\visionx-neural\main.py
"""
Ponto de entrada principal do VisionX Neural.
Inicializa o Painel de Controle.
"""
import sys

from PyQt6.QtWidgets import QApplication

from src.core.anomaly_memory_integration import install_anomaly_memory_integration
from src.core.experts.knn_expert import KNNExpert
from src.core.experts.semantic_calibration import (
    install_semantic_calibration,
    install_semantic_widget_calibration,
)
from src.core.experts.semantic_expert import SemanticExpert
from src.core.inverted_face_integration import install_inverted_face_integration
from src.core.inverted_signature_extension import install_inverted_signature_extension
from src.core.moe_orchestrator import MoEOrchestrator
from src.core.semantic_roi_extension import (
    install_semantic_roi_extension,
    install_semantic_roi_widget,
)
from src.core.strict_category_memory import install_strict_category_memory
from src.services.anomaly_learning import install_anomaly_learning
from src.ui.control_panel import ControlPanel
from src.ui.decision_panel import install_decision_panel
from src.ui.iconography import install_iconography_hooks, install_svg_iconography
from src.ui.inverted_face_panel import install_inverted_face_panel
from src.ui.missing_component_panel import install_missing_component_panel
from src.ui.operational_controls import (
    OperationalControlsPresenter,
    install_operational_controls,
)
from src.ui.test_mode_dataset_controls import install_test_mode_dataset_controls
from src.ui.widgets.semantic_dna import SemanticDNAWidget


def main():
    app = QApplication(sys.argv)

    # Os hooks precisam ser instalados antes de o controller conectar os sinais.
    install_iconography_hooks(ControlPanel, OperationalControlsPresenter)
    install_semantic_calibration(SemanticExpert)
    install_semantic_roi_extension(SemanticExpert)
    install_semantic_widget_calibration(SemanticDNAWidget)
    install_semantic_roi_widget(SemanticDNAWidget)
    install_strict_category_memory(KNNExpert)
    install_anomaly_memory_integration(MoEOrchestrator)
    install_inverted_signature_extension()
    install_inverted_face_integration(MoEOrchestrator)
    install_anomaly_learning(ControlPanel)

    panel = ControlPanel()
    install_missing_component_panel(panel)
    install_inverted_face_panel(panel)
    install_decision_panel(panel)
    install_test_mode_dataset_controls(panel)
    install_operational_controls(panel)
    install_svg_iconography(panel)
    panel.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
