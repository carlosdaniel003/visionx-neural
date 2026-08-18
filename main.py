# C:\Users\cdaniel\visionx-neural\main.py
"""
Ponto de entrada principal do VisionX Neural.
Inicializa o Painel de Controle.
"""
import sys

from PyQt6.QtWidgets import QApplication

import src.core.anomaly_memory_integration as anomaly_memory_module
import src.core.best_match_memory as best_match_memory_module
import src.services.dataset_manager as dataset_manager_module
from src.core.anomaly_memory_integration import install_anomaly_memory_integration
from src.core.best_match_memory import install_best_match_memory
from src.core.dual_scale_memory import install_dual_scale_memory
from src.core.experts.knn_expert import KNNExpert
from src.core.experts.missing_component_expert import MissingComponentExpert
from src.core.experts.semantic_calibration import (
    install_semantic_calibration,
    install_semantic_widget_calibration,
)
from src.core.experts.semantic_expert import SemanticExpert
from src.core.experts.silk_expert import SilkExpert
from src.core.experts.ssim_expert import SSIMExpert
from src.core.inverted_face_integration import install_inverted_face_integration
from src.core.inverted_signature_extension import install_inverted_signature_extension
from src.core.memory_hypothesis_contrast import install_memory_hypothesis_contrast
from src.core.moe_orchestrator import MoEOrchestrator
from src.core.prototype_memory import install_prototype_memory
from src.core.roi_input_contract import install_roi_input_contract
from src.core.roi_visual_alignment import install_roi_visual_alignment
from src.core.semantic_roi_extension import (
    install_semantic_roi_extension,
    install_semantic_roi_widget,
)
from src.core.strict_category_memory import install_strict_category_memory
from src.services.anomaly_learning import install_anomaly_learning
from src.services.dataset_manager import DatasetManager
from src.ui.capture_button_copy import install_capture_button_copy
from src.ui.control_panel import ControlPanel
from src.ui.decision_panel import install_decision_panel
from src.ui.iconography import install_iconography_hooks, install_svg_iconography
from src.ui.inverted_face_panel import install_inverted_face_panel
from src.ui.local_capture_safety import install_local_capture_safety
from src.ui.memory_status_ui import install_memory_status_ui
from src.ui.missing_component_panel import install_missing_component_panel
from src.ui.mode_selector_gate import install_mode_selector_gate
from src.ui.network_aoi_intake_filter import install_network_aoi_intake_filter
from src.ui.network_image_cycle_gate import install_network_image_cycle_gate
from src.ui.operational_controls import (
    OperationalControlsPresenter,
    install_operational_controls,
)
from src.ui.production_confidence_gate import install_production_confidence_gate
from src.ui.qt_button_signal_adapter import install_qt_button_signal_adapter
from src.ui.strict_category_memory_ui import install_strict_category_memory_ui
from src.ui.test_mode_dataset_controls import install_test_mode_dataset_controls
from src.ui.widgets.knn_spectrum import KNNSpectrumWidget
from src.ui.widgets.semantic_dna import SemanticDNAWidget


def main():
    app = QApplication(sys.argv)

    # Os hooks precisam ser instalados antes de o controller conectar os sinais.
    install_iconography_hooks(ControlPanel, OperationalControlsPresenter)
    install_capture_button_copy(OperationalControlsPresenter)
    install_semantic_calibration(SemanticExpert)
    install_semantic_roi_extension(SemanticExpert)
    install_semantic_widget_calibration(SemanticDNAWidget)
    install_semantic_roi_widget(SemanticDNAWidget)
    install_strict_category_memory(KNNExpert)
    install_strict_category_memory_ui(KNNSpectrumWidget)

    # Corrige somente o alinhamento posterior ao recorte. A caixa escolhida pelo
    # EpicenterExtractor permanece exatamente a mesma.
    install_roi_visual_alignment(SilkExpert, MissingComponentExpert)

    install_anomaly_memory_integration(MoEOrchestrator)
    # A memória consulta todos os registros da mesma categoria, mas somente a
    # melhor correspondência visual fornece o rótulo. Quantidade não vota.
    install_best_match_memory(KNNExpert, anomaly_memory_module)
    # Expande a memória sem alterar a assinatura local existente: 70% epicentro
    # + 30% contexto visual da maior caixa verde do componente. JSONs antigos
    # continuam usando somente o epicentro.
    install_dual_scale_memory(
        anomaly_memory_module,
        best_match_memory_module,
        dataset_manager_module,
    )
    # Compacta apenas padrões OK redundantes. Cada NG permanece como memória
    # individual protegida; contadores de ocorrência nunca alteram o score.
    install_prototype_memory(
        KNNExpert,
        DatasetManager,
        dataset_manager_module,
        best_match_memory_module,
    )
    # Contrasta explicitamente a melhor hipótese NG com a melhor hipótese OK.
    # Empates confiáveis dentro de 1 ponto percentual ficam inconclusivos e
    # forçam revisão humana, sem desfazer dual-scale ou protótipos.
    install_memory_hypothesis_contrast(
        KNNExpert,
        anomaly_memory_module,
        best_match_memory_module,
    )
    install_inverted_signature_extension()
    install_inverted_face_integration(MoEOrchestrator)
    # Deve ser a última extensão do orquestrador: audita o resultado final de
    # todos os motores e garante a categoria correta no Laboratório de Textura.
    install_roi_input_contract(MoEOrchestrator, SSIMExpert, SilkExpert)

    # Ordem dos wrappers operacionais:
    # 1. aprendizado humano;
    # 2. confiança mínima de produção;
    # 3. trava geral de uma única imagem ativa;
    # 4. filtro de rede: dois frames estáveis + epicentro válido;
    # 5. supervisão externa do MSS e recuperação de exceções;
    # 6. adaptador final que consome o checked(bool) dos QPushButtons;
    # 7. seletor de modo bloqueado durante o ciclo ativo.
    install_anomaly_learning(ControlPanel)
    install_production_confidence_gate(ControlPanel, OperationalControlsPresenter)
    install_network_image_cycle_gate(ControlPanel, OperationalControlsPresenter)
    install_network_aoi_intake_filter(ControlPanel)
    install_local_capture_safety(ControlPanel)
    install_qt_button_signal_adapter(ControlPanel)
    install_mode_selector_gate(OperationalControlsPresenter)

    # Camada exclusivamente visual: resume dual-scale, protótipos e contraste
    # OK x NG sem modificar analysis, score, confiança ou persistência.
    install_memory_status_ui(ControlPanel)

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
