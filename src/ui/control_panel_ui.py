# src/ui/control_panel_ui.py
"""
Módulo responsável exclusivamente pela construção da Interface Gráfica (View).
Ajuste de UX/Observabilidade: Adicionada uma Barra de Status Global no rodapé
para exibir em tempo real a saúde da rede, o estado do cérebro da IA e o log da última ação.
"""
from PyQt6.QtWidgets import (QVBoxLayout, QPushButton, QLabel,
                             QHBoxLayout, QFrame, QGridLayout,
                             QApplication, QSizePolicy, QScrollArea, QWidget, QComboBox)
from PyQt6.QtCore import Qt

from src.ui.widgets.radar_chart import RadarChartWidget
from src.ui.widgets.knn_spectrum import KNNSpectrumWidget
from src.ui.widgets.shift_debugger import ShiftDebuggerWidget 
from src.ui.widgets.silk_debugger import SilkDebuggerWidget 
from src.ui.widgets.ssim_debugger import SSIMDebuggerWidget 
from src.ui.widgets.semantic_dna import SemanticDNAWidget

class ControlPanelUI:
    def setup_ui(self, window):
        window.setWindowTitle("VisionX Neural - Deep Debugger Console")
        window.setStyleSheet("background-color: #0d1117; color: #c9d1d9;")

        screen = QApplication.primaryScreen()
        available = screen.availableGeometry()
        window.setGeometry(available)

        main_layout = QVBoxLayout(window)
        main_layout.setContentsMargins(15, 10, 15, 15) 
        main_layout.setSpacing(10)

        self._build_header(window, main_layout)
        self._build_aoi_info(window, main_layout)
        self._build_main_stage(window, main_layout)
        self._build_footer(window, main_layout)
        self._build_action_buttons(window, main_layout)
        
        # =========================================================
        # NOVO: BARRA DE STATUS GLOBAL
        # =========================================================
        self._build_status_bar(window, main_layout)

        window.setWindowState(Qt.WindowState.WindowMaximized)

    def _build_header(self, window, parent_layout):
        top_layout = QHBoxLayout()
        title = QLabel("VisionX Neural • Monitoramento IA")
        title.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #ffffff;")
        
        # =========================================================
        # SELETOR DE MODO DE OPERAÇÃO
        # =========================================================
        window.combo_mode = QComboBox()
        window.combo_mode.addItems(["Modo Sombra", "Modo Teste", "Modo Produção"])
        window.combo_mode.setCurrentText("Modo Teste") # Padrão atual do sistema
        window.combo_mode.setStyleSheet("""
            QComboBox { background-color: #21262d; color: #58a6ff; font-weight: bold; border-radius: 6px; padding: 5px 15px; border: 1px solid #30363d; font-size: 14px; min-width: 150px; }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView { background-color: #161b22; color: #c9d1d9; selection-background-color: #30363d; border: 1px solid #30363d; }
        """)
        window.combo_mode.setCursor(Qt.CursorShape.PointingHandCursor)
        
        window.lbl_timer = QLabel("Latência: 0.00s")
        window.lbl_timer.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        window.lbl_timer.setStyleSheet("font-family: Consolas, monospace; font-size: 14px; font-weight: bold; color: #58a6ff;")

        top_layout.addWidget(title, stretch=1)
        top_layout.addWidget(window.combo_mode)
        top_layout.addSpacing(20)
        top_layout.addWidget(window.lbl_timer)
        parent_layout.addLayout(top_layout)

    def _build_aoi_info(self, window, parent_layout):
        window.aoi_info_frame = QFrame()
        window.aoi_info_frame.setFixedHeight(60)
        window.aoi_info_frame.setStyleSheet("""
            QFrame { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; }
        """)
        aoi_info_layout = QHBoxLayout(window.aoi_info_frame)
        aoi_info_layout.setContentsMargins(20, 10, 20, 10)
        
        def create_info_block(label, value_lbl):
            container = QVBoxLayout()
            container.setSpacing(2)
            lbl_title = QLabel(label)
            lbl_title.setStyleSheet("color: #8b949e; font-size: 11px; border: none;")
            value_lbl.setStyleSheet("color: #ffffff; font-size: 14px; font-weight: bold; border: none;")
            container.addWidget(lbl_title)
            container.addWidget(value_lbl)
            return container

        window.lbl_board_value = QLabel("-")
        window.lbl_parts_value = QLabel("-")
        window.lbl_category_value = QLabel("-")
        window.lbl_category_value.setStyleSheet("color: #ffd33d; font-size: 14px; font-weight: bold; border: none;")
        window.lbl_value_value = QLabel("-")

        aoi_info_layout.addLayout(create_info_block("Placa / Máquina", window.lbl_board_value))
        aoi_info_layout.addStretch()
        aoi_info_layout.addLayout(create_info_block("Componente", window.lbl_parts_value))
        aoi_info_layout.addStretch()
        aoi_info_layout.addLayout(create_info_block("Categoria do Erro", window.lbl_category_value))
        aoi_info_layout.addStretch()
        aoi_info_layout.addLayout(create_info_block("Valor / OCR", window.lbl_value_value))

        parent_layout.addWidget(window.aoi_info_frame)

    def _build_main_stage(self, window, parent_layout):
        stage_layout = QHBoxLayout()
        stage_layout.setSpacing(15)

        # IMAGENS (Lado Esquerdo - Tamanho fixo)
        images_frame = QFrame()
        images_frame.setFixedWidth(300) 
        images_frame.setStyleSheet("background-color: transparent; border: none;")
        images_column_layout = QVBoxLayout(images_frame)
        images_column_layout.setContentsMargins(0, 0, 0, 0)
        
        window.lbl_sample = QLabel("Sem Sinal")
        window.lbl_sample.setAlignment(Qt.AlignmentFlag.AlignCenter)
        window.lbl_sample.setStyleSheet("background-color: #0d1117; border: 1px solid #30363d; border-radius: 6px;")
        window.lbl_sample.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        window.lbl_sample.setMinimumSize(10, 10)
        
        window.lbl_sample_focus = QLabel("Sem Foco") 
        window.lbl_sample_focus.setAlignment(Qt.AlignmentFlag.AlignCenter)
        window.lbl_sample_focus.setStyleSheet("background-color: #0d1117; border: 1px solid #1f6feb; border-radius: 6px;")
        window.lbl_sample_focus.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        window.lbl_sample_focus.setMinimumSize(10, 10)
        
        window.lbl_ng = QLabel("Sem Sinal")
        window.lbl_ng.setAlignment(Qt.AlignmentFlag.AlignCenter)
        window.lbl_ng.setStyleSheet("background-color: #0d1117; border: 1px solid #30363d; border-radius: 6px;")
        window.lbl_ng.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        window.lbl_ng.setMinimumSize(10, 10)
        
        window.lbl_ng_focus = QLabel("Sem Foco") 
        window.lbl_ng_focus.setAlignment(Qt.AlignmentFlag.AlignCenter)
        window.lbl_ng_focus.setStyleSheet("background-color: #0d1117; border: 1px solid #da3633; border-radius: 6px;")
        window.lbl_ng_focus.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        window.lbl_ng_focus.setMinimumSize(10, 10)

        images_column_layout.addWidget(QLabel("Gabarito"))
        images_column_layout.addWidget(window.lbl_sample, stretch=2)
        images_column_layout.addWidget(window.lbl_sample_focus, stretch=1)
        images_column_layout.addWidget(QLabel("Teste"))
        images_column_layout.addWidget(window.lbl_ng, stretch=2)
        images_column_layout.addWidget(window.lbl_ng_focus, stretch=1)
        stage_layout.addWidget(images_frame)

        # TELEMETRIA (Lado Direito - Scroll com trava de card)
        telemetry_layout = QVBoxLayout()
        title_telemetry = QLabel("DEBUGGERS DA IA")
        title_telemetry.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_telemetry.setStyleSheet("color: #00ffaa; font-size: 12px; font-weight: bold; background-color: #112211; padding: 4px; border: 1px solid #114411;")
        telemetry_layout.addWidget(title_telemetry)

        window.scroll_area = QScrollArea()
        window.scroll_area.setWidgetResizable(True)
        window.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        window.scroll_area.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")

        window.scroll_content = QWidget()
        window.scroll_layout = QHBoxLayout(window.scroll_content)
        window.scroll_layout.setContentsMargins(5, 5, 5, 5)
        window.scroll_layout.setSpacing(15)

        # CONFIGURAÇÃO DE CARDS
        def setup_card(widget):
            widget.setMinimumWidth(450)
            widget.setMaximumWidth(600) 
            widget.setMinimumHeight(280)
            window.scroll_layout.addWidget(widget)

        window.frame_ssim_debug = SSIMDebuggerWidget()
        window.frame_silk = SilkDebuggerWidget()
        window.frame_dna = SemanticDNAWidget()
        window.frame_shift = ShiftDebuggerWidget()
        window.frame_radar = RadarChartWidget()

        setup_card(window.frame_ssim_debug)
        setup_card(window.frame_silk)
        setup_card(window.frame_dna)
        setup_card(window.frame_shift)
        setup_card(window.frame_radar)

        window.scroll_layout.addStretch()
        window.scroll_area.setWidget(window.scroll_content)
        telemetry_layout.addWidget(window.scroll_area) 
        stage_layout.addLayout(telemetry_layout, stretch=10)
        parent_layout.addLayout(stage_layout, stretch=10)

    def _build_footer(self, window, parent_layout):
        window.confidence_frame = QFrame()
        window.confidence_frame.setFixedHeight(120) 
        window.confidence_frame.setStyleSheet("QFrame { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; }")
        footer_super_layout = QHBoxLayout(window.confidence_frame)
        footer_super_layout.setContentsMargins(15, 10, 15, 10)
        
        # VEREDITO
        verdict_layout = QVBoxLayout()
        window.lbl_verdict = QLabel("AGUARDANDO PEÇA")
        window.lbl_verdict.setAlignment(Qt.AlignmentFlag.AlignCenter)
        window.lbl_verdict.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        window.lbl_verdict.setStyleSheet("color: #8b949e; font-size: 16px; font-weight: bold;")
        
        window.lbl_reason = QLabel("A IA está inativa.")
        window.lbl_reason.setAlignment(Qt.AlignmentFlag.AlignCenter)
        window.lbl_reason.setWordWrap(True)
        window.lbl_reason.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        verdict_layout.addWidget(window.lbl_verdict)
        verdict_layout.addWidget(window.lbl_reason)
        
        # MÉTRICAS GRID
        metrics_grid = QGridLayout()
        window.metric_labels = {}
        metrics_def = [
            ("ssim", "SSIM:"), ("pct_changed", "Anomalia:"),
            ("hist_corr", "Correl.:"), ("semantic_loss", "DNA Loss:"),
            ("local_score", "Foco:"), ("ctx_score", "Ctx:"),
            ("final_score", "Ameaça:")
        ]
        
        row, col = 0, 0
        for key, label_text in metrics_def:
            lbl_name = QLabel(label_text)
            lbl_name.setStyleSheet("color: #8b949e; font-size: 10px;")
            lbl_value = QLabel("-")
            lbl_value.setStyleSheet("color: #ffffff; font-weight: bold;")
            metrics_grid.addWidget(lbl_name, row, col)
            metrics_grid.addWidget(lbl_value, row, col + 1)
            window.metric_labels[key] = lbl_value
            row += 1
            if row > 3: row = 0; col += 2

        # DATASET
        dataset_layout = QVBoxLayout()
        window.lbl_db_info = QLabel("Sem dados.")
        window.lbl_db_info.setWordWrap(True)
        window.lbl_db_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        window.lbl_db_info.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        dataset_layout.addWidget(QLabel("DATASET (KNN)"))
        dataset_layout.addWidget(window.lbl_db_info)

        footer_super_layout.addLayout(verdict_layout, stretch=2)
        footer_super_layout.addLayout(metrics_grid, stretch=2)
        footer_super_layout.addLayout(dataset_layout, stretch=1)
        parent_layout.addWidget(window.confidence_frame, stretch=0)

    def _build_action_buttons(self, window, parent_layout):
        # QWidget container para ocultar/exibir botões facilmente
        window.action_widget = QWidget()
        btn_action_layout = QHBoxLayout(window.action_widget)
        btn_action_layout.setContentsMargins(0, 0, 0, 0)
        btn_action_layout.setSpacing(10)
        
        button_style = """
            QPushButton { background-color: #21262d; color: #c9d1d9; font-weight: bold; border-radius: 6px; padding: 10px; }
            QPushButton:hover { background-color: #30363d; }
            QPushButton:disabled { background-color: #0d1117; color: #484f58; border: 1px solid #21262d; }
        """
        
        window.btn_start = QPushButton("Capturar Local (MSS)")
        window.btn_skip = QPushButton("Descartar Imagem")
        window.btn_save_ok = QPushButton("Salvar Dataset: OK")
        window.btn_save_ng = QPushButton("Confirmar Defeito (NG)")

        for btn in [window.btn_start, window.btn_skip, window.btn_save_ok, window.btn_save_ng]:
            btn.setStyleSheet(button_style)
            btn_action_layout.addWidget(btn)
        
        parent_layout.addWidget(window.action_widget, stretch=0)

        window.btn_start.clicked.connect(window.start_monitoring)
        window.btn_skip.clicked.connect(window.skip_image)
        window.btn_save_ok.clicked.connect(lambda: window.save_label("OK", source="button"))
        window.btn_save_ng.clicked.connect(lambda: window.save_label("NG", source="button"))

        # GATILHO DE INTERFACE: Oculta os botões no Modo Sombra
        def apply_mode_visibility(mode_text):
            if mode_text == "Modo Sombra":
                window.action_widget.setVisible(False)
            else:
                window.action_widget.setVisible(True)

        window.combo_mode.currentTextChanged.connect(apply_mode_visibility)
        apply_mode_visibility(window.combo_mode.currentText()) # Aplica o estado inicial

    def _build_status_bar(self, window, parent_layout):
        """
        Constrói a Barra de Status Global na base da janela.
        Exibe telemetria de Rede, Estado da IA e o Log da última ação.
        """
        window.status_frame = QFrame()
        window.status_frame.setFixedHeight(30)
        window.status_frame.setStyleSheet("""
            QFrame { background-color: transparent; border-top: 1px solid #30363d; }
            QLabel { color: #8b949e; font-size: 11px; font-weight: bold; border: none; }
        """)
        status_layout = QHBoxLayout(window.status_frame)
        status_layout.setContentsMargins(10, 0, 10, 0)
        
        # 1. Status de Rede (Esquerda)
        window.lbl_status_network = QLabel("Ouvindo AOI (Porta 5001)")
        window.lbl_status_network.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        
        # 2. Status do Cérebro / Motor IA (Centro)
        window.lbl_status_brain = QLabel("Sistema Ocioso")
        window.lbl_status_brain.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # 3. Log de Ação (Direita)
        window.lbl_status_history = QLabel("Última Peça: Nenhuma")
        window.lbl_status_history.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        
        status_layout.addWidget(window.lbl_status_network, stretch=1)
        status_layout.addWidget(window.lbl_status_brain, stretch=1)
        status_layout.addWidget(window.lbl_status_history, stretch=1)
        
        parent_layout.addWidget(window.status_frame, stretch=0)