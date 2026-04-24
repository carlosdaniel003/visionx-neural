# src/ui/control_panel_ui.py
"""
Módulo responsável exclusivamente pela construção da Interface Gráfica (View).
Aplica o Princípio da Responsabilidade Única (SRP) separando o visual da lógica.
Ajuste Layout: Scroll Horizontal Limpo. Os Debuggers são "Cards" e não são esmagados.
Ajuste Rodapé Unificado: Todas as métricas matemáticas, o Dataset (KNN) e o Veredito
foram consolidados no rodapé da tela para despoluir o Dashboard Central.
"""
from PyQt6.QtWidgets import (QVBoxLayout, QPushButton, QLabel,
                             QHBoxLayout, QFrame, QGridLayout,
                             QApplication, QSizePolicy, QScrollArea, QWidget)
from PyQt6.QtCore import Qt

# Nossos componentes Lego (Widgets de Telemetria da IA)
from src.ui.widgets.radar_chart import RadarChartWidget
from src.ui.widgets.knn_spectrum import KNNSpectrumWidget # Mantido aqui caso queira voltar a usar como widget isolado futuramente
from src.ui.widgets.shift_debugger import ShiftDebuggerWidget 
from src.ui.widgets.silk_debugger import SilkDebuggerWidget 
from src.ui.widgets.ssim_debugger import SSIMDebuggerWidget 
from src.ui.widgets.semantic_dna import SemanticDNAWidget

class ControlPanelUI:
    def setup_ui(self, window):
        window.setWindowTitle("VisionX Neural - Deep Debugger Console")
        # Cor de fundo baseada num tema escuro analítico profissional
        window.setStyleSheet("background-color: #0d1117; color: #c9d1d9;")

        screen = QApplication.primaryScreen()
        available = screen.availableGeometry()
        window.setGeometry(available)

        main_layout = QVBoxLayout(window)
        main_layout.setContentsMargins(15, 10, 15, 15) 
        main_layout.setSpacing(10)

        # ============================================================
        # === CABEÇALHO (Título + Ticker Time)
        # ============================================================
        top_layout = QHBoxLayout()
        
        title = QLabel("VisionX Neural • Monitoramento IA")
        title.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #ffffff;")
        
        window.lbl_timer = QLabel("Latência: 0.00s")
        window.lbl_timer.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        window.lbl_timer.setStyleSheet("font-family: Consolas, monospace; font-size: 14px; font-weight: bold; color: #58a6ff;")

        top_layout.addWidget(title, stretch=1)
        top_layout.addWidget(window.lbl_timer)
        
        main_layout.addLayout(top_layout)

        # ============================================================
        # === BARRA DE INFORMAÇÕES DA AOI (O HEADER)
        # ============================================================
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

        main_layout.addWidget(window.aoi_info_frame)

        # ============================================================
        # === PALCO PRINCIPAL (COLUNA IMAGENS FIXAS X CARROSSEL)
        # ============================================================
        stage_layout = QHBoxLayout()
        stage_layout.setSpacing(15)

        # --- LADO ESQUERDO: IMAGENS DA CÂMERA (Largura Fixa) ---
        images_frame = QFrame()
        images_frame.setFixedWidth(300) 
        images_frame.setStyleSheet("background-color: transparent; border: none;")
        images_column_layout = QVBoxLayout(images_frame)
        images_column_layout.setContentsMargins(0, 0, 0, 0)
        
        lbl_sample_title = QLabel("Gabarito (Placa Original)")
        lbl_sample_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_sample_title.setStyleSheet("color: #8b949e; font-size: 10px; font-weight: bold;")
        window.lbl_sample = QLabel("Sem Sinal")
        window.lbl_sample.setAlignment(Qt.AlignmentFlag.AlignCenter)
        window.lbl_sample.setStyleSheet("background-color: #0d1117; border: 1px solid #30363d; color: #484f58; border-radius: 6px;")
        window.lbl_sample.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        window.lbl_sample.setMinimumSize(10, 10)
        
        lbl_sample_focus_title = QLabel("Lupa Foco (Epicentro Gabarito)")
        lbl_sample_focus_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_sample_focus_title.setStyleSheet("color: #58a6ff; font-size: 10px; font-weight: bold;")
        window.lbl_sample_focus = QLabel("Sem Foco") 
        window.lbl_sample_focus.setAlignment(Qt.AlignmentFlag.AlignCenter)
        window.lbl_sample_focus.setStyleSheet("background-color: #0d1117; border: 1px solid #1f6feb; color: #58a6ff; border-radius: 6px;")
        window.lbl_sample_focus.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        window.lbl_sample_focus.setMinimumSize(10, 10)
        
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: #30363d;")
        
        lbl_ng_title = QLabel("Teste (Anomalia Reportada)")
        lbl_ng_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_ng_title.setStyleSheet("color: #8b949e; font-size: 10px; font-weight: bold;")
        window.lbl_ng = QLabel("Sem Sinal")
        window.lbl_ng.setAlignment(Qt.AlignmentFlag.AlignCenter)
        window.lbl_ng.setStyleSheet("background-color: #0d1117; border: 1px solid #30363d; color: #484f58; border-radius: 6px;")
        window.lbl_ng.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        window.lbl_ng.setMinimumSize(10, 10)
        
        lbl_ng_focus_title = QLabel("Lupa Foco (Epicentro Teste)")
        lbl_ng_focus_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_ng_focus_title.setStyleSheet("color: #ff7b72; font-size: 10px; font-weight: bold;")
        window.lbl_ng_focus = QLabel("Sem Foco") 
        window.lbl_ng_focus.setAlignment(Qt.AlignmentFlag.AlignCenter)
        window.lbl_ng_focus.setStyleSheet("background-color: #0d1117; border: 1px solid #da3633; color: #ff7b72; border-radius: 6px;")
        window.lbl_ng_focus.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        window.lbl_ng_focus.setMinimumSize(10, 10)

        images_column_layout.addWidget(lbl_sample_title)
        images_column_layout.addWidget(window.lbl_sample, stretch=2)
        images_column_layout.addWidget(lbl_sample_focus_title)
        images_column_layout.addWidget(window.lbl_sample_focus, stretch=1)
        images_column_layout.addWidget(line)
        images_column_layout.addWidget(lbl_ng_title)
        images_column_layout.addWidget(window.lbl_ng, stretch=2)
        images_column_layout.addWidget(lbl_ng_focus_title)
        images_column_layout.addWidget(window.lbl_ng_focus, stretch=1)
        
        stage_layout.addWidget(images_frame)

        # --- LADO DIREITO: O CARROSSEL DO DASHBOARD (Scroll Horizontal) ---
        telemetry_layout = QVBoxLayout()
        
        title_telemetry = QLabel("DEBUGGERS DA IA")
        title_telemetry.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_telemetry.setStyleSheet("color: #00ffaa; font-size: 12px; font-weight: bold; background-color: #112211; padding: 4px; border-radius: 4px; border: 1px solid #114411;")
        telemetry_layout.addWidget(title_telemetry)

        # QScrollArea Horizontal!
        window.scroll_area = QScrollArea()
        window.scroll_area.setWidgetResizable(True)
        window.scroll_area.setStyleSheet("""
            QScrollArea { border: none; background-color: transparent; }
            QScrollBar:horizontal { background: #0d1117; height: 16px; margin: 0px; border-radius: 8px;}
            QScrollBar::handle:horizontal { background: #30363d; border-radius: 8px; min-width: 60px; }
            QScrollBar::handle:horizontal:hover { background: #484f58; }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0px; }
        """)

        window.scroll_content = QWidget()
        window.scroll_content.setStyleSheet("background-color: transparent;")
        
        # MÁGICA: QHBoxLayout para colocar os módulos Lado a Lado
        window.scroll_layout = QHBoxLayout(window.scroll_content)
        window.scroll_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop) 
        window.scroll_layout.setContentsMargins(5, 5, 5, 5)
        window.scroll_layout.setSpacing(15)

        # Instancia as janelas da IA e força uma largura mínima generosa (Evita esmagamento)
        window.frame_ssim_debug = SSIMDebuggerWidget()
        window.frame_ssim_debug.setMinimumWidth(550)
        window.frame_ssim_debug.setMinimumHeight(280)
        
        window.frame_silk = SilkDebuggerWidget() 
        window.frame_silk.setMinimumWidth(550)
        window.frame_silk.setMinimumHeight(280)
        
        window.frame_dna = SemanticDNAWidget() 
        window.frame_dna.setMinimumWidth(550) 
        window.frame_dna.setMinimumHeight(280)
        
        window.frame_shift = ShiftDebuggerWidget() 
        window.frame_shift.setMinimumWidth(550)
        window.frame_shift.setMinimumHeight(280)
        
        window.frame_radar = RadarChartWidget()
        window.frame_radar.setMinimumWidth(550)
        window.frame_radar.setMinimumHeight(280)

        # Adiciona no container (começam invisíveis)
        window.scroll_layout.addWidget(window.frame_ssim_debug)
        window.scroll_layout.addWidget(window.frame_silk)
        window.scroll_layout.addWidget(window.frame_dna)
        window.scroll_layout.addWidget(window.frame_shift)
        window.scroll_layout.addWidget(window.frame_radar)
        
        # Preenche o espaço vazio pra direita
        window.scroll_layout.addStretch()

        window.scroll_area.setWidget(window.scroll_content)
        telemetry_layout.addWidget(window.scroll_area, stretch=1) 

        stage_layout.addLayout(telemetry_layout, stretch=1)
        main_layout.addLayout(stage_layout, stretch=10)

        # ============================================================
        # === RODAPÉ CONSOLIDADO (MÉTRICAS + DATASET + VEREDITO)
        # ============================================================
        window.confidence_frame = QFrame()
        window.confidence_frame.setFixedHeight(120) # Aumentamos um pouco para caber tudo bem organizado
        window.confidence_frame.setStyleSheet("""
            QFrame { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 5px; }
        """)
        footer_super_layout = QHBoxLayout(window.confidence_frame)
        footer_super_layout.setContentsMargins(15, 10, 15, 10)
        footer_super_layout.setSpacing(20)

        # 1. BLOCO DO VEREDITO (Esquerda)
        verdict_layout = QVBoxLayout()
        window.lbl_verdict = QLabel("AGUARDANDO PEÇA")
        window.lbl_verdict.setAlignment(Qt.AlignmentFlag.AlignCenter)
        window.lbl_verdict.setStyleSheet("color: #8b949e; font-size: 18px; font-weight: bold; border: none;")
        
        window.lbl_reason = QLabel("A IA está inativa.")
        window.lbl_reason.setAlignment(Qt.AlignmentFlag.AlignCenter)
        window.lbl_reason.setStyleSheet("color: #c9d1d9; font-size: 11px; border: none;")
        
        verdict_layout.addWidget(window.lbl_verdict)
        verdict_layout.addWidget(window.lbl_reason)
        
        # 2. BLOCO DAS MÉTRICAS MATEMÁTICAS (Centro)
        metrics_grid = QGridLayout()
        metrics_grid.setSpacing(5)
        window.metric_labels = {}
        metrics_def = [
            ("ssim", "SSIM (Textura):"), ("pct_changed", "Anomalia Local:"),
            ("hist_corr", "Correl. Tinta:"), ("semantic_loss", "Perda Estrutura:"),
            ("local_score", "Score de Foco:"), ("ctx_score", "Score Contexto:"),
            ("final_score", "Ameaça Final:")
        ]
        
        row, col = 0, 0
        for key, label_text in metrics_def:
            lbl_name = QLabel(label_text)
            lbl_name.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            lbl_name.setStyleSheet("color: #8b949e; font-size: 11px; border: none; background: transparent;")
            
            lbl_value = QLabel("-")
            lbl_value.setStyleSheet("color: #ffffff; font-size: 12px; font-weight: bold; border: none; background: transparent;")
            
            metrics_grid.addWidget(lbl_name, row, col)
            metrics_grid.addWidget(lbl_value, row, col + 1)
            window.metric_labels[key] = lbl_value
            
            row += 1
            if row > 3: # 4 linhas por coluna
                row = 0
                col += 2

        # 3. BLOCO DO DATASET (KNN Histórico na Direita)
        dataset_layout = QVBoxLayout()
        lbl_ds_title = QLabel("MEMÓRIA DO DATASET (KNN)")
        lbl_ds_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_ds_title.setStyleSheet("color: #58a6ff; font-size: 11px; font-weight: bold; border: none;")
        
        window.lbl_db_info = QLabel("Sem dados no momento.")
        window.lbl_db_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        window.lbl_db_info.setWordWrap(True)
        window.lbl_db_info.setStyleSheet("color: #c9d1d9; font-size: 11px; border: none;")
        
        dataset_layout.addWidget(lbl_ds_title)
        dataset_layout.addWidget(window.lbl_db_info)
        dataset_layout.addStretch()

        # Monta o Super Rodapé
        footer_super_layout.addLayout(verdict_layout, stretch=2)
        
        line_vert1 = QFrame()
        line_vert1.setFrameShape(QFrame.Shape.VLine)
        line_vert1.setStyleSheet("color: #30363d;")
        footer_super_layout.addWidget(line_vert1)
        
        footer_super_layout.addLayout(metrics_grid, stretch=2)
        
        line_vert2 = QFrame()
        line_vert2.setFrameShape(QFrame.Shape.VLine)
        line_vert2.setStyleSheet("color: #30363d;")
        footer_super_layout.addWidget(line_vert2)
        
        footer_super_layout.addLayout(dataset_layout, stretch=1)

        main_layout.addWidget(window.confidence_frame, stretch=0)

        # ============================================================
        # === BARRA DE BOTÕES DE AÇÃO
        # ============================================================
        button_style = """
            QPushButton { background-color: #21262d; color: #c9d1d9; font-weight: bold; font-size: 12px; border: 1px solid #30363d; border-radius: 6px; padding: 5px; }
            QPushButton:hover { background-color: #30363d; color: #ffffff;}
            QPushButton:disabled { background-color: #0d1117; color: #484f58; border: 1px solid #21262d; }
        """
        
        btn_action_layout = QHBoxLayout()
        btn_action_layout.setSpacing(10)
        
        window.btn_start = QPushButton("⛶ Capturar Local (MSS)")
        window.btn_start.setFixedHeight(40) 
        window.btn_start.setStyleSheet(button_style)
        
        window.btn_skip = QPushButton("Descartar Imagem")
        window.btn_skip.setFixedHeight(40) 
        window.btn_skip.setEnabled(False)
        window.btn_skip.setStyleSheet(button_style.replace("#21262d", "#3a1d1d")) # Fundo levemente avermelhado pro skip

        window.btn_save_ok = QPushButton("Salvar Dataset: Falha Falsa (OK)")
        window.btn_save_ok.setFixedHeight(40) 
        window.btn_save_ok.setEnabled(False)
        window.btn_save_ok.setStyleSheet(button_style.replace("#21262d", "#1f3a28")) # Fundo levemente verde

        window.btn_save_ng = QPushButton("Confirmar Defeito (NG)")
        window.btn_save_ng.setFixedHeight(40) 
        window.btn_save_ng.setEnabled(False)
        window.btn_save_ng.setStyleSheet(button_style.replace("#21262d", "#5c1e1e")) # Fundo vermelho forte

        btn_action_layout.addWidget(window.btn_start)
        btn_action_layout.addWidget(window.btn_skip)
        btn_action_layout.addWidget(window.btn_save_ok)
        btn_action_layout.addWidget(window.btn_save_ng)
        
        main_layout.addLayout(btn_action_layout, stretch=0)

        # Liga os gatilhos dos botões
        window.btn_start.clicked.connect(window.start_monitoring)
        window.btn_skip.clicked.connect(window.skip_image)
        window.btn_save_ok.clicked.connect(lambda: window.save_label("OK", source="button"))
        window.btn_save_ng.clicked.connect(lambda: window.save_label("NG", source="button"))

        window.setWindowState(Qt.WindowState.WindowMaximized)