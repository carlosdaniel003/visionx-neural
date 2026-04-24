# src/ui/control_panel_ui.py
"""
Módulo responsável exclusivamente pela construção da Interface Gráfica (View).
Aplica o Princípio da Responsabilidade Única (SRP) separando o visual da lógica.
Layout em Z: Imagens à esquerda, Telemetria à direita, Numéricos no rodapé.
Ajuste Overlay: Substituição do DNA Semântico pelo SSIM Debugger Visual.
NOVO AJUSTE: Construção rígida de layout para evitar botões caindo e cortes na UI.
"""
from PyQt6.QtWidgets import (QVBoxLayout, QPushButton, QLabel,
                             QHBoxLayout, QFrame, QGridLayout,
                             QApplication, QSizePolicy, QStackedWidget)
from PyQt6.QtCore import Qt

# Nossos componentes Lego (Widgets de Telemetria da IA)
from src.ui.widgets.radar_chart import RadarChartWidget
from src.ui.widgets.knn_spectrum import KNNSpectrumWidget
from src.ui.widgets.shift_debugger import ShiftDebuggerWidget 
from src.ui.widgets.silk_debugger import SilkDebuggerWidget 
from src.ui.widgets.ssim_debugger import SSIMDebuggerWidget 

class ControlPanelUI:
    def setup_ui(self, window):
        """
        Recebe a janela principal (window) e injeta todos os componentes visuais nela.
        """
        window.setWindowTitle("VisionX Neural - Console Industrial AOI")
        window.setStyleSheet("background-color: #121212; color: #ffffff;")

        screen = QApplication.primaryScreen()
        available = screen.availableGeometry()
        window.setGeometry(available)

        main_layout = QVBoxLayout(window)
        main_layout.setContentsMargins(10, 6, 10, 25) 
        main_layout.setSpacing(4)

        # ============================================================
        # === CABEÇALHO (Título + Ticker Time)
        # ============================================================
        top_layout = QHBoxLayout()
        
        title = QLabel("VisionX Neural: Monitoramento AOI")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 2px; color: #ffffff;")
        
        window.lbl_timer = QLabel("Latencia: 0.00s")
        window.lbl_timer.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        window.lbl_timer.setStyleSheet("font-family: Consolas, monospace; font-size: 14px; font-weight: bold; color: #aaaaaa;")

        spacer = QLabel("")
        spacer.setMinimumWidth(120) 
        top_layout.addWidget(spacer)
        top_layout.addWidget(title, stretch=1)
        top_layout.addWidget(window.lbl_timer)
        
        main_layout.addLayout(top_layout)

        # ============================================================
        # === BARRA DE INFORMAÇÕES DA AOI
        # ============================================================
        window.aoi_info_frame = QFrame()
        window.aoi_info_frame.setFixedHeight(35) # TRAVADO
        window.aoi_info_frame.setStyleSheet("""
            QFrame { background-color: #1a1a1a; border: 1px solid #333333; border-radius: 6px; padding: 4px; }
        """)
        aoi_info_layout = QHBoxLayout(window.aoi_info_frame)
        aoi_info_layout.setContentsMargins(12, 4, 12, 4)
        aoi_info_layout.setSpacing(20)

        info_label_style = "color: #888888; font-size: 11px; border: none;"
        info_value_style = "color: #ffffff; font-size: 12px; font-weight: bold; border: none;"
        category_value_style = "color: #55ff55; font-size: 12px; font-weight: bold; border: none;"

        lbl_board_name = QLabel("Board:")
        lbl_board_name.setStyleSheet(info_label_style)
        window.lbl_board_value = QLabel("-")
        window.lbl_board_value.setStyleSheet(info_value_style)

        lbl_parts_name = QLabel("Parts:")
        lbl_parts_name.setStyleSheet(info_label_style)
        window.lbl_parts_value = QLabel("-")
        window.lbl_parts_value.setStyleSheet(info_value_style)

        lbl_category_name = QLabel("Category:")
        lbl_category_name.setStyleSheet(info_label_style)
        window.lbl_category_value = QLabel("-")
        window.lbl_category_value.setStyleSheet(category_value_style)

        lbl_value_name = QLabel("Value:")
        lbl_value_name.setStyleSheet(info_label_style)
        window.lbl_value_value = QLabel("-")
        window.lbl_value_value.setStyleSheet(info_value_style)

        aoi_info_layout.addWidget(lbl_board_name)
        aoi_info_layout.addWidget(window.lbl_board_value)
        aoi_info_layout.addStretch()
        
        aoi_info_layout.addWidget(lbl_parts_name)
        aoi_info_layout.addWidget(window.lbl_parts_value)
        aoi_info_layout.addStretch()

        aoi_info_layout.addWidget(lbl_category_name)
        aoi_info_layout.addWidget(window.lbl_category_value)
        aoi_info_layout.addStretch()

        aoi_info_layout.addWidget(lbl_value_name)
        aoi_info_layout.addWidget(window.lbl_value_value)

        main_layout.addWidget(window.aoi_info_frame)

        # ============================================================
        # === PALCO PRINCIPAL: IMAGENS (ESQUERDA) E TELEMETRIA (DIREITA)
        # ============================================================
        stage_layout = QHBoxLayout()
        stage_layout.setSpacing(8)

        # --- LADO ESQUERDO: AS IMAGENS DA CÂMERA ---
        images_layout = QHBoxLayout()
        
        # Sample
        sample_layout = QVBoxLayout()
        lbl_sample_title = QLabel("Padrao (Sample)")
        lbl_sample_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_sample_title.setStyleSheet("color: #aaaaaa; font-size: 12px;")
        window.lbl_sample = QLabel("Aguardando capturas da Rede...")
        window.lbl_sample.setAlignment(Qt.AlignmentFlag.AlignCenter)
        window.lbl_sample.setStyleSheet("background-color: #1a1a1a; border: 1px solid #333333; color: #888888;")
        
        # POLÍTICA IGNORADA: A imagem nunca mais quebra ou empurra o layout
        window.lbl_sample.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        window.lbl_sample.setMinimumSize(10, 10)
        
        sample_layout.addWidget(lbl_sample_title)
        sample_layout.addWidget(window.lbl_sample, stretch=1)

        # NG
        ng_layout = QVBoxLayout()
        lbl_ng_title = QLabel("Analise VisaoX (NG)")
        lbl_ng_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_ng_title.setStyleSheet("color: #aaaaaa; font-size: 12px;")
        window.lbl_ng = QLabel("Aguardando capturas da Rede...")
        window.lbl_ng.setAlignment(Qt.AlignmentFlag.AlignCenter)
        window.lbl_ng.setStyleSheet("background-color: #1a1a1a; border: 1px solid #333333; color: #888888;")
        
        # POLÍTICA IGNORADA: A imagem nunca mais quebra ou empurra o layout
        window.lbl_ng.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        window.lbl_ng.setMinimumSize(10, 10)
        
        ng_layout.addWidget(lbl_ng_title)
        ng_layout.addWidget(window.lbl_ng, stretch=1)

        images_layout.addLayout(sample_layout, stretch=1)
        images_layout.addLayout(ng_layout, stretch=1)
        
        stage_layout.addLayout(images_layout, stretch=6)

        # --- LADO DIREITO: DASHBOARD SEMÂNTICO (WIDGETS) ---
        telemetry_layout = QVBoxLayout()
        
        title_telemetry = QLabel("DASHBOARD SEMANTICO DA IA")
        title_telemetry.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_telemetry.setStyleSheet("color: #aaaaaa; font-size: 12px; font-weight: bold;")
        telemetry_layout.addWidget(title_telemetry)

        # Container Dinâmico
        window.stack_central = QStackedWidget()
        window.frame_ssim_debug = SSIMDebuggerWidget()
        window.frame_radar = RadarChartWidget()
        window.frame_shift = ShiftDebuggerWidget() 
        window.frame_silk = SilkDebuggerWidget() 
        
        window.stack_central.addWidget(window.frame_ssim_debug) 
        window.stack_central.addWidget(window.frame_radar)      
        window.stack_central.addWidget(window.frame_shift)      
        window.stack_central.addWidget(window.frame_silk)       
        
        telemetry_layout.addWidget(window.stack_central, stretch=2)

        # Container 3: Espectro KNN
        window.frame_knn = KNNSpectrumWidget()
        telemetry_layout.addWidget(window.frame_knn, stretch=1)

        stage_layout.addLayout(telemetry_layout, stretch=4)
        main_layout.addLayout(stage_layout, stretch=1)

        # ============================================================
        # === RODAPÉ ANALÍTICO (VEREDITO E TEXTOS)
        # ============================================================
        window.confidence_frame = QFrame()
        window.confidence_frame.setFixedHeight(95) # TRAVADO
        window.confidence_frame.setStyleSheet("""
            QFrame { background-color: #1e1e1e; border: 1px solid #333333; border-radius: 8px; padding: 2px; }
        """)
        conf_layout = QVBoxLayout(window.confidence_frame)
        conf_layout.setSpacing(1)

        window.lbl_verdict = QLabel("Aguardando analise...")
        window.lbl_verdict.setAlignment(Qt.AlignmentFlag.AlignCenter)
        window.lbl_verdict.setStyleSheet("color: #888888; font-size: 16px; font-weight: bold; border: none;")
        conf_layout.addWidget(window.lbl_verdict)
        
        window.lbl_active_engines = QLabel("")
        window.lbl_active_engines.setAlignment(Qt.AlignmentFlag.AlignCenter)
        window.lbl_active_engines.setStyleSheet("color: #00ffaa; font-size: 10px; font-family: Consolas, monospace; border: none; margin-bottom: 2px;")
        conf_layout.addWidget(window.lbl_active_engines)

        metrics_grid = QGridLayout()
        metrics_grid.setSpacing(2)

        window.metric_labels = {}
        metrics_def = [
            ("ssim", "SSIM (Similaridade)"), ("pct_changed", "Pixels Alterados"),
            ("edge_change", "Mudanca de Bordas"), ("hist_corr", "Correlacao Histograma"),
            ("local_score", "Score Local"), ("ctx_score", "Score Contexto"),
            ("db_score", "Score Dataset"), ("final_score", "Score Final"),
        ]

        for i, (key, label_text) in enumerate(metrics_def):
            lbl_name = QLabel(label_text + ":")
            lbl_name.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            lbl_name.setStyleSheet("color: #888888; font-size: 11px; border: none; background: transparent;")
            
            lbl_value = QLabel("-")
            lbl_value.setStyleSheet("color: #ffffff; font-size: 11px; font-weight: bold; border: none; background: transparent;")
            
            metrics_grid.addWidget(lbl_name, 0, i * 2)
            metrics_grid.addWidget(lbl_value, 0, (i * 2) + 1)
            
            window.metric_labels[key] = lbl_value

        conf_layout.addLayout(metrics_grid)

        info_db_layout = QHBoxLayout()
        window.lbl_reason = QLabel("")
        window.lbl_reason.setStyleSheet("color: #aaaaaa; font-size: 10px; border: none;")
        window.lbl_reason.setWordWrap(True)
        window.lbl_reason.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        window.lbl_db_info = QLabel("")
        window.lbl_db_info.setStyleSheet("color: #777777; font-size: 10px; border: none;")
        window.lbl_db_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        info_db_layout.addWidget(window.lbl_reason, stretch=1)
        info_db_layout.addWidget(window.lbl_db_info, stretch=1)
        conf_layout.addLayout(info_db_layout)

        main_layout.addWidget(window.confidence_frame, stretch=0)

        # ============================================================
        # === BOTÕES (COMANDOS DE DECISÃO)
        # ============================================================
        button_style = """
            QPushButton { background-color: #2d2d2d; color: #ffffff; font-weight: bold; border: 1px solid #444444; border-radius: 4px; }
            QPushButton:hover { background-color: #3d3d3d; }
            QPushButton:disabled { background-color: #1a1a1a; color: #555555; border: 1px solid #333333; }
        """
        
        skip_button_style = """
            QPushButton { background-color: #4a2c2c; color: #ffaaaa; font-weight: bold; border: 1px solid #663333; border-radius: 4px; }
            QPushButton:hover { background-color: #5a3c3c; }
            QPushButton:disabled { background-color: #1a1a1a; color: #555555; border: 1px solid #333333; }
        """

        window.btn_start = QPushButton("Capturar Local Manualmente (MSS)")
        window.btn_start.setFixedHeight(40) # TRAVADO
        window.btn_start.setStyleSheet(button_style)
        window.btn_start.clicked.connect(window.start_monitoring)
        main_layout.addWidget(window.btn_start, stretch=0)

        curation_layout = QHBoxLayout()
        
        window.btn_skip = QPushButton("X - Descartar")
        window.btn_skip.setFixedHeight(40) # TRAVADO
        window.btn_skip.setEnabled(False)
        window.btn_skip.setStyleSheet(skip_button_style)
        window.btn_skip.clicked.connect(window.skip_image)

        window.btn_save_ok = QPushButton("Salvar como Falha Falsa (OK)")
        window.btn_save_ok.setFixedHeight(40) # TRAVADO
        window.btn_save_ok.setEnabled(False)
        window.btn_save_ok.setStyleSheet(button_style)
        window.btn_save_ok.clicked.connect(lambda: window.save_label("OK", source="button"))

        window.btn_save_ng = QPushButton("Confirmar Defeito Real (NG)")
        window.btn_save_ng.setFixedHeight(40) # TRAVADO
        window.btn_save_ng.setEnabled(False)
        window.btn_save_ng.setStyleSheet(button_style)
        window.btn_save_ng.clicked.connect(lambda: window.save_label("NG", source="button"))

        curation_layout.addWidget(window.btn_skip)
        curation_layout.addWidget(window.btn_save_ok)
        curation_layout.addWidget(window.btn_save_ng)
        
        main_layout.addLayout(curation_layout, stretch=0)
        window.setWindowState(Qt.WindowState.WindowMaximized)