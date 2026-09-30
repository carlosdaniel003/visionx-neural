"""Construção da interface responsiva do painel principal do VisionX Neural."""

from __future__ import annotations

from PyQt6.QtCore import QEvent, QObject, Qt
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from src.ui.network_xp_debug import (
    copy_network_debug_to_clipboard,
    copy_network_image_to_clipboard,
    sync_network_debug_controls,
)
from src.ui.responsive_layout import profile_for_width
from src.ui.theme import APP_STYLESHEET
from src.ui.widgets.knn_spectrum import KNNSpectrumWidget
from src.ui.widgets.radar_chart import RadarChartWidget
from src.ui.widgets.semantic_dna import SemanticDNAWidget
from src.ui.widgets.shift_debugger import ShiftDebuggerWidget
from src.ui.widgets.silk_debugger import SilkDebuggerWidget
from src.ui.widgets.ssim_debugger import SSIMDebuggerWidget


class _ResponsiveEventFilter(QObject):
    """Reaplica o perfil visual apenas quando o breakpoint muda."""

    def __init__(self, builder: "ControlPanelUI", window: QWidget):
        super().__init__(window)
        self.builder = builder
        self.window = window

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.Resize:
            viewport = getattr(getattr(self.window, "root_scroll", None), "viewport", lambda: None)()
            if watched is viewport:
                width = event.size().width()
                self.builder.apply_layout_profile(self.window, width)
            elif watched is self.window:
                viewport_width = viewport.width() if viewport is not None else 0
                width = viewport_width if viewport_width > 100 else event.size().width()
                self.builder.apply_layout_profile(self.window, width)
        return super().eventFilter(watched, event)


class ControlPanelUI:
    """View do painel com hierarquia visual e breakpoints para notebooks/monitores."""

    def __init__(self):
        self._active_profile_name: str | None = None
        self.info_cards: list[QWidget] = []
        self.footer_cards: list[QWidget] = []
        self.debug_wrappers: list[QWidget] = []
        self.image_viewports: list[QLabel] = []
        self.light_buttons: list[QPushButton] = []
        self.action_buttons: list[QPushButton] = []

    def setup_ui(self, window):
        window.setWindowTitle("VisionX Neural - Monitoramento IA")
        window.setObjectName("rootWindow")
        window.setStyleSheet(APP_STYLESHEET)

        screen = QApplication.primaryScreen()
        available = screen.availableGeometry()
        window.setGeometry(available)

        outer_layout = QVBoxLayout(window)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        window.root_scroll = QScrollArea()
        window.root_scroll.setWidgetResizable(True)
        window.root_scroll.setMinimumWidth(0)
        window.root_scroll.setFrameShape(QFrame.Shape.NoFrame)
        window.root_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        window.root_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        window.root_scroll.viewport().setObjectName("rootViewport")

        window.root_content = QWidget()
        window.root_content.setObjectName("rootContent")
        window.root_content.setMinimumWidth(0)
        window.root_content.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        window.content_layout = QVBoxLayout(window.root_content)
        window.content_layout.setContentsMargins(14, 12, 14, 12)
        window.content_layout.setSpacing(12)

        self._build_header(window, window.content_layout)
        self._build_aoi_info(window, window.content_layout)
        self._build_main_stage(window, window.content_layout)
        self._build_footer(window, window.content_layout)
        self._build_action_buttons(window, window.content_layout)
        self._build_ng_archive_bar(window, window.content_layout)
        self._build_network_debug_bar(window, window.content_layout)
        self._build_status_bar(window, window.content_layout)

        window.root_scroll.setWidget(window.root_content)
        outer_layout.addWidget(window.root_scroll)

        window._responsive_event_filter = _ResponsiveEventFilter(self, window)
        window.installEventFilter(window._responsive_event_filter)
        window.root_scroll.viewport().installEventFilter(window._responsive_event_filter)
        viewport_width = window.root_scroll.viewport().width()
        initial_width = viewport_width if viewport_width > 100 else available.width()
        self.apply_layout_profile(window, initial_width, force=True)
        window.setWindowState(Qt.WindowState.WindowMaximized)

    @staticmethod
    def _section_heading(title_text: str, hint_text: str = "") -> QWidget:
        heading = QWidget()
        layout = QVBoxLayout(heading)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        title = QLabel(title_text)
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        if hint_text:
            hint = QLabel(hint_text)
            hint.setObjectName("sectionHint")
            hint.setWordWrap(True)
            layout.addWidget(hint)
        return heading

    def _build_header(self, window, parent_layout):
        header = QFrame()
        header.setObjectName("headerFrame")
        header.setMinimumWidth(0)
        self.header_layout = QGridLayout(header)
        self.header_layout.setContentsMargins(18, 12, 18, 12)
        self.header_layout.setHorizontalSpacing(14)
        self.header_layout.setVerticalSpacing(8)

        title_block = QWidget()
        title_block.setMinimumWidth(0)
        title_block.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        title_layout = QVBoxLayout(title_block)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(1)

        eyebrow = QLabel("INSPEÇÃO VISUAL INDUSTRIAL")
        eyebrow.setObjectName("eyebrowLabel")
        title = QLabel("VisionX Neural")
        title.setObjectName("pageTitle")
        subtitle = QLabel("Monitoramento, diagnóstico e memória visual em tempo real")
        subtitle.setObjectName("pageSubtitle")
        subtitle.setWordWrap(True)

        title_layout.addWidget(eyebrow)
        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)

        mode_card = QFrame()
        mode_card.setObjectName("modeCard")
        mode_card.setMinimumWidth(0)
        mode_layout = QVBoxLayout(mode_card)
        mode_layout.setContentsMargins(12, 7, 12, 7)
        mode_layout.setSpacing(3)
        mode_label = QLabel("MODO DE OPERAÇÃO")
        mode_label.setObjectName("fieldLabel")
        window.combo_mode = QComboBox()
        window.combo_mode.setObjectName("modeSelector")
        window.combo_mode.addItems(["Modo Sombra", "Modo Teste", "Modo Produção"])
        window.combo_mode.setCurrentText("Modo Teste")
        window.combo_mode.setCursor(Qt.CursorShape.PointingHandCursor)
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(window.combo_mode)

        latency_card = QFrame()
        latency_card.setObjectName("latencyCard")
        latency_card.setMinimumWidth(0)
        latency_layout = QVBoxLayout(latency_card)
        latency_layout.setContentsMargins(12, 7, 12, 7)
        latency_layout.setSpacing(3)
        latency_title = QLabel("TEMPO DE ANÁLISE")
        latency_title.setObjectName("fieldLabel")
        window.lbl_timer = QLabel("Latência: 0.00s")
        window.lbl_timer.setObjectName("latencyLabel")
        window.lbl_timer.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        latency_layout.addWidget(latency_title)
        latency_layout.addWidget(window.lbl_timer)

        self.header_widgets = [title_block, mode_card, latency_card]
        self.header_layout.addWidget(title_block, 0, 0)
        self.header_layout.addWidget(mode_card, 0, 1)
        self.header_layout.addWidget(latency_card, 0, 2)
        self.header_layout.setColumnStretch(0, 1)
        self.header_layout.setColumnStretch(1, 0)
        self.header_layout.setColumnStretch(2, 0)
        parent_layout.addWidget(header)

    def _create_info_card(self, title: str, value_label: QLabel, accent: bool = False) -> QFrame:
        card = QFrame()
        card.setObjectName("infoCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(12, 9, 12, 9)
        card_layout.setSpacing(3)

        label = QLabel(title)
        label.setObjectName("fieldLabel")
        value_label.setObjectName("accentValue" if accent else "fieldValue")
        value_label.setWordWrap(True)
        value_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        card_layout.addWidget(label)
        card_layout.addWidget(value_label)
        return card

    def _build_aoi_info(self, window, parent_layout):
        window.aoi_info_frame = QFrame()
        window.aoi_info_frame.setObjectName("infoSection")
        section_layout = QVBoxLayout(window.aoi_info_frame)
        section_layout.setContentsMargins(12, 10, 12, 12)
        section_layout.setSpacing(8)
        section_layout.addWidget(
            self._section_heading(
                "IDENTIFICAÇÃO DA INSPEÇÃO",
                "Dados extraídos da AOI e contexto usado pelos especialistas.",
            )
        )

        self.info_grid = QGridLayout()
        self.info_grid.setHorizontalSpacing(8)
        self.info_grid.setVerticalSpacing(8)

        window.lbl_board_value = QLabel("-")
        window.lbl_parts_value = QLabel("-")
        window.lbl_category_value = QLabel("-")
        window.lbl_value_value = QLabel("-")
        window.lbl_light_value = QLabel("TOP")

        self.info_cards = [
            self._create_info_card("Placa / Máquina", window.lbl_board_value),
            self._create_info_card("Componente", window.lbl_parts_value),
            self._create_info_card("Categoria do Erro", window.lbl_category_value, accent=True),
            self._create_info_card("Valor / OCR", window.lbl_value_value),
            self._create_info_card("Iluminação Atual", window.lbl_light_value, accent=True),
        ]

        section_layout.addLayout(self.info_grid)
        parent_layout.addWidget(window.aoi_info_frame)

    def _create_image_card(self, title_text: str, viewport: QLabel, focused: bool = False) -> QFrame:
        card = QFrame()
        card.setObjectName("imageCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(8, 7, 8, 8)
        layout.setSpacing(5)

        title = QLabel(title_text)
        title.setObjectName("imageTitle")
        viewport.setObjectName("focusViewport" if focused else "imageViewport")
        viewport.setAlignment(Qt.AlignmentFlag.AlignCenter)
        viewport.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        viewport.setMinimumSize(80, 100)

        layout.addWidget(title)
        layout.addWidget(viewport, stretch=1)
        self.image_viewports.append(viewport)
        return card

    def _wrap_debug_widget(self, label_text: str, widget: QWidget) -> QWidget:
        # O próprio widget continua sendo o item do layout. Assim, os comandos
        # setVisible() já usados pelo controller escondem o card completo.
        widget.setObjectName("debugCard")
        widget.setToolTip(label_text)
        widget.setAccessibleName(label_text)
        widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.debug_wrappers.append(widget)
        return widget

    def _build_main_stage(self, window, parent_layout):
        window.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        window.main_splitter.setMinimumWidth(0)
        window.main_splitter.setChildrenCollapsible(False)
        window.main_splitter.setHandleWidth(6)

        window.images_section = QFrame()
        window.images_section.setObjectName("sectionPanel")
        window.images_section.setMinimumWidth(0)
        images_layout = QVBoxLayout(window.images_section)
        images_layout.setContentsMargins(10, 10, 10, 10)
        images_layout.setSpacing(8)
        images_layout.addWidget(
            self._section_heading(
                "IMAGENS DA INSPEÇÃO",
                "Visão completa e recorte do epicentro para comparação rápida.",
            )
        )

        image_grid = QGridLayout()
        image_grid.setHorizontalSpacing(8)
        image_grid.setVerticalSpacing(8)

        window.lbl_sample = QLabel("Sem Sinal")
        window.lbl_sample_focus = QLabel("Sem Foco")
        window.lbl_ng = QLabel("Sem Sinal")
        window.lbl_ng_focus = QLabel("Sem Foco")

        image_cards = [
            self._create_image_card("GABARITO • VISÃO COMPLETA", window.lbl_sample),
            self._create_image_card("GABARITO • EPICENTRO", window.lbl_sample_focus, focused=True),
            self._create_image_card("TESTE • VISÃO COMPLETA", window.lbl_ng),
            self._create_image_card("TESTE • EPICENTRO", window.lbl_ng_focus, focused=True),
        ]
        for index, card in enumerate(image_cards):
            image_grid.addWidget(card, index // 2, index % 2)
        image_grid.setColumnStretch(0, 1)
        image_grid.setColumnStretch(1, 1)
        images_layout.addLayout(image_grid, stretch=1)

        window.telemetry_section = QFrame()
        window.telemetry_section.setObjectName("sectionPanel")
        window.telemetry_section.setMinimumWidth(0)
        telemetry_layout = QVBoxLayout(window.telemetry_section)
        telemetry_layout.setContentsMargins(10, 10, 10, 10)
        telemetry_layout.setSpacing(8)
        telemetry_layout.addWidget(
            self._section_heading(
                "ANÁLISE DOS ESPECIALISTAS",
                "Percorra os cards para investigar textura, tinta, semântica, deslocamento e score.",
            )
        )

        window.scroll_area = QScrollArea()
        window.scroll_area.setWidgetResizable(True)
        window.scroll_area.setMinimumWidth(0)
        window.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        window.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        window.scroll_content = QWidget()
        window.scroll_layout = QHBoxLayout(window.scroll_content)
        window.scroll_layout.setContentsMargins(2, 2, 2, 2)
        window.scroll_layout.setSpacing(10)

        window.frame_ssim_debug = SSIMDebuggerWidget()
        window.frame_silk = SilkDebuggerWidget()
        window.frame_dna = SemanticDNAWidget()
        window.frame_shift = ShiftDebuggerWidget()
        window.frame_radar = RadarChartWidget()

        debug_items = [
            ("SSIM • TEXTURA E CALOR", window.frame_ssim_debug),
            ("XOR • TINTA E EPICENTRO", window.frame_silk),
            ("DNA • ASSINATURA SEMÂNTICA", window.frame_dna),
            ("SHIFT • DESLOCAMENTO", window.frame_shift),
            ("FUSÃO • SCORE FINAL", window.frame_radar),
        ]
        for label, widget in debug_items:
            window.scroll_layout.addWidget(self._wrap_debug_widget(label, widget))
        window.scroll_layout.addStretch()
        window.scroll_area.setWidget(window.scroll_content)
        telemetry_layout.addWidget(window.scroll_area, stretch=1)

        window.main_splitter.addWidget(window.images_section)
        window.main_splitter.addWidget(window.telemetry_section)
        window.main_splitter.setStretchFactor(0, 1)
        window.main_splitter.setStretchFactor(1, 3)
        parent_layout.addWidget(window.main_splitter, stretch=10)

    @staticmethod
    def _create_footer_card(title_text: str) -> tuple[QFrame, QVBoxLayout]:
        card = QFrame()
        card.setObjectName("footerCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 9, 12, 10)
        layout.setSpacing(7)
        title = QLabel(title_text)
        title.setObjectName("eyebrowLabel")
        layout.addWidget(title)
        return card, layout

    def _build_footer(self, window, parent_layout):
        window.confidence_frame = QFrame()
        window.confidence_frame.setObjectName("confidenceFrame")
        confidence_layout = QVBoxLayout(window.confidence_frame)
        confidence_layout.setContentsMargins(10, 10, 10, 10)
        confidence_layout.setSpacing(8)
        confidence_layout.addWidget(
            self._section_heading(
                "DECISÃO E CONFIANÇA",
                "Resumo executivo primeiro; métricas e memória local logo abaixo.",
            )
        )

        self.footer_grid = QGridLayout()
        self.footer_grid.setHorizontalSpacing(8)
        self.footer_grid.setVerticalSpacing(8)

        verdict_card, verdict_layout = self._create_footer_card("VEREDITO DA IA")
        window.lbl_verdict = QLabel("AGUARDANDO PEÇA")
        window.lbl_verdict.setAlignment(Qt.AlignmentFlag.AlignCenter)
        window.lbl_verdict.setWordWrap(True)
        window.lbl_verdict.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        window.lbl_verdict.setStyleSheet("font-size: 17px; font-weight: 800;")
        window.lbl_reason = QLabel("A IA está inativa.")
        window.lbl_reason.setAlignment(Qt.AlignmentFlag.AlignCenter)
        window.lbl_reason.setWordWrap(True)
        window.lbl_reason.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        verdict_layout.addWidget(window.lbl_verdict)
        verdict_layout.addWidget(window.lbl_reason, stretch=1)

        metrics_card, metrics_layout = self._create_footer_card("MÉTRICAS DE DECISÃO")
        metrics_grid = QGridLayout()
        metrics_grid.setHorizontalSpacing(10)
        metrics_grid.setVerticalSpacing(5)
        window.metric_labels = {}
        metrics_def = [
            ("ssim", "SSIM"),
            ("pct_changed", "Anomalia"),
            ("hist_corr", "Correlação"),
            ("semantic_loss", "DNA Loss"),
            ("local_score", "Foco"),
            ("ctx_score", "Contexto"),
            ("final_score", "Ameaça"),
        ]
        for index, (key, label_text) in enumerate(metrics_def):
            row = index % 4
            col = (index // 4) * 2
            lbl_name = QLabel(label_text)
            lbl_name.setObjectName("metricName")
            lbl_value = QLabel("-")
            lbl_value.setObjectName("fieldValue")
            lbl_value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            metrics_grid.addWidget(lbl_name, row, col)
            metrics_grid.addWidget(lbl_value, row, col + 1)
            window.metric_labels[key] = lbl_value
        metrics_grid.setColumnStretch(1, 1)
        metrics_grid.setColumnStretch(3, 1)
        metrics_layout.addLayout(metrics_grid)

        dataset_card, dataset_layout = self._create_footer_card("MEMÓRIA LOCAL • KNN")
        window.lbl_db_info = QLabel("Sem dados no momento.")
        window.lbl_db_info.setWordWrap(True)
        window.lbl_db_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        window.lbl_db_info.setObjectName("fieldValue")
        window.frame_knn = KNNSpectrumWidget()
        window.frame_knn.setMinimumHeight(80)
        dataset_layout.addWidget(window.lbl_db_info)
        dataset_layout.addWidget(window.frame_knn, stretch=1)

        self.footer_cards = [verdict_card, metrics_card, dataset_card]
        confidence_layout.addLayout(self.footer_grid)
        parent_layout.addWidget(window.confidence_frame)

    def _build_action_buttons(self, window, parent_layout):
        window.controls_section = QFrame()
        window.controls_section.setObjectName("controlsSection")
        controls_layout = QVBoxLayout(window.controls_section)
        controls_layout.setContentsMargins(10, 10, 10, 10)
        controls_layout.setSpacing(8)
        controls_layout.addWidget(
            self._section_heading(
                "CONTROLES OPERACIONAIS",
                "Ações primárias destacadas em amarelo; comandos auxiliares permanecem discretos.",
            )
        )

        window.action_widget = QWidget()
        master_action_layout = QVBoxLayout(window.action_widget)
        master_action_layout.setContentsMargins(0, 0, 0, 0)
        master_action_layout.setSpacing(8)

        light_label = QLabel("ILUMINAÇÃO DA CÂMERA")
        light_label.setObjectName("fieldLabel")
        master_action_layout.addWidget(light_label)
        self.light_grid = QGridLayout()
        self.light_grid.setHorizontalSpacing(8)
        self.light_grid.setVerticalSpacing(8)

        window.btn_light_mid = QPushButton("Luz MID • ←")
        window.btn_light_side = QPushButton("Luz SIDE • ↓")
        window.btn_light_top = QPushButton("Luz TOP • →")
        self.light_buttons = [window.btn_light_mid, window.btn_light_side, window.btn_light_top]
        for button in self.light_buttons:
            button.setObjectName("lightButton")
            button.setCursor(Qt.CursorShape.PointingHandCursor)
        master_action_layout.addLayout(self.light_grid)

        action_label = QLabel("CAPTURA E DECISÃO DO OPERADOR")
        action_label.setObjectName("fieldLabel")
        master_action_layout.addWidget(action_label)
        self.action_grid = QGridLayout()
        self.action_grid.setHorizontalSpacing(8)
        self.action_grid.setVerticalSpacing(8)

        window.btn_start = QPushButton("Capturar Local (MSS)")
        window.btn_skip = QPushButton("Descartar Imagem")
        window.btn_save_ok = QPushButton("Salvar Dataset: OK")
        window.btn_save_ng = QPushButton("Confirmar Defeito (NG)")

        window.btn_start.setObjectName("primaryButton")
        window.btn_skip.setObjectName("secondaryButton")
        window.btn_save_ok.setObjectName("outlineAccentButton")
        window.btn_save_ng.setObjectName("primaryButton")
        self.action_buttons = [window.btn_start, window.btn_skip, window.btn_save_ok, window.btn_save_ng]
        for button in self.action_buttons:
            button.setCursor(Qt.CursorShape.PointingHandCursor)
        master_action_layout.addLayout(self.action_grid)

        controls_layout.addWidget(window.action_widget)
        parent_layout.addWidget(window.controls_section)

        window.btn_start.clicked.connect(window.start_monitoring)
        window.btn_skip.clicked.connect(window.skip_image)
        window.btn_save_ok.clicked.connect(lambda: window.save_label("OK", source="button"))
        window.btn_save_ng.clicked.connect(lambda: window.save_label("NG", source="button"))

        def apply_mode_visibility(mode_text):
            visible = mode_text != "Modo Sombra"
            window.action_widget.setVisible(visible)
            window.controls_section.setVisible(visible)

        window.combo_mode.currentTextChanged.connect(apply_mode_visibility)
        apply_mode_visibility(window.combo_mode.currentText())

    def _build_ng_archive_bar(self, window, parent_layout):
        window.ng_archive_frame = QFrame()
        window.ng_archive_frame.setObjectName("ngArchiveFrame")
        window.ng_archive_frame.setMinimumWidth(0)
        window.ng_archive_frame.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        window.ng_archive_frame.setStyleSheet(
            "QFrame#ngArchiveFrame {"
            "background: #101010; border: 1px solid #303030; border-radius: 8px;"
            "}"
            "QLabel#ngArchiveTitle {"
            "color: #f5c518; font-size: 10px; font-weight: 900; letter-spacing: 1px;"
            "}"
            "QLabel#ngArchiveStatus {"
            "color: #8b949e; font-size: 11px; font-weight: 700;"
            "}"
            "QLabel#ngArchiveStatus[archiveEnabled='true'] { color: #4ade80; }"
            "QPushButton#ngArchiveToggleButton {"
            "min-height: 40px; padding: 7px 14px; border-radius: 6px; "
            "font-size: 11px; font-weight: 800; color: #b8b8b8; "
            "background: #181818; border: 1px solid #505050;"
            "}"
            "QPushButton#ngArchiveToggleButton:hover,"
            "QPushButton#ngArchiveToggleButton:focus {"
            "color: #f5c518; border-color: #f5c518; background: #242424;"
            "}"
            "QPushButton#ngArchiveToggleButton:checked {"
            "color: #07120b; background: #4ade80; border: 1px solid #4ade80;"
            "}"
            "QPushButton#ngArchiveToggleButton:checked:hover {"
            "background: #6ee79a; border-color: #b7f7cb;"
            "}"
        )

        self.ng_archive_grid = QGridLayout(window.ng_archive_frame)
        self.ng_archive_grid.setContentsMargins(12, 8, 12, 8)
        self.ng_archive_grid.setHorizontalSpacing(10)
        self.ng_archive_grid.setVerticalSpacing(6)

        window.lbl_ng_archive_title = QLabel("ARQUIVO VISUAL NG")
        window.lbl_ng_archive_title.setObjectName("ngArchiveTitle")

        window.lbl_ng_archive_status = QLabel(
            "Desativado • fluxo atual mantido"
        )
        window.lbl_ng_archive_status.setObjectName("ngArchiveStatus")
        window.lbl_ng_archive_status.setProperty("archiveEnabled", False)
        window.lbl_ng_archive_status.setWordWrap(True)
        window.lbl_ng_archive_status.setMinimumWidth(0)
        window.lbl_ng_archive_status.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )

        window.btn_toggle_ng_archive = QPushButton(
            "Salvar imagens NG • DESATIVADO"
        )
        window.btn_toggle_ng_archive.setObjectName("ngArchiveToggleButton")
        window.btn_toggle_ng_archive.setCheckable(True)
        window.btn_toggle_ng_archive.setChecked(False)
        window.btn_toggle_ng_archive.setProperty("archiveEnabled", False)
        window.btn_toggle_ng_archive.setCursor(
            Qt.CursorShape.PointingHandCursor
        )
        window.btn_toggle_ng_archive.setFocusPolicy(
            Qt.FocusPolicy.StrongFocus
        )
        window.btn_toggle_ng_archive.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        window.btn_toggle_ng_archive.setToolTip(
            "Quando ativado, salva em public/ng_archive uma cópia bruta da "
            "imagem de teste de toda decisão final NG. Não altera o dataset."
        )
        window.btn_toggle_ng_archive.toggled.connect(
            lambda checked: window.set_ng_archive_enabled(checked)
        )

        self.ng_archive_widgets = [
            window.lbl_ng_archive_title,
            window.lbl_ng_archive_status,
            window.btn_toggle_ng_archive,
        ]

        parent_layout.addWidget(window.ng_archive_frame)

    def _layout_ng_archive(self, window, compact: bool) -> None:
        grid = self.ng_archive_grid
        for widget in self.ng_archive_widgets:
            grid.removeWidget(widget)

        if compact:
            grid.addWidget(window.lbl_ng_archive_title, 0, 0)
            grid.addWidget(window.lbl_ng_archive_status, 1, 0)
            grid.addWidget(window.btn_toggle_ng_archive, 2, 0)
            grid.setColumnStretch(0, 1)
            grid.setColumnStretch(1, 0)
            grid.setColumnStretch(2, 0)
        else:
            grid.addWidget(window.lbl_ng_archive_title, 0, 0)
            grid.addWidget(window.lbl_ng_archive_status, 0, 1)
            grid.addWidget(window.btn_toggle_ng_archive, 0, 2)
            grid.setColumnStretch(0, 0)
            grid.setColumnStretch(1, 1)
            grid.setColumnStretch(2, 0)

        window.btn_toggle_ng_archive.setMinimumWidth(0 if compact else 250)

    def _build_network_debug_bar(self, window, parent_layout):
        window.network_debug_frame = QFrame()
        window.network_debug_frame.setObjectName("networkDebugFrame")
        window.network_debug_frame.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        window.network_debug_frame.setStyleSheet(
            "QFrame#networkDebugFrame {"
            "background: #101010; border: 1px solid #303030; border-radius: 8px;"
            "}"
            "QLabel#networkDebugTitle {"
            "color: #f5c518; font-size: 10px; font-weight: 900; letter-spacing: 1px;"
            "}"
            "QLabel#networkDebugState {"
            "color: #9a9a9a; font-size: 11px; font-weight: 700;"
            "}"
            "QLabel#networkDebugState[state='ready'] { color: #4ade80; }"
            "QLabel#networkDebugState[state='partial'] { color: #f5c518; }"
            "QLabel#networkDebugState[state='rejected'] { color: #ff7777; }"
            "QWidget#networkDebugActions {"
            "background: transparent; border: none;"
            "}"
            "QPushButton#networkDebugTextButton,"
            "QPushButton#networkDebugImageButton {"
            "min-height: 40px; min-width: 128px; padding: 7px 14px; "
            "border-radius: 6px; font-size: 11px; font-weight: 800;"
            "}"
            "QPushButton#networkDebugTextButton {"
            "color: #e2e2e2; background: #181818; border: 1px solid #505050;"
            "}"
            "QPushButton#networkDebugTextButton:hover,"
            "QPushButton#networkDebugTextButton:focus {"
            "color: #f5c518; border-color: #f5c518; background: #242424;"
            "}"
            "QPushButton#networkDebugImageButton {"
            "color: #f5c518; background: #211d0d; border: 1px solid #8f7a18;"
            "}"
            "QPushButton#networkDebugImageButton:hover,"
            "QPushButton#networkDebugImageButton:focus {"
            "color: #101010; background: #f5c518; border-color: #ffd84d;"
            "}"
            "QPushButton#networkDebugTextButton:pressed,"
            "QPushButton#networkDebugImageButton:pressed {"
            "padding-top: 8px; padding-bottom: 6px;"
            "}"
            "QPushButton#networkDebugTextButton:disabled,"
            "QPushButton#networkDebugImageButton:disabled {"
            "color: #666666; background: #111111; border: 1px solid #292929;"
            "}"
        )

        self.network_debug_grid = QGridLayout(window.network_debug_frame)
        self.network_debug_grid.setContentsMargins(12, 8, 12, 8)
        self.network_debug_grid.setHorizontalSpacing(10)
        self.network_debug_grid.setVerticalSpacing(6)

        window.lbl_network_debug_title = QLabel("DIAGNÓSTICO XP")
        window.lbl_network_debug_title.setObjectName("networkDebugTitle")
        window.lbl_network_debug_title.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )

        window.lbl_network_debug_state = QLabel(
            "Aguardando a primeira imagem do Windows XP"
        )
        window.lbl_network_debug_state.setObjectName("networkDebugState")
        window.lbl_network_debug_state.setProperty("state", "idle")
        window.lbl_network_debug_state.setWordWrap(True)
        window.lbl_network_debug_state.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )

        window.network_debug_actions = QWidget(window.network_debug_frame)
        window.network_debug_actions.setObjectName("networkDebugActions")
        window.network_debug_actions.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        self.network_debug_actions_layout = QHBoxLayout(window.network_debug_actions)
        self.network_debug_actions_layout.setContentsMargins(0, 0, 0, 0)
        self.network_debug_actions_layout.setSpacing(8)

        window.btn_copy_network_debug = QPushButton("Copiar debug")
        window.btn_copy_network_debug.setObjectName("networkDebugTextButton")
        window.btn_copy_network_debug.setCursor(Qt.CursorShape.PointingHandCursor)
        window.btn_copy_network_debug.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        window.btn_copy_network_debug.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        window.btn_copy_network_debug.setToolTip(
            "Copia o relatório técnico da última imagem recebida do Windows XP."
        )
        window.btn_copy_network_debug.clicked.connect(
            lambda _checked=False: copy_network_debug_to_clipboard(window)
        )

        window.btn_copy_network_image = QPushButton("Copiar imagem XP")
        window.btn_copy_network_image.setObjectName("networkDebugImageButton")
        window.btn_copy_network_image.setCursor(Qt.CursorShape.PointingHandCursor)
        window.btn_copy_network_image.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        window.btn_copy_network_image.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        window.btn_copy_network_image.setToolTip(
            "Copia exatamente o frame enviado pelo Windows XP que está vinculado "
            "ao diagnóstico atual."
        )
        window.btn_copy_network_image.clicked.connect(
            lambda _checked=False: copy_network_image_to_clipboard(window)
        )

        self.network_debug_actions_layout.addWidget(
            window.btn_copy_network_debug,
            stretch=1,
        )
        self.network_debug_actions_layout.addWidget(
            window.btn_copy_network_image,
            stretch=1,
        )

        self.network_debug_widgets = [
            window.lbl_network_debug_title,
            window.lbl_network_debug_state,
            window.network_debug_actions,
        ]

        parent_layout.addWidget(window.network_debug_frame)
        sync_network_debug_controls(window)

    def _layout_network_debug(self, window, compact: bool) -> None:
        grid = self.network_debug_grid
        for widget in self.network_debug_widgets:
            grid.removeWidget(widget)

        if compact:
            grid.addWidget(window.lbl_network_debug_title, 0, 0)
            grid.addWidget(window.lbl_network_debug_state, 1, 0)
            grid.addWidget(window.network_debug_actions, 2, 0)
            grid.setColumnStretch(0, 1)
            window.network_debug_frame.setMinimumHeight(138)
        else:
            grid.addWidget(window.lbl_network_debug_title, 0, 0)
            grid.addWidget(window.lbl_network_debug_state, 0, 1)
            grid.addWidget(window.network_debug_actions, 0, 2)
            grid.setColumnStretch(0, 0)
            grid.setColumnStretch(1, 1)
            grid.setColumnStretch(2, 0)
            window.network_debug_frame.setMinimumHeight(62)

        window.network_debug_actions.setMinimumWidth(0 if compact else 300)

    def _build_status_bar(self, window, parent_layout):
        window.status_frame = QFrame()
        window.status_frame.setObjectName("statusBar")
        window.status_frame.setMinimumWidth(0)
        window.status_frame.setMinimumHeight(34)
        self.status_layout = QGridLayout(window.status_frame)
        self.status_layout.setContentsMargins(12, 5, 12, 5)
        self.status_layout.setHorizontalSpacing(10)
        self.status_layout.setVerticalSpacing(4)

        window.lbl_status_network = QLabel("Ouvindo AOI (Porta 5001)")
        window.lbl_status_brain = QLabel("Sistema Ocioso")
        window.lbl_status_history = QLabel("Última Peça: Nenhuma")

        self.status_widgets = [
            window.lbl_status_network,
            window.lbl_status_brain,
            window.lbl_status_history,
        ]
        window.status_layout_items = list(self.status_widgets)
        for label in self.status_widgets:
            label.setMinimumWidth(0)
            label.setWordWrap(True)
            label.setSizePolicy(
                QSizePolicy.Policy.Ignored,
                QSizePolicy.Policy.Preferred,
            )

        self.lbl_status_network = window.lbl_status_network
        self.lbl_status_brain = window.lbl_status_brain
        self.lbl_status_history = window.lbl_status_history

        self.status_layout.addWidget(window.lbl_status_network, 0, 0)
        self.status_layout.addWidget(window.lbl_status_brain, 0, 1)
        self.status_layout.addWidget(window.lbl_status_history, 0, 2)
        parent_layout.addWidget(window.status_frame)

    def _layout_header(self, window, compact: bool) -> None:
        for widget in self.header_widgets:
            self.header_layout.removeWidget(widget)
            widget.setMinimumWidth(0)

        if compact:
            self.header_layout.addWidget(self.header_widgets[0], 0, 0, 1, 2)
            self.header_layout.addWidget(self.header_widgets[1], 1, 0)
            self.header_layout.addWidget(self.header_widgets[2], 1, 1)
            self.header_layout.setColumnStretch(0, 1)
            self.header_layout.setColumnStretch(1, 1)
            self.header_layout.setColumnStretch(2, 0)
        else:
            self.header_layout.addWidget(self.header_widgets[0], 0, 0)
            self.header_layout.addWidget(self.header_widgets[1], 0, 1)
            self.header_layout.addWidget(self.header_widgets[2], 0, 2)
            self.header_layout.setColumnStretch(0, 1)
            self.header_layout.setColumnStretch(1, 0)
            self.header_layout.setColumnStretch(2, 0)

    def _layout_status_bar(self, window, compact: bool) -> None:
        while self.status_layout.count():
            self.status_layout.takeAt(0)

        layout_items = list(
            getattr(window, "status_layout_items", self.status_widgets)
        )
        alignments = (
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            Qt.AlignmentFlag.AlignCenter,
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
        )

        if compact:
            for row, (item, label) in enumerate(
                zip(layout_items, self.status_widgets)
            ):
                label.setAlignment(
                    Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
                )
                self.status_layout.addWidget(
                    item,
                    row,
                    0,
                    1,
                    1,
                    Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                )
            self.status_layout.setColumnStretch(0, 1)
            self.status_layout.setColumnStretch(1, 0)
            self.status_layout.setColumnStretch(2, 0)
            window.status_frame.setMinimumHeight(72)
        else:
            for column, (item, label, alignment) in enumerate(
                zip(layout_items, self.status_widgets, alignments)
            ):
                label.setAlignment(alignment)
                self.status_layout.addWidget(
                    item,
                    0,
                    column,
                    1,
                    1,
                    alignment,
                )
                self.status_layout.setColumnStretch(column, 1)
            window.status_frame.setMinimumHeight(34)

    @staticmethod
    def _reflow_grid(grid: QGridLayout, widgets: list[QWidget], columns: int) -> None:
        for widget in widgets:
            grid.removeWidget(widget)
        for index, widget in enumerate(widgets):
            grid.addWidget(widget, index // columns, index % columns)
        for column in range(columns):
            grid.setColumnStretch(column, 1)

    def apply_layout_profile(self, window, width: int, force: bool = False) -> None:
        width = max(int(width), 1)
        profile = profile_for_width(width)
        profile_changed = force or profile.name != self._active_profile_name
        self._active_profile_name = profile.name
        compact = profile.name == "compact"

        window.content_layout.setContentsMargins(
            profile.outer_margin,
            profile.outer_margin,
            profile.outer_margin,
            profile.outer_margin,
        )
        window.content_layout.setSpacing(profile.section_spacing)

        if profile_changed:
            self._layout_header(window, compact=compact)
            self._reflow_grid(self.info_grid, self.info_cards, profile.info_columns)
            self._reflow_grid(self.footer_grid, self.footer_cards, profile.footer_columns)
            self._reflow_grid(
                self.light_grid,
                self.light_buttons,
                min(3, profile.action_columns),
            )
            self._reflow_grid(
                self.action_grid,
                self.action_buttons,
                profile.action_columns,
            )
            self._layout_ng_archive(window, compact=compact)
            self._layout_network_debug(window, compact=compact)
            self._layout_status_bar(window, compact=compact)

        orientation = (
            Qt.Orientation.Vertical
            if profile.splitter_vertical
            else Qt.Orientation.Horizontal
        )
        window.main_splitter.setOrientation(orientation)
        window.main_splitter.setMinimumWidth(0)
        window.images_section.setMinimumWidth(0)
        window.telemetry_section.setMinimumWidth(0)
        window.telemetry_section.setMaximumWidth(16777215)
        window.scroll_area.setMinimumWidth(0)

        for viewport in self.image_viewports:
            viewport.setMinimumHeight(profile.image_min_height)

        for wrapper in self.debug_wrappers:
            wrapper.setMinimumWidth(profile.debugger_min_width)
            wrapper.setMaximumWidth(profile.debugger_max_width)
            wrapper.setMinimumHeight(265 if compact else 300)

        if profile.splitter_vertical:
            window.images_section.setMaximumWidth(16777215)
            window.images_section.setMinimumHeight(320)
            window.telemetry_section.setMinimumHeight(330)
            window.main_splitter.setSizes([340, 430])
        else:
            image_width = 400 if profile.name == "wide" else 340
            window.images_section.setMaximumWidth(image_width)
            window.images_section.setMinimumHeight(0)
            window.telemetry_section.setMinimumHeight(360)
            available_for_telemetry = max(1, width - image_width)
            window.main_splitter.setSizes(
                [image_width, available_for_telemetry]
            )

        window.root_content.setMinimumWidth(0)
        window.root_content.updateGeometry()
        window.updateGeometry()
