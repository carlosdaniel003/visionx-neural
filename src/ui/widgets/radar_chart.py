# src/ui/widgets/radar_chart.py
import math
from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QColor, QPen, QPolygonF, QFont
from PyQt6.QtCore import Qt, QPointF

class RadarChartWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(150)
        self.setMinimumWidth(150)
        
        # Inicializa o grafico vazio
        self.axes = [
            ("Estrutura", 0.0),
            ("Pixels", 0.0),
            ("Bordas", 0.0),
            ("Cores", 0.0),
            ("Score Local", 0.0),
            ("Contexto", 0.0)
        ]

    def update_data(self, detail: dict):
        """ 
        Recebe os detalhes da analise e converte para a escala de anomalia.
        0.0 = Perfeito (Centro) | 1.0 = Defeito Grave (Borda)
        """
        if not detail:
            return
            
        # 1. Estrutura (SSIM: 1.0 eh perfeito. Convertendo para anomalia: 1.0 - SSIM)
        ssim_val = detail.get("ssim", 1.0)
        anomalia_ssim = max(0.0, min(1.0, 1.0 - ssim_val))
        
        # 2. Pixels (0.15 de mudanca ja aciona o alarme maximo)
        pct = detail.get("pct_changed", 0.0)
        anomalia_pct = max(0.0, min(1.0, pct / 0.15))
        
        # 3. Bordas (0.08 de mudanca ja aciona o alarme maximo)
        edge = detail.get("edge_change", 0.0)
        anomalia_edge = max(0.0, min(1.0, edge / 0.08))
        
        # 4. Cores/Histograma (1.0 eh perfeito. Convertendo para anomalia: 1.0 - Hist)
        hist = detail.get("hist_corr", 1.0)
        anomalia_hist = max(0.0, min(1.0, 1.0 - hist))
        
        # 5. Score Local Bruto (Ja vem de 0.0 a 1.0 como defeito)
        local_score = detail.get("local_score", 0.0)
        
        # 6. Contexto (Ja vem formatado)
        ctx_score = detail.get("ctx_score", 0.0)
        
        # Atualiza a lista de desenho
        self.axes = [
            ("Estrutura", anomalia_ssim),
            ("Pixels", anomalia_pct),
            ("Bordas", anomalia_edge),
            ("Cores", anomalia_hist),
            ("Score Local", local_score),
            ("Contexto", ctx_score)
        ]
        
        self.update() # Forca o redesenho na tela

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        w = self.width()
        h = self.height()
        
        # Fundo Escuro
        painter.fillRect(0, 0, w, h, QColor("#1a1a1a"))
        
        # Calculo de Geometria
        cx = w / 2.0
        cy = (h / 2.0) + 10  # Desloca um pouco para baixo para nao sobrepor o titulo
        radius = min(cx, cy) - 25 # Margem de 25px para os textos dos eixos
        
        num_axes = len(self.axes)
        if num_axes == 0:
            return
            
        angle_step = (2 * math.pi) / num_axes
        
        # Desenha a grade de fundo (As linhas da teia de aranha)
        painter.setPen(QPen(QColor("#444444"), 1, Qt.PenStyle.DashLine))
        for step in range(1, 6): # 5 aneis indicando 20%, 40%, 60%, 80%, 100%
            r = radius * (step / 5.0)
            poly = QPolygonF()
            for i in range(num_axes):
                angle = i * angle_step - (math.pi / 2) # Comeca do topo (12 horas)
                px = cx + r * math.cos(angle)
                py = cy + r * math.sin(angle)
                poly.append(QPointF(px, py))
            painter.drawPolygon(poly)
            
        # Desenha as hastes centrais e os rotulos
        painter.setPen(QPen(QColor("#555555"), 1, Qt.PenStyle.SolidLine))
        painter.setFont(QFont("Consolas", 7, QFont.Weight.Bold))
        
        for i in range(num_axes):
            angle = i * angle_step - (math.pi / 2)
            px = cx + radius * math.cos(angle)
            py = cy + radius * math.sin(angle)
            painter.drawLine(QPointF(cx, cy), QPointF(px, py))
            
            # Posicionamento inteligente do texto
            label, val = self.axes[i]
            text_x = cx + (radius + 12) * math.cos(angle)
            text_y = cy + (radius + 12) * math.sin(angle)
            
            flags = Qt.AlignmentFlag.AlignCenter
            if math.cos(angle) > 0.1:
                flags = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            elif math.cos(angle) < -0.1:
                flags = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                
            text_rect = painter.boundingRect(int(text_x - 30), int(text_y - 10), 60, 20, flags, label)
            painter.setPen(QColor("#aaaaaa"))
            painter.drawText(text_rect, flags, label)

        # Desenha o poligono de dados (O tamanho da anomalia)
        data_poly = QPolygonF()
        area_total = 0.0
        
        for i in range(num_axes):
            angle = i * angle_step - (math.pi / 2)
            val = self.axes[i][1]
            area_total += val
            r = radius * val
            px = cx + r * math.cos(angle)
            py = cy + r * math.sin(angle)
            data_poly.append(QPointF(px, py))
            
        # Logica de cores: Verde se a anomalia for muito pequena, Vermelho se for grande
        if area_total < 1.5:
            cor_borda = QColor("#55ff55")
            cor_fundo = QColor(85, 255, 85, 80)
        else:
            cor_borda = QColor("#ff5555")
            cor_fundo = QColor(255, 85, 85, 100)

        # Preenchimento Translucido
        painter.setBrush(cor_fundo)
        painter.setPen(QPen(cor_borda, 2, Qt.PenStyle.SolidLine))
        painter.drawPolygon(data_poly)
        
        # Pontos nas extremidades
        painter.setBrush(cor_borda)
        for i in range(num_axes):
            angle = i * angle_step - (math.pi / 2)
            val = self.axes[i][1]
            r = radius * val
            px = cx + r * math.cos(angle)
            py = cy + r * math.sin(angle)
            painter.drawEllipse(QPointF(px, py), 3, 3)

        # Titulo
        painter.setPen(QColor("#dddddd"))
        painter.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
        painter.drawText(5, 15, "Morfologia do Defeito")
        
        painter.end()