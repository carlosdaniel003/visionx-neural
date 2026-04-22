# src/ui/widgets/knn_spectrum.py
from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QColor, QFont, QLinearGradient, QPolygonF, QPen
from PyQt6.QtCore import Qt, QPointF, QRectF

class KNNSpectrumWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(80)
        
        # Dados iniciais vazios
        self.is_active = False # Flag MoE: O K-NN rodou?
        self.has_memory = False
        self.vote = 0.5
        self.best_sim = 0.0
        self.n_neighbors = 0

    def update_data(self, detail: dict):
        """ Recebe o dicionario de analise e extrai os dados do banco (KNN) """
        # MoE: Verifica os novos nomes das chaves do Expert ou os antigos por segurança
        if not detail or ("vote_defect" not in detail and "db_vote" not in detail):
            self.is_active = False
            self.update()
            return
            
        self.is_active = True
        
        # Busca pelas chaves novas do Orquestrador, ou as antigas (Backward compatibility)
        self.has_memory = detail.get("has_memory", detail.get("db_has_memory", False))
        self.vote = detail.get("vote_defect", detail.get("db_vote", 0.5)) 
        self.best_sim = detail.get("best_similarity", detail.get("db_best_sim", 0.0))
        self.n_neighbors = detail.get("n_neighbors", detail.get("db_neighbors", 0))

        self.update() # Forca o redesenho na tela

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()

        # Fundo do componente
        painter.fillRect(0, 0, w, h, QColor("#1a1a1a"))

        font_title = QFont("Consolas", 8, QFont.Weight.Bold)
        font_text = QFont("Consolas", 8)

        # MoE: Se o K-NN não foi ativado, acinza o painel
        if not self.is_active:
            painter.setPen(QColor("#555555"))
            font = QFont("Consolas", 10, QFont.Weight.Bold)
            painter.setFont(font)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Motor K-NN Inativo (MoE)")
            painter.end()
            return

        # Se nao houver ninguem no banco de dados ainda
        if not self.has_memory:
            painter.setPen(QColor("#555555"))
            painter.setFont(font_title)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Dataset Vazio (Sem Vizinhos)")
            painter.end()
            return

        # Titulo superior
        painter.setPen(QColor("#dddddd"))
        painter.setFont(font_title)
        painter.drawText(5, 15, "Espectro KNN (Voto da Memoria)")

        # Dimensoes e margens da barra de espectro
        pad_x = 20
        bar_h = 12
        bar_y = (h / 2.0) - (bar_h / 2.0)
        bar_w = w - (pad_x * 2)

        # Criacao do Gradiente (Verde -> Cinza -> Vermelho)
        grad = QLinearGradient(pad_x, bar_y, pad_x + bar_w, bar_y)
        grad.setColorAt(0.0, QColor("#55ff55")) # Extremo OK
        grad.setColorAt(0.5, QColor("#666666")) # Neutro / Duvida
        grad.setColorAt(1.0, QColor("#ff5555")) # Extremo NG

        # Desenha a barra base
        painter.setBrush(grad)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(QRectF(pad_x, bar_y, bar_w, bar_h), 4, 4)

        # Calculo da posicao do ponteiro (Marcador)
        marker_x = pad_x + (bar_w * self.vote)
        marker_y = bar_y

        # Desenha o ponteiro (Um triangulo apontando para baixo)
        poly = QPolygonF([
            QPointF(marker_x - 6, marker_y - 8),
            QPointF(marker_x + 6, marker_y - 8),
            QPointF(marker_x, marker_y + 2)
        ])
        painter.setBrush(QColor("#ffffff"))
        painter.setPen(QPen(QColor("#000000"), 1))
        painter.drawPolygon(poly)

        # Linha de corte dentro da barra
        painter.setPen(QPen(QColor("#ffffff"), 2))
        painter.drawLine(QPointF(marker_x, bar_y + 2), QPointF(marker_x, bar_y + bar_h))

        # --- Textos Inferiores ---
        painter.setFont(font_text)
        
        # Rotulo Esquerdo (OK)
        painter.setPen(QColor("#55ff55"))
        painter.drawText(pad_x, int(bar_y + bar_h + 16), "OK")
        
        # Rotulo Direito (NG)
        painter.setPen(QColor("#ff5555"))
        text_ng = "NG"
        ng_width = painter.fontMetrics().horizontalAdvance(text_ng)
        painter.drawText(int(pad_x + bar_w - ng_width), int(bar_y + bar_h + 16), text_ng)

        # Texto Central com os detalhes tecnicos (Precisao)
        painter.setPen(QColor("#aaaaaa"))
        info_text = f"Voto NG: {self.vote*100:.0f}% | Amostras: {self.n_neighbors} | Sim. Max: {self.best_sim*100:.0f}%"
        info_width = painter.fontMetrics().horizontalAdvance(info_text)
        
        # Centraliza o texto
        text_x = (w / 2.0) - (info_width / 2.0)
        painter.drawText(int(text_x), int(bar_y + bar_h + 16), info_text)

        painter.end()