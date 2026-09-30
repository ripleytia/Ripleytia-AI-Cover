import sys
import math
import random
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QGraphicsDropShadowEffect, QFrame, QProgressBar, QSlider)
from PyQt6.QtGui import QPainter, QPainterPath, QPen, QColor, QFont, QPixmap
from PyQt6.QtCore import Qt, QTimer, QPointF, QUrl
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput

from gui_theme import RipleytiaDesignSystem
from audio_mixer import RipleytiaCoverPipeline

class RipleytiaVFXLightningWidget(QWidget):
    """
    CPU/GPU optimizasyonlu, dinamik olarak hesaplanan ve sürekli hareket eden 
    'Neon Mor Şimşek / Enerji Arkı' (VFX Lightning & Energy Arcs) animasyon sınıfı.
    Midpoint Displacement Algoritması ile procedural çizim yapar.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        
        self.is_processing = False
        self.time_counter = 0.0
        
        # Smooth 60 FPS Loop
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_vfx)
        self.timer.start(16) 
        
        self.bolts = []
        
    def set_processing_state(self, state: bool):
        self.is_processing = state
        
    def update_vfx(self):
        # İşlem sırasında hız 3 katına çıkar (Process-Reactive)
        self.time_counter += 0.15 if self.is_processing else 0.05
        
        chance = 0.25 if self.is_processing else 0.02
        if random.random() < chance:
            self._generate_bolt()
            
        for bolt in self.bolts:
            bolt['alpha'] -= 20 if self.is_processing else 8
            
        self.bolts = [b for b in self.bolts if b['alpha'] > 0]
        self.update()

    def _midpoint_displacement(self, p1, p2, displacement, iterations):
        points = [p1, p2]
        for _ in range(iterations):
            new_points = []
            for i in range(len(points) - 1):
                mid_x = (points[i].x() + points[i+1].x()) / 2
                mid_y = (points[i].y() + points[i+1].y()) / 2
                mid_x += random.uniform(-displacement, displacement)
                mid_y += random.uniform(-displacement, displacement)
                new_points.append(points[i])
                new_points.append(QPointF(mid_x, mid_y))
            new_points.append(points[-1])
            points = new_points
            displacement *= 0.5
        return points

    def _generate_bolt(self):
        w, h = self.width(), self.height()
        if w < 10 or h < 10: return
        
        edge = random.choice(['top', 'bottom', 'left', 'right'])
        if edge == 'top':
            p1 = QPointF(0, random.uniform(0, 30))
            p2 = QPointF(w, random.uniform(0, 30))
        elif edge == 'bottom':
            p1 = QPointF(0, h - random.uniform(0, 30))
            p2 = QPointF(w, h - random.uniform(0, 30))
        elif edge == 'left':
            p1 = QPointF(random.uniform(0, 30), 0)
            p2 = QPointF(random.uniform(0, 30), h)
        else:
            p1 = QPointF(w - random.uniform(0, 30), 0)
            p2 = QPointF(w - random.uniform(0, 30), h)
            
        iterations = 5 if self.is_processing else 4
        displacement = 60.0 if self.is_processing else 25.0
        
        path_points = self._midpoint_displacement(p1, p2, displacement, iterations)
        
        self.bolts.append({
            'points': path_points,
            'alpha': 255,
            'thickness': random.uniform(2.0, 4.5) if self.is_processing else random.uniform(0.5, 1.5)
        })

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        sine_val = (math.sin(self.time_counter) + 1.0) / 2.0 
        
        for bolt in self.bolts:
            alpha = bolt['alpha']
            thick = bolt['thickness']
            
            core_color = QColor(RipleytiaDesignSystem.COLORS.CYBER_PINK)
            core_color.setAlpha(alpha)
            core_pen = QPen(core_color)
            core_pen.setWidthF(thick)
            
            aura_color = QColor(RipleytiaDesignSystem.COLORS.NEON_PURPLE)
            aura_color.setAlpha(int(alpha * 0.45))
            aura_pen = QPen(aura_color)
            aura_pen.setWidthF(thick * 4.0)
            
            path = QPainterPath()
            pts = bolt['points']
            path.moveTo(pts[0])
            for p in pts[1:]:
                path.lineTo(p)
                
            painter.setPen(aura_pen)
            painter.drawPath(path)
            painter.setPen(core_pen)
            painter.drawPath(path)


class RipleytiaAppWindow(QMainWindow):
    """Ripleytia AI Cover Masaüstü Arayüzü (Stüdyo Kalitesi)"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ripleytia AI Automated Cover - v1.0.0")
        self.resize(1000, 700)
        self.setStyleSheet(RipleytiaDesignSystem.get_main_stylesheet())
        
        self.pipeline = RipleytiaCoverPipeline()
        self._init_ui()
        
    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(50, 50, 50, 50)
        
        glass_panel = QFrame()
        glass_panel.setObjectName("GlassPanel")
        panel_layout = QVBoxLayout(glass_panel)
        
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(RipleytiaDesignSystem.GLOW_IDLE_RADIUS)
        shadow.setColor(QColor(RipleytiaDesignSystem.COLORS.NEON_PURPLE))
        shadow.setOffset(0, 0)
        glass_panel.setGraphicsEffect(shadow)
        self.panel_shadow = shadow
        
        # LOGO EKLENTİSİ
        self.logo_label = QLabel()
        self.logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo_path = RipleytiaDesignSystem.get_resource_path("logo.jpg")
        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path).scaled(120, 120, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.logo_label.setPixmap(pixmap)
        panel_layout.addWidget(self.logo_label)

        title = QLabel("RIPLEYTIA AI COVER STUDIO")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont("Impact", 24)
        title.setFont(font)
        title.setStyleSheet(f"color: {RipleytiaDesignSystem.COLORS.NEON_PURPLE};")
        panel_layout.addWidget(title)

        # İlerleme Çubuğu (Kıvılcım Etkileşimli)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: {RipleytiaDesignSystem.COLORS.DARK_BG};
                border: 1px solid {RipleytiaDesignSystem.COLORS.GLOW_PURPLE};
                border-radius: 4px;
                text-align: center;
                color: white;
            }}
            QProgressBar::chunk {{
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                                  stop:0 {RipleytiaDesignSystem.COLORS.GLOW_PURPLE}, 
                                                  stop:1 {RipleytiaDesignSystem.COLORS.CYBER_PINK});
                border-radius: 3px;
            }}
        """)
        panel_layout.addWidget(self.progress_bar)
        
        self.btn_process = QPushButton("ŞARKIYI İŞLE (TEST)")
        self.btn_process.clicked.connect(self.toggle_processing)
        panel_layout.addWidget(self.btn_process, alignment=Qt.AlignmentFlag.AlignCenter)

        # Entegre Stüdyo Önizleme Oyuncusu (Audio Preview Player)
        self.player_panel = QFrame()
        self.player_panel.hide() # İşlem bitene kadar gizli
        player_layout = QHBoxLayout(self.player_panel)
        
        self.btn_play = QPushButton("▶ OYNAT")
        self.btn_play.setStyleSheet(f"background-color: {RipleytiaDesignSystem.COLORS.NEON_PURPLE};")
        self.btn_play.clicked.connect(self.toggle_playback)
        player_layout.addWidget(self.btn_play)
        
        self.lbl_time = QLabel("00:00 / 00:00")
        self.lbl_time.setStyleSheet(f"color: {RipleytiaDesignSystem.COLORS.CYBER_PINK}; font-weight: bold;")
        player_layout.addWidget(self.lbl_time)
        
        panel_layout.addWidget(self.player_panel)
        main_layout.addWidget(glass_panel)
        
        # QMediaPlayer Kurulumu (PyQt6.QtMultimedia)
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(1.0)
        
        self.player.positionChanged.connect(self.update_player_time)
        self.player.durationChanged.connect(self.update_player_duration)
        self.player.playbackStateChanged.connect(self.player_state_changed)
        self._duration = 0
        
        self.vfx_layer = RipleytiaVFXLightningWidget(self)
        self.vfx_layer.resize(self.width(), self.height())
        self.vfx_layer.show()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'vfx_layer'):
            self.vfx_layer.resize(self.width(), self.height())

    def toggle_processing(self):
        is_proc = not self.vfx_layer.is_processing
        self.vfx_layer.set_processing_state(is_proc)
        
        if is_proc:
            self.btn_process.setText("İŞLENİYOR... (VFX AKTİF)")
            self.panel_shadow.setBlurRadius(RipleytiaDesignSystem.GLOW_PROCESSING_RADIUS)
            self.panel_shadow.setColor(QColor(RipleytiaDesignSystem.COLORS.CYBER_PINK))
            self.progress_bar.setValue(50) # Örnek İlerleme
            self.player_panel.hide()
            self.player.stop()
        else:
            self.btn_process.setText("ŞARKIYI İŞLE")
            self.panel_shadow.setBlurRadius(RipleytiaDesignSystem.GLOW_IDLE_RADIUS)
            self.panel_shadow.setColor(QColor(RipleytiaDesignSystem.COLORS.NEON_PURPLE))
            self.progress_bar.setValue(100)
            self.player_panel.show()
            # Örnek ses yükleme
            # self.player.setSource(QUrl.fromLocalFile("path/to/final_cover.wav"))

    def toggle_playback(self):
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def player_state_changed(self, state):
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.btn_play.setText("⏸ DURAKLAT")
        else:
            self.btn_play.setText("▶ OYNAT")

    def update_player_duration(self, duration):
        self._duration = duration
        self.update_time_label(self.player.position(), duration)

    def update_player_time(self, position):
        self.update_time_label(position, self._duration)

    def update_time_label(self, position, duration):
        def format_time(ms):
            s = int(ms / 1000)
            m = s // 60
            s = s % 60
            return f"{m:02d}:{s:02d}"
        self.lbl_time.setText(f"{format_time(position)} / {format_time(duration)}")
