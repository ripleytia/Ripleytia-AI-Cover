import sys
import os
import math
import random
import logging
import traceback
from pydub import AudioSegment
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QMessageBox
from PyQt6.QtCore import Qt, QTimer, QPointF
from PyQt6.QtGui import QPainter, QPainterPath, QPen, QColor, QFont
from gui_app import RipleytiaAppWindow
from gui_theme import RipleytiaDesignSystem

# ... (Splash Screen sınıfı kodları aynı kalır)

class RipleytiaSplashScreen(QWidget):
    """
    3 Saniyelik Animasyonlu Giriş Ekranı.
    Çerçevesiz (Frameless), saydam arka planlı ve dışa doğru yayılan 
    procedural mor şimşek emisyonlarına sahip.
    """
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.SplashScreen)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.resize(500, 500)
        
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.title = QLabel("RIPLEYTIA\nAI COVER")
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont("Impact", 42)
        self.title.setFont(font)
        self.title.setStyleSheet(f"color: {RipleytiaDesignSystem.COLORS.CYBER_PINK};")
        layout.addWidget(self.title)
        
        self.time_counter = 0.0
        self.bolts = []
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_vfx)
        self.timer.start(16)
        
        QTimer.singleShot(3000, self.close_splash)
        
    def close_splash(self):
        self.timer.stop()
        self.main_window = RipleytiaAppWindow()
        self.main_window.show()
        self.close()

    def update_vfx(self):
        self.time_counter += 0.1
        if random.random() < 0.3:
            self._generate_bolt()
        for bolt in self.bolts:
            bolt['alpha'] -= 10
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
                new_points.extend([points[i], QPointF(mid_x, mid_y)])
            new_points.append(points[-1])
            points = new_points
            displacement *= 0.5
        return points

    def _generate_bolt(self):
        w, h = self.width(), self.height()
        cx, cy = w / 2, h / 2
        angle = random.uniform(0, math.pi * 2)
        radius = random.uniform(100, 250)
        dx = cx + math.cos(angle) * radius
        dy = cy + math.sin(angle) * radius
        
        p1 = QPointF(cx, cy)
        p2 = QPointF(dx, dy)
        path = self._midpoint_displacement(p1, p2, 40, 4)
        
        self.bolts.append({'points': path, 'alpha': 255})

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        sine_val = (math.sin(self.time_counter) + 1.0) / 2.0
        glow_alpha = int(40 + 60 * sine_val)
        glow_radius = int(100 + 30 * sine_val)
        cx, cy = self.width() / 2, self.height() / 2
        
        painter.setBrush(QColor(138, 43, 226, glow_alpha))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(cx, cy), glow_radius, glow_radius)
        
        for bolt in self.bolts:
            alpha = bolt['alpha']
            core_pen = QPen(QColor(255, 0, 127, alpha))
            core_pen.setWidthF(2.0)
            aura_pen = QPen(QColor(138, 43, 226, int(alpha * 0.5)))
            aura_pen.setWidthF(6.0)
            
            path = QPainterPath()
            pts = bolt['points']
            path.moveTo(pts[0])
            for p in pts[1:]:
                path.lineTo(p)
                
            painter.setPen(aura_pen)
            painter.drawPath(path)
            painter.setPen(core_pen)
            painter.drawPath(path)

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    logging.info("Ripleytia AI Automated Cover - Başlatılıyor...")
    
    # Dinamik MEIPASS (PyInstaller Dağıtım) Yolu Çözücü
    if hasattr(sys, '_MEIPASS'):
        base_dir = sys._MEIPASS
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
    # FFmpeg Dinamik Bağlayıcı (Kullanıcının bilgisayarında kurulu olmasa bile çalışır)
    ffmpeg_path = os.path.join(base_dir, 'assets', 'ffmpeg.exe')
    if os.path.exists(ffmpeg_path):
        AudioSegment.converter = ffmpeg_path
        logging.info("Gömülü FFmpeg binary bulundu ve Pydub'a bağlandı.")

    app = QApplication(sys.argv)
    try:
        splash = RipleytiaSplashScreen()
        splash.show()
        sys.exit(app.exec())
    except Exception as e:
        logging.error(f"[KRİTİK HATA] Uygulama çöktü: {str(e)}")
        logging.error(traceback.format_exc())
        error_box = QMessageBox()
        error_box.setIcon(QMessageBox.Icon.Critical)
        error_box.setWindowTitle("Ripleytia Studio - Kritik Sistem Hatası")
        error_box.setText("Uygulama başlatılamadı. Lütfen logları kontrol edin.")
        error_box.setDetailedText(traceback.format_exc())
        error_box.setStyleSheet("background-color: #0B0B0E; color: #FFFFFF;")
        error_box.exec()
        sys.exit(1)

if __name__ == "__main__":
    main()
