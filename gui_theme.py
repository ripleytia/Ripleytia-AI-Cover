import os
from dataclasses import dataclass

@dataclass
class RipleytiaColors:
    # Derin mat siyah ve koyu gri (Background / Panels)
    DARK_BG = "#0B0B0E"
    PANEL_BG = "#13131A"
    
    # Siberpunk Vurgular
    NEON_PURPLE = "#8A2BE2"  # Ana marka rengi
    CYBER_PINK = "#FF007F"   # Şimşek çekirdeği ve kritik uyarılar
    GLOW_PURPLE = "#4B0082"  # Derin gölge / Hacimsel parlama
    
    # Metin
    TEXT_PRIMARY = "#FFFFFF"
    TEXT_SECONDARY = "#A0A0B0"

class RipleytiaDesignSystem:
    """
    Ripleytia markasının görsel kimliği, PyQt6 stil şablonları 
    ve 3D parlama parametrelerini barındıran mimari tasarım sınıfı.
    """
    COLORS = RipleytiaColors()
    
    # Animasyon ve Glow Parametreleri
    GLOW_IDLE_RADIUS = 15
    GLOW_PROCESSING_RADIUS = 40
    
    @classmethod
    def get_main_stylesheet(cls) -> str:
        """PyQt6 QMainWindow ve bileşenleri için evrensel stil."""
        return f"""
        QMainWindow {{
            background-color: {cls.COLORS.DARK_BG};
        }}
        QWidget {{
            color: {cls.COLORS.TEXT_PRIMARY};
            font-family: 'Segoe UI', Arial, sans-serif;
        }}
        QFrame#GlassPanel {{
            background-color: rgba(19, 19, 26, 0.7);
            border: 1px solid {cls.COLORS.GLOW_PURPLE};
            border-radius: 8px;
        }}
        QPushButton {{
            background-color: {cls.COLORS.PANEL_BG};
            border: 2px solid {cls.COLORS.NEON_PURPLE};
            border-radius: 6px;
            padding: 8px 16px;
            font-weight: bold;
        }}
        QPushButton:hover {{
            background-color: {cls.COLORS.GLOW_PURPLE};
            border: 2px solid {cls.COLORS.CYBER_PINK};
        }}
        QPushButton:pressed {{
            background-color: {cls.COLORS.NEON_PURPLE};
        }}
        """

    @classmethod
    def get_resource_path(cls, filename: str) -> str:
        """Dinamik dizin yönetimi (Evrensel kural)"""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base_dir, "assets", filename)
