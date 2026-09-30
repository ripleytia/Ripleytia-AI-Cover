import os
import logging
import traceback
import subprocess
from PyQt6.QtCore import QThread, pyqtSignal

class RipleytiaSeparator(QThread):
    """
    Meta'nın Demucs v4 kütüphanesini kullanarak şarkıyı Vokal ve Enstrümantal
    olarak 2 kök (stem) parçaya ayıran ve arayüzü dondurmamak için
    QThread (Background) üzerinde koşan mimari.
    """
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool, str, str, str) # success, msg, vocals_path, inst_path
    
    def __init__(self, input_path: str, temp_dir: str, device: str = 'cuda', parent=None):
        super().__init__(parent)
        self.input_path = input_path
        self.temp_dir = temp_dir
        self.device = device
        
    def run(self):
        try:
            self.progress.emit(10, "[Ayırma] Ses dosyası hazırlanıyor...")
            
            # Dosya adını güvenli almak
            base_name = os.path.splitext(os.path.basename(self.input_path))[0]
            
            self.progress.emit(30, "[Ayırma] Demucs v4 HT (2 Stem) modeli çalışıyor... Lütfen bekleyin.")
            
            # Demucs CLI çağrısı (subprocess memory leak riskini tamamen sıfırlar)
            cmd = [
                "demucs",
                "--two-stems=vocals",
                "-n", "htdemucs",
                "-o", self.temp_dir,
                "--device", self.device,
                self.input_path
            ]
            
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                raise Exception(f"Demucs işlemi çöktü: {stderr}")
                
            self.progress.emit(90, "[Ayırma] İşlem tamamlandı. Dosyalar doğrulanıyor...")
            
            # Demucs çıktısı: {temp_dir}/htdemucs/{base_name}/vocals.wav
            out_folder = os.path.join(self.temp_dir, "htdemucs", base_name)
            vocals_path = os.path.join(out_folder, "vocals.wav")
            inst_path = os.path.join(out_folder, "no_vocals.wav")
            
            if not os.path.exists(vocals_path) or not os.path.exists(inst_path):
                raise FileNotFoundError("Demucs çıktı dosyalarını oluşturamadı (Yetersiz VRAM veya Bozuk Ses Dosyası).")
                
            self.progress.emit(100, "[Ayırma] Başarılı!")
            self.finished.emit(True, "Başarılı", vocals_path, inst_path)
            
        except Exception as e:
            logging.error(f"[SPLITTER HATA] {str(e)}")
            logging.error(traceback.format_exc())
            self.finished.emit(False, str(e), "", "")
