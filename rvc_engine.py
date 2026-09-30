import os
import logging
import traceback
from PyQt6.QtCore import QThread, pyqtSignal

class RipleytiaRVCManager(QThread):
    """
    RVC v2 Ses Dönüştürme Motoru. 
    Kullanıcının yüklediği .pth model ağırlıklarını ve .index dosyasını kullanarak
    orijinal vokal izini hedef sese (RMVPE algoritmasıyla) dönüştürür.
    """
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool, str, str) # success, msg, converted_path
    
    def __init__(self, vocal_path: str, model_path: str, index_path: str, output_path: str, pitch_algo: str = 'rmvpe', pitch_shift: int = 0, device: str = 'cuda', parent=None):
        super().__init__(parent)
        self.vocal_path = vocal_path
        self.model_path = model_path
        self.index_path = index_path
        self.output_path = output_path
        self.pitch_algo = pitch_algo
        self.pitch_shift = pitch_shift
        self.device = device
        
    def run(self):
        try:
            self.progress.emit(10, f"[RVC] {os.path.basename(self.model_path)} modeli yükleniyor...")
            
            # Dinamik Import (GUI açılışında kilitlenmeyi önler, sadece çalışınca yükler)
            try:
                from rvc_python.infer import RVCInference
            except ImportError:
                raise ImportError("rvc-python kütüphanesi bulunamadı! 'pip install -r requirements.txt' çalıştırın.")
                
            self.progress.emit(40, "[RVC] Vokal verisi analiz ediliyor (RMVPE Pitch Extraction)...")
            
            # RVC Infer Pipeline
            rvc = RVCInference(device=self.device)
            rvc.load_model(self.model_path)
            
            if self.index_path and os.path.exists(self.index_path):
                rvc.set_index(self.index_path)
                self.progress.emit(50, "[RVC] Index dosyası (accent/timbre düzeltmesi) aktif edildi.")
            
            self.progress.emit(70, "[RVC] İnferans (Yapay Zeka Ses Dönüşümü) sürüyor. Lütfen bekleyin...")
            
            rvc.infer_file(
                input_path=self.vocal_path,
                output_path=self.output_path,
                pitch_algo=self.pitch_algo,   # rmvpe (En iyi kalite, f0 algılayıcı)
                pitch_shift=self.pitch_shift  # Kadın-Erkek arası geçişler için +/- 12
            )
            
            if not os.path.exists(self.output_path):
                raise FileNotFoundError("RVC motoru dönüşüm dosyasını oluşturamadı! CUDA bellek hatası olabilir.")
                
            self.progress.emit(100, "[RVC] Vokal Dönüşümü Tamamlandı!")
            self.finished.emit(True, "Başarılı", self.output_path)
            
        except Exception as e:
            logging.error(f"[RVC HATA] {str(e)}")
            logging.error(traceback.format_exc())
            self.finished.emit(False, str(e), "")
