import os
import logging
import traceback
import numpy as np

# Bağımsız modüllerin (ileride eklenecek) sahte (mock) importları
# from audio_splitter import RipleytiaSeparator
# from rvc_engine import RipleytiaRVCManager

# Dinamik ana dizin (Evrensel Dağıtım Kuralı)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from pydub import AudioSegment
import pydub.effects

class RipleytiaAudioMixer:
    """Sesleri senkronize birleştiren ve mastering uygulayan stüdyo kalitesinde sınıf."""
    
    @staticmethod
    def _strip_silence(audio: AudioSegment, silence_thresh=-50.0, chunk_size=10) -> AudioSegment:
        """Başlangıçtaki boşluğu kırparak RVC'nin oluşturduğu faz kaymasını engeller."""
        trim_ms = 0
        assert chunk_size > 0
        while trim_ms < len(audio) and audio[trim_ms:trim_ms+chunk_size].dBFS < silence_thresh:
            trim_ms += chunk_size
        return audio[trim_ms:]

    @staticmethod
    def mix_and_master(
        instrumental_path: str, 
        vocal_path: str, 
        output_path: str, 
        vocal_boost_db: float = 3.0, 
        inst_boost_db: float = 0.0
    ):
        """
        Gelişmiş Pydub Stüdyo Masteringi:
        - Faz Hizalama (Temporal Alignment)
        - RMS Tabanlı Otomatik Gain Dengelemesi
        - Clipping Önleyici Kompresör ve Limiter
        - Fade-In (500ms) / Fade-Out (1500ms)
        """
        try:
            # 1. Sesleri Yükle
            logging.info("[Mastering] Ses izleri belleğe alınıyor (pydub)...")
            inst = AudioSegment.from_file(instrumental_path)
            vocal = AudioSegment.from_file(vocal_path)

            # 2. Faz Hizalama (Temporal Alignment)
            logging.info("[Mastering] Faz hizalaması yapılıyor (Silence stripping)...")
            # RVC dönüştürücüleri vokalin başına sessizlik (padding) ekleyebilir, bunu buduyoruz.
            vocal = RipleytiaAudioMixer._strip_silence(vocal, silence_thresh=-45.0)

            # 3. Auto-Gain & Volume Balancing (RMS)
            logging.info("[Mastering] RMS enerji seviyeleri taranıyor (Auto-Gain)...")
            inst_db = inst.dBFS
            vocal_db = vocal.dBFS
            
            # Vokali enstrümantale göre eşitle ve altın oran (vocal_boost_db) kadar üste taşı
            if vocal_db != float('-inf'):
                target_vocal_db = inst_db + vocal_boost_db
                gain_difference = target_vocal_db - vocal_db
                vocal = vocal.apply_gain(gain_difference)
            
            inst = inst.apply_gain(inst_boost_db)

            # 4. Mix (Üst Üste Bindirme)
            logging.info("[Mastering] İzler üst üste bindiriliyor (Overlay)...")
            # Vokal müzikten uzunsa veya kısaysa instrumental süresine sabitlenir
            mixed = inst.overlay(vocal)

            # 5. Kırpma Koruması & Digital Limiter (-1.0 dBFS True Peak Limiter)
            logging.info("[Mastering] Anti-Clipping Limitör ve Kompresör uygulanıyor...")
            # Stüdyo tipi dinamik aralık kompresörü (Ses patlamalarını yumuşatır)
            mixed = pydub.effects.compress_dynamic_range(mixed, threshold=-12.0, ratio=4.0, attack=5.0, release=50.0)
            
            # -1.0 dBFS Limiter (Kesinlikle distorsiyona girmesini engeller)
            peak_db = mixed.max_dBFS
            if peak_db > -1.0:
                mixed = mixed.apply_gain(-1.0 - peak_db)

            # 6. Fade-In / Fade-Out
            logging.info("[Mastering] Giriş-Çıkış yumuşatıcıları (Fade) ekleniyor...")
            mixed = mixed.fade_in(500).fade_out(1500)

            # 7. Çıktıyı Kaydet
            logging.info(f"[Mastering] Nihai stüdyo dosyası dışa aktarılıyor: {output_path}")
            mixed.export(output_path, format="wav")
            
            return True, output_path
            
        except Exception as e:
            logging.error(f"[MASTERING HATA] {str(e)}")
            logging.error(traceback.format_exc())
            raise e


class RipleytiaCoverPipeline:
    """
    Tüm ses dönüştürme adımlarını (Split -> Convert -> Mix) 
    dinamik dizin koruması ve CUDA/CPU otomatik geçişi ile yöneten ana işlem motoru.
    """
    
    def __init__(self):
        self._setup_device_guard()
        self._setup_directories()
        
    def _setup_device_guard(self):
        """Donanım ivmesi tespiti (CUDA vs CPU)"""
        try:
            import torch
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                self.device_name = torch.cuda.get_device_name(0)
                logging.info(f"[Device Guard] NVIDIA GPU tespit edildi. İşlemler CUDA üzerinde yürütülecek: {self.device_name}")
            else:
                self.device = torch.device('cpu')
                logging.warning("[Device Guard] UYARI: NVIDIA GPU bulunamadı. Ağır işlemler CPU (Fallback) modunda çalışacak.")
        except ImportError:
            self.device = 'cpu'
            logging.warning("[Device Guard] UYARI: PyTorch yüklü değil, varsayılan CPU (Fallback) modu aktif.")

    def _setup_directories(self):
        """Uygulama klasör hiyerarşisini dinamik olarak oluşturur."""
        self.dirs = {
            'models': os.path.join(BASE_DIR, "models"),
            'temp': os.path.join(BASE_DIR, "temp"),
            'output': os.path.join(BASE_DIR, "output"),
            'assets': os.path.join(BASE_DIR, "assets")
        }
        for d in self.dirs.values():
            os.makedirs(d, exist_ok=True)
            
    def run_pipeline(self, input_audio: str, model_path: str, index_path: str, pitch_val: int, algo_val: str, output_dir: str = None):
        """
        Try-except bypass duvarları içeren kilitlenme korumalı ana Cover akışı.
        """
        logging.info(f"=== RIPLEYTIA AI COVER PIPELINE BAŞLATILDI ===")
        logging.info(f"Hedef Model: {model_path} | Algoritma: {algo_val} | Pitch: {pitch_val}")
        
        try:
            import subprocess
            
            # ADIM 1: AI Stem Splitting
            logging.info("[Adım 1/3] Vokal / Enstrümantal ayrıştırması başlatılıyor (Demucs)...")
            base_name = os.path.splitext(os.path.basename(input_audio))[0]
            
            cmd_demucs = [
                "demucs", "--two-stems=vocals", "-n", "htdemucs",
                "-o", self.dirs['temp'],
                input_audio
            ]
            process = subprocess.Popen(cmd_demucs, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate()
            
            out_folder = os.path.join(self.dirs['temp'], "htdemucs", base_name)
            vocals_path = os.path.join(out_folder, "vocals.wav")
            inst_path = os.path.join(out_folder, "no_vocals.wav")
            
            if not os.path.exists(vocals_path) or not os.path.exists(inst_path):
                raise Exception(f"Demucs ayırma işlemi başarısız oldu! (Belki Demucs yüklü değil?)\nHata: {stderr}")
            
            # ADIM 2: RVC Voice Inference
            logging.info("[Adım 2/3] RVC Motoru Vokal dönüşümünü uyguluyor...")
            converted_vocal = os.path.join(self.dirs['temp'], "converted_vocal.wav")
            
            try:
                from rvc_python.infer import RVCInference
                rvc = RVCInference(device="cuda" if self.device_name else "cpu")
                rvc.load_model(model_path)
                if index_path and os.path.exists(index_path):
                    rvc.set_index(index_path)
                    
                # Parametreleri rvc-python yapısına uygun şekilde ata
                rvc.set_params(
                    f0up_key=pitch_val,
                    f0method=algo_val
                )
                
                rvc.infer_file(
                    input_path=vocals_path,
                    output_path=converted_vocal
                )
            except ImportError:
                raise ImportError("Yapay Zeka (RVC) Motoru bulunamadı! Lütfen 'pip install rvc-python' komutu ile kurunuz.")
                
            if not os.path.exists(converted_vocal):
                raise Exception("RVC dönüştürme işlemi başarısız oldu!")
            
            # ADIM 3: Esports Studio Mastering
            logging.info("[Adım 3/3] Audio Mixer senkronizasyon ve mastering uyguluyor...")
            if not output_dir:
                output_dir = self.dirs['output']
            os.makedirs(output_dir, exist_ok=True)
            
            final_output = os.path.join(output_dir, f"Ripleytia_Cover_{os.path.basename(input_audio)}")
            RipleytiaAudioMixer.mix_and_master(inst_path, converted_vocal, final_output)
            
            logging.info("=== PIPELINE BAŞARIYLA TAMAMLANDI ===")
            return True, final_output
            
        except Exception as e:
            logging.error(f"[KRİTİK HATA] Pipeline kilitlenmesi yakalandı: {str(e)}")
            logging.error(traceback.format_exc())
            return False, str(e)
