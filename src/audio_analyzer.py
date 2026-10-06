import librosa
import numpy as np

class AudioAnalyzer:
    @staticmethod
    def analyze_audio(orig_path, vocal_path, inst_path):
        print("[Auto Mode] Şarkı analiz ediliyor...")
        results = {}
        
        try:
            # 1. Analyze Original Song (BPM)
            y_orig, sr = librosa.load(orig_path, sr=22050, duration=30) # Load 30 seconds to speed up
            tempo, _ = librosa.beat.beat_track(y=y_orig, sr=sr)
            bpm = tempo[0] if isinstance(tempo, (list, np.ndarray)) else tempo
            results['bpm'] = float(bpm)
            print(f"  -> Tespit edilen Tempo (BPM): {results['bpm']:.1f}")
        except Exception as e:
            print(f"  -> Tempo analizi hatası: {e}")
            results['bpm'] = 120.0

        try:
            # 2. Analyze Vocals (Dynamics, Brightness, Sibilance)
            y_voc, sr = librosa.load(vocal_path, sr=22050)
            
            # RMS (Dynamics)
            rms = librosa.feature.rms(y=y_voc)[0]
            rms_std = np.std(rms)
            rms_mean = np.mean(rms)
            dynamic_range = rms_std / (rms_mean + 1e-6)
            results['dynamic_range'] = float(dynamic_range)
            print(f"  -> Vokal Dinamik Aralığı: {results['dynamic_range']:.2f}")

            # Zero Crossing Rate (Sibilance/Breathiness)
            zcr = librosa.feature.zero_crossing_rate(y_voc)[0]
            zcr_mean = np.mean(zcr)
            results['zcr'] = float(zcr_mean)
            print(f"  -> Vokal Nefes/Sessiz Harf Yoğunluğu (ZCR): {results['zcr']:.4f}")
            
            # Spectral Centroid (Brightness)
            cent = librosa.feature.spectral_centroid(y=y_voc, sr=sr)[0]
            cent_mean = np.mean(cent)
            results['brightness'] = float(cent_mean)
            
        except Exception as e:
            print(f"  -> Vokal analizi hatası: {e}")
            results['dynamic_range'] = 0.5
            results['zcr'] = 0.05
            results['brightness'] = 2000.0

        return AudioAnalyzer.calculate_optimal_settings(results)

    @staticmethod
    def calculate_optimal_settings(features):
        settings = {}
        
        # 1. RMS Mix Rate
        # High dynamic range -> higher RMS mix to preserve emotion
        dr = features.get('dynamic_range', 0.5)
        if dr > 1.2:
            settings['rms_mix_rate'] = 0.85
        elif dr > 0.8:
            settings['rms_mix_rate'] = 0.75
        else:
            settings['rms_mix_rate'] = 0.65

        # 2. Protect (0.33 default. Lower = more protection for voiceless consonants)
        # High ZCR means lots of sibilance/breath
        zcr = features.get('zcr', 0.05)
        if zcr > 0.08:
            settings['protect'] = 0.20 # heavy protect
        elif zcr > 0.05:
            settings['protect'] = 0.25 # medium protect
        else:
            settings['protect'] = 0.33 # normal protect

        # 3. Index Rate (0.0 to 1.0)
        # Bright vocals might artifact more with high index.
        brightness = features.get('brightness', 2000)
        if brightness > 3000:
            settings['index_rate'] = 0.55
        elif brightness > 2000:
            settings['index_rate'] = 0.65
        else:
            settings['index_rate'] = 0.75

        # 4. Filter Radius & Reverb
        bpm = features.get('bpm', 120.0)
        if bpm > 130:
            # Fast song: tight filter, small room, less reverb
            settings['filter_radius'] = 3
            settings['reverb_rm_size'] = 0.10
            settings['reverb_wet'] = 0.10
            settings['reverb_dry'] = 0.90
        elif bpm < 90:
            # Slow song/Ballad: smooth filter, bigger room, more reverb
            settings['filter_radius'] = 4
            settings['reverb_rm_size'] = 0.25
            settings['reverb_wet'] = 0.25
            settings['reverb_dry'] = 0.80
        else:
            # Mid tempo
            settings['filter_radius'] = 3
            settings['reverb_rm_size'] = 0.15
            settings['reverb_wet'] = 0.15
            settings['reverb_dry'] = 0.85
            
        settings['reverb_damping'] = 0.7 # Keep damping constant for natural sound

        print("\n[Auto Mode] Hesaplanan Optimal Ayarlar:")
        for k, v in settings.items():
            print(f"  * {k}: {v}")
            
        return settings

