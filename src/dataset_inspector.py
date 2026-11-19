"""
Dataset Quality Inspector for Ripleytia AI Studio.
Calculates dataset duration, segment count, SNR, clipping detection, and overall Quality Grade.
"""

import os
import glob
import numpy as np
import soundfile as sf
import librosa


def inspect_dataset_directory(dataset_dir: str) -> dict:
    """
    Scans dataset directory and returns comprehensive quality report.
    """
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset klasörü bulunamadı: {dataset_dir}")

    audio_files = []
    for ext in ("*.wav", "*.mp3", "*.flac", "*.ogg"):
        audio_files.extend(glob.glob(os.path.join(dataset_dir, "**", ext), recursive=True))

    if not audio_files:
        return {
            "total_files": 0,
            "total_duration_sec": 0,
            "duration_str": "00:00",
            "grade": "Veri Yok",
            "snr_db": 0.0,
            "clipping_pct": 0.0,
            "advice": "Klasörde ses dosyası (.wav, .mp3) bulunamadı."
        }

    total_sec = 0.0
    snr_list = []
    clipped_samples = 0
    total_samples = 0

    # Sample up to 50 files for swift inspection
    sample_files = audio_files[:50]
    for fp in sample_files:
        try:
            info = sf.info(fp)
            total_sec += info.duration
            
            # Read first 15 seconds to check SNR and clipping
            data, sr = sf.read(fp, dtype="float32", always_2d=True)
            mono = np.mean(data, axis=1)
            total_samples += len(mono)
            
            # Check clipping (samples near +/- 0.999)
            clipped_samples += int(np.sum(np.abs(mono) >= 0.995))

            # SNR estimation
            rms = librosa.feature.rms(y=mono)[0]
            if len(rms) > 4:
                db_rms = librosa.amplitude_to_db(rms, ref=np.max)
                act = db_rms > -40
                if np.sum(act) > 0 and np.sum(~act) > 0:
                    snr_list.append(float(np.mean(db_rms[act]) - np.mean(db_rms[~act])))
        except Exception:
            continue

    # Scale total duration if we sampled a subset
    if len(audio_files) > len(sample_files):
        avg_dur = total_sec / len(sample_files) if sample_files else 0
        total_sec = avg_dur * len(audio_files)

    avg_snr = float(np.mean(snr_list)) if snr_list else 30.0
    clipping_pct = (clipped_samples / max(1, total_samples)) * 100.0

    # Minutes and Seconds format
    mins = int(total_sec // 60)
    secs = int(total_sec % 60)
    dur_str = f"{mins} dk {secs:02d} sn"

    # Grading logic
    if total_sec >= 15 * 60 and avg_snr >= 26 and clipping_pct < 0.5:
        grade = "A+ (Kusursuz Stüdyo)"
        badge_color = "#00d2d3"
        advice = "Veri setiniz ultra yüksek kalitede! 140 - 180 Epoch 'Maksimum Kalite' eğitimi ile dünya standartlarında model elde edebilirsiniz."
    elif total_sec >= 8 * 60 and avg_snr >= 20:
        grade = "A (Çok Başarılı)"
        badge_color = "#7B68EE"
        advice = "Harika bir vokal havuzu. 120 - 150 Epoch 'Dengeli' preset ile modeliniz oturacaktır."
    elif total_sec >= 3 * 60:
        grade = "B (Yeterli / Orta)"
        badge_color = "#f39c12"
        advice = "Kabul edilebilir süre. Aşırı öğrenmeyi (Overfitting) önlemek için 100 Epoch civarı eğitim önerilir."
    else:
        grade = "C (Düşük / Kısa Süre)"
        badge_color = "#e74c3c"
        advice = "Veri seti 3 dakikadan az. Ses modelinin tınıyı tam öğrenmesi için YouTube'dan ek şarkı veya konuşma eklemeniz önerilir."

    return {
        "total_files": len(audio_files),
        "total_duration_sec": round(total_sec, 1),
        "duration_str": dur_str,
        "grade": grade,
        "badge_color": badge_color,
        "snr_db": round(avg_snr, 1),
        "clipping_pct": round(clipping_pct, 2),
        "advice": advice
    }
