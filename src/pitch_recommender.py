"""
Intelligent Pitch & Key Recommendation System for Ripleytia AI Studio.
Analyzes musical key and vocal F0 range of the source audio,
and calculates the ideal pitch shift (+/- semitones) for the target RVC voice model.
"""

import os
import math
import numpy as np
import soundfile as sf
import librosa


NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Heuristic gender classification based on model naming
FEMALE_KEYWORDS = [
    'kadin', 'kadın', 'female', 'girl', 'woman', 'pelin', 'pqueen', 'pquenn',
    'azki', 'miku', 'lady', 'nesil', 'eslem', 'sebnem', 'ferah', 'queen'
]
MALE_KEYWORDS = [
    'erkek', 'male', 'man', 'boy', 'adam', 'batu', 'baturay', 'ripley',
    'baris', 'manco', 'cem', 'karaca', 'sagopa', 'ceza', 'king'
]


def detect_musical_key(y: np.ndarray, sr: int) -> str:
    """
    Estimates the musical key (e.g. C Major, A Minor) using Krumhansl-Schmuckler chroma correlation.
    """
    try:
        # Compute chromagram
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_avg = np.mean(chroma, axis=1)

        # Standard major/minor profile templates
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

        major_profile = (major_profile - np.mean(major_profile)) / np.std(major_profile)
        minor_profile = (minor_profile - np.mean(minor_profile)) / np.std(minor_profile)

        best_score = -999.0
        best_key = "Bilinmiyor"

        for i in range(12):
            shifted_chroma = np.roll(chroma_avg, -i)
            shifted_chroma = (shifted_chroma - np.mean(shifted_chroma)) / (np.std(shifted_chroma) + 1e-8)

            score_maj = np.corrcoef(shifted_chroma, major_profile)[0, 1]
            score_min = np.corrcoef(shifted_chroma, minor_profile)[0, 1]

            if score_maj > best_score:
                best_score = score_maj
                best_key = f"{NOTE_NAMES[i]} Majör"
            if score_min > best_score:
                best_score = score_min
                best_key = f"{NOTE_NAMES[i]} Minör"

        return best_key
    except Exception:
        return "Bilinmiyor"


def estimate_voice_gender(model_name: str) -> str:
    """
    Guesses the target model gender from model name. Returns 'female', 'male', or 'neutral'.
    """
    lower = model_name.lower()
    for kw in FEMALE_KEYWORDS:
        if kw in lower:
            return 'female'
    for kw in MALE_KEYWORDS:
        if kw in lower:
            return 'male'
    return 'neutral'


def analyze_pitch_and_recommend(
    audio_path: str,
    target_model_name: str = "",
    max_duration: float = 60.0
) -> dict:
    """
    Analyzes the source audio file and recommends the optimal pitch shift.
    
    Returns dict:
      - recommended_pitch (int)
      - detected_key (str)
      - avg_f0 (float)
      - source_gender_est (str)
      - target_gender_est (str)
      - explanation (str)
    """
    if not os.path.exists(audio_path):
        return {
            "recommended_pitch": 0,
            "detected_key": "Dosya Bulunamadı",
            "avg_f0": 0.0,
            "source_gender_est": "Bilinmiyor",
            "target_gender_est": "Bilinmiyor",
            "explanation": "Kaynak ses dosyası okunamadı."
        }

    try:
        # Load up to max_duration for rapid analysis (~10-30s is more than enough)
        y, sr = librosa.load(audio_path, sr=22050, mono=True, duration=max_duration)
        if len(y) == 0:
            return {"recommended_pitch": 0, "detected_key": "-", "avg_f0": 0.0, "explanation": "Ses boş."}

        # 1. Detect Musical Key
        key = detect_musical_key(y, sr)

        # 2. Estimate F0 (Fundamental Frequency)
        # Using pyin or piptrack
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr,
            frame_length=2048
        )
        valid_f0 = f0[voiced_flag & ~np.isnan(f0)]

        if len(valid_f0) > 0:
            avg_f0 = float(np.median(valid_f0))
        else:
            avg_f0 = 175.0  # fallback neutral

        # Determine source vocal range
        if avg_f0 > 175.0:
            source_gender = "Kadın (Tiz / Soprano-Alto)"
            src_cat = "female"
        else:
            source_gender = "Erkek (Pes / Tenor-Bariton)"
            src_cat = "male"

        # Determine target model gender
        target_cat = estimate_voice_gender(target_model_name)
        if target_cat == 'female':
            target_gender = "Kadın Model"
            target_typical_f0 = 220.0
        elif target_cat == 'male':
            target_gender = "Erkek Model"
            target_typical_f0 = 125.0
        else:
            target_gender = "Nötr / Belirsiz"
            target_typical_f0 = avg_f0  # keep original

        # Calculate ideal semitones shift
        if target_cat != 'neutral':
            semitones = 12.0 * math.log2(target_typical_f0 / max(avg_f0, 50.0))
        else:
            semitones = 0.0

        # Intelligent quantize to standard vocal pitch shifts (-12, 0, +12) or closest semitone
        if src_cat == 'male' and target_cat == 'female':
            # Male to Female generally needs +12 (1 octave) or +10
            recommended = 12
            reason = "Erkek vokalinden kadın ses modeline geçiş için standart +12 (1 oktav tizleştirme) önerildi."
        elif src_cat == 'female' and target_cat == 'male':
            # Female to Male generally needs -12 (1 octave) or -10
            recommended = -12
            reason = "Kadın vokalinden erkek ses modeline geçiş için standart -12 (1 oktav pesleştirme) önerildi."
        else:
            # Same gender or neutral: small pitch correction or 0
            rounded = int(round(semitones))
            if abs(rounded) <= 2:
                recommended = 0
                reason = "Kaynak vokal ve hedef model aynı ses karakterinde; orijinal ton (0) korundu."
            elif rounded > 6:
                recommended = 12
                reason = f"Tizleştirme farkı tespit edildi ({avg_f0:.1f}Hz ➔ {target_typical_f0:.1f}Hz); +12 oktav önerildi."
            elif rounded < -6:
                recommended = -12
                reason = f"Pesleştirme farkı tespit edildi ({avg_f0:.1f}Hz ➔ {target_typical_f0:.1f}Hz); -12 oktav önerildi."
            else:
                recommended = rounded
                reason = f"Hassas ton dengesi için {recommended:+d} yarım ses (semitone) ayarı önerildi."

        return {
            "recommended_pitch": recommended,
            "detected_key": key,
            "avg_f0": round(avg_f0, 1),
            "source_gender_est": source_gender,
            "target_gender_est": target_gender,
            "explanation": reason
        }

    except Exception as e:
        return {
            "recommended_pitch": 0,
            "detected_key": "Hata",
            "avg_f0": 0.0,
            "source_gender_est": "-",
            "target_gender_est": "-",
            "explanation": f"Pitch analizi sırasında hata: {e}"
        }
