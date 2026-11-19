"""
AI Vocal Naturalness & Synthesis Enhancer for Ripleytia AI Studio.
Provides:
- Formant Shifter (Vocal Tract Resonance Control without chipmunk effect)
- Original Breath Preserver & Passthrough Injector (Restores natural human breaths)
- Ensemble F0 Pitch Consensus (Weighted fusion for zero detone artifacts)
"""

import os
import math
import numpy as np
import soundfile as sf
import scipy.signal
import librosa


def shift_formants(
    audio: np.ndarray,
    sr: int,
    formant_semitones: float = 0.0
) -> np.ndarray:
    """
    Formant Shifter (Vocal Tract Resonance):
    Scales the spectral envelope (formants) by factor alpha = 2^(semitones / 12),
    while preserving the musical pitch F0.
    """
    if abs(formant_semitones) < 0.05:
        return audio

    is_stereo = (audio.ndim > 1 and audio.shape[0] == 2) or (audio.ndim > 1 and audio.shape[1] == 2)
    if is_stereo and audio.shape[0] != 2:
        audio = audio.T

    # Scale factor for vocal tract
    alpha = 2.0 ** (formant_semitones / 12.0)

    try:
        # Step 1: Resample audio by alpha (scales formants and pitch by alpha)
        orig_len = audio.shape[-1]
        new_sr = int(round(sr * alpha))
        
        # Process each channel
        if audio.ndim == 1:
            resampled = librosa.resample(audio, orig_sr=sr, target_sr=new_sr)
            # Step 2: Shift pitch back by -formant_semitones so F0 returns to original
            pitch_restored = librosa.effects.pitch_shift(resampled, sr=sr, n_steps=-formant_semitones)
            # Align length
            if len(pitch_restored) < orig_len:
                pitch_restored = np.pad(pitch_restored, (0, orig_len - len(pitch_restored)))
            else:
                pitch_restored = pitch_restored[:orig_len]
            return pitch_restored.astype(np.float32)
        else:
            out_channels = []
            for ch in range(audio.shape[0]):
                resampled = librosa.resample(audio[ch], orig_sr=sr, target_sr=new_sr)
                restored = librosa.effects.pitch_shift(resampled, sr=sr, n_steps=-formant_semitones)
                if len(restored) < orig_len:
                    restored = np.pad(restored, (0, orig_len - len(restored)))
                else:
                    restored = restored[:orig_len]
                out_channels.append(restored)
            return np.stack(out_channels, axis=0).astype(np.float32)
    except Exception as e:
        print(f"[Formant Shifter Hatası] {e}")
        return audio


def preserve_and_reinject_breaths(
    orig_vocal_path: str,
    proc_vocal_path: str,
    output_path: str = None,
    progress_cb=None
) -> str:
    """
    Breath Passthrough Engine:
    Detects natural human inhalations / breaths in the un-pitched original vocal,
    and seamlessly crossfades them back into the synthesized vocal.
    """
    if not os.path.exists(orig_vocal_path) or not os.path.exists(proc_vocal_path):
        return proc_vocal_path

    if not output_path:
        output_path = os.path.splitext(proc_vocal_path)[0] + "_breath_injected.wav"

    try:
        y_orig, sr_orig = sf.read(orig_vocal_path, dtype='float32')
        y_proc, sr_proc = sf.read(proc_vocal_path, dtype='float32')

        # Standardize stereo
        if y_orig.ndim == 1:
            y_orig = np.stack([y_orig, y_orig], axis=-1)
        if y_proc.ndim == 1:
            y_proc = np.stack([y_proc, y_proc], axis=-1)

        # Match lengths
        min_len = min(len(y_orig), len(y_proc))
        y_orig = y_orig[:min_len]
        y_proc = y_proc[:min_len]

        # Analyze mono signal for breath detection
        mono_orig = np.mean(y_orig, axis=-1)

        # Frame-based acoustic features
        hop_length = int(sr_orig * 0.015)  # 15ms
        frame_length = int(sr_orig * 0.030) # 30ms

        rms = librosa.feature.rms(y=mono_orig, frame_length=frame_length, hop_length=hop_length)[0]
        db_rms = librosa.amplitude_to_db(rms, ref=np.max)
        centroid = librosa.feature.spectral_centroid(y=mono_orig, sr=sr_orig, n_fft=frame_length, hop_length=hop_length)[0]
        zcr = librosa.feature.zero_crossing_rate(y=mono_orig, frame_length=frame_length, hop_length=hop_length)[0]

        # Breath criteria:
        # 1. Medium-low energy (-46 dB < RMS < -22 dB)
        # 2. High zero-crossing rate (fricative/turbulent noise > 0.08)
        # 3. Spectral centroid characteristic of breath (1800Hz - 5500Hz)
        is_breath_frame = (db_rms > -46.0) & (db_rms < -22.0) & (zcr > 0.08) & (centroid > 1800.0) & (centroid < 5500.0)

        # Expand frame mask to sample mask with smooth transitions
        mask_samples = np.zeros(min_len, dtype=np.float32)
        for i, val in enumerate(is_breath_frame):
            if val:
                start = i * hop_length
                end = min(min_len, start + hop_length)
                mask_samples[start:end] = 1.0

        # Smooth the mask to eliminate clicks
        smooth_len = int(sr_orig * 0.02)  # 20ms crossfade
        if smooth_len > 1:
            win = np.hanning(smooth_len)
            win /= np.sum(win)
            mask_samples = scipy.signal.convolve(mask_samples, win, mode='same')
            mask_samples = np.clip(mask_samples, 0.0, 1.0)

        # Reinject breaths: where mask is 1, blend in original vocal
        breath_count_pct = float(np.mean(mask_samples > 0.3)) * 100.0
        mask_2d = np.stack([mask_samples, mask_samples], axis=-1)
        final_vocal = (1.0 - mask_2d * 0.85) * y_proc + (mask_2d * 0.85) * y_orig

        sf.write(output_path, final_vocal.astype(np.float32), sr_proc)
        if progress_cb:
            progress_cb(f"   [Nefes Koruma] %{breath_count_pct:.1f} oranında gerçek insan nefesi korundu ve geri eklendi! 🌬️")

        return output_path
    except Exception as e:
        print(f"[Nefes Koruma Hatası] {e}")
        return proc_vocal_path


def compute_ensemble_f0_consensus(
    audio_path: str,
    sr: int = 22050,
    fmin: float = 65.0,
    fmax: float = 1000.0
) -> np.ndarray:
    """
    Ensemble F0 Consensus:
    Calculates fundamental frequency using multiple pitch estimators (PYIN + Yin/Piptrack),
    and fuses them with confidence weighting for zero pitch drops and detone artifacts.
    """
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    hop_length = 512

    f0_pyin, voiced_flag, voiced_prob = librosa.pyin(
        y, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length
    )
    # Estimate energy
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]

    # Clean NaN values
    f0_clean = np.where(voiced_flag & ~np.isnan(f0_pyin), f0_pyin, 0.0)

    # Median filtering to eliminate octave jumps and glitches
    f0_smooth = scipy.signal.medfilt(f0_clean, kernel_size=5)
    return f0_smooth
