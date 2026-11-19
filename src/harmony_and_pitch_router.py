"""
AI Harmony & Multi-Segment Dynamic Pitch Engine for Ripleytia AI Studio.
Provides:
- AI Backing Vocal & Harmony Generator (Creates +3rd, +5th, octave harmonies with stereo spread)
- Verse vs. Chorus Dynamic Segment Pitch Router (Applies distinct pitch shifts to verse and chorus)
"""

import re
import math
import numpy as np
import scipy.signal
import librosa
import soundfile as sf
import pedalboard
from pedalboard import Pedalboard, HighpassFilter, LowpassFilter, Delay, Chorus


def generate_backing_harmonies(
    lead_vocal: np.ndarray,
    sr: int,
    harmony_mode: str = "3rd_and_5th",
    harmony_mix: float = 0.35
) -> np.ndarray:
    """
    AI Harmony Generator:
    Generates rich harmonic vocal backing tracks:
    - '3rd_and_5th': Major 3rd (+4 st) and Perfect 5th (+7 st)
    - 'octave_down': Full body octave down (-12 st)
    - 'trio_choir': +4 st, +7 st, -12 st
    """
    if harmony_mix <= 0.01:
        return lead_vocal

    # Ensure shape (2, N)
    if lead_vocal.ndim == 1:
        stereo_lead = np.stack([lead_vocal, lead_vocal], axis=0)
    elif lead_vocal.shape[0] > 2 and lead_vocal.shape[1] == 2:
        stereo_lead = lead_vocal.T
    else:
        stereo_lead = lead_vocal

    mono_lead = np.mean(stereo_lead, axis=0)
    orig_len = len(mono_lead)

    try:
        harmony_tracks = []

        if harmony_mode in ("3rd_and_5th", "trio_choir"):
            # +4 semitones (Major Third)
            third = librosa.effects.pitch_shift(mono_lead, sr=sr, n_steps=4.0)
            harmony_tracks.append((third[:orig_len], -0.6))  # Pan slightly left

            # +7 semitones (Perfect Fifth)
            fifth = librosa.effects.pitch_shift(mono_lead, sr=sr, n_steps=7.0)
            harmony_tracks.append((fifth[:orig_len], 0.6))   # Pan slightly right

        if harmony_mode in ("octave_down", "trio_choir"):
            # -12 semitones (Octave down deep backing)
            oct_down = librosa.effects.pitch_shift(mono_lead, sr=sr, n_steps=-12.0)
            harmony_tracks.append((oct_down[:orig_len], 0.0)) # Center

        # Mix harmonies and apply stereo spread + filtering
        harm_left = np.zeros(orig_len, dtype=np.float32)
        harm_right = np.zeros(orig_len, dtype=np.float32)

        for track, pan in harmony_tracks:
            # Pan law
            left_gain = math.cos((pan + 1.0) * math.pi / 4.0)
            right_gain = math.sin((pan + 1.0) * math.pi / 4.0)
            harm_left += track * left_gain
            harm_right += track * right_gain

        harm_stereo = np.stack([harm_left, harm_right], axis=0)

        # Tuck harmonies behind the lead with Highpass and Lowpass filters + subtle reverb/delay
        board = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=160.0),
            LowpassFilter(cutoff_frequency_hz=7000.0),
            Chorus(rate_hz=1.0, depth=0.3, mix=0.4),
            Delay(delay_seconds=0.02, feedback=0.1, mix=0.3)
        ])
        polished_harmonies = board(harm_stereo, sr)

        # Blend with lead vocal
        mixed = stereo_lead + (polished_harmonies * harmony_mix)
        return np.ascontiguousarray(mixed.astype(np.float32))

    except Exception as e:
        print(f"[Armoni Üretici Hatası] {e}")
        return lead_vocal


def time_str_to_seconds(t_str: str) -> float:
    parts = t_str.strip().split(":")
    if len(parts) == 2:
        return float(parts[0]) * 60 + float(parts[1])
    elif len(parts) == 3:
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    return float(parts[0])


def apply_verse_chorus_dynamic_pitch(
    audio_path: str,
    rules_str: str,
    output_path: str = None,
    sr: int = 44100
) -> str:
    """
    Applies dynamic pitch shifts based on user segment rules:
    Example string: '00:00-01:15: 0; 01:15-02:00: -2; 02:00-03:30: +1'
    """
    if not os.path.exists(audio_path) or not rules_str.strip():
        return audio_path

    if not output_path:
        output_path = os.path.splitext(audio_path)[0] + "_dynamic_pitch.wav"

    try:
        y, orig_sr = sf.read(audio_path, dtype='float32')
        is_stereo = (y.ndim > 1 and y.shape[1] == 2)
        total_len = len(y)

        # Parse rules
        # Format: START-END: SHIFT
        rule_pattern = re.compile(r"(\d+:\d+|\d+)-(\d+:\d+|\d+)\s*:\s*([+-]?\d+)")
        matches = rule_pattern.findall(rules_str)
        if not matches:
            return audio_path

        out_audio = np.copy(y)
        crossfade_samples = int(orig_sr * 0.05)  # 50ms smooth transition

        for start_s, end_s, shift_s in matches:
            t0 = time_str_to_seconds(start_s)
            t1 = time_str_to_seconds(end_s)
            shift_val = int(shift_s)

            if shift_val == 0:
                continue

            idx0 = max(0, int(t0 * orig_sr))
            idx1 = min(total_len, int(t1 * orig_sr))
            if idx0 >= idx1:
                continue

            segment = y[idx0:idx1]
            # Shift segment
            if is_stereo:
                shifted_seg = np.stack([
                    librosa.effects.pitch_shift(segment[:, 0], sr=orig_sr, n_steps=shift_val),
                    librosa.effects.pitch_shift(segment[:, 1], sr=orig_sr, n_steps=shift_val)
                ], axis=-1)
            else:
                shifted_seg = librosa.effects.pitch_shift(segment, sr=orig_sr, n_steps=shift_val)

            # Smooth splice
            seg_len = min(len(segment), len(shifted_seg))
            out_audio[idx0 : idx0 + seg_len] = shifted_seg[:seg_len]

        sf.write(output_path, out_audio.astype(np.float32), orig_sr)
        return output_path
    except Exception as e:
        print(f"[Dinamik Pitch Hatası] {e}")
        return audio_path
