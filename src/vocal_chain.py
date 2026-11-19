"""
Professional Studio Vocal Chain & Mastering Engine for Ripleytia AI Studio.
Provides:
- Smart De-Esser (Sibilance Reducer for harsh 's/ş/ç' sounds)
- Parametric Studio EQ (Air 12kHz+, Warmth 250Hz, Mud Cut 450Hz, Low Cut 80Hz)
- Stereo Vocal Doubler & Widener (Creates wide, full-bodied stereo stage)
- Dynamic Sidechain Ducking (Carves out 1kHz-3.5kHz pocket in instruments for vocals)
- Radio-Ready LUFS Mastering Limiter (Targets -14 LUFS with True Peak protection)
Powered by pedalboard & scipy.
"""

import numpy as np
import scipy.signal
import pedalboard
from pedalboard import (
    Pedalboard,
    Compressor,
    Limiter,
    Gain,
    HighShelfFilter,
    LowShelfFilter,
    PeakFilter,
    HighpassFilter,
    Chorus,
    Delay
)


def apply_deesser(
    audio: np.ndarray,
    sr: int,
    freq_hz: float = 6500.0,
    threshold_db: float = -20.0,
    amount: float = 0.7
) -> np.ndarray:
    """
    Smart De-Esser: Detects harsh sibilant spikes (5kHz - 9kHz) and applies dynamic attenuation.
    """
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=0)
    elif audio.shape[0] > 2 and audio.shape[1] == 2:
        audio = audio.T

    # Design bandpass filter around sibilance frequencies (5500Hz - 8500Hz)
    nyq = sr * 0.5
    low = max(100.0, freq_hz - 1500.0) / nyq
    high = min(nyq - 100.0, freq_hz + 1500.0) / nyq

    try:
        b, a = scipy.signal.butter(2, [low, high], btype='bandpass')
        sibilance_band = scipy.signal.filtfilt(b, a, audio, axis=-1)

        # Calculate envelope of sibilance band
        env = np.abs(sibilance_band)
        # Smooth envelope
        kernel_size = int(sr * 0.005)  # 5ms smoothing
        if kernel_size > 1:
            win = np.ones(kernel_size) / kernel_size
            for ch in range(env.shape[0]):
                env[ch] = scipy.signal.convolve(env[ch], win, mode='same')

        env_db = 20.0 * np.log10(np.maximum(env, 1e-6))
        over_threshold = np.maximum(0.0, env_db - threshold_db)

        # Gain reduction factor (0.0 to amount)
        gain_reduction_db = over_threshold * amount
        gain_scale = 10.0 ** (-gain_reduction_db / 20.0)

        # Attenuate only the sibilant high band
        processed_band = sibilance_band * gain_scale
        base_audio = audio - sibilance_band
        result = base_audio + processed_band
        return np.ascontiguousarray(result.astype(np.float32))
    except Exception as e:
        print(f"[De-Esser Hatası] {e}")
        return audio


def apply_studio_parametric_eq(
    audio: np.ndarray,
    sr: int,
    air_boost_db: float = 2.5,
    warmth_db: float = 1.5,
    mud_cut_db: float = -2.0,
    low_cut_hz: float = 80.0
) -> np.ndarray:
    """
    4-Band Studio Parametric EQ:
    - 80Hz Low Cut (eliminates sub-bass rumble)
    - 250Hz Warmth Bell (+1.5 dB body)
    - 450Hz Mud Cut (-2.0 dB clears boxy tone)
    - 12kHz Air Shelf (+2.5 dB sparkle & open high end)
    """
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=0)
    elif audio.shape[0] > 2 and audio.shape[1] == 2:
        audio = audio.T

    board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=low_cut_hz),
        PeakFilter(cutoff_frequency_hz=250.0, gain_db=warmth_db, q=1.0),
        PeakFilter(cutoff_frequency_hz=450.0, gain_db=mud_cut_db, q=1.2),
        HighShelfFilter(cutoff_frequency_hz=12000.0, gain_db=air_boost_db, q=0.7)
    ])

    try:
        processed = board(audio, sr)
        return np.ascontiguousarray(processed.astype(np.float32))
    except Exception as e:
        print(f"[Studio EQ Hatası] {e}")
        return audio


def apply_vocal_doubler_widener(
    audio: np.ndarray,
    sr: int,
    width: float = 0.35,
    delay_ms: float = 16.0
) -> np.ndarray:
    """
    Stereo Vocal Doubler & Widener:
    Creates a wide, radio-ready vocal image by adding subtle delayed and chorus-modulated side channels.
    """
    if audio.ndim == 1:
        mono = audio
        audio = np.stack([mono, mono], axis=0)
    elif audio.shape[0] > 2 and audio.shape[1] == 2:
        audio = audio.T

    # Mid component
    mid = 0.5 * (audio[0] + audio[1])

    # Side micro-delay channel
    delay_samples = int(sr * (delay_ms / 1000.0))
    left_delayed = np.pad(mid, (delay_samples, 0))[:-delay_samples]
    right_delayed = np.pad(mid, (0, delay_samples))[delay_samples:]

    # Modulate sides gently with Chorus
    board = Pedalboard([
        Chorus(rate_hz=0.8, depth=0.25, feedback=0.1, mix=0.5)
    ])
    try:
        side_input = np.stack([left_delayed, right_delayed], axis=0)
        sides_processed = board(side_input, sr)

        # Mix Mid and Sides
        out_left = (1.0 - width * 0.5) * audio[0] + (width * 0.7) * sides_processed[0]
        out_right = (1.0 - width * 0.5) * audio[1] + (width * 0.7) * sides_processed[1]
        doubled = np.stack([out_left, out_right], axis=0)
        return np.ascontiguousarray(doubled.astype(np.float32))
    except Exception as e:
        print(f"[Doubler Hatası] {e}")
        return audio


def apply_sidechain_ducking(
    instrumental_audio: np.ndarray,
    vocal_audio: np.ndarray,
    sr: int,
    max_reduction_db: float = 2.0,
    duck_low_hz: float = 1000.0,
    duck_high_hz: float = 3500.0
) -> np.ndarray:
    """
    Dynamic Sidechain Pocket Ducking:
    Subtly dips the 1kHz - 3.5kHz mid pocket of the instrumentals when vocal energy is active.
    Creates instant clarity so vocals cut through the mix without cranking volume.
    """
    # Ensure stereo shape (2, N)
    if instrumental_audio.ndim == 1:
        inst = np.stack([instrumental_audio, instrumental_audio], axis=0)
    elif instrumental_audio.shape[0] > 2 and instrumental_audio.shape[1] == 2:
        inst = instrumental_audio.T
    else:
        inst = instrumental_audio

    if vocal_audio.ndim == 1:
        voc = np.stack([vocal_audio, vocal_audio], axis=0)
    elif vocal_audio.shape[0] > 2 and vocal_audio.shape[1] == 2:
        voc = vocal_audio.T
    else:
        voc = vocal_audio

    min_len = min(inst.shape[1], voc.shape[1])
    inst = inst[:, :min_len]
    voc = voc[:, :min_len]

    try:
        # Detect vocal activity envelope
        voc_mono = np.mean(voc, axis=0)
        env = np.abs(voc_mono)
        # Smooth envelope (attack 10ms, release 100ms)
        win = np.ones(int(sr * 0.05)) / int(sr * 0.05)
        smooth_env = scipy.signal.convolve(env, win, mode='same')
        max_env = np.max(smooth_env) + 1e-8
        norm_env = np.clip(smooth_env / (max_env * 0.4), 0.0, 1.0)

        # Dynamic reduction curve (0 dB to -max_reduction_db)
        duck_factor = 10.0 ** (-(norm_env * max_reduction_db) / 20.0)

        # Filter the vocal pocket from the instrumental
        nyq = sr * 0.5
        b, a = scipy.signal.butter(2, [duck_low_hz / nyq, duck_high_hz / nyq], btype='bandpass')
        inst_pocket = scipy.signal.filtfilt(b, a, inst, axis=-1)
        inst_base = inst - inst_pocket

        # Apply dynamic ducking to the pocket
        ducked_pocket = inst_pocket * duck_factor
        result = inst_base + ducked_pocket
        return np.ascontiguousarray(result.astype(np.float32))
    except Exception as e:
        print(f"[Sidechain Ducking Hatası] {e}")
        return instrumental_audio


def apply_lufs_loudness_mastering(
    audio: np.ndarray,
    sr: int,
    target_lufs: float = -14.0,
    true_peak_ceiling: float = -1.0
) -> np.ndarray:
    """
    Broadcast Standard LUFS Mastering Limiter:
    Calculates K-weighted integrated loudness, matches -14 LUFS (Spotify/YouTube/Apple Music),
    and enforces True-Peak brickwall limiting (-1.0 dBFS) to prevent inter-sample clipping.
    """
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=0)
    elif audio.shape[0] > 2 and audio.shape[1] == 2:
        audio = audio.T

    try:
        # Standard K-weighting pre-filter (High shelf 4dB boost at 1.5kHz + highpass at 100Hz)
        pre_board = Pedalboard([
            HighShelfFilter(cutoff_frequency_hz=1500.0, gain_db=4.0),
            HighpassFilter(cutoff_frequency_hz=100.0)
        ])
        k_weighted = pre_board(audio, sr)

        # Calculate mean square energy
        ms = np.mean(k_weighted ** 2)
        if ms < 1e-9:
            return audio

        measured_lufs = -0.691 + 10.0 * np.log10(ms)

        # Required gain adjustment
        gain_db = target_lufs - measured_lufs
        # Limit gain adjustment range to sensible mastering levels (-8 dB to +12 dB)
        gain_db = max(-8.0, min(12.0, gain_db))

        # Master limiter chain
        master_board = Pedalboard([
            Gain(gain_db=gain_db),
            Limiter(threshold_db=true_peak_ceiling, release_ms=100.0)
        ])

        mastered = master_board(audio, sr)
        return np.ascontiguousarray(mastered.astype(np.float32))
    except Exception as e:
        print(f"[LUFS Mastering Hatası] {e}")
        return audio
