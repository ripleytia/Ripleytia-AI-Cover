"""
Vocal Analyzer & Visualization Engine for Ripleytia AI Studio.
Calculates pitch range, dynamic range, silence ratio, SNR, and renders vocal analysis plots.
"""

import os
import math
import numpy as np
import soundfile as sf
import librosa
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt


ANALYSIS_OUTPUT_IMG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "song_output",
    "vocal_analysis.png"
)


def hz_to_note_name(hz: float) -> str:
    if hz <= 0:
        return "-"
    try:
        midi = librosa.hz_to_midi(hz)
        return librosa.midi_to_note(int(round(midi)))
    except Exception:
        return f"{hz:.1f}Hz"


def analyze_vocal_file(audio_path: str, max_duration: float = 90.0) -> dict:
    """
    Performs comprehensive vocal acoustics analysis and creates a plot image.
    Returns dictionary with metrics and plot path.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError("Ses dosyası bulunamadı.")

    y, sr = librosa.load(audio_path, sr=22050, mono=True, duration=max_duration)
    if len(y) == 0:
        raise ValueError("Ses dosyası boş.")

    # 1. Pitch / F0 Analysis
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=sr,
        frame_length=2048
    )
    valid_f0 = f0[voiced_flag & ~np.isnan(f0)]

    if len(valid_f0) > 0:
        min_hz = float(np.min(valid_f0))
        max_hz = float(np.max(valid_f0))
        avg_hz = float(np.median(valid_f0))
        min_note = hz_to_note_name(min_hz)
        max_note = hz_to_note_name(max_hz)
        avg_note = hz_to_note_name(avg_hz)
    else:
        min_hz = max_hz = avg_hz = 0.0
        min_note = max_note = avg_note = "-"

    # 2. Dynamic Range & Energy (RMS)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    peak = float(np.max(np.abs(y)))
    avg_rms = float(np.mean(rms))
    
    # Silence vs Active singing ratio (threshold: -42 dB)
    db_rms = librosa.amplitude_to_db(rms, ref=np.max)
    active_mask = db_rms > -42.0
    singing_pct = float(np.mean(active_mask)) * 100.0
    silence_pct = 100.0 - singing_pct

    # Dynamic Range in dB
    if avg_rms > 1e-6:
        dynamic_range_db = float(20 * np.log10(max(peak, 1e-4) / avg_rms))
    else:
        dynamic_range_db = 0.0

    # 3. Spectral Centroid (Vocal Brightness)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    avg_centroid = float(np.mean(centroid))
    if avg_centroid > 2400:
        timbre_desc = "Parlak & Tiz (Bright / Treble)"
    elif avg_centroid < 1400:
        timbre_desc = "Koyu & Sıcak (Warm / Bass)"
    else:
        timbre_desc = "Dengeli (Balanced / Natural)"

    # 4. Estimated SNR (Signal to Noise Ratio)
    noise_part = db_rms[~active_mask]
    signal_part = db_rms[active_mask]
    if len(noise_part) > 0 and len(signal_part) > 0:
        snr_est = float(np.mean(signal_part) - np.mean(noise_part))
    else:
        snr_est = 35.0

    # 5. Render Gothic-themed Matplotlib Plot
    os.makedirs(os.path.dirname(ANALYSIS_OUTPUT_IMG), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
    fig.patch.set_facecolor('#12121A')

    times = librosa.times_like(f0, sr=sr, hop_length=512)
    
    # Subplot 1: Waveform amplitude
    ax1.set_facecolor('#181824')
    ax1.plot(np.linspace(0, len(y)/sr, len(y)), y, color='#7B68EE', alpha=0.7, lw=0.6)
    ax1.set_ylabel('Genlik', color='#A390E4', fontsize=9)
    ax1.tick_params(colors='#8888AA', labelsize=8)
    ax1.grid(True, color='#2A2A3E', linestyle='--', alpha=0.5)
    ax1.set_title(f"Dalga Formu & Ses Dinamikleri (Aktif Vokal: %{singing_pct:.1f})", color='#E0E0FF', fontsize=10, pad=6)

    # Subplot 2: Pitch Tracking Contour
    ax2.set_facecolor('#181824')
    ax2.plot(times, f0, color='#00d2d3', lw=1.5, label='F0 Pitch (Hz)')
    ax2.set_ylabel('Pitch (Hz)', color='#00d2d3', fontsize=9)
    ax2.set_xlabel('Zaman (Saniye)', color='#A390E4', fontsize=9)
    ax2.tick_params(colors='#8888AA', labelsize=8)
    ax2.grid(True, color='#2A2A3E', linestyle='--', alpha=0.5)
    ax2.set_ylim(max(40, min_hz - 20), min(900, max_hz + 40))

    plt.tight_layout()
    plt.savefig(ANALYSIS_OUTPUT_IMG, dpi=120, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)

    return {
        "min_note": min_note,
        "max_note": max_note,
        "avg_note": avg_note,
        "avg_f0_hz": round(avg_hz, 1),
        "min_f0_hz": round(min_hz, 1),
        "max_f0_hz": round(max_hz, 1),
        "singing_pct": round(singing_pct, 1),
        "silence_pct": round(silence_pct, 1),
        "dynamic_range_db": round(dynamic_range_db, 1),
        "snr_db": round(snr_est, 1),
        "timbre_desc": timbre_desc,
        "plot_path": ANALYSIS_OUTPUT_IMG
    }
