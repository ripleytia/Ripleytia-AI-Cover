"""
Audio Comparator & SDR Evaluation Engine for Ripleytia AI Studio.
Calculates Signal-to-Distortion Ratio (SDR), spectral similarity, and renders side-by-side waveform comparisons.
"""

import os
import numpy as np
import soundfile as sf
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


COMP_OUTPUT_IMG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "song_output",
    "audio_comparison.png"
)


def compare_audio_tracks(ref_path: str, target_path: str, max_duration: float = 60.0) -> dict:
    """
    Compares reference audio with target audio, calculating SDR, correlation, and comparison image.
    """
    if not os.path.exists(ref_path) or not os.path.exists(target_path):
        raise FileNotFoundError("Karşılaştırılacak dosyalardan biri bulunamadı.")

    y_ref, sr = librosa.load(ref_path, sr=22050, mono=True, duration=max_duration)
    y_tgt, _ = librosa.load(target_path, sr=22050, mono=True, duration=max_duration)

    # Align lengths
    min_len = min(len(y_ref), len(y_tgt))
    if min_len == 0:
        raise ValueError("Ses dosyası verisi yetersiz.")

    y_ref = y_ref[:min_len]
    y_tgt = y_tgt[:min_len]

    # Normalize amplitudes
    ref_norm = y_ref / (np.max(np.abs(y_ref)) + 1e-8)
    tgt_norm = y_tgt / (np.max(np.abs(y_tgt)) + 1e-8)

    # 1. Pearson Correlation
    corr = float(np.corrcoef(ref_norm, tgt_norm)[0, 1])
    sim_pct = max(0.0, min(100.0, (corr + 1.0) / 2.0 * 100.0))

    # 2. SDR Estimation (Signal-to-Distortion Ratio)
    error = ref_norm - tgt_norm
    sig_power = np.sum(ref_norm ** 2)
    err_power = np.sum(error ** 2) + 1e-8
    sdr_db = float(10.0 * np.log10(sig_power / err_power))
    sdr_db = max(-10.0, min(35.0, sdr_db))

    # 3. Spectral Similarity (Mel Spectrogram distance)
    S_ref = librosa.feature.melspectrogram(y=y_ref, sr=sr, n_mels=64)
    S_tgt = librosa.feature.melspectrogram(y=y_tgt, sr=sr, n_mels=64)
    spec_diff = np.mean(np.abs(librosa.power_to_db(S_ref) - librosa.power_to_db(S_tgt)))
    fidelity_score = max(50.0, min(99.0, 100.0 - (spec_diff * 1.5)))

    # 4. Generate Side-by-Side Waveform Comparison Image
    os.makedirs(os.path.dirname(COMP_OUTPUT_IMG), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 3.8), sharex=True)
    fig.patch.set_facecolor('#12121A')

    t = np.linspace(0, min_len / sr, min_len)

    ax1.set_facecolor('#181824')
    ax1.plot(t, y_ref, color='#A390E4', lw=0.6, alpha=0.9)
    ax1.set_title("Orijinal Referans Ses", color='#E0E0FF', fontsize=10, pad=4)
    ax1.set_ylabel("Genlik", color='#A390E4', fontsize=8)
    ax1.tick_params(colors='#8888AA', labelsize=8)
    ax1.grid(True, color='#2A2A3E', linestyle='--', alpha=0.5)

    ax2.set_facecolor('#181824')
    ax2.plot(t, y_tgt, color='#00d2d3', lw=0.6, alpha=0.9)
    ax2.set_title("Üretilen Yapay Zeka Sesi", color='#00d2d3', fontsize=10, pad=4)
    ax2.set_xlabel("Zaman (sn)", color='#A390E4', fontsize=8)
    ax2.set_ylabel("Genlik", color='#00d2d3', fontsize=8)
    ax2.tick_params(colors='#8888AA', labelsize=8)
    ax2.grid(True, color='#2A2A3E', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(COMP_OUTPUT_IMG, dpi=120, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)

    return {
        "sdr_db": round(sdr_db, 1),
        "similarity_pct": round(sim_pct, 1),
        "fidelity_score": round(fidelity_score, 1),
        "plot_path": COMP_OUTPUT_IMG
    }
