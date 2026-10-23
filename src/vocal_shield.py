import os
import numpy as np
import soundfile as sf

def apply_vocal_rescue_shield(
    ref_path: str,
    proc_path: str,
    output_path: str = None,
    min_energy_ratio: float = 0.40,
    silence_threshold_db: float = -48.0,
    progress_cb=None
) -> str:
    """
    Intelligent Vocal Rescue & Ghosting Fallback Shield:
    Compares the processed vocal (after Karaoke/De-Reverb) against the reference Stage-1
    isolated vocal (Becruily Mel-Band Roformer). If the processed vocal dropped out,
    lost body, or had syllables/words swallowed by over-aggressive filtering,
    this shield seamlessly restores and blends them from the reference vocal without
    any clicks, phase distortion, or pops.
    """
    if not os.path.exists(ref_path) or not os.path.exists(proc_path):
        return proc_path

    if output_path is None:
        output_path = proc_path

    try:
        ref_data, ref_sr = sf.read(ref_path, dtype='float32')
        proc_data, proc_sr = sf.read(proc_path, dtype='float32')
    except Exception as e:
        if progress_cb:
            progress_cb(f"[Vocal Shield Uyarısı] Ses okunamadı, işlem atlandı: {e}")
        return proc_path

    # Ensure equal sample rates
    if ref_sr != proc_sr:
        # Match sample rates if necessary
        try:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(ref_sr, proc_sr)
            up = ref_sr // g
            down = proc_sr // g
            proc_data = resample_poly(proc_data, up, down, axis=0)
            proc_sr = ref_sr
        except Exception:
            pass

    # Ensure matching channels
    if ref_data.ndim == 1 and proc_data.ndim == 2:
        ref_data = np.stack([ref_data, ref_data], axis=-1)
    elif ref_data.ndim == 2 and proc_data.ndim == 1:
        proc_data = np.stack([proc_data, proc_data], axis=-1)

    min_len = min(len(ref_data), len(proc_data))
    if min_len < 1000:
        return proc_path

    ref_aligned = ref_data[:min_len]
    proc_aligned = proc_data[:min_len]

    # Mono representation for energy envelope analysis
    is_2d = ref_aligned.ndim > 1
    ref_mono = np.mean(ref_aligned, axis=1) if is_2d else ref_aligned
    proc_mono = np.mean(proc_aligned, axis=1) if is_2d else proc_aligned

    # Analysis window parameters: 50ms window with 25ms hop
    win_size = int(ref_sr * 0.050)
    hop_size = int(ref_sr * 0.025)
    silence_thres = 10.0 ** (silence_threshold_db / 20.0)

    num_frames = max(1, (min_len - win_size) // hop_size + 1)
    frame_centers = []
    w_ref = []

    # Thresholds: below dropout_ratio, 100% restored. Above keep_ratio, 100% processed kept.
    dropout_ratio = 0.15
    keep_ratio = float(min_energy_ratio)

    for i in range(num_frames):
        s = i * hop_size
        e = s + win_size
        frame_centers.append(s + win_size // 2)

        r_rms = float(np.sqrt(np.mean(ref_mono[s:e] ** 2)))
        p_rms = float(np.sqrt(np.mean(proc_mono[s:e] ** 2)))

        if r_rms > silence_thres:
            ratio = p_rms / (r_rms + 1e-7)
            if ratio < dropout_ratio:
                # Severe dropout/eaten word -> 100% restore from ref
                w = 1.0
            elif ratio < keep_ratio:
                # Partial thinning / muffling -> smooth cosine/linear crossfade
                w = (keep_ratio - ratio) / (keep_ratio - dropout_ratio)
            else:
                # Vocal is healthy and punchy -> keep processed de-reverbed/isolated
                w = 0.0
        else:
            w = 0.0

        w_ref.append(w)

    frame_centers = np.array(frame_centers)
    w_ref = np.array(w_ref)

    if len(frame_centers) < 2:
        return proc_path

    # Sample-accurate linear interpolation of gain envelope
    w_samples = np.interp(np.arange(min_len), frame_centers, w_ref)
    w_samples = np.clip(w_samples, 0.0, 1.0)

    # 5% subtle baseline body blend: preserves natural harmonic presence
    w_samples = np.maximum(w_samples, 0.05)

    if is_2d:
        w_samples = w_samples[:, np.newaxis]

    # Reconstruct protected audio
    rescued_audio = (1.0 - w_samples) * proc_aligned + w_samples * ref_aligned

    # Prevent accidental digital clipping
    max_peak = float(np.max(np.abs(rescued_audio)))
    if max_peak > 0.999:
        rescued_audio = rescued_audio * (0.999 / max_peak)

    # Calculate rescued percentage for feedback
    rescued_pct = float(np.mean(w_samples > 0.15)) * 100.0
    if progress_cb:
        if rescued_pct > 0.5:
            progress_cb(f"   [Vocal Shield] %{rescued_pct:.1f} oranında yutulmuş vokal/hece başarıyla kurtarıldı ve onarıldı!")
        else:
            progress_cb("   [Vocal Shield] Vokal dengeli ve temiz; herhangi bir kayıp/yutulma tespit edilmedi.")

    # Save output
    tmp_out = output_path + ".tmp.wav"
    sf.write(tmp_out, rescued_audio, ref_sr)
    if os.path.exists(output_path):
        os.remove(output_path)
    os.replace(tmp_out, output_path)

    return output_path
