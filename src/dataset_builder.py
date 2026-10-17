import os
import sys
import re
import gc
import json
import time
import shutil
import subprocess
import unicodedata
import numpy as np
import soundfile as sf
import librosa
import yt_dlp
from pedalboard import Pedalboard, HighpassFilter, Compressor, NoiseGate

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS_ROOT = os.path.join(BASE_DIR, "datasets")
APPLIO_DIR = r"F:\Yapay_Zeka_Uygulamalari\Applio"
APPLIO_DATASETS_DIR = os.path.join(APPLIO_DIR, "assets", "datasets")
UVR5_MODELS_DIR = r"C:\Users\Ripleytia\AppData\Local\Programs\Ultimate Vocal Remover\models\MDX_Net_Models"
MDXNET_MODELS_DIR = os.path.join(BASE_DIR, "mdxnet_models")

os.makedirs(DATASETS_ROOT, exist_ok=True)


def sanitize_name(name: str) -> str:
    tr_map = str.maketrans("çÇğĞıİöÖşŞüÜ", "cCgGiIoOsSuU")
    name = name.translate(tr_map)
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    name = re.sub(r"[^a-zA-Z0-9_\-]", "_", name.strip())
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "Yeni_Model"


def format_duration(seconds: float) -> str:
    total_sec = int(round(seconds))
    mins, secs = divmod(total_sec, 60)
    hours, mins = divmod(mins, 60)
    if hours > 0:
        return f"{hours} sa {mins} dk {secs} sn"
    return f"{mins} dk {secs} sn"


def calculate_recommended_epochs(duration_seconds: float, preset: str = "Dengeli") -> dict:
    """
    Calculates optimal RVC training epochs & save frequency based on clean vocal dataset duration.
    """
    minutes = max(0.5, duration_seconds / 60.0)

    if minutes < 3:
        base_epochs = 350
        quality_note = "Kısa Dataset (<3 dk): Overfitting'i önlemek için 300-400 epoch idealdir."
    elif minutes <= 8:
        base_epochs = 300
        quality_note = "Orta-Kısa Dataset (3-8 dk): 250-350 epoch arası en temiz sonucu verir."
    elif minutes <= 15:
        base_epochs = 250
        quality_note = "İdeal Dataset (8-15 dk): 200-300 epoch stüdyo kalitesi için altın standarttır."
    elif minutes <= 30:
        base_epochs = 200
        quality_note = "Geniş Dataset (15-30 dk): Zengin veri sayesinde 180-250 epoch mükemmel detay sağlar."
    else:
        base_epochs = 160
        quality_note = "Dev Dataset (30+ dk): Çok geniş veri seti, 150-200 epoch fazlasıyla yeterlidir."

    if preset == "Hızlı Render":
        epochs = max(80, int(base_epochs * 0.55))
        save_every = 20
    elif preset == "Maksimum Kalite":
        epochs = int(base_epochs * 1.25)
        save_every = 10
    elif preset == "W-Okada (Canlı Konuşma)":
        epochs = int(base_epochs * 1.15)
        save_every = 10
    else:  # Dengeli
        epochs = base_epochs
        save_every = 10 if epochs <= 250 else 15

    # Round epochs to nearest 10
    epochs = int(round(epochs / 10.0) * 10)
    return {
        "recommended_epochs": epochs,
        "save_every_epoch": save_every,
        "duration_minutes": round(minutes, 2),
        "note": quality_note,
    }


def _ensure_mdxc_patched():
    from audio_separator.separator import Separator
    try:
        from audio_separator.separator.architectures.mdxc_separator import MDXCSeparator
        if not hasattr(MDXCSeparator, "_patched_by_ripleytia"):
            original_demix = MDXCSeparator.demix

            def patched_demix(self, mix):
                res = original_demix(self, mix)
                if isinstance(res, dict):
                    return res
                return {self.primary_stem_name: res, self.secondary_stem_name: res}

            MDXCSeparator.demix = patched_demix
            MDXCSeparator._patched_by_ripleytia = True
    except Exception:
        pass
    return Separator


def get_modern_ytdlp_cmd() -> list:
    """
    Returns command prefix for a modern yt-dlp (Python 3.12/3.14, 2026+) so YouTube downloads
    never fail due to AICoverGen's legacy Python 3.9 environment, without touching AICoverGen's venv.
    """
    candidates = [
        r"C:\Users\Ripleytia\AppData\Local\Programs\Python\Python314\Scripts\yt-dlp.exe",
        os.path.join(APPLIO_DIR, "env", "Scripts", "yt-dlp.exe"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return [c]

    applio_py = os.path.join(APPLIO_DIR, "env", "python.exe")
    if os.path.exists(applio_py):
        return [applio_py, "-m", "yt_dlp"]

    sys_ytdlp = shutil.which("yt-dlp")
    if sys_ytdlp and "AICoverGen\\venv" not in sys_ytdlp:
        return [sys_ytdlp]

    return [sys.executable, "-m", "yt_dlp"]


def download_youtube_audios(urls: list, raw_dir: str, progress_cb=None) -> list:
    """
    Downloads one or multiple YouTube URLs into raw_dir as WAV files using modern yt-dlp (2026+).
    """
    os.makedirs(raw_dir, exist_ok=True)
    downloaded_files = []
    ytdlp_base_cmd = get_modern_ytdlp_cmd()

    for idx, url in enumerate(urls, 1):
        url = url.strip()
        if not url:
            continue
        if progress_cb:
            progress_cb(f"[YouTube {idx}/{len(urls)}] İndiriliyor: {url}")

        out_template = os.path.join(raw_dir, f"yt_{idx:02d}_%(id)s.%(ext)s")
        cmd = ytdlp_base_cmd + [
            "--no-playlist",
            "--no-check-certificate",
            "--no-warnings",
            "-f",
            "bestaudio/best",
            "-x",
            "--audio-format",
            "wav",
            "-o",
            out_template,
            url,
        ]
        try:
            res = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            found_wav = None
            for f in sorted(os.listdir(raw_dir)):
                if f.startswith(f"yt_{idx:02d}_") and f.lower().endswith(".wav"):
                    found_wav = os.path.join(raw_dir, f)
                    break

            if found_wav and os.path.exists(found_wav):
                downloaded_files.append(found_wav)
                if progress_cb:
                    progress_cb(f"[YouTube {idx}/{len(urls)}] Başarıyla indirildi: {os.path.basename(found_wav)}")
            else:
                err_tail = (res.stdout or "").strip().splitlines()[-3:]
                raise RuntimeError(" | ".join(err_tail) if err_tail else f"Çıkış kodu: {res.returncode}")
        except Exception as e:
            if progress_cb:
                progress_cb(f"[Uyarı] YouTube linki indirilemedi ({url}): {e}")

    return downloaded_files


def convert_to_stereo_wav(input_path: str, out_wav_path: str, target_sr: int = 48000) -> str:
    """
    Uses FFmpeg to convert any audio/video file (mp3, mp4, mkv, flac, m4a, wav) into a clean 48kHz stereo WAV.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        input_path,
        "-vn",
        "-ac",
        "2",
        "-ar",
        str(target_sr),
        "-f",
        "wav",
        out_wav_path,
    ]
    subprocess.run(cmd, check=True)
    return out_wav_path


def isolate_clean_vocals(
    input_wav: str,
    work_dir: str,
    clean_mode: str = "UVR5 Special 2 (Ultra Kalite)",
    remove_backups: bool = True,
    apply_dereverb: bool = True,
    progress_cb=None,
) -> str:
    """
    Isolates vocals from background music, strips backing harmonies/crowd noise, and removes reverb.
    """
    if clean_mode == "Temizleme Yapma (Zaten Temiz Kayıt)":
        return input_wav

    Separator = _ensure_mdxc_patched()
    current_audio = input_wav

    # Step 1: Vocal Separation from Background Music / Noise
    if clean_mode in ("UVR5 Special 2 (Ultra Kalite)", "UVR5 Special (Hızlı Roformer)"):
        if clean_mode == "UVR5 Special 2 (Ultra Kalite)":
            if progress_cb:
                progress_cb("   -> [1/3] Mel-Band Roformer (Becruily) ile arka plan müziği ve gürültü ayrıştırılıyor...")
            sep1 = Separator(
                log_level=30,
                output_dir=work_dir,
                model_file_dir=UVR5_MODELS_DIR,
                output_single_stem="Vocals",
                mdxc_params={"segment_size": 256, "batch_size": 1, "overlap": 4, "override_model_segment_size": False},
            )
            sep1.load_model(model_filename="mel_band_roformer_vocals_becruily.ckpt")
            res1 = sep1.separate(current_audio)
            voc_files = [os.path.join(work_dir, f) for f in res1 if "Vocals" in f]
            if voc_files:
                current_audio = voc_files[0]
            del sep1
            gc.collect()
        else:
            if progress_cb:
                progress_cb("   -> [1/3] BS-Roformer (Viperx 1297) ile arka plan müziği ayrıştırılıyor...")
            sep1 = Separator(
                log_level=30,
                output_dir=work_dir,
                model_file_dir=UVR5_MODELS_DIR,
                output_single_stem="Vocals",
                mdxc_params={"segment_size": 256, "batch_size": 1, "overlap": 4, "override_model_segment_size": False},
            )
            sep1.load_model(model_filename="model_bs_roformer_ep_317_sdr_12.9755.ckpt")
            res1 = sep1.separate(current_audio)
            voc_files = [os.path.join(work_dir, f) for f in res1 if "Vocals" in f]
            if voc_files:
                current_audio = voc_files[0]
            del sep1
            gc.collect()

        # Step 2: Lead Vocal Isolation (Karaoke Model) to strip backing vocals / other speakers
        if remove_backups:
            if progress_cb:
                progress_cb("   -> [2/3] Mel-Band Roformer Karaoke (Aufr33-Viperx) ile geri vokaller/diğer sesler temizleniyor...")
            sep2 = Separator(
                log_level=30,
                output_dir=work_dir,
                model_file_dir=UVR5_MODELS_DIR,
                output_single_stem="Vocals",
                mdxc_params={"segment_size": 256, "batch_size": 1, "overlap": 4, "override_model_segment_size": False},
            )
            sep2.load_model(model_filename="mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt")
            res2 = sep2.separate(current_audio)
            lead_files = [os.path.join(work_dir, f) for f in res2 if "_(Vocals)_mel_band_roformer_karaoke" in f or "Vocals" in f]
            if lead_files:
                current_audio = lead_files[0]
            del sep2
            gc.collect()

    # Step 3: De-Reverb (Anvuew Mel-Band Roformer SDR 19.17)
    if apply_dereverb or clean_mode == "Sadece Yankı & Dip Ses Temizle (Konuşma)":
        if progress_cb:
            progress_cb("   -> [3/3] Anvuew Mel-Band Roformer De-Reverb (SDR 19.17) ile oda yankısı siliniyor...")
        sep3 = Separator(
            log_level=30,
            output_dir=work_dir,
            model_file_dir=UVR5_MODELS_DIR,
            output_single_stem="Vocals",
            mdxc_params={"segment_size": 256, "batch_size": 1, "overlap": 2, "override_model_segment_size": False},
        )
        sep3.load_model(model_filename="dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt")
        res3 = sep3.separate(current_audio)
        dereverb_files = [os.path.join(work_dir, f) for f in res3 if "dereverb_mel_band_roformer" in f or "Vocals" in f]
        if dereverb_files:
            current_audio = dereverb_files[0]
        del sep3
        gc.collect()

    return current_audio


def polish_and_trim_vocal(
    wav_path: str,
    target_sr: int = 48000,
    silence_top_db: int = 35,
    min_segment_sec: float = 1.0,
    apply_highpass: bool = True,
    apply_normalization: bool = True,
) -> tuple:
    """
    Loads isolated vocal, converts to mono at target_sr, applies 80Hz Highpass + gentle gate,
    strips dead silence intervals, normalizes loudness, and returns (concatenated_audio_array, raw_duration, clean_duration, segments_list).
    """
    y, sr = librosa.load(wav_path, sr=target_sr, mono=True)
    raw_duration = len(y) / float(sr)

    if len(y) == 0:
        return np.array([], dtype=np.float32), 0.0, 0.0, []

    # Studio Highpass (removes sub-bass rumble below 75Hz) & gentle NoiseGate
    if apply_highpass:
        board = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=75.0),
            NoiseGate(threshold_db=-48.0, ratio=2.5, release_ms=150.0),
        ])
        y = board(y, sr)

    # Split non-silent intervals
    intervals = librosa.effects.split(y, top_db=silence_top_db, frame_length=2048, hop_length=512)
    clean_segments = []
    short_pause = np.zeros(int(sr * 0.15), dtype=np.float32)  # 150ms natural breathing buffer between segments

    for start_idx, end_idx in intervals:
        seg = y[start_idx:end_idx]
        seg_dur = len(seg) / float(sr)
        if seg_dur < min_segment_sec:
            continue

        # Normalize individual segment peak if requested so quiet phrases match loud phrases
        if apply_normalization:
            peak = np.max(np.abs(seg))
            if peak > 1e-4:
                # Target -1.5 dBFS peak (~0.84)
                seg = seg * (0.84 / peak)

        # Apply tiny 15ms fade-in/fade-out to prevent clicks at splice points
        fade_len = min(int(sr * 0.015), len(seg) // 4)
        if fade_len > 1:
            fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
            fade_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
            seg[:fade_len] *= fade_in
            seg[-fade_len:] *= fade_out

        clean_segments.append(seg)

    if not clean_segments:
        return np.array([], dtype=np.float32), raw_duration, 0.0, []

    # Concatenate with 150ms natural pause
    combined_parts = []
    for i, seg in enumerate(clean_segments):
        combined_parts.append(seg)
        if i < len(clean_segments) - 1:
            combined_parts.append(short_pause)

    concatenated = np.concatenate(combined_parts)
    clean_duration = sum(len(s) for s in clean_segments) / float(sr)
    return concatenated, raw_duration, clean_duration, clean_segments


def build_automatic_dataset(
    dataset_name: str,
    local_files: list = None,
    youtube_urls: list = None,
    clean_mode: str = "UVR5 Special 2 (Ultra Kalite)",
    remove_backups: bool = True,
    apply_dereverb: bool = True,
    target_sr: int = 48000,
    silence_top_db: int = 35,
    max_duration_minutes: float = 0.0,  # 0 = unlimited
    export_mode: str = "Hem Birleştirilmiş Master Hem Parçalı (Önerilen)",
    progress_cb=None,
) -> dict:
    """
    Full end-to-end Automatic Dataset Builder pipeline:
    1. Downloads YouTube links (if any) + gathers local audio/video files.
    2. Runs UVR5 Special 2 / Roformer vocal extraction, karaoke lead isolation, and de-reverb on each source.
    3. Removes silence, normalizes volume, filters sub-bass rumble, and merges clean vocals.
    4. Saves dataset to both AICoverGen/datasets/<dataset_name> and Applio/assets/datasets/<dataset_name>.
    5. Returns detailed summary (raw duration, clean duration, file paths, recommended epochs).
    """
    safe_name = sanitize_name(dataset_name)
    dataset_dir = os.path.join(DATASETS_ROOT, safe_name)
    applio_dataset_dir = os.path.join(APPLIO_DATASETS_DIR, safe_name)
    work_dir = os.path.join(dataset_dir, "_temp_work")
    raw_dir = os.path.join(work_dir, "raw_inputs")

    # Clean previous output files in dataset_dir (except _temp_work while building)
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(applio_dataset_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    for old_f in os.listdir(dataset_dir):
        old_p = os.path.join(dataset_dir, old_f)
        if os.path.isfile(old_p) and old_f.endswith((".wav", ".mp3", ".flac", ".json")):
            try:
                os.remove(old_p)
            except Exception:
                pass

    for old_f in os.listdir(applio_dataset_dir):
        old_p = os.path.join(applio_dataset_dir, old_f)
        if os.path.isfile(old_p):
            try:
                os.remove(old_p)
            except Exception:
                pass

    local_files = [f for f in (local_files or []) if f and os.path.exists(f)]
    youtube_urls = [u.strip() for u in (youtube_urls or []) if u and u.strip()]

    if not local_files and not youtube_urls:
        raise ValueError("Lütfen en az bir yerel ses/video dosyası seçin veya YouTube linki girin!")

    if progress_cb:
        progress_cb("=" * 65)
        progress_cb(f"[Dataset Hazırlayıcı] '{safe_name}' için otomatik stüdyo hattı başlatıldı...")
        progress_cb(f" -> Yerel Dosya Sayısı: {len(local_files)} | YouTube Link Sayısı: {len(youtube_urls)}")
        progress_cb(f" -> Temizleme Modu: {clean_mode} | Örnekleme Hızı: {target_sr} Hz")
        progress_cb("=" * 65)

    all_input_sources = list(local_files)

    # 1. Download YouTube links if provided
    if youtube_urls:
        yt_files = download_youtube_audios(youtube_urls, raw_dir, progress_cb=progress_cb)
        all_input_sources.extend(yt_files)

    if not all_input_sources:
        raise RuntimeError("Hiçbir ses kaynağı indirilemedi veya bulunamadı.")

    total_raw_duration = 0.0
    total_clean_duration = 0.0
    master_audio_chunks = []
    all_segments = []
    max_seconds = max_duration_minutes * 60.0 if max_duration_minutes and max_duration_minutes > 0 else 0.0

    # 2. Process each file through UVR5 Roformer + Silence Trimmer + Normalizer
    for idx, src_path in enumerate(all_input_sources, 1):
        src_name = os.path.basename(src_path)
        if progress_cb:
            progress_cb(f"\n[Kaynak {idx}/{len(all_input_sources)}] İşleniyor: {src_name}")

        file_work_dir = os.path.join(work_dir, f"src_{idx:03d}")
        os.makedirs(file_work_dir, exist_ok=True)

        # Convert source to standard stereo WAV
        std_wav = os.path.join(file_work_dir, f"source_{idx:03d}.wav")
        convert_to_stereo_wav(src_path, std_wav, target_sr=44100)

        # Isolate vocals
        clean_vocal_wav = isolate_clean_vocals(
            input_wav=std_wav,
            work_dir=file_work_dir,
            clean_mode=clean_mode,
            remove_backups=remove_backups,
            apply_dereverb=apply_dereverb,
            progress_cb=progress_cb,
        )

        # Trim silence, highpass & normalize
        if progress_cb:
            progress_cb(f"   -> Sessizlikler kesiliyor, 75Hz Highpass ve ses seviyesi dengeleme (Normalization) uygulanıyor...")

        concat_audio, raw_dur, clean_dur, segs = polish_and_trim_vocal(
            wav_path=clean_vocal_wav,
            target_sr=target_sr,
            silence_top_db=silence_top_db,
            min_segment_sec=1.0,
            apply_highpass=True,
            apply_normalization=True,
        )

        total_raw_duration += raw_dur
        if len(concat_audio) > 0:
            master_audio_chunks.append(concat_audio)
            all_segments.extend(segs)
            total_clean_duration += clean_dur
            if progress_cb:
                progress_cb(
                    f"   [OK] {src_name}: Ham Süre = {format_duration(raw_dur)} -> Net Temiz Vokal = {format_duration(clean_dur)} ({len(segs)} parça)"
                )
        else:
            if progress_cb:
                progress_cb(f"   [Uyarı] {src_name} içinde yeterli net vokal bulunamadı, atlandı.")

        # Check if we reached target max duration
        if max_seconds > 0 and total_clean_duration >= max_seconds:
            if progress_cb:
                progress_cb(f"\n[Bilgi] Hedeflenen maksimum dataset süresine ({format_duration(max_seconds)}) ulaşıldı!")
            break

    if not master_audio_chunks:
        raise RuntimeError("Seçilen dosyalardan hiç temiz vokal çıkarılamadı!")

    # 3. Combine & Export clean audio to dataset_dir and applio_dataset_dir
    pause_between_files = np.zeros(int(target_sr * 0.25), dtype=np.float32)
    final_blocks = []
    for i, chunk in enumerate(master_audio_chunks):
        final_blocks.append(chunk)
        if i < len(master_audio_chunks) - 1:
            final_blocks.append(pause_between_files)

    full_master_audio = np.concatenate(final_blocks)

    # Cap to exact max_seconds if requested
    if max_seconds > 0 and (len(full_master_audio) / float(target_sr)) > max_seconds:
        full_master_audio = full_master_audio[: int(max_seconds * target_sr)]
        total_clean_duration = max_seconds

    # Split master into 5-minute WAV files so neither Applio nor librosa ever chokes on a single huge >1GB WAV file
    max_chunk_samples = int(target_sr * 300)  # 5 minutes per master part
    saved_files = []

    for part_idx, start_sample in enumerate(range(0, len(full_master_audio), max_chunk_samples), 1):
        part_audio = full_master_audio[start_sample : start_sample + max_chunk_samples]
        part_filename = f"{safe_name}_clean_part{part_idx:02d}.wav"
        out_path_local = os.path.join(dataset_dir, part_filename)
        out_path_applio = os.path.join(applio_dataset_dir, part_filename)

        sf.write(out_path_local, part_audio, target_sr, subtype="PCM_16")
        shutil.copy2(out_path_local, out_path_applio)
        saved_files.append(out_path_local)

    # Cleanup temporary work folder to save disk space
    try:
        shutil.rmtree(work_dir, ignore_errors=True)
    except Exception:
        pass

    epoch_advice = calculate_recommended_epochs(total_clean_duration, preset="Dengeli")

    manifest = {
        "dataset_name": safe_name,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_count": len(all_input_sources),
        "clean_mode": clean_mode,
        "sample_rate": target_sr,
        "raw_duration_seconds": round(total_raw_duration, 2),
        "clean_duration_seconds": round(total_clean_duration, 2),
        "raw_duration_str": format_duration(total_raw_duration),
        "clean_duration_str": format_duration(total_clean_duration),
        "segment_count": len(all_segments),
        "saved_files": saved_files,
        "applio_dataset_dir": applio_dataset_dir,
        "local_dataset_dir": dataset_dir,
        "recommended_epochs": epoch_advice["recommended_epochs"],
        "recommended_save_every": epoch_advice["save_every_epoch"],
        "quality_note": epoch_advice["note"],
    }

    with open(os.path.join(dataset_dir, "dataset_info.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    if progress_cb:
        progress_cb("\n" + "=" * 65)
        progress_cb(f"✅ OTOMATİK DATASET HAZIRLAMA TAMAMLANDI: {safe_name}")
        progress_cb(f" 📊 Toplam Ham Ses Süresi   : {manifest['raw_duration_str']}")
        progress_cb(f" 💎 Net Temiz Vokal Süresi  : {manifest['clean_duration_str']} ({len(all_segments)} temiz vokal cümlesi)")
        progress_cb(f" 🧠 Önerilen Eğitim Epoch   : {manifest['recommended_epochs']} Epoch ({manifest['quality_note']})")
        progress_cb(f" 📂 Dataset Klasörü         : {dataset_dir}")
        progress_cb("=" * 65)

    return manifest
