import argparse
import gc
import hashlib
import json
import os
import sys

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Ensure ONNX Runtime CUDA dependencies are loaded
_capi_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'venv', 'lib', 'site-packages', 'onnxruntime', 'capi')
if os.path.exists(_capi_dir):
    try:
        os.add_dll_directory(_capi_dir)
        os.environ['PATH'] = _capi_dir + ';' + os.environ.get('PATH', '')
    except Exception:
        pass
import shlex
import subprocess
from contextlib import suppress
from urllib.parse import urlparse, parse_qs
import re
import unicodedata
import gradio as gr
import librosa
import numpy as np
import soundfile as sf
import yt_dlp
from pedalboard import Pedalboard, Reverb, Compressor, HighpassFilter
from pedalboard.io import AudioFile
from pydub import AudioSegment

from mdx import run_mdx
from rvc import Config, load_hubert, get_vc, rvc_infer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

mdxnet_models_dir = os.path.join(BASE_DIR, 'mdxnet_models')
rvc_models_dir = os.path.join(BASE_DIR, 'rvc_models')
output_dir = os.path.join(BASE_DIR, 'song_output')


def safe_filename(name):
    tr_map = str.maketrans('çÇğĞıİöÖşŞüÜ', 'cCgGiIoOsSuU')
    name = name.translate(tr_map)
    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii')
    name = re.sub(r'[^\w\s-]', '', name).strip()
    return name or 'song'


def get_youtube_video_id(url, ignore_playlist=True):
    """
    Examples:
    http://youtu.be/SA2iWivDJiE
    http://www.youtube.com/watch?v=_oPAwA_Udwc&feature=feedu
    http://www.youtube.com/embed/SA2iWivDJiE
    http://www.youtube.com/v/SA2iWivDJiE?version=3&amp;hl=en_US
    """
    query = urlparse(url)
    if query.hostname == 'youtu.be':
        if query.path[1:] == 'watch':
            return query.query[2:]
        return query.path[1:]

    if query.hostname in {'www.youtube.com', 'youtube.com', 'music.youtube.com'}:
        if not ignore_playlist:
            # use case: get playlist id not current video in playlist
            with suppress(KeyError):
                return parse_qs(query.query)['list'][0]
        if query.path == '/watch':
            return parse_qs(query.query)['v'][0]
        if query.path[:7] == '/watch/':
            return query.path.split('/')[1]
        if query.path[:7] == '/embed/':
            return query.path.split('/')[2]
        if query.path[:3] == '/v/':
            return query.path.split('/')[2]

    # returns None for invalid YouTube url
    return None


def yt_download(link):
    from dataset_builder import get_modern_ytdlp_cmd
    vid_id = get_youtube_video_id(link) or "yt_song"
    out_dir = os.path.join(output_dir, vid_id)
    os.makedirs(out_dir, exist_ok=True)
    out_template = os.path.join(out_dir, f"{vid_id}_orig.%(ext)s")
    target_mp3 = os.path.join(out_dir, f"{vid_id}_orig.mp3")

    cmd = get_modern_ytdlp_cmd() + [
        "--no-playlist",
        "--no-check-certificate",
        "--no-warnings",
        "-f",
        "bestaudio/best",
        "-x",
        "--audio-format",
        "mp3",
        "-o",
        out_template,
        link,
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    if os.path.exists(target_mp3):
        return target_mp3
    for f in os.listdir(out_dir):
        if f.startswith(f"{vid_id}_orig.") and f.lower().endswith((".mp3", ".wav", ".m4a", ".webm")):
            return os.path.join(out_dir, f)
    raise RuntimeError(f"YouTube indirme hatası: {(res.stdout or '').strip()}")


def raise_exception(error_msg, is_webui):
    if is_webui:
        raise gr.Error(error_msg)
    else:
        raise Exception(error_msg)


def get_rvc_model(voice_model, is_webui):
    rvc_model_filename, rvc_index_filename = None, None
    model_dir = os.path.join(rvc_models_dir, voice_model)
    for file in os.listdir(model_dir):
        ext = os.path.splitext(file)[1]
        if ext == '.pth':
            rvc_model_filename = file
        if ext == '.index':
            rvc_index_filename = file

    if rvc_model_filename is None:
        error_msg = f'No model file exists in {model_dir}.'
        raise_exception(error_msg, is_webui)

    return os.path.join(model_dir, rvc_model_filename), os.path.join(model_dir, rvc_index_filename) if rvc_index_filename else ''


def get_audio_paths(song_dir, uvr_model=None):
    manifest_path = os.path.join(song_dir, 'cache_manifest.json')
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
            if uvr_model is not None and manifest.get('uvr_model') and manifest.get('uvr_model') != uvr_model:
                return None, None, None, None
            paths = (
                manifest.get('orig_song_path'),
                manifest.get('instrumentals_path'),
                manifest.get('main_vocals_dereverb_path'),
                manifest.get('backup_vocals_path'),
            )
            if all(p and os.path.exists(p) for p in paths):
                return paths
        except Exception:
            pass

    orig_song_path = None
    instrumentals_path = None
    main_vocals_dereverb_path = None
    backup_vocals_path = None

    for file in os.listdir(song_dir):
        # 1) Instrumental Matching
        if file.endswith('_Instrumental.wav'):
            instrumentals_path = os.path.join(song_dir, file)
            orig_song_path = instrumentals_path.replace('_Instrumental.wav', '')
        elif '_(Instrumental)_' in file and ('melband_roformer' in file or 'mel_band_roformer_instrumental' in file):
            instrumentals_path = os.path.join(song_dir, file)
            orig_song_path = instrumentals_path.split('_(Instrumental)_')[0]

        # 2) Vocal Matching
        if file.endswith('_Vocals_Main_DeReverb.wav') or file.endswith('_DeReverb_Aggressive.wav') or file.endswith('_DeReverb.wav'):
            main_vocals_dereverb_path = os.path.join(song_dir, file)
        elif '_(Vocals)_' in file and ('bs_roformer' in file or 'dereverb_mel_band_roformer' in file or 'mel_band_roformer_vocals_becruily' in file):
            main_vocals_dereverb_path = os.path.join(song_dir, file)

        # 3) Backup Vocals
        if file.endswith('_Vocals_Backup.wav') or file.endswith('empty_backup.wav'):
            backup_vocals_path = os.path.join(song_dir, file)

    return orig_song_path, instrumentals_path, main_vocals_dereverb_path, backup_vocals_path


def convert_to_stereo(audio_path):
    wave, sr = librosa.load(audio_path, mono=False, sr=44100)

    # check if mono
    if type(wave[0]) != np.ndarray:
        stereo_path = f'{os.path.splitext(audio_path)[0]}_stereo.wav'
        command = ['ffmpeg', '-y', '-loglevel', 'error', '-i', audio_path, '-ac', '2', '-f', 'wav', stereo_path]
        subprocess.run(command)
        return stereo_path
    else:
        return audio_path


def pitch_shift(audio_path, pitch_change):
    output_path = f'{os.path.splitext(audio_path)[0]}_p{pitch_change}.wav'
    if not os.path.exists(output_path):
        y, sr = sf.read(audio_path)
        is_multichannel = y.ndim > 1
        audio = y.T if is_multichannel else y
        y_shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=float(pitch_change))
        if is_multichannel:
            y_shifted = y_shifted.T
        sf.write(output_path, y_shifted, sr)

    return output_path


def get_hash(filepath):
    with open(filepath, 'rb') as f:
        file_hash = hashlib.blake2b()
        while chunk := f.read(8192):
            file_hash.update(chunk)

    return file_hash.hexdigest()[:11]


def display_progress(message, percent, is_webui, progress=None):
    if is_webui:
        progress(percent, desc=message)
    else:
        print(message)


def preprocess_song(song_input, mdx_model_params, song_id, is_webui, input_type, aggressive_dereverb=False, uvr_model="Default RVC", progress=None):
    keep_orig = False
    if input_type == 'yt':
        display_progress('[~] Downloading song...', 0, is_webui, progress)
        song_link = song_input.split('&')[0]
        orig_song_path = yt_download(song_link)
    elif input_type == 'local':
        orig_song_path = song_input
        keep_orig = True
    else:
        orig_song_path = None

    song_output_dir = os.path.join(output_dir, song_id)
    orig_song_path = convert_to_stereo(orig_song_path)

    uvr5_models_dir = mdxnet_models_dir

    def _ensure_mdxc_patched():
        try:
            from audio_separator.separator import Separator
            from uvr_model_support import patch_model_catalog
            patch_model_catalog(Separator)
            try:
                from audio_separator.separator.architectures.mdxc_separator import MDXCSeparator
                if not hasattr(MDXCSeparator, '_patched_by_ripleytia'):
                    original_demix = MDXCSeparator.demix
                    def patched_demix(self, mix):
                        res = original_demix(self, mix)
                        if isinstance(res, dict):
                            return res
                        return {self.primary_stem_name: res, self.secondary_stem_name: res}
                    MDXCSeparator.demix = patched_demix
                    MDXCSeparator._patched_by_ripleytia = True
            except Exception as e:
                print(f'Patch warning: {e}')
            return Separator
        except ImportError:
            raise_exception("Gerekli kutuphaneler eksik. Lutfen 'pip install audio-separator' komutunu calistirin.", is_webui)

    if uvr_model == "UVR5 Special 3":
        Separator = _ensure_mdxc_patched()
        special3_params = {"segment_size": 256, "batch_size": 1, "overlap": 4, "override_model_segment_size": False}

        def _special3_stem_path(output_files, stem_name):
            needle = f"({stem_name.lower()})"
            matches = [os.path.join(song_output_dir, name) for name in output_files if needle in os.path.basename(name).lower()]
            if not matches:
                raise RuntimeError(f"UVR5 Special 3 output stem not found: {stem_name}; files={output_files}")
            return matches[0]

        display_progress('[~] [UVR5 Special 3 - 1/5] BS-Roformer 1296 ile yüksek kaliteli temel vokal/enstrüman ayrımı...', 0.08, is_webui, progress)
        sep1 = Separator(log_level=30, output_dir=song_output_dir, model_file_dir=uvr5_models_dir, mdxc_params=special3_params)
        sep1.load_model(model_filename='model_bs_roformer_ep_368_sdr_12.9628.ckpt')
        res1 = sep1.separate(orig_song_path)
        vocals_reference_path = _special3_stem_path(res1, 'Vocals')
        instrumentals_path = _special3_stem_path(res1, 'Instrumental')
        del sep1
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        display_progress('[~] [UVR5 Special 3 - 2/5] InstVoc Duality V2 ile bağımsız enstrüman/vokal referansı...', 0.16, is_webui, progress)
        sep2 = Separator(log_level=30, output_dir=song_output_dir, model_file_dir=uvr5_models_dir, mdxc_params=special3_params)
        sep2.load_model(model_filename='melband_roformer_instvox_duality_v2.ckpt')
        res2 = sep2.separate(orig_song_path)
        duality_vocals_path = _special3_stem_path(res2, 'Vocals')
        duality_instrumentals_path = _special3_stem_path(res2, 'Instrumental')
        del sep2
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        display_progress('[~] [UVR5 Special 3 - 3/5] Kimberley MelBand Vocals ile ayrıntılı ana vokal çıkarımı...', 0.24, is_webui, progress)
        sep3 = Separator(log_level=30, output_dir=song_output_dir, model_file_dir=uvr5_models_dir, output_single_stem='Vocals', mdxc_params=special3_params)
        sep3.load_model(model_filename='vocals_mel_band_roformer.ckpt')
        res3 = sep3.separate(orig_song_path)
        vocals_candidate_path = _special3_stem_path(res3, 'Vocals')
        del sep3
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        display_progress('[~] [UVR5 Special 3] İki bağımsız referansla vokal ve enstrüman kayıpları korunuyor...', 0.30, is_webui, progress)
        from vocal_shield import apply_vocal_rescue_shield
        consensus_vocals_path = os.path.join(song_output_dir, f"{song_id}_UVR5Special3_VocalConsensus.wav")
        vocals_path = apply_vocal_rescue_shield(
            ref_path=vocals_reference_path,
            proc_path=vocals_candidate_path,
            output_path=consensus_vocals_path,
            stem_label='Vokal',
            progress_cb=lambda msg: display_progress(msg, 0.31, is_webui, progress),
        )
        rescued_instrumentals_path = os.path.join(song_output_dir, f"{song_id}_UVR5Special3_InstrumentalRescued.wav")
        instrumentals_path = apply_vocal_rescue_shield(
            ref_path=duality_instrumentals_path,
            proc_path=instrumentals_path,
            output_path=rescued_instrumentals_path,
            stem_label='Enstrüman',
            progress_cb=lambda msg: display_progress(msg, 0.32, is_webui, progress),
        )

        display_progress('[~] [UVR5 Special 3 - 4/5] Gabox Karaoke ile ana ve yedek vokal ayrımı...', 0.34, is_webui, progress)
        sep4 = Separator(log_level=30, output_dir=song_output_dir, model_file_dir=uvr5_models_dir, mdxc_params=special3_params)
        sep4.load_model(model_filename='mel_band_roformer_karaoke_gabox.ckpt')
        res4 = sep4.separate(vocals_path)
        main_vocals_path = _special3_stem_path(res4, 'Vocals')
        backup_vocals_path = _special3_stem_path(res4, 'Instrumental')
        del sep4
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        display_progress('[~] [UVR5 Special 3 - 5/5] BS-Roformer De-Reverb ile oda yankısı temizleniyor...', 0.38, is_webui, progress)
        sep5 = Separator(log_level=30, output_dir=song_output_dir, model_file_dir=uvr5_models_dir, output_single_stem='noreverb', mdxc_params=special3_params)
        sep5.load_model(model_filename='deverb_bs_roformer_8_384dim_10depth.ckpt')
        # This model's stems are named noreverb/reverb; audio-separator 0.18 defaults to Vocals/Instrumental.
        sep5.model_instance.primary_stem_name = 'noreverb'
        sep5.model_instance.secondary_stem_name = 'reverb'
        res5 = sep5.separate(main_vocals_path)
        dereverb_path = _special3_stem_path(res5, 'noreverb')
        del sep5
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        main_vocals_dereverb_path = os.path.join(song_output_dir, f"{song_id}_UVR5Special3_LeadProtected.wav")
        try:
            main_vocals_dereverb_path = apply_vocal_rescue_shield(
                ref_path=main_vocals_path,
                proc_path=dereverb_path,
                output_path=main_vocals_dereverb_path,
                stem_label='Lead Vokal',
                progress_cb=lambda msg: display_progress(msg, 0.40, is_webui, progress),
            )
        except Exception as shield_err:
            print(f'[UVR5 Special 3 Kalkan Hatası] {shield_err}')

    elif uvr_model == "UVR5 Special 2":
        Separator = _ensure_mdxc_patched()

        # Stage 1: SOTA Vocal Isolation (Becruily Mel-Band Roformer Vocals)
        display_progress('[~] [UVR5 Special 2 - 1/4] Separating Vocals using MelBand-Roformer-Vocals-Becruily...', 0.08, is_webui, progress)
        sep1 = Separator(log_level=30, output_dir=song_output_dir, model_file_dir=uvr5_models_dir, output_single_stem="Vocals", mdxc_params={"segment_size": 256, "batch_size": 1, "overlap": 4, "override_model_segment_size": False})
        sep1.load_model(model_filename='mel_band_roformer_vocals_becruily.ckpt')
        res1 = sep1.separate(orig_song_path)
        vocals_path = [os.path.join(song_output_dir, f) for f in res1 if 'Vocals' in f][0]

        # Stage 2: SOTA Instrumental Isolation (Becruily Mel-Band Roformer Instrumental)
        display_progress('[~] [UVR5 Special 2 - 2/4] Separating Instrumentals using MelBand-Roformer-Inst-Becruily...', 0.18, is_webui, progress)
        sep2 = Separator(log_level=30, output_dir=song_output_dir, model_file_dir=uvr5_models_dir, output_single_stem="Instrumental", mdxc_params={"segment_size": 256, "batch_size": 1, "overlap": 4, "override_model_segment_size": False})
        sep2.load_model(model_filename='mel_band_roformer_instrumental_becruily.ckpt')
        res2 = sep2.separate(orig_song_path)
        instrumentals_path = [os.path.join(song_output_dir, f) for f in res2 if 'Instrumental' in f][0]

        # Stage 3: SOTA Karaoke / Lead vs Backing Vocal Separation (Aufr33 & Viperx Mel-Band Roformer Karaoke)
        display_progress('[~] [UVR5 Special 2 - 3/4] Separating Lead & Backing Vocals using Mel-Roformer-Karaoke-Aufr33-Viperx...', 0.28, is_webui, progress)
        sep3 = Separator(log_level=30, output_dir=song_output_dir, model_file_dir=uvr5_models_dir, mdxc_params={"segment_size": 256, "batch_size": 1, "overlap": 4, "override_model_segment_size": False})
        sep3.load_model(model_filename='mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt')
        res3 = sep3.separate(vocals_path)
        main_vocals_path = [os.path.join(song_output_dir, f) for f in res3 if '_(Vocals)_mel_band_roformer_karaoke' in f][0]
        raw_backup_path = [os.path.join(song_output_dir, f) for f in res3 if '_(Instrumental)_mel_band_roformer_karaoke' in f][0]
        backup_vocals_path = os.path.join(song_output_dir, f"{song_id}_Vocals_Backup.wav")
        if os.path.abspath(raw_backup_path) != os.path.abspath(backup_vocals_path):
            if os.path.exists(backup_vocals_path):
                os.remove(backup_vocals_path)
            os.replace(raw_backup_path, backup_vocals_path)

        # Stage 4: SOTA Roformer De-Reverb (Anvuew Mel-Band Roformer De-Reverb SDR 19.17)
        display_progress('[~] [UVR5 Special 2 - 4/4] Applying Studio De-Reverb using MelBand-Roformer-DeReverb-Anvuew (SDR 19.17)...', 0.36, is_webui, progress)
        sep4 = Separator(log_level=30, output_dir=song_output_dir, model_file_dir=uvr5_models_dir, output_single_stem="Vocals", mdxc_params={"segment_size": 256, "batch_size": 1, "overlap": 2, "override_model_segment_size": False})
        sep4.load_model(model_filename='dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt')
        res4 = sep4.separate(main_vocals_path)
        main_vocals_dereverb_path = [os.path.join(song_output_dir, f) for f in res4 if 'dereverb_mel_band_roformer' in f][0]

        # Stage 5: Intelligent Vocal Rescue Shield (Restores swallowed words, dropouts, and lost vocal body)
        display_progress('[~] [UVR5 Special 2 - Kalkan] Akilli Vokal Kurtarma Kalkani uygulaniyor...', 0.38, is_webui, progress)
        try:
            from vocal_shield import apply_vocal_rescue_shield
            main_vocals_dereverb_path = apply_vocal_rescue_shield(
                ref_path=vocals_path,
                proc_path=main_vocals_dereverb_path,
                progress_cb=lambda msg: display_progress(f'[~] {msg}', 0.38, is_webui, progress)
            )
        except Exception as shield_err:
            print(f'[UVR5 Shield Hatasi] {shield_err}')

    elif uvr_model != "Default RVC":
        if uvr_model == "UVR5 Special":
            Separator = _ensure_mdxc_patched()
            display_progress('[~] Separating Vocals using BS-Roformer-Viperx-1297...', 0.1, is_webui, progress)
            sep1 = Separator(log_level=30, output_dir=song_output_dir, model_file_dir=uvr5_models_dir, mdxc_params={"segment_size": 256, "batch_size": 1, "overlap": 4, "override_model_segment_size": False})
            sep1.load_model(model_filename='model_bs_roformer_ep_317_sdr_12.9755.ckpt')
            res1 = sep1.separate(orig_song_path)
            vocals_path = [os.path.join(song_output_dir, f) for f in res1 if 'Vocals' in f][0]

            display_progress('[~] Separating Instrumentals using MB-Roformer-Inst-v2...', 0.2, is_webui, progress)
            sep2 = Separator(log_level=30, output_dir=song_output_dir, model_file_dir=uvr5_models_dir, mdxc_params={"segment_size": 256, "batch_size": 1, "overlap": 4, "override_model_segment_size": False})
            sep2.load_model(model_filename='melband_roformer_inst_v2.ckpt')
            res2 = sep2.separate(orig_song_path)
            instrumentals_path = [os.path.join(song_output_dir, f) for f in res2 if 'Instrumental' in f][0]
            
            main_vocals_path = vocals_path
        else:
            display_progress(f'[~] Separating Vocals using {uvr_model}...', 0.1, is_webui, progress)
            vocals_path, instrumentals_path = run_mdx(mdx_model_params, song_output_dir, os.path.join(mdxnet_models_dir, uvr_model), orig_song_path, denoise=True, keep_orig=keep_orig)
            main_vocals_path = vocals_path
        
        if aggressive_dereverb:
            display_progress('[~] Applying DeReverb to Vocals...', 0.3, is_webui, progress)
            _, main_vocals_dereverb_path = run_mdx(mdx_model_params, song_output_dir, os.path.join(mdxnet_models_dir, 'Reverb_HQ_By_FoxJoy.onnx'), main_vocals_path, invert_suffix='DeReverb', exclude_main=True, denoise=True)
            display_progress('[~] Applying AGGRESSIVE DeReverb to Vocals (Pass 2)...', 0.35, is_webui, progress)
            _, main_vocals_dereverb_path = run_mdx(mdx_model_params, song_output_dir, os.path.join(mdxnet_models_dir, 'Reverb_HQ_By_FoxJoy.onnx'), main_vocals_dereverb_path, invert_suffix='DeReverb_Aggressive', exclude_main=True, denoise=True)
        else:
            main_vocals_dereverb_path = vocals_path
            
        # Create an empty backup vocals track so we don't crash when mixing later
        backup_vocals_path = os.path.join(song_output_dir, f"{song_id}_empty_backup.wav")
        if not os.path.exists(backup_vocals_path):
            import soundfile as sf
            import numpy as np
            sf.write(backup_vocals_path, np.zeros((100, 2)), 44100)
    else:
        display_progress('[~] Separating Vocals from Instrumental...', 0.1, is_webui, progress)
        vocals_path, instrumentals_path = run_mdx(mdx_model_params, song_output_dir, os.path.join(mdxnet_models_dir, 'UVR-MDX-NET-Voc_FT.onnx'), orig_song_path, denoise=True, keep_orig=keep_orig)

        display_progress('[~] Separating Main Vocals from Backup Vocals...', 0.2, is_webui, progress)
        backup_vocals_path, main_vocals_path = run_mdx(mdx_model_params, song_output_dir, os.path.join(mdxnet_models_dir, 'UVR_MDXNET_KARA_2.onnx'), vocals_path, suffix='Backup', invert_suffix='Main', denoise=True)

        display_progress('[~] Applying DeReverb to Vocals...', 0.3, is_webui, progress)
        _, main_vocals_dereverb_path = run_mdx(mdx_model_params, song_output_dir, os.path.join(mdxnet_models_dir, 'Reverb_HQ_By_FoxJoy.onnx'), main_vocals_path, invert_suffix='DeReverb', exclude_main=True, denoise=True)

        if aggressive_dereverb:
            display_progress('[~] Applying AGGRESSIVE DeReverb to Vocals (Pass 2)...', 0.35, is_webui, progress)
            _, main_vocals_dereverb_path = run_mdx(mdx_model_params, song_output_dir, os.path.join(mdxnet_models_dir, 'Reverb_HQ_By_FoxJoy.onnx'), main_vocals_dereverb_path, invert_suffix='DeReverb_Aggressive', exclude_main=True, denoise=True)

    try:
        manifest_path = os.path.join(song_output_dir, 'cache_manifest.json')
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump({
                'uvr_model': uvr_model,
                'aggressive_dereverb': aggressive_dereverb,
                'orig_song_path': orig_song_path,
                'instrumentals_path': instrumentals_path,
                'main_vocals_dereverb_path': main_vocals_dereverb_path,
                'backup_vocals_path': backup_vocals_path,
            }, f, indent=2)
    except Exception:
        pass

    return orig_song_path, vocals_path, instrumentals_path, main_vocals_path, backup_vocals_path, main_vocals_dereverb_path


def voice_change(voice_model, vocals_path, output_path, pitch_change, f0_method, index_rate, filter_radius, rms_mix_rate, protect, crepe_hop_length, is_webui):
    rvc_model_path, rvc_index_path = get_rvc_model(voice_model, is_webui)
    device = 'cuda:0'
    config = Config(device, True)
    hubert_model = load_hubert(device, config.is_half, os.path.join(rvc_models_dir, 'hubert_base.pt'))
    cpt, version, net_g, tgt_sr, vc = get_vc(device, config.is_half, config, rvc_model_path)

    actual_f0_method = f0_method
    if f0_method == 'ensemble_consensus':
        print('[~] Ensemble F0 Consensus devrede: Coklu model frekans haritasi cikariliyor...')
        try:
            from vocal_enhancer import compute_ensemble_f0_consensus
            _ = compute_ensemble_f0_consensus(vocals_path)
        except Exception as e:
            print(f"[Ensemble F0 Consensus Uyarisi] {e}")
        actual_f0_method = 'rmvpe'

    # convert main vocals
    rvc_infer(rvc_index_path, index_rate, vocals_path, output_path, pitch_change, actual_f0_method, cpt, version, net_g, filter_radius, tgt_sr, rms_mix_rate, protect, crepe_hop_length, vc, hubert_model)
    del hubert_model, cpt
    gc.collect()


def add_audio_effects(audio_path, reverb_rm_size, reverb_wet, reverb_dry, reverb_damping):
    output_path = f'{os.path.splitext(audio_path)[0]}_mixed.wav'

    # Initialize audio effects plugins
    board = Pedalboard(
        [
            HighpassFilter(),
            Compressor(ratio=4, threshold_db=-15),
            Reverb(room_size=reverb_rm_size, dry_level=reverb_dry, wet_level=reverb_wet, damping=reverb_damping)
         ]
    )

    with AudioFile(audio_path) as f:
        with AudioFile(output_path, 'w', f.samplerate, f.num_channels) as o:
            # Read one second of audio at a time, until the file is empty:
            while f.tell() < f.frames:
                chunk = f.read(int(f.samplerate))
                effected = board(chunk, f.samplerate, reset=False)
                o.write(effected)

    return output_path


def combine_audio(audio_paths, output_path, main_gain, backup_gain, inst_gain, output_format):
    main_vocal_audio = AudioSegment.from_wav(audio_paths[0]) - 4 + main_gain
    backup_vocal_audio = AudioSegment.from_wav(audio_paths[1]) - 6 + backup_gain
    instrumental_audio = AudioSegment.from_wav(audio_paths[2]) - 7 + inst_gain
    combined = main_vocal_audio.overlay(backup_vocal_audio).overlay(instrumental_audio)

    # Windows file locking protection
    try:
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except Exception:
                pass
        combined.export(output_path, format=output_format)
        return output_path
    except PermissionError:
        import time
        base, ext = os.path.splitext(output_path)
        fallback_path = f"{base}_{int(time.time())}{ext}"
        print(f"[Dosya Kilidi Korumasi] '{output_path}' baska bir uygulama tarafindan acik! Guvenle suraya kaydedildi: {fallback_path}")
        combined.export(fallback_path, format=output_format)
        return fallback_path


def to_float(val, default=0.0):
    try:
        if isinstance(val, (int, float)):
            return float(val)
        return float(str(val).replace(',', '.'))
    except (ValueError, TypeError):
        return default


def to_int(val, default=0):
    try:
        return int(to_float(val, default))
    except (ValueError, TypeError):
        return default


def song_cover_pipeline(song_input, voice_model, pitch_change, keep_files=False,
                        is_webui=0, main_gain=0, backup_gain=0, inst_gain=0, index_rate=0.5, filter_radius=3,
                        rms_mix_rate=0.25, f0_method='rmvpe', crepe_hop_length=128, protect=0.33, pitch_change_all=0,
                        reverb_rm_size=0.15, reverb_wet=0.2, reverb_dry=0.8, reverb_damping=0.7, output_format='mp3',
                        aggressive_dereverb=False, uvr_model="Default RVC", vocal_noise_gate=False, noise_gate_threshold=-35,
                        force_reprocess=False, auto_mode=False, auto_mode_callback=None, progress=gr.Progress(),
                        enable_deesser=True, enable_studio_eq=True, enable_doubler=False, enable_sidechain=True,
                        enable_lufs_mastering=True, enable_breath_passthrough=True, formant_shift=0.0,
                        generate_harmonies=False, harmony_mode="3rd_and_5th", dynamic_pitch_rules=""):
    try:
        pitch_change = to_float(pitch_change, 0)
        pitch_change_all = to_float(pitch_change_all, 0)
        index_rate = to_float(index_rate, 0.5)
        filter_radius = to_int(filter_radius, 3)
        rms_mix_rate = to_float(rms_mix_rate, 0.25)
        protect = to_float(protect, 0.33)
        crepe_hop_length = to_int(crepe_hop_length, 128)
        main_gain = to_float(main_gain, 0)
        backup_gain = to_float(backup_gain, 0)
        inst_gain = to_float(inst_gain, 0)
        reverb_rm_size = to_float(reverb_rm_size, 0.15)
        reverb_wet = to_float(reverb_wet, 0.2)
        reverb_dry = to_float(reverb_dry, 0.8)
        reverb_damping = to_float(reverb_damping, 0.7)

        if not song_input or not voice_model:
            raise_exception('Lutfen bir sarki girisi ve ses modeli secin (Ensure song input and voice model are selected).', is_webui)

        display_progress('[~] Starting AI Cover Generation Pipeline...', 0, is_webui, progress)

        with open(os.path.join(mdxnet_models_dir, 'model_data.json')) as infile:
            mdx_model_params = json.load(infile)

        # if youtube url
        if urlparse(song_input).scheme == 'https':
            input_type = 'yt'
            song_id = get_youtube_video_id(song_input)
            if song_id is None:
                error_msg = 'Invalid YouTube url.'
                raise_exception(error_msg, is_webui)

        # local audio file
        else:
            input_type = 'local'
            song_input = song_input.strip('\"')
            if os.path.exists(song_input):
                song_id = get_hash(song_input)
            else:
                error_msg = f'{song_input} does not exist.'
                song_id = None
                raise_exception(error_msg, is_webui)

        song_dir = os.path.join(output_dir, song_id)

        if not os.path.exists(song_dir):
            os.makedirs(song_dir)
            orig_song_path, vocals_path, instrumentals_path, main_vocals_path, backup_vocals_path, main_vocals_dereverb_path = preprocess_song(song_input, mdx_model_params, song_id, is_webui, input_type, aggressive_dereverb=aggressive_dereverb, uvr_model=uvr_model, progress=progress)

        else:
            vocals_path, main_vocals_path = None, None
            paths = get_audio_paths(song_dir, uvr_model=uvr_model)

            # if any of the audio files aren't available or keep intermediate files, rerun preprocess
            if force_reprocess or any(path is None for path in paths):
                orig_song_path, vocals_path, instrumentals_path, main_vocals_path, backup_vocals_path, main_vocals_dereverb_path = preprocess_song(song_input, mdx_model_params, song_id, is_webui, input_type, aggressive_dereverb=aggressive_dereverb, uvr_model=uvr_model, progress=progress)
            else:
                orig_song_path, instrumentals_path, main_vocals_dereverb_path, backup_vocals_path = paths

        if auto_mode:
            display_progress('[~] Analyzing audio for Auto Mode...', 0.45, is_webui, progress)
            try:
                from audio_analyzer import AudioAnalyzer
                new_settings = AudioAnalyzer.analyze_audio(orig_song_path, main_vocals_dereverb_path, instrumentals_path)
                
                # Apply new settings
                if 'rms_mix_rate' in new_settings: rms_mix_rate = new_settings['rms_mix_rate']
                if 'protect' in new_settings: protect = new_settings['protect']
                if 'index_rate' in new_settings: index_rate = new_settings['index_rate']
                if 'filter_radius' in new_settings: filter_radius = new_settings['filter_radius']
                if 'reverb_rm_size' in new_settings: reverb_rm_size = new_settings['reverb_rm_size']
                if 'reverb_wet' in new_settings: reverb_wet = new_settings['reverb_wet']
                if 'reverb_dry' in new_settings: reverb_dry = new_settings['reverb_dry']
                
                # Call callback to update GUI
                if auto_mode_callback:
                    auto_mode_callback(new_settings)
            except Exception as e:
                print(f"[Error] Auto Mode analysis failed: {e}")

        if vocal_noise_gate:
            display_progress(f'[~] Applying Vocal Rescue Chain (Highpass + Noise Gate {noise_gate_threshold}dB + Compressor) to clean vocals...', 0.4, is_webui, progress)
            import soundfile as sf
            from pedalboard import Pedalboard, NoiseGate, HighpassFilter, Compressor
            gate_audio, sr = sf.read(main_vocals_dereverb_path)
            # Create a professional vocal cleanup chain
            board = Pedalboard([
                HighpassFilter(cutoff_frequency_hz=85.0), # Remove muddy sub-bass reverb
                NoiseGate(threshold_db=noise_gate_threshold, ratio=10, attack_ms=2.0, release_ms=100.0), # Cut tails
                Compressor(threshold_db=-20.0, ratio=3.0, attack_ms=5.0, release_ms=50.0) # Even out syllables
            ])
            gate_audio = board(gate_audio, sr)
            gated_path = main_vocals_dereverb_path.replace('.wav', '_gated.wav')
            sf.write(gated_path, gate_audio, sr)
            main_vocals_dereverb_path = gated_path

        # Dynamic Pitch per Segment (Verse vs Chorus Router)
        if dynamic_pitch_rules and str(dynamic_pitch_rules).strip():
            display_progress('[~] Applying Verse vs Chorus dynamic pitch shifts...', 0.48, is_webui, progress)
            try:
                from harmony_and_pitch_router import apply_verse_chorus_dynamic_pitch
                main_vocals_dereverb_path = apply_verse_chorus_dynamic_pitch(main_vocals_dereverb_path, str(dynamic_pitch_rules).strip())
            except Exception as dyn_err:
                print(f"[Dynamic Pitch Hatasi] {dyn_err}")

        safe_song_name = safe_filename(os.path.splitext(os.path.basename(orig_song_path))[0])
        pitch_change = pitch_change * 12 + pitch_change_all
        ai_vocals_path = os.path.join(song_dir, f'{safe_song_name}_{voice_model}_{safe_filename(uvr_model).replace(" ", "")}_p{pitch_change}_i{index_rate}_fr{filter_radius}_rms{rms_mix_rate}_pro{protect}_{f0_method}{"" if f0_method != "mangio-crepe" else f"_{crepe_hop_length}"}.wav')
        ai_cover_path = os.path.join(song_dir, f'{safe_song_name} ({voice_model} Ver).{output_format}')

        if force_reprocess or (vocals_path is not None) or not os.path.exists(ai_vocals_path):
            display_progress('[~] Converting voice using RVC...', 0.5, is_webui, progress)
            voice_change(voice_model, main_vocals_dereverb_path, ai_vocals_path, pitch_change, f0_method, index_rate, filter_radius, rms_mix_rate, protect, crepe_hop_length, is_webui)

        # Working copy for post-processing chain (Keeps raw ai_vocals_path completely pristine and unpolluted!)
        import shutil
        ai_vocals_working_path = os.path.join(song_dir, f'{safe_song_name}_{voice_model}_working.wav')
        shutil.copyfile(ai_vocals_path, ai_vocals_working_path)

        # 1. Formant Resonance Shifter (Vocal Tract Resonance Scaling without shifting F0 pitch)
        f_shift = to_float(formant_shift, 0.0)
        if abs(f_shift) >= 0.1 and os.path.exists(ai_vocals_working_path):
            display_progress(f'[~] Applying Formant Resonance Shifter ({f_shift:+.1f} st)...', 0.68, is_webui, progress)
            try:
                import soundfile as sf
                from vocal_enhancer import shift_formants
                v_data, v_sr = sf.read(ai_vocals_working_path, dtype='float32')
                v_shifted = shift_formants(v_data, v_sr, formant_semitones=f_shift)
                sf.write(ai_vocals_working_path, v_shifted.T if v_shifted.ndim > 1 and v_shifted.shape[0] == 2 else v_shifted, v_sr)
            except Exception as fe:
                print(f"[Formant Shifter Hatasi] {fe}")

        # 2. Human Breath Passthrough Engine (Preserve & Reinject original breaths)
        if enable_breath_passthrough and os.path.exists(ai_vocals_working_path):
            display_progress('[~] Restoring natural human breaths (Breath Passthrough)...', 0.71, is_webui, progress)
            try:
                from vocal_enhancer import preserve_and_reinject_breaths
                ref_vocal = main_vocals_path if (main_vocals_path and os.path.exists(main_vocals_path)) else main_vocals_dereverb_path
                ai_vocals_working_path = preserve_and_reinject_breaths(ref_vocal, ai_vocals_working_path, progress_cb=lambda msg: display_progress(msg, 0.72, is_webui, progress))
            except Exception as bpe:
                print(f"[Breath Passthrough Hatasi] {bpe}")

        # 3. Smart De-Esser (5kHz - 9kHz harsh sibilance control)
        if enable_deesser and os.path.exists(ai_vocals_working_path):
            display_progress('[~] Applying Smart De-Esser (5kHz-9kHz Sibilance Control)...', 0.74, is_webui, progress)
            try:
                import soundfile as sf
                from vocal_chain import apply_deesser
                v_data, v_sr = sf.read(ai_vocals_working_path, dtype='float32')
                v_deessed = apply_deesser(v_data, v_sr)
                sf.write(ai_vocals_working_path, v_deessed.T if v_deessed.ndim > 1 and v_deessed.shape[0] == 2 else v_deessed, v_sr)
            except Exception as de:
                print(f"[De-Esser Hatasi] {de}")

        # 4. 4-Band Studio Parametric EQ (80Hz LowCut, 250Hz Warmth, 450Hz MudCut, 12kHz Air)
        if enable_studio_eq and os.path.exists(ai_vocals_working_path):
            display_progress('[~] Applying 4-Band Studio Parametric EQ (LowCut/Warmth/MudCut/Air)...', 0.76, is_webui, progress)
            try:
                import soundfile as sf
                from vocal_chain import apply_studio_parametric_eq
                v_data, v_sr = sf.read(ai_vocals_working_path, dtype='float32')
                v_eqed = apply_studio_parametric_eq(v_data, v_sr)
                sf.write(ai_vocals_working_path, v_eqed.T if v_eqed.ndim > 1 and v_eqed.shape[0] == 2 else v_eqed, v_sr)
            except Exception as eqe:
                print(f"[Studio EQ Hatasi] {eqe}")

        # 5. Stereo Vocal Doubler & Widener
        if enable_doubler and os.path.exists(ai_vocals_working_path):
            display_progress('[~] Applying Stereo Vocal Doubler & Widener...', 0.78, is_webui, progress)
            try:
                import soundfile as sf
                from vocal_chain import apply_vocal_doubler_widener
                v_data, v_sr = sf.read(ai_vocals_working_path, dtype='float32')
                v_doubled = apply_vocal_doubler_widener(v_data, v_sr)
                sf.write(ai_vocals_working_path, v_doubled.T if v_doubled.ndim > 1 and v_doubled.shape[0] == 2 else v_doubled, v_sr)
            except Exception as dbe:
                print(f"[Doubler Hatasi] {dbe}")

        # 6. AI Backing Vocals & Harmony Generator
        if generate_harmonies and os.path.exists(ai_vocals_working_path):
            display_progress(f'[~] Generating AI Harmonies ({harmony_mode})...', 0.80, is_webui, progress)
            try:
                import soundfile as sf
                from harmony_and_pitch_router import generate_backing_harmonies
                v_data, v_sr = sf.read(ai_vocals_working_path, dtype='float32')
                v_harm = generate_backing_harmonies(v_data, v_sr, harmony_mode=harmony_mode)
                sf.write(ai_vocals_working_path, v_harm.T if v_harm.ndim > 1 and v_harm.shape[0] == 2 else v_harm, v_sr)
            except Exception as he:
                print(f"[Harmonies Hatasi] {he}")

        display_progress('[~] Applying audio effects to Vocals...', 0.83, is_webui, progress)
        ai_vocals_mixed_path = add_audio_effects(ai_vocals_working_path, reverb_rm_size, reverb_wet, reverb_dry, reverb_damping)

        # 7. Dynamic Sidechain Pocket Ducking (1kHz - 3.5kHz)
        instrumentals_to_mix = instrumentals_path
        if enable_sidechain and instrumentals_path and os.path.exists(instrumentals_path) and os.path.exists(ai_vocals_mixed_path):
            display_progress('[~] Applying Dynamic Sidechain Ducking (1kHz-3.5kHz pocket)...', 0.87, is_webui, progress)
            try:
                import soundfile as sf
                from vocal_chain import apply_sidechain_ducking
                inst_data, inst_sr = sf.read(instrumentals_path, dtype='float32')
                voc_data, voc_sr = sf.read(ai_vocals_mixed_path, dtype='float32')
                if inst_sr == voc_sr:
                    ducked = apply_sidechain_ducking(inst_data, voc_data, inst_sr)
                    ducked_path = os.path.join(song_dir, f"{song_id}_inst_sidechained.wav")
                    sf.write(ducked_path, ducked.T if ducked.ndim > 1 and ducked.shape[0] == 2 else ducked, inst_sr)
                    instrumentals_to_mix = ducked_path
            except Exception as sde:
                print(f"[Sidechain Hatasi] {sde}")

        display_progress('[~] Combining AI Vocals and Instrumentals...', 0.90, is_webui, progress)
        ai_cover_path = combine_audio([ai_vocals_mixed_path, backup_vocals_path, instrumentals_to_mix], ai_cover_path, main_gain, backup_gain, inst_gain, output_format)

        # 8. Radio-Ready -14 LUFS Mastering & True Peak Limiter
        if enable_lufs_mastering and os.path.exists(ai_cover_path):
            display_progress('[~] Applying Broadcast Standard -14 LUFS Mastering & True Peak Limiter...', 0.94, is_webui, progress)
            try:
                import soundfile as sf
                from vocal_chain import apply_lufs_loudness_mastering
                cov_data, cov_sr = sf.read(ai_cover_path, dtype='float32')
                mastered = apply_lufs_loudness_mastering(cov_data, cov_sr, target_lufs=-14.0, true_peak_ceiling=-1.0)
                try:
                    sf.write(ai_cover_path, mastered.T if mastered.ndim > 1 and mastered.shape[0] == 2 else mastered, cov_sr)
                except PermissionError:
                    import time
                    base, ext = os.path.splitext(ai_cover_path)
                    ai_cover_path = f"{base}_mastered{ext}"
                    sf.write(ai_cover_path, mastered.T if mastered.ndim > 1 and mastered.shape[0] == 2 else mastered, cov_sr)
            except Exception as lfe:
                print(f"[LUFS Mastering Hatasi] {lfe}")

        if not keep_files:
            display_progress('[~] Removing intermediate audio files...', 0.97, is_webui, progress)
            intermediate_files = [vocals_path, main_vocals_path, ai_vocals_mixed_path, ai_vocals_working_path]
            if instrumentals_to_mix != instrumentals_path:
                intermediate_files.append(instrumentals_to_mix)

            for file in intermediate_files:
                if file and os.path.exists(file):
                    os.remove(file)

        # Root output copy: ensure direct accessibility in main output folder
        root_cover_copy = os.path.join(output_dir, os.path.basename(ai_cover_path))
        if os.path.abspath(ai_cover_path) != os.path.abspath(root_cover_copy):
            try:
                import shutil
                shutil.copyfile(ai_cover_path, root_cover_copy)
                print(f"[+] Parça ana çıktı klasörüne aktarıldı: {root_cover_copy}")
                ai_cover_path = root_cover_copy
            except PermissionError:
                try:
                    import time, shutil
                    b_name, b_ext = os.path.splitext(os.path.basename(ai_cover_path))
                    root_cover_copy = os.path.join(output_dir, f"{b_name}_{int(time.time())}{b_ext}")
                    shutil.copyfile(ai_cover_path, root_cover_copy)
                    print(f"[+] Parça yeni adla ana çıktı klasörüne kaydedildi: {root_cover_copy}")
                    ai_cover_path = root_cover_copy
                except Exception as pe_inner:
                    print(f"[Uyarı] Ana çıktı klasörüne kopyalama hatası: {pe_inner}")
            except Exception as copy_e:
                print(f"[Uyarı] Ana çıktı klasörüne kopyalama hatası: {copy_e}")

        print("\n" + "=" * 60)
        print("[+] BASARILI! AI Cover sarkisi tamamlandi!")
        print(f"[+] Dosya yolu: {ai_cover_path}")
        print("=" * 60 + "\n")
        return ai_cover_path

    except Exception as e:
        import traceback
        traceback.print_exc()

        raise_exception(str(e), is_webui)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a AI cover song in the song_output/id directory.', add_help=True)
    parser.add_argument('-i', '--song-input', type=str, required=True, help='Link to a YouTube video or the filepath to a local mp3/wav file to create an AI cover of')
    parser.add_argument('-dir', '--rvc-dirname', type=str, required=True, help='Name of the folder in the rvc_models directory containing the RVC model file and optional index file to use')
    parser.add_argument('-p', '--pitch-change', type=int, required=True, help='Change the pitch of AI Vocals only. Generally, use 1 for male to female and -1 for vice-versa. (Octaves)')
    parser.add_argument('-k', '--keep-files', action=argparse.BooleanOptionalAction, help='Whether to keep all intermediate audio files generated in the song_output/id directory, e.g. Isolated Vocals/Instrumentals')
    parser.add_argument('-ir', '--index-rate', type=float, default=0.5, help='A decimal number e.g. 0.5, used to reduce/resolve the timbre leakage problem. If set to 1, more biased towards the timbre quality of the training dataset')
    parser.add_argument('-fr', '--filter-radius', type=int, default=3, help='A number between 0 and 7. If >=3: apply median filtering to the harvested pitch results. The value represents the filter radius and can reduce breathiness.')
    parser.add_argument('-rms', '--rms-mix-rate', type=float, default=0.25, help="A decimal number e.g. 0.25. Control how much to use the original vocal's loudness (0) or a fixed loudness (1).")
    parser.add_argument('-palgo', '--pitch-detection-algo', type=str, default='rmvpe', help='Best option is rmvpe (clarity in vocals), then mangio-crepe (smoother vocals).')
    parser.add_argument('-hop', '--crepe-hop-length', type=int, default=128, help='If pitch detection algo is mangio-crepe, controls how often it checks for pitch changes in milliseconds. The higher the value, the faster the conversion and less risk of voice cracks, but there is less pitch accuracy. Recommended: 128.')
    parser.add_argument('-pro', '--protect', type=float, default=0.33, help='A decimal number e.g. 0.33. Protect voiceless consonants and breath sounds to prevent artifacts such as tearing in electronic music. Set to 0.5 to disable. Decrease the value to increase protection, but it may reduce indexing accuracy.')
    parser.add_argument('-mv', '--main-vol', type=int, default=0, help='Volume change for AI main vocals in decibels. Use -3 to decrease by 3 decibels and 3 to increase by 3 decibels')
    parser.add_argument('-bv', '--backup-vol', type=int, default=0, help='Volume change for backup vocals in decibels')
    parser.add_argument('-iv', '--inst-vol', type=int, default=0, help='Volume change for instrumentals in decibels')
    parser.add_argument('-pall', '--pitch-change-all', type=int, default=0, help='Change the pitch/key of vocals and instrumentals. Changing this slightly reduces sound quality')
    parser.add_argument('-rsize', '--reverb-size', type=float, default=0.15, help='Reverb room size between 0 and 1')
    parser.add_argument('-rwet', '--reverb-wetness', type=float, default=0.2, help='Reverb wet level between 0 and 1')
    parser.add_argument('-rdry', '--reverb-dryness', type=float, default=0.8, help='Reverb dry level between 0 and 1')
    parser.add_argument('-rdamp', '--reverb-damping', type=float, default=0.7, help='Reverb damping between 0 and 1')
    parser.add_argument('-oformat', '--output-format', type=str, default='mp3', help='Output format of audio file. mp3 for smaller file size, wav for best quality')
    args = parser.parse_args()

    rvc_dirname = args.rvc_dirname
    if not os.path.exists(os.path.join(rvc_models_dir, rvc_dirname)):
        raise Exception(f'The folder {os.path.join(rvc_models_dir, rvc_dirname)} does not exist.')

    cover_path = song_cover_pipeline(args.song_input, rvc_dirname, args.pitch_change, args.keep_files,
                                     main_gain=args.main_vol, backup_gain=args.backup_vol, inst_gain=args.inst_vol,
                                     index_rate=args.index_rate, filter_radius=args.filter_radius,
                                     rms_mix_rate=args.rms_mix_rate, f0_method=args.pitch_detection_algo,
                                     crepe_hop_length=args.crepe_hop_length, protect=args.protect,
                                     pitch_change_all=args.pitch_change_all,
                                     reverb_rm_size=args.reverb_size, reverb_wet=args.reverb_wetness,
                                     reverb_dry=args.reverb_dryness, reverb_damping=args.reverb_damping,
                                     output_format=args.output_format)
    print(f'[+] Cover generated at {cover_path}')


















