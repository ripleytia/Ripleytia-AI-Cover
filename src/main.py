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
import sox
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
    ydl_opts = {
        'format': 'bestaudio',
        'outtmpl': '%(title)s',
        'nocheckcertificate': True,
        'ignoreerrors': True,
        'no_warnings': True,
        'quiet': True,
        'extractaudio': True,
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3'}],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(link, download=True)
        download_path = ydl.prepare_filename(result, outtmpl='%(title)s.mp3')

    return download_path


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


def get_audio_paths(song_dir):
    orig_song_path = None
    instrumentals_path = None
    main_vocals_dereverb_path = None
    backup_vocals_path = None

    for file in os.listdir(song_dir):
        if file.endswith('_Instrumental.wav'):
            instrumentals_path = os.path.join(song_dir, file)
            orig_song_path = instrumentals_path.replace('_Instrumental', '')

        elif file.endswith('_Vocals_Main_DeReverb.wav'):
            main_vocals_dereverb_path = os.path.join(song_dir, file)

        elif file.endswith('_Vocals_Backup.wav'):
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
        tfm = sox.Transformer()
        tfm.pitch(pitch_change)
        y_shifted = tfm.build_array(input_array=y, sample_rate_in=sr)
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

    if uvr_model != "Default RVC":
        if uvr_model == "UVR5 Special":
            display_progress('[~] Separating Vocals using BS-Roformer-Viperx-1297...', 0.1, is_webui, progress)
            try:
                from audio_separator.separator import Separator
                import glob
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


            except ImportError:
                raise_exception("Gerekli kutuphaneler eksik. Lutfen 'pip install audio-separator' komutunu calistirin.", is_webui)

            sep1 = Separator(log_level=30, output_dir=song_output_dir, model_file_dir=r"C:\Users\Ripleytia\AppData\Local\Programs\Ultimate Vocal Remover\models\MDX_Net_Models", mdxc_params={"segment_size": 256, "batch_size": 1, "overlap": 4, "override_model_segment_size": False})
            sep1.load_model(model_filename='model_bs_roformer_ep_317_sdr_12.9755.ckpt')
            res1 = sep1.separate(orig_song_path)
            vocals_path = [os.path.join(song_output_dir, f) for f in res1 if 'Vocals' in f][0]

            display_progress('[~] Separating Instrumentals using MB-Roformer-Inst-v2...', 0.2, is_webui, progress)
            sep2 = Separator(log_level=30, output_dir=song_output_dir, model_file_dir=r"C:\Users\Ripleytia\AppData\Local\Programs\Ultimate Vocal Remover\models\MDX_Net_Models", mdxc_params={"segment_size": 256, "batch_size": 1, "overlap": 4, "override_model_segment_size": False})
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

    return orig_song_path, vocals_path, instrumentals_path, main_vocals_path, backup_vocals_path, main_vocals_dereverb_path


def voice_change(voice_model, vocals_path, output_path, pitch_change, f0_method, index_rate, filter_radius, rms_mix_rate, protect, crepe_hop_length, is_webui):
    rvc_model_path, rvc_index_path = get_rvc_model(voice_model, is_webui)
    device = 'cuda:0'
    config = Config(device, True)
    hubert_model = load_hubert(device, config.is_half, os.path.join(rvc_models_dir, 'hubert_base.pt'))
    cpt, version, net_g, tgt_sr, vc = get_vc(device, config.is_half, config, rvc_model_path)

    # convert main vocals
    rvc_infer(rvc_index_path, index_rate, vocals_path, output_path, pitch_change, f0_method, cpt, version, net_g, filter_radius, tgt_sr, rms_mix_rate, protect, crepe_hop_length, vc, hubert_model)
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
    main_vocal_audio.overlay(backup_vocal_audio).overlay(instrumental_audio).export(output_path, format=output_format)


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


def song_cover_pipeline(song_input, voice_model, pitch_change, keep_files,
                        is_webui=0, main_gain=0, backup_gain=0, inst_gain=0, index_rate=0.5, filter_radius=3,
                        rms_mix_rate=0.25, f0_method='rmvpe', crepe_hop_length=128, protect=0.33, pitch_change_all=0,
                        reverb_rm_size=0.15, reverb_wet=0.2, reverb_dry=0.8, reverb_damping=0.7, output_format='mp3',
                        aggressive_dereverb=False, uvr_model="Default RVC", vocal_noise_gate=False, noise_gate_threshold=-35, progress=gr.Progress()):
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
            paths = get_audio_paths(song_dir)

            # if any of the audio files aren't available or keep intermediate files, rerun preprocess
            if any(path is None for path in paths) or keep_files or uvr_model != "Default RVC" or aggressive_dereverb:
                orig_song_path, vocals_path, instrumentals_path, main_vocals_path, backup_vocals_path, main_vocals_dereverb_path = preprocess_song(song_input, mdx_model_params, song_id, is_webui, input_type, aggressive_dereverb=aggressive_dereverb, uvr_model=uvr_model, progress=progress)
            else:
                orig_song_path, instrumentals_path, main_vocals_dereverb_path, backup_vocals_path = paths

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

        safe_song_name = safe_filename(os.path.splitext(os.path.basename(orig_song_path))[0])
        pitch_change = pitch_change * 12 + pitch_change_all
        ai_vocals_path = os.path.join(song_dir, f'{safe_song_name}_{voice_model}_p{pitch_change}_i{index_rate}_fr{filter_radius}_rms{rms_mix_rate}_pro{protect}_{f0_method}{"" if f0_method != "mangio-crepe" else f"_{crepe_hop_length}"}.wav')
        ai_cover_path = os.path.join(song_dir, f'{safe_song_name} ({voice_model} Ver).{output_format}')

        if not os.path.exists(ai_vocals_path):
            display_progress('[~] Converting voice using RVC...', 0.5, is_webui, progress)
            voice_change(voice_model, main_vocals_dereverb_path, ai_vocals_path, pitch_change, f0_method, index_rate, filter_radius, rms_mix_rate, protect, crepe_hop_length, is_webui)

        display_progress('[~] Applying audio effects to Vocals...', 0.8, is_webui, progress)
        ai_vocals_mixed_path = add_audio_effects(ai_vocals_path, reverb_rm_size, reverb_wet, reverb_dry, reverb_damping)


        display_progress('[~] Combining AI Vocals and Instrumentals...', 0.9, is_webui, progress)
        combine_audio([ai_vocals_mixed_path, backup_vocals_path, instrumentals_path], ai_cover_path, main_gain, backup_gain, inst_gain, output_format)

        if not keep_files:
            display_progress('[~] Removing intermediate audio files...', 0.95, is_webui, progress)
            intermediate_files = [vocals_path, main_vocals_path, ai_vocals_mixed_path]

            for file in intermediate_files:
                if file and os.path.exists(file):
                    os.remove(file)

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












