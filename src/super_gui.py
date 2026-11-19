# ==============================================================================
# Ripleytia Gothic AI Sound Studio - V4.0 Studio Ultimate Edition
# ==============================================================================

import os
import sys
import threading
import queue
import json
import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog
import subprocess
import shutil
from PIL import Image, ImageTk
import numpy as np

# Ensure src is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import song_cover_pipeline
from youtube_searcher import (
    search_youtube,
    download_youtube_audio,
    get_cached_thumbnail,
    is_playlist_url,
    get_playlist_videos,
)
from dataset_builder import (
    build_automatic_dataset,
    calculate_recommended_epochs,
    format_duration,
    sanitize_name,
    DATASETS_ROOT,
    APPLIO_DATASETS_DIR,
)
from voice_trainer import (
    TRAINING_PRESETS,
    PRETRAIN_MAP,
    run_full_voice_training_pipeline,
    stop_active_training,
    deploy_trained_model_to_studio,
    APPLIO_LOGS_DIR,
    generate_training_loss_plot,
    run_quick_audio_test,
)
from audio_player import AudioPlayerWidget
from pitch_recommender import analyze_pitch_and_recommend
from cover_history import add_history_entry, get_history_entries, clear_history, delete_history_entry
from batch_cover_manager import BatchQueueManager
from model_downloader import search_curated_models, download_and_extract_model
from vocal_analyzer import analyze_vocal_file
from audio_comparator import compare_audio_tracks
from style_advisor import advise_style_parameters
from quality_evaluator import evaluate_clone_quality
from dataset_inspector import inspect_dataset_directory
from lyrics_translator import fetch_lyrics, translate_lyrics_text
from notifier import send_windows_notification

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rvc_models_dir = os.path.join(BASE_DIR, 'rvc_models')
mdxnet_models_dir = os.path.join(BASE_DIR, 'mdxnet_models')
output_dir = os.path.join(BASE_DIR, 'song_output')
os.makedirs(output_dir, exist_ok=True)
logo_path = os.path.join(BASE_DIR, 'logo.jpg')
settings_path = os.path.join(BASE_DIR, 'settings.json')


class RedirectText:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, string):
        try:
            self.text_widget.after(0, self._safe_insert, string)
        except Exception:
            pass

    def _safe_insert(self, string):
        try:
            self.text_widget.insert("end", string)
            self.text_widget.see("end")
        except Exception:
            pass

    def flush(self):
        pass


def get_current_models(models_dir):
    if not os.path.exists(models_dir):
        return []
    return [
        item for item in os.listdir(models_dir)
        if item not in ['hubert_base.pt', 'MODELS.txt', 'public_models.json', 'rmvpe.pt']
    ]


def get_mdx_models():
    if not os.path.exists(mdxnet_models_dir):
        return []
    return [
        item for item in os.listdir(mdxnet_models_dir)
        if item.endswith(".onnx") and item not in ['UVR_MDXNET_KARA_2.onnx', 'Reverb_HQ_By_FoxJoy.onnx']
    ]


def get_available_datasets():
    found = []
    for root_dir in [DATASETS_ROOT, APPLIO_DATASETS_DIR]:
        if os.path.exists(root_dir):
            for d in sorted(os.listdir(root_dir)):
                full_p = os.path.join(root_dir, d)
                if os.path.isdir(full_p) and not d.startswith("_"):
                    if any(f.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a")) for f in os.listdir(full_p)):
                        if full_p not in found:
                            found.append(full_p)
    return found


THEMES = {
    "Gothic Purple": {
        "primary": "#5D3FD3",
        "hover": "#7B68EE",
        "bg": "#121212",
        "card": "#181824",
        "accent": "#9966FF"
    },
    "Cyber Cyan": {
        "primary": "#00d2d3",
        "hover": "#0abde3",
        "bg": "#0b1320",
        "card": "#101e33",
        "accent": "#48dbfb"
    },
    "Midnight OLED": {
        "primary": "#ff3838",
        "hover": "#ff5252",
        "bg": "#000000",
        "card": "#151515",
        "accent": "#ff7675"
    }
}

translations = {
    "tr": {
        "title": "Ripleytia AI Sound Studio - V4.0 Ultimate Edition",
        "lang_switch": "Dil",
        "refresh": "Yenile",
        "save_settings": "Ayarları Kaydet",
        "mode_cover": "🎵 AI Cover",
        "mode_batch": "📑 Toplu Cover",
        "mode_history": "🎧 Çalar & Geçmiş",
        "mode_analysis": "📊 Vokal Analiz",
        "mode_models": "🌐 Model İndirici",
        "mode_dataset": "📦 Dataset Hazırlayıcı",
        "mode_train": "🧠 Ses Eğitimi",
        "yt_search_toggle": "🔍 YouTube'da Ara",
        "yt_panel_title": "🎵 YouTube Müzik / Video Arama ve Hızlı Cover",
        "yt_search_placeholder": "Şarkı, sanatçı adı veya YouTube linki (Örn: Tarkan, Sil Baştan)...",
        "yt_btn_search": "🔍 Ara",
        "yt_btn_download_select": "📥 İndir ve Seç",
        "yt_btn_download_cover": "⚡ İndir & Cover Yap",
        "yt_searching": "🔍 YouTube'da aranıyor, lütfen bekleyin...",
        "yt_search_hint": "YouTube'da aramak için yukarıya şarkı adı yazıp 'Ara' butonuna veya Enter'a basın.",
        "yt_no_results": "Sonuç bulunamadı. Lütfen farklı kelimelerle tekrar deneyin.",
        "yt_downloading": "📥 Ses indiriliyor...",
        "yt_download_done": "✅ İndirildi ve şarkı girişine eklendi!",
        "model_select": "1. Ses Modelini Seçin",
        "input_select": "2. Şarkı (YouTube Linki veya Yerel Dosya)",
        "output_select": "Çıktı Klasörü (Varsayılan için boş bırakın)",
        "browse": "Dosya Seç",
        "browse_dir": "Klasör Seç",
        "pipeline_select": "3. Vokal Ayırma / UVR5 Modeli Seçimi",
        "upload_model": "Yeni Model Yükle",
        "tab_tuning": "Ses & Ton Ayarları",
        "tab_mix": "Ses Karıştırma (Mix)",
        "tab_fx": "Efektler (Reverb)",
        "tab_adv": "Gelişmiş Ayarlar",
        "tab_mastering": "🎛️ Stüdyo & Mastering",
        "vocal_pitch": "Vokal Ton (Pitch) Değişimi (Oktav)",
        "overall_pitch": "Vokal Ton Değişimi (Yarım Sesler)",
        "main_gain": "Ana Vokal Ses Seviyesi (dB)",
        "backup_gain": "Arka Vokal Ses Seviyesi (dB)",
        "inst_gain": "Enstrüman Ses Seviyesi (dB)",
        "mute_backup": "Arka Vokalleri Tamamen Sustur",
        "room_size": "Oda Büyüklüğü (Yankı Hacmi)",
        "wetness": "Islaklık (Yankı Yoğunluğu)",
        "dryness": "Kuruluk (Ham Ses Yoğunluğu)",
        "damping": "Sönümleme (Tiz Ses Emme)",
        "index_rate": "İndeks Oranı (Model Karakteri vs Orijinal)",
        "filter_radius": "Filtre Yarıçapı (Nefesliliği Azaltma)",
        "rms_mix": "RMS Mix Oranı (Ses Gürlüğü Taklidi)",
        "protect": "Koruma Oranı (Sessiz Harfleri Koruma)",
        "f0_method": "F0 Metodu (Ses Tonu Algılama)",
        "crepe_hop": "Crepe Atlama Uzunluğu (Hop Length)",
        "noise_gate": "Vokal Noise Gate (Yankı Kuyruğu Kesici)",
        "noise_gate_thresh": "Noise Gate Eşiği (dB)",
        "agg_dereverb": "Agresif Yankı Temizleme (Roformer)",
        "keep_files": "Ara Dosyaları Sakla (Vokal/Enstrüman)",
        "force_reprocess": "Sıfırdan Vokal Ayır (Önbelleği Yoksay)",
        "no_cache": " (Daha Önce İşlenmemiş)",
        "output_format": "Çıktı Formatı",
        "generate": "Yapay Zeka Cover Üret",
        "auto_mode": "Otomatik Ses Modu (Şarkıyı analiz eder, ayarları otomatik yapar)",
        "open_folder": "Çıktı Klasörünü Aç",
        "play_last": "Son Çıktıyı Çal",
        "clear_log": "Konsolu Temizle",
        "logs": "Stüdyo İşlem Kayıtları (Log) Konsolu:",
        "generating": "İşleniyor... Lütfen bekleyin",
        "settings_saved": "[Sistem] Ayarlar başarıyla kaydedildi!"
    },
    "en": {
        "title": "Ripleytia AI Sound Studio - V4.0 Ultimate Edition",
        "lang_switch": "Language",
        "refresh": "Refresh",
        "save_settings": "Save Settings",
        "mode_cover": "🎵 AI Cover",
        "mode_batch": "📑 Batch Queue",
        "mode_history": "🎧 Player & History",
        "mode_analysis": "📊 Vocal Lab",
        "mode_models": "🌐 Model Downloader",
        "mode_dataset": "📦 Dataset Builder",
        "mode_train": "🧠 Voice Trainer",
        "yt_search_toggle": "🔍 Search on YouTube",
        "yt_panel_title": "🎵 YouTube Search & Quick AI Cover",
        "yt_search_placeholder": "Enter song, artist name or YouTube URL...",
        "yt_btn_search": "🔍 Search",
        "yt_btn_download_select": "📥 Download & Select",
        "yt_btn_download_cover": "⚡ Download & Cover Now",
        "yt_searching": "🔍 Searching YouTube...",
        "yt_search_hint": "Type a song or artist name above and press Enter or click Search.",
        "yt_no_results": "No videos found. Please try with different keywords.",
        "yt_downloading": "📥 Downloading audio...",
        "yt_download_done": "✅ Downloaded and set as input song!",
        "model_select": "1. Select Voice Model",
        "input_select": "2. Input Song (YouTube Link or Local File)",
        "output_select": "Output Directory (Leave empty for default)",
        "browse": "Browse File",
        "browse_dir": "Select Folder",
        "pipeline_select": "3. Vocal Extraction / UVR5 Pipeline",
        "upload_model": "Upload New Model",
        "tab_tuning": "Pitch & Tuning",
        "tab_mix": "Audio Mixing",
        "tab_fx": "Effects (Reverb)",
        "tab_adv": "Advanced Settings",
        "tab_mastering": "🎛️ Studio & Mastering",
        "vocal_pitch": "Vocal Pitch Change (Octaves)",
        "overall_pitch": "Vocal Pitch Change (Semitones)",
        "main_gain": "Main Vocals Gain (dB)",
        "backup_gain": "Backup Vocals Gain (dB)",
        "inst_gain": "Instrumentals Gain (dB)",
        "mute_backup": "Mute Backup Vocals",
        "room_size": "Room Size",
        "wetness": "Wetness (Reverb Level)",
        "dryness": "Dryness (Dry Vocal Level)",
        "damping": "Damping",
        "index_rate": "Index Rate (Timbre vs Original)",
        "filter_radius": "Filter Radius",
        "rms_mix": "RMS Mix Rate",
        "protect": "Protect Rate",
        "f0_method": "F0 Method",
        "crepe_hop": "Crepe Hop Length",
        "noise_gate": "Vocal Noise Gate",
        "noise_gate_thresh": "Noise Gate Threshold (dB)",
        "agg_dereverb": "Aggressive DeReverb",
        "keep_files": "Keep Intermediate Files",
        "force_reprocess": "Force Re-Extract Vocals",
        "no_cache": " (No Cache Found)",
        "output_format": "Output Format",
        "generate": "Generate AI Cover",
        "auto_mode": "Auto Audio Mode",
        "open_folder": "Open Output Folder",
        "play_last": "Play Last Output",
        "clear_log": "Clear Console",
        "logs": "Studio Process Console:",
        "generating": "Generating... Please wait",
        "settings_saved": "[System] Settings saved successfully!"
    },
    "ja": {
        "title": "Ripleytia AI Sound Studio - V4.0 アルティメット版",
        "lang_switch": "言語",
        "refresh": "更新",
        "save_settings": "設定保存",
        "mode_cover": "🎵 AIカバー",
        "mode_batch": "📑 一括キュー",
        "mode_history": "🎧 再生・履歴",
        "mode_analysis": "📊 音声分析",
        "mode_models": "🌐 モデルDL",
        "mode_dataset": "📦 データセット",
        "mode_train": "🧠 音声学習",
        "yt_search_toggle": "🔍 YouTube検索",
        "yt_panel_title": "🎵 YouTube音楽検索＆即時AIカバー",
        "yt_search_placeholder": "曲名、アーティスト名、またはURLを入力...",
        "yt_btn_search": "🔍 検索",
        "yt_btn_download_select": "📥 ダウンロード",
        "yt_btn_download_cover": "⚡ DL＆即座にカバー",
        "yt_searching": "🔍 YouTube検索中...",
        "yt_search_hint": "曲名を入力して検索ボタンを押してください。",
        "yt_no_results": "見つかりませんでした。",
        "yt_downloading": "📥 音声をダウンロード中...",
        "yt_download_done": "✅ ダウンロード完了！",
        "model_select": "1. 音声モデルの選択",
        "input_select": "2. 入力音源 (YouTubeまたはローカル)",
        "output_select": "出力先フォルダ",
        "browse": "ファイル参照",
        "browse_dir": "フォルダ選択",
        "pipeline_select": "3. UVR5ボーカル分離モデル",
        "upload_model": "モデル追加",
        "tab_tuning": "ピッチ設定",
        "tab_mix": "音量ミキシング",
        "tab_fx": "リバーブ効果",
        "tab_adv": "詳細設定",
        "tab_mastering": "🎛️ スタジオ＆マスタリング",
        "vocal_pitch": "ピッチ変更 (オクターブ)",
        "overall_pitch": "ピッチ微調整 (半音)",
        "main_gain": "メインボーカル音量",
        "backup_gain": "バックボーカル音量",
        "inst_gain": "伴奏音量",
        "mute_backup": "バックボーカルをミュート",
        "room_size": "ルームサイズ",
        "wetness": "リバーブ量",
        "dryness": "ドライ音量",
        "damping": "減衰",
        "index_rate": "インデックス率",
        "filter_radius": "フィルタ半径",
        "rms_mix": "音圧模倣率",
        "protect": "子音保護率",
        "f0_method": "ピッチ検出法",
        "crepe_hop": "Crepeホップ長",
        "noise_gate": "ノイズゲート",
        "noise_gate_thresh": "ゲート閾値 (dB)",
        "agg_dereverb": "高度な残響除去",
        "keep_files": "中間ファイルを保持",
        "force_reprocess": "再抽出を強制",
        "no_cache": " (キャッシュなし)",
        "output_format": "出力形式",
        "generate": "AIカバー生成開始",
        "auto_mode": "自動オーディオモード",
        "open_folder": "出力フォルダを開く",
        "play_last": "再生",
        "clear_log": "ログ消去",
        "logs": "スタジオコンソールログ:",
        "generating": "処理中... お待ちください",
        "settings_saved": "[システム] 設定が保存されました！"
    },
    "ko": {
        "title": "Ripleytia AI Sound Studio - V4.0 얼티밋 에디션",
        "lang_switch": "언어",
        "refresh": "새로고침",
        "save_settings": "설정 저장",
        "mode_cover": "🎵 AI 커버",
        "mode_batch": "📑 일괄 대기열",
        "mode_history": "🎧 재생 및 기록",
        "mode_analysis": "📊 보컬 랩",
        "mode_models": "🌐 모델 다운로더",
        "mode_dataset": "📦 데이터셋 빌더",
        "mode_train": "🧠 음성 모델 학습",
        "yt_search_toggle": "🔍 유튜브 검색",
        "yt_panel_title": "🎵 유튜브 검색 및 빠른 AI 커버",
        "yt_search_placeholder": "노래 제목, 아티스트 또는 URL 입력...",
        "yt_btn_search": "🔍 검색",
        "yt_btn_download_select": "📥 다운로드 후 선택",
        "yt_btn_download_cover": "⚡ 다운로드 & 즉시 커버",
        "yt_searching": "🔍 유튜브 검색 중...",
        "yt_search_hint": "노래 제목을 입력하고 엔터를 누르세요.",
        "yt_no_results": "검색 결과가 없습니다.",
        "yt_downloading": "📥 오디오 다운로드 중...",
        "yt_download_done": "✅ 다운로드 완료 및 입력 곡 설정!",
        "model_select": "1. 음성 모델 선택",
        "input_select": "2. 노래 입력 (유튜브 또는 로컬 파일)",
        "output_select": "출력 디렉터리",
        "browse": "파일 찾기",
        "browse_dir": "폴더 선택",
        "pipeline_select": "3. UVR5 보컬 분리 모델",
        "upload_model": "새 모델 업로드",
        "tab_tuning": "음정 및 피치",
        "tab_mix": "오디오 믹싱",
        "tab_fx": "리버브 효과",
        "tab_adv": "고급 설정",
        "tab_mastering": "🎛️ 스튜디오 및 마스터링",
        "vocal_pitch": "피치 변경 (옥타브)",
        "overall_pitch": "피치 미세조정 (반음)",
        "main_gain": "메인 보컬 볼륨",
        "backup_gain": "백 보컬 볼륨",
        "inst_gain": "반주 볼륨",
        "mute_backup": "백 보컬 음소거",
        "room_size": "공간 크기",
        "wetness": "리버브 강도",
        "dryness": "원음 강도",
        "damping": "감쇠",
        "index_rate": "인덱스 비율",
        "filter_radius": "필터 반경",
        "rms_mix": "음량 모방 비율",
        "protect": "자음 보호 비율",
        "f0_method": "F0 피치 검출법",
        "crepe_hop": "Crepe 홉 길이",
        "noise_gate": "노이즈 게이트",
        "noise_gate_thresh": "게이트 임계값 (dB)",
        "agg_dereverb": "강력한 리버브 제거",
        "keep_files": "중간 파일 보존",
        "force_reprocess": "재추출 강제",
        "no_cache": " (캐시 없음)",
        "output_format": "출력 형식",
        "generate": "AI 커버 생성 시작",
        "auto_mode": "자동 오디오 모드",
        "open_folder": "출력 폴더 열기",
        "play_last": "마지막 출력 재생",
        "clear_log": "로그 지우기",
        "logs": "스튜디오 처리 로그:",
        "generating": "처리 중... 잠시 기다려주세요",
        "settings_saved": "[시스템] 설정이 저장되었습니다!"
    }
}

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")
PRIMARY_COLOR = "#5D3FD3"
HOVER_COLOR = "#7B68EE"


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.lang = "tr"
        self.ui_queue = queue.Queue()
        self._check_ui_queue()
        self.title(translations[self.lang]["title"])
        self.geometry("1120x990")

        # Configure layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Set Window and Taskbar Icon
        ico_path = os.path.join(BASE_DIR, 'logo.ico')
        if os.path.exists(ico_path):
            try:
                self.iconbitmap(ico_path)
            except Exception:
                pass

        self.main_frame = ctk.CTkScrollableFrame(self, fg_color="#121212")
        self.main_frame.grid(row=0, column=0, sticky="nsew")

        # --- V4 Multi-feature Managers ---
        self.batch_manager = BatchQueueManager()
        self.theme_name_var = ctk.StringVar(value="Gothic Purple")
        self.compare_model_var = ctk.StringVar(value="")
        self.enable_compare_var = ctk.BooleanVar(value=False)

        # --- Cover Variables ---
        self.model_var = ctk.StringVar()
        self.pipeline_var = ctk.StringVar(value="UVR5 Special 2")
        self.song_input_var = ctk.StringVar()
        self.output_dir_var = ctk.StringVar(value=output_dir)
        self.pitch_var = tk.DoubleVar(value=0.0)
        self.pitch_all_var = tk.DoubleVar(value=0.0)
        self.main_gain_var = tk.IntVar(value=0)
        self.backup_gain_var = tk.IntVar(value=0)
        self.inst_gain_var = tk.IntVar(value=0)
        self.mute_backup_var = ctk.BooleanVar(value=True)
        self.reverb_size_var = tk.DoubleVar(value=0.15)
        self.reverb_wet_var = tk.DoubleVar(value=0.2)
        self.reverb_dry_var = tk.DoubleVar(value=0.8)
        self.reverb_damp_var = tk.DoubleVar(value=0.7)
        self.index_rate_var = tk.DoubleVar(value=0.5)
        self.filter_radius_var = tk.IntVar(value=3)
        self.rms_mix_var = tk.DoubleVar(value=0.25)
        self.protect_var = tk.DoubleVar(value=0.33)
        self.f0_method_var = ctk.StringVar(value="rmvpe")
        self.crepe_hop_var = tk.IntVar(value=128)
        self.noise_gate_var = ctk.BooleanVar(value=False)
        self.noise_gate_thresh_var = tk.IntVar(value=-35)
        self.aggressive_dereverb_var = ctk.BooleanVar(value=True)
        self.keep_files_var = ctk.BooleanVar(value=False)
        self.force_reprocess_var = ctk.BooleanVar(value=False)
        self.output_format_var = ctk.StringVar(value="mp3")

        # --- Studio Mastering & Vocal Chain Variables ---
        self.deesser_var = ctk.BooleanVar(value=True)
        self.studio_eq_var = ctk.BooleanVar(value=True)
        self.sidechain_var = ctk.BooleanVar(value=True)
        self.lufs_master_var = ctk.BooleanVar(value=True)
        self.breath_passthrough_var = ctk.BooleanVar(value=True)
        self.doubler_var = ctk.BooleanVar(value=False)
        self.harmonies_var = ctk.BooleanVar(value=False)
        self.harmony_mode_var = ctk.StringVar(value="3rd_and_5th (+Maj 3rd & 5th)")
        self.formant_shift_var = tk.DoubleVar(value=0.0)
        self.dynamic_pitch_rules_var = ctk.StringVar(value="")

        # --- Batch View Variables ---
        self.batch_input_var = ctk.StringVar()
        self.batch_model_var = ctk.StringVar()
        self.batch_pitch_var = tk.DoubleVar(value=0.0)
        self.batch_night_mode_var = ctk.BooleanVar(value=True)

        # --- Vocal Analysis Variables ---
        self.analysis_file_var = ctk.StringVar()

        # --- Dataset Builder Variables ---
        self.ds_name_var = ctk.StringVar(value="Yeni_Ses_Modelim")
        self.ds_local_files = []
        self.ds_clean_mode_var = ctk.StringVar(value="UVR5 Special 2 (Ultra Kalite)")
        self.ds_remove_backups_var = ctk.BooleanVar(value=True)
        self.ds_apply_dereverb_var = ctk.BooleanVar(value=True)
        self.ds_sample_rate_var = ctk.StringVar(value="48000")
        self.ds_silence_db_var = tk.IntVar(value=35)
        self.ds_max_minutes_var = tk.IntVar(value=0)
        self.ds_auto_send_train_var = ctk.BooleanVar(value=True)
        self.ds_auto_start_train_var = ctk.BooleanVar(value=False)

        # --- Voice Training Variables ---
        self.tr_model_name_var = ctk.StringVar(value="Yeni_Ses_Modelim")
        self.tr_dataset_path_var = ctk.StringVar(value="")
        self.tr_preset_var = ctk.StringVar(value="Dengeli (Önerilen - Stüdyo Standart)")
        self.tr_sample_rate_var = ctk.StringVar(value="40000")
        self.tr_vocoder_var = ctk.StringVar(value="HiFi-GAN")
        self.tr_f0_method_var = ctk.StringVar(value="rmvpe")
        self.tr_embedder_var = ctk.StringVar(value="contentvec")
        self.tr_pretrain_var = ctk.StringVar(value="Standart Applio (Varsayılan)")
        self.tr_batch_size_var = tk.IntVar(value=6)
        self.tr_total_epoch_var = tk.IntVar(value=250)
        self.tr_save_every_var = tk.IntVar(value=10)
        self.tr_cut_mode_var = ctk.StringVar(value="Automatic")
        self.tr_chunk_len_var = tk.DoubleVar(value=3.0)
        self.tr_overlap_len_var = tk.DoubleVar(value=0.3)
        self.tr_norm_mode_var = ctk.StringVar(value="post")
        self.tr_process_fx_var = ctk.BooleanVar(value=False)
        self.tr_noise_red_var = ctk.BooleanVar(value=False)
        self.tr_include_mutes_var = tk.IntVar(value=2)
        self.tr_index_algo_var = ctk.StringVar(value="Auto")
        self.tr_save_only_latest_var = ctk.BooleanVar(value=True)
        self.tr_save_every_weights_var = ctk.BooleanVar(value=True)
        self.tr_cache_gpu_var = ctk.BooleanVar(value=False)
        self.tr_checkpointing_var = ctk.BooleanVar(value=False)
        self.tr_cleanup_fresh_var = ctk.BooleanVar(value=True)
        self.tr_auto_deploy_var = ctk.BooleanVar(value=True)
        self.tr_cpu_cores_var = tk.IntVar(value=min(os.cpu_count() or 8, 16))

        # --- Header ---
        self.header_frame = ctk.CTkFrame(self.main_frame, fg_color="#181824", corner_radius=15, border_color="#5D3FD3", border_width=1)
        self.header_frame.pack(fill="x", pady=(10, 5), padx=10)

        if os.path.exists(logo_path):
            try:
                img = Image.open(logo_path)
                self.logo_image = ctk.CTkImage(light_image=img, dark_image=img, size=(75, 75))
                ctk.CTkLabel(self.header_frame, image=self.logo_image, text="").pack(side="left", padx=15, pady=10)
            except Exception:
                pass

        title_col = ctk.CTkFrame(self.header_frame, fg_color="transparent")
        title_col.pack(side="left", padx=5, pady=12)
        self.header_title = ctk.CTkLabel(
            title_col,
            text="Ripleytia AI Sound Studio V4.0",
            font=("Century Gothic", 24, "bold"),
            text_color=PRIMARY_COLOR,
        )
        self.header_title.pack(anchor="w")
        self.header_subtitle = ctk.CTkLabel(
            title_col,
            text="Cover  •  Toplu Kuyruk  •  Vokal Lab  •  Model İndirici  •  Otomatik Dataset  •  Ses Eğitimi",
            font=("Arial", 11),
            text_color="#AAA",
        )
        self.header_subtitle.pack(anchor="w")

        # Top Right Controls: Theme & Language & Save Settings
        right_header = ctk.CTkFrame(self.header_frame, fg_color="transparent")
        right_header.pack(side="right", padx=10)

        # Theme Selector
        ctk.CTkLabel(right_header, text="🎨", font=("Arial", 13)).pack(side="left", padx=2)
        self.theme_menu = ctk.CTkOptionMenu(
            right_header,
            values=["Gothic Purple", "Cyber Cyan", "Midnight OLED"],
            variable=self.theme_name_var,
            width=120,
            height=30,
            command=self.change_theme,
            fg_color="#2A2A3E",
            button_color="#3D3D58"
        )
        self.theme_menu.pack(side="left", padx=5)

        # Language Selector
        self.lang_menu = ctk.CTkOptionMenu(
            right_header,
            values=["🇹🇷 TR", "🇬🇧 EN", "🇯🇵 JA", "🇰🇷 KO"],
            width=90,
            height=30,
            command=self.on_lang_select,
            fg_color="#2A2A3E",
            button_color="#3D3D58"
        )
        self.lang_menu.pack(side="left", padx=5)

        self.btn_save_settings = ctk.CTkButton(
            right_header,
            text=translations[self.lang]["save_settings"],
            command=self.save_settings,
            width=110,
            height=30,
            fg_color="#333",
            hover_color="#555",
        )
        self.btn_save_settings.pack(side="left", padx=5)

        # --- Top Studio Mode Navigation Bar ---
        self.mode_bar_frame = ctk.CTkFrame(self.main_frame, fg_color="#181824", corner_radius=12)
        self.mode_bar_frame.pack(fill="x", pady=6, padx=10)

        self.studio_mode_var = ctk.StringVar(value="cover")

        def make_nav_btn(text_key, mode_val):
            return ctk.CTkButton(
                self.mode_bar_frame,
                text=translations[self.lang][text_key],
                font=("Century Gothic", 13, "bold"),
                height=38,
                fg_color=PRIMARY_COLOR if mode_val == "cover" else "#2B2B2B",
                hover_color=HOVER_COLOR,
                command=lambda: self.switch_studio_mode(mode_val),
            )

        self.mode_btn_cover = make_nav_btn("mode_cover", "cover")
        self.mode_btn_cover.pack(side="left", expand=True, fill="x", padx=3, pady=5)

        self.mode_btn_batch = make_nav_btn("mode_batch", "batch")
        self.mode_btn_batch.pack(side="left", expand=True, fill="x", padx=3, pady=5)

        self.mode_btn_history = make_nav_btn("mode_history", "history")
        self.mode_btn_history.pack(side="left", expand=True, fill="x", padx=3, pady=5)

        self.mode_btn_analysis = make_nav_btn("mode_analysis", "analysis")
        self.mode_btn_analysis.pack(side="left", expand=True, fill="x", padx=3, pady=5)

        self.mode_btn_models = make_nav_btn("mode_models", "models")
        self.mode_btn_models.pack(side="left", expand=True, fill="x", padx=3, pady=5)

        self.mode_btn_dataset = make_nav_btn("mode_dataset", "dataset")
        self.mode_btn_dataset.pack(side="left", expand=True, fill="x", padx=3, pady=5)

        self.mode_btn_train = make_nav_btn("mode_train", "train")
        self.mode_btn_train.pack(side="left", expand=True, fill="x", padx=3, pady=5)

        # --- Container Frames for the Studio Views ---
        self.cover_view_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.batch_view_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.history_view_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.vocal_analysis_view_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.models_view_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.dataset_view_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.train_view_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")

        self._build_cover_view()
        self._build_batch_view()
        self._build_history_view()
        self._build_vocal_analysis_view()
        self._build_models_view()
        self._build_dataset_view()
        self._build_train_view()

        # Show Cover View by default
        self.cover_view_frame.pack(fill="x")

        # Load Settings after defining widgets
        self.load_settings()

        # --- Shared Tools & Log Console at Bottom ---
        self.tools_frame = ctk.CTkFrame(self.main_frame, fg_color="#181824", corner_radius=10)
        self.tools_frame.pack(fill="x", pady=5, padx=10)

        self.btn_open_folder = ctk.CTkButton(
            self.tools_frame,
            text=translations[self.lang]["open_folder"],
            command=self.open_output_folder,
            fg_color="#333",
            hover_color="#555",
        )
        self.btn_open_folder.pack(side="left", padx=10, pady=10)

        self.btn_open_datasets = ctk.CTkButton(
            self.tools_frame,
            text="📂 Dataset Klasörünü Aç",
            command=lambda: os.startfile(DATASETS_ROOT) if os.path.exists(DATASETS_ROOT) else None,
            fg_color="#333",
            hover_color="#555",
        )
        self.btn_open_datasets.pack(side="left", padx=10, pady=10)

        self.play_btn = ctk.CTkButton(
            self.tools_frame,
            text=translations[self.lang]["play_last"],
            command=self.play_output,
            state="disabled",
            fg_color="#333",
            hover_color="#555",
        )
        self.play_btn.pack(side="left", padx=10, pady=10)

        self.btn_clear_log = ctk.CTkButton(
            self.tools_frame,
            text=translations[self.lang]["clear_log"],
            command=self.clear_logs,
            fg_color="#8b0000",
            hover_color="#5a0000",
        )
        self.btn_clear_log.pack(side="right", padx=10, pady=10)

        self.last_output_path = None

        # --- Log Console ---
        self.lbl_logs = ctk.CTkLabel(self.main_frame, text=translations[self.lang]["logs"], font=("Arial", 12, "bold"))
        self.lbl_logs.pack(anchor="w", padx=10, pady=(10, 0))

        self.log_textbox = ctk.CTkTextbox(
            self.main_frame,
            height=140,
            fg_color="#0D0D15",
            text_color="#B0B0D0",
            font=("Consolas", 11),
            corner_radius=8,
        )
        self.log_textbox.pack(fill="x", padx=10, pady=(5, 15))

        sys.stdout = RedirectText(self.log_textbox)
        sys.stderr = RedirectText(self.log_textbox)

        # Initial updates
        self.update_texts()
        self.refresh_datasets_dropdown()
        print(f"[{translations[self.lang]['title']}] Hazır!")
        print(f"Çıktı Klasörü: {self.output_dir_var.get()}")

    def change_theme(self, theme_name: str):
        if theme_name in THEMES:
            t = THEMES[theme_name]
            global PRIMARY_COLOR, HOVER_COLOR
            PRIMARY_COLOR = t["primary"]
            HOVER_COLOR = t["hover"]
            self.main_frame.configure(fg_color=t["bg"])
            self.header_frame.configure(border_color=PRIMARY_COLOR)
            self.header_title.configure(text_color=PRIMARY_COLOR)
            self.generate_btn.configure(fg_color=PRIMARY_COLOR, hover_color=HOVER_COLOR)
            print(f"[Tema] {theme_name} teması uygulandı! 🎨")

    def on_lang_select(self, choice: str):
        mapping = {"🇹🇷 TR": "tr", "🇬🇧 EN": "en", "🇯🇵 JA": "ja", "🇰🇷 KO": "ko"}
        self.lang = mapping.get(choice, "tr")
        self.update_texts()
        print(f"[Dil] Aktif dil: {self.lang.upper()}")

    def switch_studio_mode(self, mode: str):
        self.studio_mode_var.set(mode)
        for vf in [
            self.cover_view_frame, self.batch_view_frame, self.history_view_frame,
            self.vocal_analysis_view_frame, self.models_view_frame,
            self.dataset_view_frame, self.train_view_frame
        ]:
            vf.pack_forget()

        modes_map = {
            "cover": self.mode_btn_cover,
            "batch": self.mode_btn_batch,
            "history": self.mode_btn_history,
            "analysis": self.mode_btn_analysis,
            "models": self.mode_btn_models,
            "dataset": self.mode_btn_dataset,
            "train": self.mode_btn_train
        }
        for m, btn in modes_map.items():
            btn.configure(fg_color=PRIMARY_COLOR if mode == m else "#2B2B2B")

        if mode == "cover":
            self.cover_view_frame.pack(fill="x", before=self.tools_frame)
        elif mode == "batch":
            self._refresh_batch_ui()
            self.batch_view_frame.pack(fill="x", before=self.tools_frame)
        elif mode == "history":
            self._refresh_history_ui()
            self.history_view_frame.pack(fill="x", before=self.tools_frame)
        elif mode == "analysis":
            self.vocal_analysis_view_frame.pack(fill="x", before=self.tools_frame)
        elif mode == "models":
            self._refresh_models_ui()
            self.models_view_frame.pack(fill="x", before=self.tools_frame)
        elif mode == "dataset":
            self.dataset_view_frame.pack(fill="x", before=self.tools_frame)
        elif mode == "train":
            self.refresh_datasets_dropdown()
            self.train_view_frame.pack(fill="x", before=self.tools_frame)

    # =========================================================================
    # VIEW 1: AI COVER STUDIO
    # =========================================================================
    def _build_cover_view(self):
        # --- Models ---
        self.model_frame = ctk.CTkFrame(self.cover_view_frame, fg_color="#181824", corner_radius=10)
        self.model_frame.pack(fill="x", pady=5, padx=10)

        self.lbl_model_select = ctk.CTkLabel(
            self.model_frame, text=translations[self.lang]["model_select"], font=("Arial", 16, "bold"), text_color="#DDD"
        )
        self.lbl_model_select.pack(pady=5)

        self.models = get_current_models(rvc_models_dir)
        if not self.model_var.get() and self.models:
            self.model_var.set(self.models[0])
        self.model_dropdown = ctk.CTkOptionMenu(
            self.model_frame,
            values=self.models if self.models else [""],
            variable=self.model_var,
            width=320,
            fg_color=PRIMARY_COLOR,
            button_color=HOVER_COLOR,
        )
        self.model_dropdown.pack(pady=5)

        self.model_btn_frame = ctk.CTkFrame(self.model_frame, fg_color="transparent")
        self.model_btn_frame.pack(pady=5)
        self.btn_refresh = ctk.CTkButton(
            self.model_btn_frame,
            text=translations[self.lang]["refresh"],
            fg_color="#444",
            command=self.refresh_models,
        )
        self.btn_refresh.pack(side="left", padx=5)
        self.btn_upload = ctk.CTkButton(
            self.model_btn_frame,
            text=translations[self.lang]["upload_model"],
            fg_color=PRIMARY_COLOR,
            command=self.upload_model_gui,
        )
        self.btn_upload.pack(side="left", padx=5)

        # --- Pipeline / UVR5 Models ---
        self.pipeline_frame = ctk.CTkFrame(self.cover_view_frame, fg_color="#181824", corner_radius=10)
        self.pipeline_frame.pack(fill="x", pady=5, padx=10)

        self.lbl_pipeline_select = ctk.CTkLabel(
            self.pipeline_frame, text=translations[self.lang]["pipeline_select"], font=("Arial", 16, "bold"), text_color="#DDD"
        )
        self.lbl_pipeline_select.pack(pady=5)

        self.pipelines = [
            "UVR5 Special 2",
            "UVR5 Special 3",
            "UVR-MDX-NET-Voc_FT.onnx",
            "UVR_MDXNET_Main.onnx",
            "Kim_Vocal_2.onnx",
        ]
        self.pipeline_dropdown = ctk.CTkOptionMenu(
            self.pipeline_frame,
            values=self.pipelines,
            variable=self.pipeline_var,
            width=320,
            fg_color=PRIMARY_COLOR,
            button_color=HOVER_COLOR,
        )
        self.pipeline_dropdown.pack(pady=5)
        self.pipeline_var.trace_add("write", self.check_file_cache)

        # --- Input Selection ---
        self.input_frame = ctk.CTkFrame(self.cover_view_frame, fg_color="#181824", corner_radius=10)
        self.input_frame.pack(fill="x", pady=5, padx=10)

        input_inner = ctk.CTkFrame(self.input_frame, fg_color="transparent")
        input_inner.pack(fill="x", padx=10, pady=5)

        self.lbl_input = ctk.CTkLabel(
            input_inner, text=translations[self.lang]["input_select"], font=("Arial", 14, "bold"), text_color="#DDD"
        )
        self.lbl_input.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.input_entry = ctk.CTkEntry(input_inner, textvariable=self.song_input_var, width=440)
        self.input_entry.grid(row=0, column=1, padx=10, pady=5)

        btn_row_input = ctk.CTkFrame(input_inner, fg_color="transparent")
        btn_row_input.grid(row=0, column=2, padx=10, pady=5, sticky="w")

        self.btn_browse = ctk.CTkButton(
            btn_row_input,
            text=translations[self.lang]["browse"],
            fg_color=PRIMARY_COLOR,
            hover_color=HOVER_COLOR,
            width=85,
            command=self.browse_file,
        )
        self.btn_browse.pack(side="left", padx=(0, 6))

        self.btn_yt_toggle = ctk.CTkButton(
            btn_row_input,
            text=translations[self.lang]["yt_search_toggle"],
            fg_color="#CC0000",
            hover_color="#990000",
            width=130,
            font=("Arial", 11, "bold"),
            command=self.toggle_yt_panel,
        )
        self.btn_yt_toggle.pack(side="left")

        self.lbl_output_dir = ctk.CTkLabel(
            input_inner, text=translations[self.lang]["output_select"], font=("Arial", 14, "bold"), text_color="#DDD"
        )
        self.lbl_output_dir.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.output_dir_entry = ctk.CTkEntry(input_inner, textvariable=self.output_dir_var, width=440)
        self.output_dir_entry.grid(row=1, column=1, padx=10, pady=5)
        self.btn_browse_dir = ctk.CTkButton(
            input_inner,
            text=translations[self.lang]["browse_dir"],
            fg_color=PRIMARY_COLOR,
            hover_color=HOVER_COLOR,
            width=85,
            command=self.browse_dir,
        )
        self.btn_browse_dir.grid(row=1, column=2, padx=10, pady=5, sticky="w")

        # --- YouTube Search & Instant Cover Panel ---
        self.yt_panel = ctk.CTkFrame(self.input_frame, fg_color="#12121A", corner_radius=10, border_color="#2A2A3E", border_width=1)
        self.yt_panel.pack(fill="x", padx=10, pady=(5, 10))

        yt_header = ctk.CTkFrame(self.yt_panel, fg_color="transparent")
        yt_header.pack(fill="x", padx=10, pady=(6, 2))

        self.lbl_yt_title = ctk.CTkLabel(
            yt_header,
            text=translations[self.lang]["yt_panel_title"],
            font=("Arial", 13, "bold"),
            text_color="#FF4444",
        )
        self.lbl_yt_title.pack(side="left")

        yt_bar = ctk.CTkFrame(self.yt_panel, fg_color="transparent")
        yt_bar.pack(fill="x", padx=10, pady=4)

        self.yt_search_var = ctk.StringVar()
        self.yt_search_entry = ctk.CTkEntry(
            yt_bar,
            textvariable=self.yt_search_var,
            placeholder_text=translations[self.lang]["yt_search_placeholder"],
            height=34,
        )
        self.yt_search_entry.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self.yt_search_entry.bind("<Return>", lambda e: self.do_yt_search())

        self.btn_yt_search = ctk.CTkButton(
            yt_bar,
            text=translations[self.lang]["yt_btn_search"],
            width=85,
            height=34,
            fg_color="#CC0000",
            hover_color="#990000",
            font=("Arial", 12, "bold"),
            command=self.do_yt_search,
        )
        self.btn_yt_search.pack(side="left", padx=4)

        self.btn_yt_clear = ctk.CTkButton(
            yt_bar,
            text="✕",
            width=34,
            height=34,
            fg_color="#2A2A2A",
            hover_color="#444444",
            command=self.clear_yt_search,
        )
        self.btn_yt_clear.pack(side="left")

        self.lbl_yt_status = ctk.CTkLabel(
            self.yt_panel,
            text=translations[self.lang]["yt_search_hint"],
            font=("Arial", 11),
            text_color="#AAAAAA",
            anchor="w",
        )
        self.lbl_yt_status.pack(fill="x", padx=15, pady=(0, 4))

        self.yt_results_frame = ctk.CTkScrollableFrame(self.yt_panel, fg_color="#0A0A10", height=200, corner_radius=8)
        self.yt_results_frame.pack(fill="x", padx=10, pady=(0, 10))

        # --- Advanced Options TabView ---
        self.tabview = ctk.CTkTabview(
            self.cover_view_frame,
            width=850,
            fg_color="#181824",
            segmented_button_selected_color=PRIMARY_COLOR,
            segmented_button_selected_hover_color=HOVER_COLOR,
        )
        self.tabview.pack(fill="x", pady=10, padx=10)

        self.tab_names = ["tab_tuning", "tab_mix", "tab_fx", "tab_adv", "tab_mastering"]
        self.tabs = {}
        self.current_tab_titles = {}
        for t in self.tab_names:
            title = translations[self.lang][t]
            self.tabs[t] = self.tabview.add(title)
            self.current_tab_titles[t] = title

        # - Pitch & Tuning Tab with Smart Pitch & Style Advisor
        tab_pitch = self.tabs["tab_tuning"]

        # Smart Recommendation Tools Row
        smart_tools = ctk.CTkFrame(tab_pitch, fg_color="#12121A", corner_radius=8)
        smart_tools.pack(fill="x", padx=10, pady=(6, 8))

        btn_smart_pitch = ctk.CTkButton(
            smart_tools,
            text="🎯 Akıllı Pitch Öner (Ton & Ses Aralığı)",
            font=("Arial", 12, "bold"),
            fg_color="#00695C",
            hover_color="#004D40",
            command=self.run_smart_pitch_recommendation
        )
        btn_smart_pitch.pack(side="left", padx=8, pady=8)

        btn_style_advisor = ctk.CTkButton(
            smart_tools,
            text="💡 Şarkı Stilini Analiz Et (BPM & Mix)",
            font=("Arial", 12, "bold"),
            fg_color="#5D3FD3",
            hover_color="#7B68EE",
            command=self.run_style_advisor
        )
        btn_style_advisor.pack(side="left", padx=6, pady=8)

        self.lbl_pitch_rec_status = ctk.CTkLabel(
            smart_tools,
            text="Şarkıyı seçip akıllı pitch veya stil analizi yapabilirsiniz.",
            font=("Arial", 11),
            text_color="#A390E4"
        )
        self.lbl_pitch_rec_status.pack(side="left", padx=10)

        self.lbl_vocal_pitch = ctk.CTkLabel(tab_pitch, text=translations[self.lang]["vocal_pitch"])
        self.lbl_vocal_pitch.pack(pady=(4, 0))
        ctk.CTkSlider(
            tab_pitch, from_=-12, to=12, number_of_steps=240, variable=self.pitch_var, button_color=PRIMARY_COLOR, button_hover_color=HOVER_COLOR
        ).pack()
        # Pitch label + manual entry row
        _pitch_row1 = ctk.CTkFrame(tab_pitch, fg_color="transparent")
        _pitch_row1.pack()
        self.pitch_label = ctk.CTkLabel(_pitch_row1, text=f"{self.pitch_var.get():.2f}", width=55)
        self.pitch_label.pack(side="left", padx=(0, 4))
        self._pitch_entry = ctk.CTkEntry(_pitch_row1, textvariable=self.pitch_var, width=65)
        self._pitch_entry.pack(side="left")
        ctk.CTkLabel(_pitch_row1, text="st", text_color="#888", font=("Arial", 11)).pack(side="left", padx=2)

        def _update_pitch_label(*args):
            try:
                val = round(float(self.pitch_var.get()), 2)
                self.pitch_label.configure(text=f"{val:+.2f}")
            except Exception:
                pass
        self.pitch_var.trace_add("write", _update_pitch_label)

        self.lbl_overall_pitch = ctk.CTkLabel(tab_pitch, text=translations[self.lang]["overall_pitch"])
        self.lbl_overall_pitch.pack(pady=(6, 0))
        ctk.CTkSlider(
            tab_pitch, from_=-12, to=12, number_of_steps=240, variable=self.pitch_all_var, button_color=PRIMARY_COLOR, button_hover_color=HOVER_COLOR
        ).pack()
        # Pitch all label + manual entry row
        _pitch_row2 = ctk.CTkFrame(tab_pitch, fg_color="transparent")
        _pitch_row2.pack()
        self.pitch_all_label = ctk.CTkLabel(_pitch_row2, text=f"{self.pitch_all_var.get():.2f}", width=55)
        self.pitch_all_label.pack(side="left", padx=(0, 4))
        self._pitch_all_entry = ctk.CTkEntry(_pitch_row2, textvariable=self.pitch_all_var, width=65)
        self._pitch_all_entry.pack(side="left")
        ctk.CTkLabel(_pitch_row2, text="st", text_color="#888", font=("Arial", 11)).pack(side="left", padx=2)

        def _update_pitch_all_label(*args):
            try:
                val = round(float(self.pitch_all_var.get()), 2)
                self.pitch_all_label.configure(text=f"{val:+.2f}")
            except Exception:
                pass
        self.pitch_all_var.trace_add("write", _update_pitch_all_label)

        # - Audio Mixing
        tab_mix = self.tabs["tab_mix"]
        self.lbl_main_gain = ctk.CTkLabel(tab_mix, text=translations[self.lang]["main_gain"])
        self.lbl_main_gain.pack(pady=(5, 0))
        ctk.CTkSlider(
            tab_mix, from_=-20, to=20, number_of_steps=40, variable=self.main_gain_var, button_color=PRIMARY_COLOR, button_hover_color=HOVER_COLOR
        ).pack()
        self.main_gain_label = ctk.CTkLabel(tab_mix, text=str(self.main_gain_var.get()))
        self.main_gain_label.pack()
        self.main_gain_var.trace_add("write", lambda *args: self.main_gain_label.configure(text=str(self.main_gain_var.get())))

        self.lbl_backup_gain = ctk.CTkLabel(tab_mix, text=translations[self.lang]["backup_gain"])
        self.lbl_backup_gain.pack(pady=(5, 0))
        ctk.CTkSlider(
            tab_mix, from_=-20, to=20, number_of_steps=40, variable=self.backup_gain_var, button_color=PRIMARY_COLOR, button_hover_color=HOVER_COLOR
        ).pack()
        self.backup_gain_label = ctk.CTkLabel(tab_mix, text=str(self.backup_gain_var.get()))
        self.backup_gain_label.pack()
        self.backup_gain_var.trace_add("write", lambda *args: self.backup_gain_label.configure(text=str(self.backup_gain_var.get())))

        self.cb_mute_backup = ctk.CTkCheckBox(
            tab_mix, text=translations[self.lang]["mute_backup"], variable=self.mute_backup_var, fg_color=PRIMARY_COLOR, hover_color=HOVER_COLOR
        )
        self.cb_mute_backup.pack(pady=10)

        self.lbl_inst_gain = ctk.CTkLabel(tab_mix, text=translations[self.lang]["inst_gain"])
        self.lbl_inst_gain.pack(pady=(5, 0))
        ctk.CTkSlider(
            tab_mix, from_=-20, to=20, number_of_steps=40, variable=self.inst_gain_var, button_color=PRIMARY_COLOR, button_hover_color=HOVER_COLOR
        ).pack()
        self.inst_gain_label = ctk.CTkLabel(tab_mix, text=str(self.inst_gain_var.get()))
        self.inst_gain_label.pack()
        self.inst_gain_var.trace_add("write", lambda *args: self.inst_gain_label.configure(text=str(self.inst_gain_var.get())))

        # - Effects (Reverb)
        tab_fx = self.tabs["tab_fx"]
        self.lbl_room_size = ctk.CTkLabel(tab_fx, text=translations[self.lang]["room_size"])
        self.lbl_room_size.pack(pady=(5, 0))
        ctk.CTkSlider(
            tab_fx, from_=0, to=1, number_of_steps=100, variable=self.reverb_size_var, button_color=PRIMARY_COLOR, button_hover_color=HOVER_COLOR
        ).pack()
        self.reverb_size_label = ctk.CTkLabel(tab_fx, text=str(round(self.reverb_size_var.get(), 2)))
        self.reverb_size_label.pack()
        self.reverb_size_var.trace_add("write", lambda *args: self.reverb_size_label.configure(text=str(round(self.reverb_size_var.get(), 2))))

        self.lbl_wetness = ctk.CTkLabel(tab_fx, text=translations[self.lang]["wetness"])
        self.lbl_wetness.pack(pady=(5, 0))
        ctk.CTkSlider(
            tab_fx, from_=0, to=1, number_of_steps=100, variable=self.reverb_wet_var, button_color=PRIMARY_COLOR, button_hover_color=HOVER_COLOR
        ).pack()
        self.reverb_wet_label = ctk.CTkLabel(tab_fx, text=str(round(self.reverb_wet_var.get(), 2)))
        self.reverb_wet_label.pack()
        self.reverb_wet_var.trace_add("write", lambda *args: self.reverb_wet_label.configure(text=str(round(self.reverb_wet_var.get(), 2))))

        self.lbl_dryness = ctk.CTkLabel(tab_fx, text=translations[self.lang]["dryness"])
        self.lbl_dryness.pack(pady=(5, 0))
        ctk.CTkSlider(
            tab_fx, from_=0, to=1, number_of_steps=100, variable=self.reverb_dry_var, button_color=PRIMARY_COLOR, button_hover_color=HOVER_COLOR
        ).pack()
        self.reverb_dry_label = ctk.CTkLabel(tab_fx, text=str(round(self.reverb_dry_var.get(), 2)))
        self.reverb_dry_label.pack()
        self.reverb_dry_var.trace_add("write", lambda *args: self.reverb_dry_label.configure(text=str(round(self.reverb_dry_var.get(), 2))))

        self.lbl_damping = ctk.CTkLabel(tab_fx, text=translations[self.lang]["damping"])
        self.lbl_damping.pack(pady=(5, 0))
        ctk.CTkSlider(
            tab_fx, from_=0, to=1, number_of_steps=100, variable=self.reverb_damp_var, button_color=PRIMARY_COLOR, button_hover_color=HOVER_COLOR
        ).pack()
        self.reverb_damp_label = ctk.CTkLabel(tab_fx, text=str(round(self.reverb_damp_var.get(), 2)))
        self.reverb_damp_label.pack()
        self.reverb_damp_var.trace_add("write", lambda *args: self.reverb_damp_label.configure(text=str(round(self.reverb_damp_var.get(), 2))))

        # - Advanced Conversion
        tab_adv = self.tabs["tab_adv"]
        self.lbl_index_rate = ctk.CTkLabel(tab_adv, text=translations[self.lang]["index_rate"])
        self.lbl_index_rate.pack(pady=(5, 0))
        ctk.CTkSlider(
            tab_adv, from_=0, to=1, number_of_steps=100, variable=self.index_rate_var, button_color=PRIMARY_COLOR, button_hover_color=HOVER_COLOR
        ).pack()
        self.index_rate_label = ctk.CTkLabel(tab_adv, text=str(round(self.index_rate_var.get(), 2)))
        self.index_rate_label.pack()
        self.index_rate_var.trace_add("write", lambda *args: self.index_rate_label.configure(text=str(round(self.index_rate_var.get(), 2))))

        self.lbl_filter_radius = ctk.CTkLabel(tab_adv, text=translations[self.lang]["filter_radius"])
        self.lbl_filter_radius.pack(pady=(5, 0))
        ctk.CTkSlider(
            tab_adv, from_=0, to=7, number_of_steps=7, variable=self.filter_radius_var, button_color=PRIMARY_COLOR, button_hover_color=HOVER_COLOR
        ).pack()
        self.filter_radius_label = ctk.CTkLabel(tab_adv, text=str(self.filter_radius_var.get()))
        self.filter_radius_label.pack()
        self.filter_radius_var.trace_add("write", lambda *args: self.filter_radius_label.configure(text=str(self.filter_radius_var.get())))

        self.lbl_rms_mix = ctk.CTkLabel(tab_adv, text=translations[self.lang]["rms_mix"])
        self.lbl_rms_mix.pack(pady=(5, 0))
        ctk.CTkSlider(
            tab_adv, from_=0, to=1, number_of_steps=100, variable=self.rms_mix_var, button_color=PRIMARY_COLOR, button_hover_color=HOVER_COLOR
        ).pack()
        self.rms_mix_label = ctk.CTkLabel(tab_adv, text=str(round(self.rms_mix_var.get(), 2)))
        self.rms_mix_label.pack()
        self.rms_mix_var.trace_add("write", lambda *args: self.rms_mix_label.configure(text=str(round(self.rms_mix_var.get(), 2))))

        self.lbl_protect = ctk.CTkLabel(tab_adv, text=translations[self.lang]["protect"])
        self.lbl_protect.pack(pady=(5, 0))
        ctk.CTkSlider(
            tab_adv, from_=0, to=0.5, number_of_steps=50, variable=self.protect_var, button_color=PRIMARY_COLOR, button_hover_color=HOVER_COLOR
        ).pack()
        self.protect_label = ctk.CTkLabel(tab_adv, text=str(round(self.protect_var.get(), 2)))
        self.protect_label.pack()
        self.protect_var.trace_add("write", lambda *args: self.protect_label.configure(text=str(round(self.protect_var.get(), 2))))

        self.lbl_f0_method = ctk.CTkLabel(tab_adv, text=translations[self.lang]["f0_method"])
        self.lbl_f0_method.pack(pady=(5, 0))
        self.f0_dropdown = ctk.CTkOptionMenu(
            tab_adv,
            values=["rmvpe", "crepe", "pm", "harvest", "ensemble_consensus"],
            variable=self.f0_method_var,
            fg_color=PRIMARY_COLOR,
            button_color=HOVER_COLOR,
        )
        self.f0_dropdown.pack()

        self.lbl_crepe_hop = ctk.CTkLabel(tab_adv, text=translations[self.lang]["crepe_hop"])
        self.crepe_slider = ctk.CTkSlider(
            tab_adv, from_=32, to=512, number_of_steps=15, variable=self.crepe_hop_var, button_color=PRIMARY_COLOR, button_hover_color=HOVER_COLOR
        )
        self.crepe_hop_label = ctk.CTkLabel(tab_adv, text=str(self.crepe_hop_var.get()))
        self.crepe_hop_var.trace_add("write", lambda *args: self.crepe_hop_label.configure(text=str(self.crepe_hop_var.get())))

        def toggle_crepe(*args):
            if self.f0_method_var.get() == "crepe":
                self.lbl_crepe_hop.pack(pady=(5, 0))
                self.crepe_slider.pack()
                self.crepe_hop_label.pack()
            else:
                self.lbl_crepe_hop.pack_forget()
                self.crepe_slider.pack_forget()
                self.crepe_hop_label.pack_forget()

        self._toggle_crepe_fn = toggle_crepe
        self.f0_method_var.trace_add("write", toggle_crepe)

        self.cb_noise_gate = ctk.CTkCheckBox(
            tab_adv, text=translations[self.lang]["noise_gate"], variable=self.noise_gate_var, fg_color=PRIMARY_COLOR, hover_color=HOVER_COLOR
        )
        self.cb_noise_gate.pack(pady=10)
        self.lbl_noise_gate_thresh = ctk.CTkLabel(tab_adv, text=translations[self.lang]["noise_gate_thresh"])
        self.ng_slider = ctk.CTkSlider(
            tab_adv, from_=-60, to=-10, number_of_steps=50, variable=self.noise_gate_thresh_var, button_color=PRIMARY_COLOR, button_hover_color=HOVER_COLOR
        )
        self.ng_thresh_label = ctk.CTkLabel(tab_adv, text=str(self.noise_gate_thresh_var.get()))
        self.noise_gate_thresh_var.trace_add("write", lambda *args: self.ng_thresh_label.configure(text=str(self.noise_gate_thresh_var.get())))

        def toggle_ng(*args):
            if self.noise_gate_var.get():
                self.lbl_noise_gate_thresh.pack(pady=(5, 0))
                self.ng_slider.pack()
                self.ng_thresh_label.pack()
            else:
                self.lbl_noise_gate_thresh.pack_forget()
                self.ng_slider.pack_forget()
                self.ng_thresh_label.pack_forget()

        self.noise_gate_var.trace_add("write", toggle_ng)

        self.cb_agg_dereverb = ctk.CTkCheckBox(
            tab_adv, text=translations[self.lang]["agg_dereverb"], variable=self.aggressive_dereverb_var, fg_color=PRIMARY_COLOR, hover_color=HOVER_COLOR
        )
        self.cb_agg_dereverb.pack(pady=10)

        self.cb_keep_files = ctk.CTkCheckBox(
            tab_adv, text=translations[self.lang]["keep_files"], variable=self.keep_files_var, fg_color=PRIMARY_COLOR, hover_color=HOVER_COLOR
        )
        self.cb_keep_files.pack(pady=10)

        self.cb_force_reprocess = ctk.CTkCheckBox(
            tab_adv, text=translations[self.lang]["force_reprocess"], variable=self.force_reprocess_var, fg_color=PRIMARY_COLOR, hover_color=HOVER_COLOR
        )
        self.cb_force_reprocess.pack(pady=10)
        self.song_input_var.trace_add("write", self.check_file_cache)

        self.lbl_out_format = ctk.CTkLabel(tab_adv, text=translations[self.lang]["output_format"])
        self.lbl_out_format.pack(pady=(5, 0))
        ctk.CTkSegmentedButton(
            tab_adv, values=["mp3", "wav"], variable=self.output_format_var, selected_color=PRIMARY_COLOR, selected_hover_color=HOVER_COLOR
        ).pack(pady=5)

        # - Studio Mastering & Vocal Chain Tab
        tab_mast = self.tabs["tab_mastering"]

        mast_scroll = ctk.CTkScrollableFrame(tab_mast, fg_color="transparent", height=280)
        mast_scroll.pack(fill="both", expand=True, padx=5, pady=5)

        # 1. Section: Miksaj & Vokal İşleme
        sec1_frame = ctk.CTkFrame(mast_scroll, fg_color="#12121A", corner_radius=8, border_color="#5D3FD3", border_width=1)
        sec1_frame.pack(fill="x", padx=5, pady=5)
        ctk.CTkLabel(sec1_frame, text="🎤 Kategori 1: Miksaj & Vokal İşleme Zinciri", font=("Arial", 12, "bold"), text_color="#A390E4").pack(anchor="w", padx=10, pady=(6, 4))

        cb_row1 = ctk.CTkFrame(sec1_frame, fg_color="transparent")
        cb_row1.pack(fill="x", padx=10, pady=4)
        self.cb_deesser = ctk.CTkCheckBox(cb_row1, text="Akıllı De-Esser (5kHz-9kHz Tıslama & Sibilance Kontrolü)", variable=self.deesser_var, fg_color=PRIMARY_COLOR, hover_color=HOVER_COLOR)
        self.cb_deesser.pack(side="left", padx=(0, 20))
        self.cb_studio_eq = ctk.CTkCheckBox(cb_row1, text="4-Bant Stüdyo Parametrik EQ (80Hz LowCut, 250Hz Gövde, 450Hz MudCut, 12kHz Air)", variable=self.studio_eq_var, fg_color=PRIMARY_COLOR, hover_color=HOVER_COLOR)
        self.cb_studio_eq.pack(side="left")

        cb_row2 = ctk.CTkFrame(sec1_frame, fg_color="transparent")
        cb_row2.pack(fill="x", padx=10, pady=(2, 8))
        self.cb_doubler = ctk.CTkCheckBox(cb_row2, text="Stereo Vokal Genişletici & Doubler (Vocal Widener - Geniş Sahne)", variable=self.doubler_var, fg_color=PRIMARY_COLOR, hover_color=HOVER_COLOR)
        self.cb_doubler.pack(side="left", padx=(0, 20))
        self.cb_sidechain = ctk.CTkCheckBox(cb_row2, text="Dinamik Sidechain Ducking (1kHz-3.5kHz Enstrüman Cebi Açar)", variable=self.sidechain_var, fg_color=PRIMARY_COLOR, hover_color=HOVER_COLOR)
        self.cb_sidechain.pack(side="left")

        # 2. Section: AI Vokal Doğallığı & Dönüşüm
        sec2_frame = ctk.CTkFrame(mast_scroll, fg_color="#12121A", corner_radius=8, border_color="#00ADB5", border_width=1)
        sec2_frame.pack(fill="x", padx=5, pady=5)
        ctk.CTkLabel(sec2_frame, text="✨ Kategori 2: AI Vokal Doğallığı & Dönüşüm", font=("Arial", 12, "bold"), text_color="#00ADB5").pack(anchor="w", padx=10, pady=(6, 4))

        cb_row3 = ctk.CTkFrame(sec2_frame, fg_color="transparent")
        cb_row3.pack(fill="x", padx=10, pady=4)
        self.cb_breath_pass = ctk.CTkCheckBox(cb_row3, text="Orijinal İnsan Nefeslerini Koruma & Geri Enjekte Etme (Breath Passthrough)", variable=self.breath_passthrough_var, fg_color="#00ADB5", hover_color="#007F85")
        self.cb_breath_pass.pack(side="left")

        formant_row = ctk.CTkFrame(sec2_frame, fg_color="transparent")
        formant_row.pack(fill="x", padx=10, pady=(4, 8))
        ctk.CTkLabel(formant_row, text="Formant Shifter (Ses Yolu Rezonansı):", font=("Arial", 11)).pack(side="left", padx=(0, 10))
        self.formant_slider = ctk.CTkSlider(formant_row, from_=-6.0, to=6.0, number_of_steps=24, variable=self.formant_shift_var, width=220, button_color="#00ADB5", button_hover_color="#007F85")
        self.formant_slider.pack(side="left", padx=5)
        self.lbl_formant_val = ctk.CTkLabel(formant_row, text="0.0 st (Doğal)", font=("Arial", 11, "bold"), text_color="#00ADB5")
        self.lbl_formant_val.pack(side="left", padx=10)

        def _update_formant_label(*args):
            val = round(self.formant_shift_var.get(), 1)
            desc = " (Doğal)" if abs(val) < 0.1 else (" (Kalın/Erkeksi Gövde)" if val < 0 else " (İnce/Feminen Rezonans)")
            self.lbl_formant_val.configure(text=f"{val:+.1f} st{desc}")

        self.formant_shift_var.trace_add("write", _update_formant_label)

        # 3. Section: Yayın Standardı & Mastering
        sec3_frame = ctk.CTkFrame(mast_scroll, fg_color="#12121A", corner_radius=8, border_color="#FFB300", border_width=1)
        sec3_frame.pack(fill="x", padx=5, pady=5)
        ctk.CTkLabel(sec3_frame, text="🏆 Kategori 3: Yayın Standardı Mastering & Armoni", font=("Arial", 12, "bold"), text_color="#FFB300").pack(anchor="w", padx=10, pady=(6, 4))

        cb_row4 = ctk.CTkFrame(sec3_frame, fg_color="transparent")
        cb_row4.pack(fill="x", padx=10, pady=4)
        self.cb_lufs_master = ctk.CTkCheckBox(cb_row4, text="Otomatik -14 LUFS Loudness Matcher & True Peak Limiter (-1.0 dBFS - Spotify/YT Uyumlu)", variable=self.lufs_master_var, fg_color="#FFB300", hover_color="#FFA000")
        self.cb_lufs_master.pack(side="left")

        harm_row = ctk.CTkFrame(sec3_frame, fg_color="transparent")
        harm_row.pack(fill="x", padx=10, pady=4)
        self.cb_harmonies = ctk.CTkCheckBox(harm_row, text="AI Destekli Armoni & Çift Ses (Backing Vocals):", variable=self.harmonies_var, fg_color="#FFB300", hover_color="#FFA000")
        self.cb_harmonies.pack(side="left", padx=(0, 10))
        self.harmony_menu = ctk.CTkOptionMenu(
            harm_row,
            values=["3rd_and_5th (+Maj 3rd & 5th)", "octave_down (Alt Oktav Gövde)", "trio_choir (Koro: +3rd, +5th, -Oktav)"],
            variable=self.harmony_mode_var,
            width=260,
            fg_color="#332A15",
            button_color="#FFB300"
        )
        self.harmony_menu.pack(side="left")

        dyn_pitch_frame = ctk.CTkFrame(sec3_frame, fg_color="transparent")
        dyn_pitch_frame.pack(fill="x", padx=10, pady=(4, 8))
        ctk.CTkLabel(dyn_pitch_frame, text="Bölüm Bazlı Dinamik Pitch (Verse vs Chorus):", font=("Arial", 11)).pack(side="left", padx=(0, 8))
        self.entry_dyn_pitch = ctk.CTkEntry(
            dyn_pitch_frame,
            textvariable=self.dynamic_pitch_rules_var,
            placeholder_text="Örn: 00:00-01:15: 0; 01:15-02:00: -2; 02:00-03:30: +1",
            width=420
        )
        self.entry_dyn_pitch.pack(side="left", fill="x", expand=True)

        # --- Lyrics & Translation Bar ---
        lyrics_bar = ctk.CTkFrame(self.cover_view_frame, fg_color="#181824", corner_radius=10)
        lyrics_bar.pack(fill="x", padx=10, pady=(6, 4))

        self.btn_toggle_lyrics = ctk.CTkButton(
            lyrics_bar,
            text="📜 Şarkı Sözleri & Çeviri Panelini Aç / Kapat",
            font=("Arial", 12, "bold"),
            fg_color="#3D3D58",
            hover_color="#5D3FD3",
            command=self.toggle_lyrics_panel
        )
        self.btn_toggle_lyrics.pack(side="left", padx=10, pady=8)

        # A/B Comparison Checkbox
        self.cb_compare_models = ctk.CTkCheckBox(
            lyrics_bar,
            text="⚖️ Model Karşılaştırma Modu (A/B Testi)",
            variable=self.enable_compare_var,
            font=("Arial", 12, "bold"),
            fg_color="#00d2d3",
            hover_color="#0abde3"
        )
        self.cb_compare_models.pack(side="left", padx=15)

        self.compare_dropdown = ctk.CTkOptionMenu(
            lyrics_bar,
            values=self.models if self.models else [""],
            variable=self.compare_model_var,
            width=200,
            fg_color="#2A2A3E"
        )
        self.compare_dropdown.pack(side="left", padx=5)

        # Collapsible Lyrics Frame
        self.lyrics_panel = ctk.CTkFrame(self.cover_view_frame, fg_color="#12121A", corner_radius=10)
        self.lyrics_text = ctk.CTkTextbox(
            self.lyrics_panel,
            height=120,
            fg_color="#0A0A10",
            text_color="#DDD",
            font=("Segoe UI", 11)
        )
        self.lyrics_text.pack(fill="x", padx=10, pady=10)

        # --- Auto Mode Switch ---
        self.auto_mode_var = ctk.BooleanVar(value=False)
        self.lock_settings_var = ctk.BooleanVar(value=False)

        self.auto_mode_frame = ctk.CTkFrame(self.cover_view_frame, fg_color="transparent")
        self.auto_mode_frame.pack(pady=(10, 0))

        self.sw_auto_mode = ctk.CTkSwitch(
            self.auto_mode_frame,
            text=translations[self.lang]["auto_mode"],
            variable=self.auto_mode_var,
            fg_color=PRIMARY_COLOR,
            progress_color=PRIMARY_COLOR,
            font=("Century Gothic", 14, "bold"),
        )
        self.sw_auto_mode.pack(side="left", padx=(0, 20))

        def toggle_lock():
            if self.lock_settings_var.get():
                self.auto_mode_var.set(False)
                self.sw_auto_mode.configure(state="disabled")
            else:
                self.sw_auto_mode.configure(state="normal")

        self.chk_lock = ctk.CTkCheckBox(
            self.auto_mode_frame,
            text="🔒 Ayarları Sabitle",
            variable=self.lock_settings_var,
            fg_color="#E74C3C",
            hover_color="#C0392B",
            border_color="#E74C3C",
            font=("Century Gothic", 12, "bold"),
            command=toggle_lock,
        )
        self.chk_lock.pack(side="left")

        # --- Generate Button ---
        self.generate_btn = ctk.CTkButton(
            self.cover_view_frame,
            text=translations[self.lang]["generate"],
            fg_color=PRIMARY_COLOR,
            hover_color=HOVER_COLOR,
            font=("Century Gothic", 18, "bold"),
            command=self.start_generation,
            height=50,
        )
        self.generate_btn.pack(pady=(15, 10), fill="x", padx=50)

        # --- Embedded Waveform Audio Player ---
        self.audio_player = AudioPlayerWidget(self.cover_view_frame)
        self.audio_player.pack(fill="x", padx=10, pady=6)

        # --- AI Quality Scorecard Card ---
        self.quality_frame = ctk.CTkFrame(self.cover_view_frame, fg_color="#181824", corner_radius=10, border_color="#5D3FD3", border_width=1)
        self.lbl_quality_stars = ctk.CTkLabel(
            self.quality_frame,
            text="⭐ AI Ses Klonlama Kalite Raporu",
            font=("Arial", 14, "bold"),
            text_color="#00d2d3"
        )
        self.lbl_quality_stars.pack(anchor="w", padx=14, pady=(8, 2))

        self.lbl_quality_metrics = ctk.CTkLabel(
            self.quality_frame,
            text="Cover tamamlandığında doğallık, ton sadakati ve SDR skorları burada listelenecektir.",
            font=("Consolas", 11),
            text_color="#B0B0C0",
            anchor="w"
        )
        self.lbl_quality_metrics.pack(fill="x", padx=14, pady=(2, 8))
        self.quality_frame.pack(fill="x", padx=10, pady=(4, 10))

        self._toggle_crepe_fn()

    # =========================================================================
    # VIEW 2: BATCH COVER QUEUE (TOPLU COVER KUYRUĞU)
    # =========================================================================
    def _build_batch_view(self):
        banner = ctk.CTkFrame(self.batch_view_frame, fg_color="#181824", corner_radius=12)
        banner.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(
            banner,
            text="📑 Toplu Cover Kuyruğu (Gece Boyu Kesintisiz Üretim)",
            font=("Century Gothic", 18, "bold"),
            text_color="#9966FF",
        ).pack(anchor="w", padx=15, pady=(10, 4))
        ctk.CTkLabel(
            banner,
            text="Birden fazla şarkı veya YouTube linkini sıraya ekleyin; stüdyo hepsini art arda işlesin.",
            font=("Arial", 12),
            text_color="#AAA",
        ).pack(anchor="w", padx=15, pady=(0, 10))

        # Add Item Card
        add_card = ctk.CTkFrame(self.batch_view_frame, fg_color="#181824", corner_radius=10)
        add_card.pack(fill="x", padx=10, pady=6)

        row1 = ctk.CTkFrame(add_card, fg_color="transparent")
        row1.pack(fill="x", padx=12, pady=8)

        ctk.CTkLabel(row1, text="Şarkı / Link:", font=("Arial", 12, "bold")).pack(side="left", padx=4)
        ctk.CTkEntry(row1, textvariable=self.batch_input_var, placeholder_text="Dosya yolu veya YouTube URL...", width=360).pack(side="left", padx=6)
        ctk.CTkButton(
            row1, text="Gözat", width=65, fg_color="#333",
            command=lambda: self.batch_input_var.set(filedialog.askopenfilename(filetypes=[("Audio/Video", "*.mp3 *.wav *.flac *.mp4 *.mkv")]) or self.batch_input_var.get())
        ).pack(side="left", padx=4)

        ctk.CTkLabel(row1, text="Model:", font=("Arial", 12, "bold")).pack(side="left", padx=(12, 4))
        self.batch_model_dropdown = ctk.CTkOptionMenu(
            row1, values=self.models if self.models else [""], variable=self.batch_model_var, width=160, fg_color=PRIMARY_COLOR
        )
        self.batch_model_dropdown.pack(side="left", padx=4)

        ctk.CTkLabel(row1, text="Pitch:", font=("Arial", 12, "bold")).pack(side="left", padx=(10, 4))
        ctk.CTkEntry(row1, textvariable=self.batch_pitch_var, width=45).pack(side="left", padx=4)

        ctk.CTkButton(
            row1, text="➕ Kuyruğa Ekle", fg_color="#00695C", hover_color="#004D40", font=("Arial", 12, "bold"),
            command=self.add_batch_item_gui
        ).pack(side="left", padx=10)

        # Controls & Night Mode
        ctrl_card = ctk.CTkFrame(self.batch_view_frame, fg_color="#181824", corner_radius=10)
        ctrl_card.pack(fill="x", padx=10, pady=4)

        self.btn_batch_start = ctk.CTkButton(
            ctrl_card, text="▶️ Kuyruğu Başlat", fg_color=PRIMARY_COLOR, hover_color=HOVER_COLOR,
            font=("Arial", 14, "bold"), height=36, width=160, command=self.start_batch_queue_gui
        )
        self.btn_batch_start.pack(side="left", padx=12, pady=8)

        self.btn_batch_stop = ctk.CTkButton(
            ctrl_card, text="⏹️ Durdur", fg_color="#8b0000", hover_color="#5a0000",
            font=("Arial", 12), height=36, width=90, command=lambda: self.batch_manager.stop_processing()
        )
        self.btn_batch_stop.pack(side="left", padx=6)

        ctk.CTkButton(
            ctrl_card, text="🗑️ Kuyruğu Temizle", fg_color="#333", hover_color="#444",
            font=("Arial", 12), height=36, width=120, command=self.clear_batch_queue_gui
        ).pack(side="left", padx=6)

        ctk.CTkCheckBox(
            ctrl_card, text="🌙 Gece Modu: Bittiğinde Windows Bildirimi Gönder",
            variable=self.batch_night_mode_var, fg_color=PRIMARY_COLOR
        ).pack(side="right", padx=15)

        # Queue Scroll Area
        self.batch_list_frame = ctk.CTkScrollableFrame(self.batch_view_frame, fg_color="#12121A", height=280, corner_radius=10)
        self.batch_list_frame.pack(fill="both", expand=True, padx=10, pady=6)

    def add_batch_item_gui(self):
        s_in = self.batch_input_var.get().strip()
        mod = self.batch_model_var.get() or self.model_var.get()
        if not s_in:
            print("[Hata] Lütfen bir şarkı veya bağlantı girin.")
            return
        self.batch_manager.add_item(
            song_input=s_in,
            song_name=os.path.basename(s_in),
            model_name=mod,
            pitch=self.batch_pitch_var.get(),
            algo=self.f0_method_var.get(),
            uvr_model=self.pipeline_var.get()
        )
        self.batch_input_var.set("")
        self._refresh_batch_ui()
        print(f"[Kuyruk] '{s_in}' başarıyla kuyruğa eklendi!")

    def clear_batch_queue_gui(self):
        self.batch_manager.clear_queue()
        self._refresh_batch_ui()

    def _refresh_batch_ui(self):
        for w in self.batch_list_frame.winfo_children():
            w.destroy()
        items = self.batch_manager.get_items()
        if not items:
            ctk.CTkLabel(
                self.batch_list_frame, text="Kuyrukta bekleyen şarkı yok. Yukarıdan şarkı ekleyebilirsiniz.",
                text_color="#888", font=("Arial", 12)
            ).pack(pady=40)
            return

        for item in items:
            card = ctk.CTkFrame(self.batch_list_frame, fg_color="#181824", corner_radius=8, border_color="#2A2A3E", border_width=1)
            card.pack(fill="x", padx=6, pady=4)

            # Left Details
            col1 = ctk.CTkFrame(card, fg_color="transparent")
            col1.pack(side="left", fill="both", expand=True, padx=10, pady=8)

            lbl_t = ctk.CTkLabel(col1, text=f"🎵 {item['song_name']}", font=("Arial", 13, "bold"), text_color="#FFF", anchor="w")
            lbl_t.pack(fill="x")
            lbl_sub = ctk.CTkLabel(
                col1, text=f"Ses Modeli: {item['model_name']}  |  Pitch: {item['pitch']:+.2f}  |  Durum: {item['status']}",
                font=("Arial", 11), text_color="#A390E4", anchor="w"
            )
            lbl_sub.pack(fill="x")

            # Status pill
            status_color = "#00d2d3" if item["status"] == "Tamamlandı" else ("#E67E22" if item["status"] == "İşleniyor" else "#888")
            if item["status"] == "Hata":
                status_color = "#ff3838"

            lbl_status = ctk.CTkLabel(card, text=f"[{item['status']}]", font=("Arial", 12, "bold"), text_color=status_color)
            lbl_status.pack(side="right", padx=14)

            if item["status"] == "Tamamlandı" and item.get("output_path"):
                ctk.CTkButton(
                    card, text="▶️ Dinle", width=70, fg_color="#5D3FD3",
                    command=lambda p=item["output_path"]: self.audio_player.load_audio(p, autoplay=True)
                ).pack(side="right", padx=4)

    def start_batch_queue_gui(self):
        if self.batch_manager.is_running:
            return

        if hasattr(self, "audio_player") and self.audio_player:
            try:
                self.audio_player.unload()
            except Exception:
                pass

        def _batch_process(item, prog_cb):
            # Download YouTube if needed
            s_in = item["song_input"]
            if s_in.startswith("http"):
                prog_cb("YouTube parçası indiriliyor...", 0.1)
                s_in = download_youtube_audio(s_in, os.path.join(output_dir, "youtube_downloads"))

            prog_cb("Vokal ayrıştırma ve RVC işlemi...", 0.4)
            out_p = song_cover_pipeline(
                song_input=s_in,
                voice_model=item["model_name"],
                pitch_change=item["pitch"],
                f0_method=item["algo"],
                uvr_model=item["uvr_model"]
            )
            target_out_dir = self.output_dir_var.get().strip() if hasattr(self, "output_dir_var") else ""
            if target_out_dir and os.path.exists(target_out_dir):
                dest_p = os.path.join(target_out_dir, os.path.basename(out_p))
                if os.path.abspath(out_p) != os.path.abspath(dest_p):
                    try:
                        import shutil
                        shutil.copyfile(out_p, dest_p)
                        out_p = dest_p
                    except Exception:
                        pass
            return out_p

        def _on_finish():
            self.dispatch_to_ui(self._refresh_batch_ui)
            if self.batch_night_mode_var.get():
                send_windows_notification("Ripleytia AI Studio", "Toplu cover kuyruğundaki tüm parçalar başarıyla tamamlandı! ✅")

        self.batch_manager.start_processing(
            process_fn=_batch_process,
            on_item_update=lambda it: self.dispatch_to_ui(self._refresh_batch_ui),
            on_queue_finished=_on_finish
        )

    # =========================================================================
    # VIEW 3: COVER HISTORY & LIBRARY (GEÇMİŞ & ARŞİV)
    # =========================================================================
    def _build_history_view(self):
        banner = ctk.CTkFrame(self.history_view_frame, fg_color="#181824", corner_radius=12)
        banner.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(
            banner,
            text="🎧 Cover Geçmişi & Ses Kütüphanesi",
            font=("Century Gothic", 18, "bold"),
            text_color="#00d2d3",
        ).pack(anchor="w", padx=15, pady=(10, 4))
        ctk.CTkLabel(
            banner,
            text="Daha önce ürettiğiniz tüm yapay zeka coverlarına tek tıkla ulaşın, dinleyin veya ayarları yükleyin.",
            font=("Arial", 12),
            text_color="#AAA",
        ).pack(anchor="w", padx=15, pady=(0, 10))

        hist_ctrl = ctk.CTkFrame(self.history_view_frame, fg_color="#181824", corner_radius=10)
        hist_ctrl.pack(fill="x", padx=10, pady=4)

        ctk.CTkButton(
            hist_ctrl, text="🔄 Geçmişi Yenile", fg_color=PRIMARY_COLOR, width=120,
            command=self._refresh_history_ui
        ).pack(side="left", padx=10, pady=8)

        ctk.CTkButton(
            hist_ctrl, text="🗑️ Tüm Geçmişi Temizle", fg_color="#8b0000", hover_color="#5a0000", width=140,
            command=lambda: (clear_history(), self._refresh_history_ui())
        ).pack(side="right", padx=10, pady=8)

        self.history_scroll = ctk.CTkScrollableFrame(self.history_view_frame, fg_color="#12121A", height=380, corner_radius=10)
        self.history_scroll.pack(fill="both", expand=True, padx=10, pady=6)

    def _refresh_history_ui(self):
        for w in self.history_scroll.winfo_children():
            w.destroy()
        entries = get_history_entries(limit=50)
        if not entries:
            ctk.CTkLabel(
                self.history_scroll, text="Henüz kaydedilmiş cover geçmişi bulunmuyor.",
                text_color="#888", font=("Arial", 12)
            ).pack(pady=40)
            return

        for e in entries:
            card = ctk.CTkFrame(self.history_scroll, fg_color="#181824", corner_radius=8, border_color="#2A2A3E", border_width=1)
            card.pack(fill="x", padx=6, pady=4)

            info = ctk.CTkFrame(card, fg_color="transparent")
            info.pack(side="left", fill="both", expand=True, padx=10, pady=8)

            lbl_t = ctk.CTkLabel(info, text=f"🎵 {e['song_name']}", font=("Arial", 13, "bold"), text_color="#FFF", anchor="w")
            lbl_t.pack(fill="x")
            lbl_meta = ctk.CTkLabel(
                info, text=f"Model: {e['model_name']}  |  Pitch: {e['pitch']}  |  {e['timestamp']}",
                font=("Arial", 11), text_color="#A390E4", anchor="w"
            )
            lbl_meta.pack(fill="x")

            btn_box = ctk.CTkFrame(card, fg_color="transparent")
            btn_box.pack(side="right", padx=10, pady=6)

            if e.get("file_exists"):
                ctk.CTkButton(
                    btn_box, text="▶️ Çal", width=65, fg_color="#5D3FD3",
                    command=lambda p=e["output_path"]: self.audio_player.load_audio(p, autoplay=True)
                ).pack(side="left", padx=4)

                ctk.CTkButton(
                    btn_box, text="📁 Aç", width=65, fg_color="#333",
                    command=lambda p=e["output_path"]: os.system(f'explorer /select,"{os.path.normpath(p)}"')
                ).pack(side="left", padx=4)

            ctk.CTkButton(
                btn_box, text="🗑️", width=35, fg_color="#8b0000",
                command=lambda eid=e["id"]: (delete_history_entry(eid), self._refresh_history_ui())
            ).pack(side="left", padx=4)

    # =========================================================================
    # VIEW 4: VOCAL & AUDIO LAB (VOKAL & SES ANALİZİ)
    # =========================================================================
    def _build_vocal_analysis_view(self):
        banner = ctk.CTkFrame(self.vocal_analysis_view_frame, fg_color="#181824", corner_radius=12)
        banner.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(
            banner,
            text="📊 Vokal & Ses Karakteristiği Laboratuvarı",
            font=("Century Gothic", 18, "bold"),
            text_color="#48dbfb",
        ).pack(anchor="w", padx=15, pady=(10, 4))
        ctk.CTkLabel(
            banner,
            text="Vokal frekans aralığı (F0), nota dağılımı, SNR ve dinamik dalga formunu grafiklerle inceleyin.",
            font=("Arial", 12),
            text_color="#AAA",
        ).pack(anchor="w", padx=15, pady=(0, 10))

        picker_card = ctk.CTkFrame(self.vocal_analysis_view_frame, fg_color="#181824", corner_radius=10)
        picker_card.pack(fill="x", padx=10, pady=6)

        ctk.CTkLabel(picker_card, text="Ses Dosyası:", font=("Arial", 12, "bold")).pack(side="left", padx=12, pady=10)
        ctk.CTkEntry(picker_card, textvariable=self.analysis_file_var, width=450).pack(side="left", padx=6)
        ctk.CTkButton(
            picker_card, text="Gözat", width=75, fg_color="#333",
            command=lambda: self.analysis_file_var.set(filedialog.askopenfilename(filetypes=[("Audio", "*.mp3 *.wav *.flac")]) or self.analysis_file_var.get())
        ).pack(side="left", padx=4)

        ctk.CTkButton(
            picker_card, text="🔍 Analiz Yap", width=120, fg_color="#00695C", hover_color="#004D40", font=("Arial", 12, "bold"),
            command=self.run_vocal_analysis_gui
        ).pack(side="left", padx=10)

        # Metrics Scorecards
        self.analysis_metrics_frame = ctk.CTkFrame(self.vocal_analysis_view_frame, fg_color="#181824", corner_radius=10)
        self.analysis_metrics_frame.pack(fill="x", padx=10, pady=6)

        self.lbl_vocal_stats = ctk.CTkLabel(
            self.analysis_metrics_frame,
            text="Analiz için bir ses dosyası seçip 'Analiz Yap' butonuna basın.",
            font=("Consolas", 12),
            text_color="#DDD"
        )
        self.lbl_vocal_stats.pack(padx=15, pady=12)

        # Plot Image display
        self.lbl_vocal_plot = ctk.CTkLabel(self.vocal_analysis_view_frame, text="")
        self.lbl_vocal_plot.pack(pady=8)

    def run_vocal_analysis_gui(self):
        f = self.analysis_file_var.get().strip() or self.song_input_var.get().strip()
        if not f or not os.path.exists(f):
            print("[Hata] Analiz edilecek ses dosyası bulunamadı.")
            return

        print(f"[Vokal Lab] '{os.path.basename(f)}' analiz ediliyor...")

        def _worker():
            try:
                res = analyze_vocal_file(f)
                stat_txt = (
                    f"🎼 Nota Aralığı: {res['min_note']} ➔ {res['max_note']} (Medyan: {res['avg_note']})\n"
                    f"⚡ Ortalama F0: {res['avg_f0_hz']} Hz  |  Tını: {res['timbre_desc']}\n"
                    f"🎤 Şarkı Söyleme Oranı: %{res['singing_pct']}  |  Sessizlik: %{res['silence_pct']}\n"
                    f"📈 Dinamik Aralık: {res['dynamic_range_db']} dB  |  Tahmini SNR: {res['snr_db']} dB"
                )

                def _ui():
                    self.lbl_vocal_stats.configure(text=stat_txt)
                    p_img = res.get("plot_path")
                    if p_img and os.path.exists(p_img):
                        im = Image.open(p_img)
                        ctk_im = ctk.CTkImage(light_image=im, dark_image=im, size=(680, 320))
                        self.lbl_vocal_plot.configure(image=ctk_im)
                    print("[Vokal Lab] Akustik analiz tamamlandı! ✅")

                self.dispatch_to_ui(_ui)
            except Exception as e:
                print(f"[Vokal Lab Hatası] {e}")

        threading.Thread(target=_worker, daemon=True).start()

    # =========================================================================
    # VIEW 5: MODEL DOWNLOADER (MODEL İNDİRİCİ)
    # =========================================================================
    def _build_models_view(self):
        banner = ctk.CTkFrame(self.models_view_frame, fg_color="#181824", corner_radius=12)
        banner.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(
            banner,
            text="🌐 Topluluk & Hugging Face RVC Ses Modeli İndirici",
            font=("Century Gothic", 18, "bold"),
            text_color="#9966FF",
        ).pack(anchor="w", padx=15, pady=(10, 4))
        ctk.CTkLabel(
            banner,
            text="Özenle seçilmiş popüler ses modellerini tek tıkla kurun veya herhangi bir ZIP/PTH linki yapıştırın.",
            font=("Arial", 12),
            text_color="#AAA",
        ).pack(anchor="w", padx=15, pady=(0, 10))

        # Direct URL Download Card
        url_card = ctk.CTkFrame(self.models_view_frame, fg_color="#181824", corner_radius=10)
        url_card.pack(fill="x", padx=10, pady=6)

        ctk.CTkLabel(url_card, text="Özel Model İndirme Linki (URL):", font=("Arial", 12, "bold")).pack(side="left", padx=12, pady=10)
        self.model_url_entry = ctk.CTkEntry(url_card, placeholder_text="Hugging Face, Pixeldrain, Mega veya doğrudan .zip linki...", width=440)
        self.model_url_entry.pack(side="left", padx=6)

        self.btn_url_download = ctk.CTkButton(
            url_card, text="📥 İndir ve Kur", fg_color=PRIMARY_COLOR, font=("Arial", 12, "bold"),
            command=self.install_custom_url_model_gui
        )
        self.btn_url_download.pack(side="left", padx=8)

        # Curated Catalog List
        self.models_catalog_scroll = ctk.CTkScrollableFrame(self.models_view_frame, fg_color="#12121A", height=320, corner_radius=10)
        self.models_catalog_scroll.pack(fill="both", expand=True, padx=10, pady=6)

    def _refresh_models_ui(self):
        for w in self.models_catalog_scroll.winfo_children():
            w.destroy()
        models = search_curated_models("")
        for m in models:
            card = ctk.CTkFrame(self.models_catalog_scroll, fg_color="#181824", corner_radius=8, border_color="#2A2A3E", border_width=1)
            card.pack(fill="x", padx=6, pady=4)

            left = ctk.CTkFrame(card, fg_color="transparent")
            left.pack(side="left", fill="both", expand=True, padx=10, pady=8)

            lbl_t = ctk.CTkLabel(left, text=f"🎙️ {m['name']}", font=("Arial", 13, "bold"), text_color="#FFF", anchor="w")
            lbl_t.pack(fill="x")
            lbl_d = ctk.CTkLabel(left, text=f"{m['gender']}  •  Geliştirici: {m['creator']}  •  {m['description']}", font=("Arial", 11), text_color="#A390E4", anchor="w")
            lbl_d.pack(fill="x")

            # Check if already installed
            is_installed = os.path.exists(os.path.join(rvc_models_dir, m["name"].split()[0]))
            btn_txt = "✅ Kurulu" if is_installed else "📥 Modeli Kur"
            btn_color = "#333" if is_installed else PRIMARY_COLOR

            ctk.CTkButton(
                card, text=btn_txt, fg_color=btn_color, width=110,
                command=lambda mod=m: self.install_curated_model_gui(mod)
            ).pack(side="right", padx=12, pady=10)

    def install_curated_model_gui(self, mod):
        u = mod.get("download_url")
        if not u:
            print(f"[Model İndirici] Bu model yerel koleksiyonunuzda mevcut: {mod['name']}")
            return

        def _worker():
            try:
                download_and_extract_model(u, custom_model_name=mod["name"].split()[0], progress_cb=lambda msg, p: print(f"[İndirme] {msg}"))
                self.dispatch_to_ui(self.refresh_models)
                self.dispatch_to_ui(self._refresh_models_ui)
                send_windows_notification("Ripleytia AI Studio", f"'{mod['name']}' modeli başarıyla yüklendi! 🎙️")
            except Exception as e:
                print(f"[Model İndirme Hatası] {e}")

        threading.Thread(target=_worker, daemon=True).start()

    def install_custom_url_model_gui(self):
        u = self.model_url_entry.get().strip()
        if not u:
            return

        def _worker():
            try:
                installed_name = download_and_extract_model(u, progress_cb=lambda msg, p: print(f"[İndirme] {msg}"))
                self.dispatch_to_ui(self.refresh_models)
                self.dispatch_to_ui(lambda: self.model_url_entry.delete(0, "end"))
                send_windows_notification("Ripleytia AI Studio", f"'{installed_name}' modeli başarıyla kuruldu! 🎙️")
            except Exception as e:
                print(f"[Model Kurulum Hatası] {e}")

        threading.Thread(target=_worker, daemon=True).start()

    # =========================================================================
    # VIEW 6: AUTOMATIC DATASET BUILDER
    # =========================================================================
    def _build_dataset_view(self):
        banner = ctk.CTkFrame(self.dataset_view_frame, fg_color="#181824", corner_radius=12)
        banner.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(
            banner,
            text="📦 Otomatik Dataset Hazırlayıcı (Çoklu Dosya & Çoklu YouTube Stüdyosu)",
            font=("Century Gothic", 18, "bold"),
            text_color="#A982FA",
        ).pack(anchor="w", padx=15, pady=(10, 4))
        ctk.CTkLabel(
            banner,
            text="Birden fazla ses/video dosyası veya YouTube linki girin. UVR5 Special 2 ile vokalleri ayırıp stüdyo kalitesinde eğitime hazır veri seti oluşturun.",
            font=("Arial", 12),
            text_color="#AAA",
        ).pack(anchor="w", padx=15, pady=(0, 10))

        # Dataset Name & Target Folder
        name_card = ctk.CTkFrame(self.dataset_view_frame, fg_color="#181824", corner_radius=10)
        name_card.pack(fill="x", padx=10, pady=5)

        row_name = ctk.CTkFrame(name_card, fg_color="transparent")
        row_name.pack(fill="x", padx=15, pady=8)

        ctk.CTkLabel(row_name, text="Oluşturulacak Model / Dataset Adı:", font=("Arial", 13, "bold")).pack(side="left", padx=5)
        self.ds_name_entry = ctk.CTkEntry(row_name, textvariable=self.ds_name_var, width=280)
        self.ds_name_entry.pack(side="left", padx=10)

        # Dataset Quality Inspector Button
        btn_inspect = ctk.CTkButton(
            row_name, text="📈 Dataset Kalite & SNR Denetle", fg_color="#00695C", hover_color="#004D40", font=("Arial", 12, "bold"),
            command=self.run_dataset_inspection_gui
        )
        btn_inspect.pack(side="right", padx=10)

        # Sources Card
        src_card = ctk.CTkFrame(self.dataset_view_frame, fg_color="#181824", corner_radius=10)
        src_card.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(src_card, text="1. Adım: Veri Kaynakları (Yerel Dosyalar ve YouTube Linkleri)", font=("Arial", 14, "bold"), text_color="#A982FA").pack(
            anchor="w", padx=15, pady=(10, 4)
        )

        row_files = ctk.CTkFrame(src_card, fg_color="transparent")
        row_files.pack(fill="x", padx=15, pady=4)

        self.btn_ds_pick_files = ctk.CTkButton(
            row_files, text="📁 Çoklu Ses / Video Dosyası Seç (WAV, MP3, MP4, MKV...)",
            command=self.select_dataset_local_files, fg_color=PRIMARY_COLOR, hover_color=HOVER_COLOR
        )
        self.btn_ds_pick_files.pack(side="left", padx=5)

        self.btn_ds_clear_files = ctk.CTkButton(row_files, text="Seçimleri Temizle", command=self.clear_dataset_local_files, fg_color="#333", width=120)
        self.btn_ds_clear_files.pack(side="left", padx=10)

        self.lbl_ds_files_summary = ctk.CTkLabel(row_files, text="Seçilen Yerel Dosya: 0 adet", text_color="#DDD")
        self.lbl_ds_files_summary.pack(side="left", padx=10)

        ctk.CTkLabel(src_card, text="YouTube Linkleri (Her satıra bir adet video veya playlist linki yapıştırın):", font=("Arial", 12, "bold")).pack(
            anchor="w", padx=15, pady=(8, 2)
        )
        self.ds_yt_textbox = ctk.CTkTextbox(src_card, height=75, fg_color="#0A0A10")
        self.ds_yt_textbox.pack(fill="x", padx=15, pady=(0, 10))

        # Audio Cleaning Options Card
        clean_card = ctk.CTkFrame(self.dataset_view_frame, fg_color="#181824", corner_radius=10)
        clean_card.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(clean_card, text="2. Adım: Stüdyo Temizleme & Roformer Seçenekleri", font=("Arial", 14, "bold"), text_color="#A982FA").pack(
            anchor="w", padx=15, pady=(10, 4)
        )

        grid_clean = ctk.CTkFrame(clean_card, fg_color="transparent")
        grid_clean.pack(fill="x", padx=15, pady=4)

        ctk.CTkLabel(grid_clean, text="Vokal Ayırma Modeli:", font=("Arial", 13, "bold")).grid(row=0, column=0, sticky="w", padx=5, pady=6)
        self.ds_clean_dropdown = ctk.CTkOptionMenu(
            grid_clean,
            values=[
                "UVR5 Special 2 (Ultra Kalite)",
                "UVR5 Special 3 (Konsensüs + MelBand)",
                "UVR-MDX-NET-Voc_FT (Hızlı MDX)",
                "Temiz Ses / Vokal (Ayrıştırma Yapma)",
            ],
            variable=self.ds_clean_mode_var,
            width=280,
            fg_color=PRIMARY_COLOR,
            button_color=HOVER_COLOR,
        )
        self.ds_clean_dropdown.grid(row=0, column=1, sticky="w", padx=10, pady=6)

        ctk.CTkLabel(grid_clean, text="Örnekleme Hızı (Sample Rate):", font=("Arial", 13, "bold")).grid(row=0, column=2, sticky="w", padx=15, pady=6)
        ctk.CTkOptionMenu(
            grid_clean,
            values=["48000", "44100", "40000", "32000"],
            variable=self.ds_sample_rate_var,
            width=120,
            fg_color=PRIMARY_COLOR,
            button_color=HOVER_COLOR,
        ).grid(row=0, column=3, sticky="w", padx=5, pady=6)

        chk_row = ctk.CTkFrame(clean_card, fg_color="transparent")
        chk_row.pack(fill="x", padx=15, pady=5)
        ctk.CTkCheckBox(
            chk_row,
            text="Geri Vokalleri / Diğer Konuşmacıları Ayıkla (Karaoke Roformer)",
            variable=self.ds_remove_backups_var,
            fg_color=PRIMARY_COLOR,
            hover_color=HOVER_COLOR,
        ).pack(side="left", padx=(5, 25))
        ctk.CTkCheckBox(
            chk_row,
            text="Oda Yankısını (Reverb) Sil (Anvuew De-Reverb SDR 19.17)",
            variable=self.ds_apply_dereverb_var,
            fg_color=PRIMARY_COLOR,
            hover_color=HOVER_COLOR,
        ).pack(side="left", padx=5)

        sliders_row = ctk.CTkFrame(clean_card, fg_color="transparent")
        sliders_row.pack(fill="x", padx=15, pady=(5, 12))

        # Silence threshold slider
        s1_col = ctk.CTkFrame(sliders_row, fg_color="transparent")
        s1_col.pack(side="left", expand=True, fill="x", padx=5)
        self.lbl_ds_silence = ctk.CTkLabel(s1_col, text=f"Sessizlik Kesme Hassasiyeti: {self.ds_silence_db_var.get()} dB (Boşlukları Siler)")
        self.lbl_ds_silence.pack(anchor="w")
        ctk.CTkSlider(
            s1_col, from_=20, to=50, number_of_steps=30, variable=self.ds_silence_db_var, button_color=PRIMARY_COLOR, button_hover_color=HOVER_COLOR
        ).pack(fill="x", pady=2)
        self.ds_silence_db_var.trace_add(
            "write", lambda *a: self.lbl_ds_silence.configure(text=f"Sessizlik Kesme Hassasiyeti: {self.ds_silence_db_var.get()} dB (Boşlukları Siler)")
        )

        # Max duration cap slider
        s2_col = ctk.CTkFrame(sliders_row, fg_color="transparent")
        s2_col.pack(side="left", expand=True, fill="x", padx=15)
        self.lbl_ds_max_min = ctk.CTkLabel(s2_col, text="Hedef Maksimum Süre Sınırı: Sınırsız (Tümünü Kullan)")
        self.lbl_ds_max_min.pack(anchor="w")
        ctk.CTkSlider(
            s2_col, from_=0, to=60, number_of_steps=12, variable=self.ds_max_minutes_var, button_color=PRIMARY_COLOR, button_hover_color=HOVER_COLOR
        ).pack(fill="x", pady=2)

        def _update_max_min_lbl(*args):
            val = self.ds_max_minutes_var.get()
            txt = "Hedef Maksimum Süre Sınırı: Sınırsız (Tümünü Kullan)" if val == 0 else f"Hedef Maksimum Süre Sınırı: En Fazla {val} Dakika"
            self.lbl_ds_max_min.configure(text=txt)

        self.ds_max_minutes_var.trace_add("write", _update_max_min_lbl)

        # Automation chain checkboxes
        chain_frame = ctk.CTkFrame(self.dataset_view_frame, fg_color="#181824", corner_radius=12)
        chain_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkCheckBox(
            chain_frame,
            text="✨ Dataset Hazırlanınca Eğitim Sekmesine Aktar ve Optimal Epoch Sayısını Otomatik Ayarla",
            variable=self.ds_auto_send_train_var,
            fg_color=PRIMARY_COLOR,
            hover_color=HOVER_COLOR,
            font=("Arial", 13, "bold"),
        ).pack(anchor="w", padx=15, pady=(10, 5))
        ctk.CTkCheckBox(
            chain_frame,
            text="🚀 Dataset Hazırlanır Hazırlanmaz Tam Otomatik Model Eğitimini de Başlat (Gece Bırakma Modu)",
            variable=self.ds_auto_start_train_var,
            fg_color="#E67E22",
            hover_color="#D35400",
            font=("Arial", 13, "bold"),
        ).pack(anchor="w", padx=15, pady=(2, 10))

        # Summary Banner + Start Button
        self.lbl_ds_summary = ctk.CTkLabel(
            self.dataset_view_frame,
            text="📊 Hazır Dataset Durumu: Henüz işlem başlatılmadı.",
            font=("Consolas", 13, "bold"),
            text_color="#00E676",
        )
        self.lbl_ds_summary.pack(pady=(8, 2))

        self.btn_build_dataset = ctk.CTkButton(
            self.dataset_view_frame,
            text="🚀 TAM OTOMATİK DATASETİ HAZIRLA VE TEMİZLE",
            font=("Century Gothic", 16, "bold"),
            height=46,
            fg_color=PRIMARY_COLOR,
            hover_color=HOVER_COLOR,
            command=self.start_dataset_builder,
        )
        self.btn_build_dataset.pack(fill="x", padx=30, pady=(6, 12))

    def run_dataset_inspection_gui(self):
        # Scan datasets root
        target = self.tr_dataset_path_var.get() or os.path.join(DATASETS_ROOT, sanitize_name(self.ds_name_var.get()))
        if not os.path.exists(target):
            # Fallback to general datasets directory
            target = DATASETS_ROOT

        print(f"[Dataset Denetçisi] '{target}' taranıyor...")
        try:
            res = inspect_dataset_directory(target)
            msg = (
                f"📊 Kalite Derecesi: {res['grade']}\n"
                f"⏱️ Toplam Temiz Vokal: {res['duration_str']} ({res['total_files']} parça)\n"
                f"📈 Ortalama SNR: {res['snr_db']} dB  |  Kırpılma (Clipping): %{res['clipping_pct']}\n"
                f"💡 Tavsiye: {res['advice']}"
            )
            self.lbl_ds_summary.configure(text=msg, text_color=res["badge_color"])
            print(f"[Dataset Denetçisi] Sonuç: {res['grade']} ({res['duration_str']})")
        except Exception as e:
            print(f"[Dataset Denetçi Hatası] {e}")

    # =========================================================================
    # VIEW 7: VOICE MODEL TRAINING
    # =========================================================================
    def _build_train_view(self):
        banner = ctk.CTkFrame(self.train_view_frame, fg_color="#181824", corner_radius=12)
        banner.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(
            banner,
            text="🧠 Profesyonel RVC v2 Ses Modeli Eğitimi (Applio Entegrasyonu)",
            font=("Century Gothic", 18, "bold"),
            text_color="#00E5FF",
        ).pack(anchor="w", padx=15, pady=(10, 4))
        ctk.CTkLabel(
            banner,
            text="Hazırladığınız vokal veri setini Applio'nun en güçlü ön eğitim modelleri (Titan, Snowie, Ov2Super) ile eğitin.",
            font=("Arial", 12),
            text_color="#AAA",
        ).pack(anchor="w", padx=15, pady=(0, 10))

        # Dataset & Model Picker
        src_card = ctk.CTkFrame(self.train_view_frame, fg_color="#181824", corner_radius=10)
        src_card.pack(fill="x", padx=10, pady=5)

        g1 = ctk.CTkFrame(src_card, fg_color="transparent")
        g1.pack(fill="x", padx=15, pady=8)

        ctk.CTkLabel(g1, text="Eğitilecek Model Adı:", font=("Arial", 13, "bold")).grid(row=0, column=0, sticky="w", padx=5, pady=6)
        self.tr_name_entry = ctk.CTkEntry(g1, textvariable=self.tr_model_name_var, width=240)
        self.tr_name_entry.grid(row=0, column=1, sticky="w", padx=10, pady=6)

        ctk.CTkLabel(g1, text="Kullanılacak Dataset:", font=("Arial", 13, "bold")).grid(row=0, column=2, sticky="w", padx=15, pady=6)
        self.tr_ds_dropdown = ctk.CTkOptionMenu(g1, values=["Dataset Bekleniyor..."], width=280, fg_color=PRIMARY_COLOR, button_color=HOVER_COLOR)
        self.tr_ds_dropdown.grid(row=0, column=3, sticky="w", padx=5, pady=6)

        ctk.CTkButton(g1, text="Gözat", width=65, fg_color="#333", command=self.browse_training_dataset).grid(row=0, column=4, padx=6)
        ctk.CTkButton(g1, text="🔄", width=35, fg_color="#444", command=self.refresh_datasets_dropdown).grid(row=0, column=5, padx=2)

        # Presets Card
        preset_card = ctk.CTkFrame(self.train_view_frame, fg_color="#181824", corner_radius=10)
        preset_card.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(preset_card, text="Akıllı Eğitim Presetleri:", font=("Arial", 13, "bold"), text_color="#00E5FF").pack(anchor="w", padx=15, pady=(8, 4))

        p_row = ctk.CTkFrame(preset_card, fg_color="transparent")
        p_row.pack(fill="x", padx=15, pady=(0, 6))

        for pname in TRAINING_PRESETS.keys():
            short_p = pname.split("(")[0].strip()
            ctk.CTkButton(
                p_row, text=short_p, font=("Arial", 11, "bold"), height=30, fg_color="#3D3D58", hover_color="#5D3FD3",
                command=lambda p=pname: self.apply_training_preset(p)
            ).pack(side="left", padx=4)

        # Smart Epoch Calculation & Inspection Tools Row
        epoch_tool_row = ctk.CTkFrame(self.train_view_frame, fg_color="#181824", corner_radius=10)
        epoch_tool_row.pack(fill="x", padx=10, pady=5)

        self.btn_smart_epoch = ctk.CTkButton(
            epoch_tool_row, text="🎯 Dataset Süresini Analiz Et ve İdeal Epoch'u Ayarla",
            font=("Arial", 12, "bold"), fg_color="#00695C", hover_color="#004D40", height=32,
            command=self.smart_calculate_epochs_for_selected_dataset
        )
        self.btn_smart_epoch.pack(side="left", padx=12, pady=8)

        # Training Loss Curve Plot Button
        btn_loss_plot = ctk.CTkButton(
            epoch_tool_row, text="📈 Kayıp (Loss) Grafiği Çıkar",
            font=("Arial", 12, "bold"), fg_color="#5D3FD3", hover_color="#7B68EE", height=32,
            command=self.run_training_loss_plot_gui
        )
        btn_loss_plot.pack(side="left", padx=6, pady=8)

        # 5-Second Quick Test Button
        btn_quick_test = ctk.CTkButton(
            epoch_tool_row, text="⚡ 5s Hızlı Ses Testi (Önizleme)",
            font=("Arial", 12, "bold"), fg_color="#E67E22", hover_color="#D35400", height=32,
            command=self.run_quick_audio_test_gui
        )
        btn_quick_test.pack(side="left", padx=6, pady=8)

        # Training Tabs: Easy vs Pro
        self.train_tabs = ctk.CTkTabview(self.train_view_frame, width=880, fg_color="#181824", segmented_button_selected_color=PRIMARY_COLOR)
        self.train_tabs.pack(fill="x", padx=10, pady=5)

        tab_easy = self.train_tabs.add("⚡ Temel Ayarlar (Kolay Mod)")
        tab_pro = self.train_tabs.add("🔬 Pro Mimari & Model Ayarları")

        # Easy Grid
        eg = ctk.CTkFrame(tab_easy, fg_color="transparent")
        eg.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(eg, text="Toplam Epoch:").grid(row=0, column=0, sticky="w", padx=5, pady=6)
        ctk.CTkEntry(eg, textvariable=self.tr_total_epoch_var, width=100).grid(row=0, column=1, sticky="w", padx=5)

        ctk.CTkLabel(eg, text="Kaydetme Sıklığı:").grid(row=0, column=2, sticky="w", padx=15, pady=6)
        ctk.CTkEntry(eg, textvariable=self.tr_save_every_var, width=100).grid(row=0, column=3, sticky="w", padx=5)

        ctk.CTkLabel(eg, text="Batch Size (RTX 4060 için 6 önerilir):").grid(row=1, column=0, sticky="w", padx=5, pady=6)
        ctk.CTkSlider(eg, from_=2, to=16, number_of_steps=7, variable=self.tr_batch_size_var).grid(row=1, column=1, sticky="ew", padx=5)
        self.lbl_tr_batch = ctk.CTkLabel(eg, text=str(self.tr_batch_size_var.get()))
        self.lbl_tr_batch.grid(row=1, column=2, sticky="w", padx=5)
        self.tr_batch_size_var.trace_add("write", lambda *a: self.lbl_tr_batch.configure(text=str(self.tr_batch_size_var.get())))

        # Action Cards
        action_card = ctk.CTkFrame(self.train_view_frame, fg_color="#181824", corner_radius=10)
        action_card.pack(fill="x", padx=10, pady=8)

        self.btn_oneclick_train = ctk.CTkButton(
            action_card,
            text="🚀 TEK TIKLA TAM OTOMATİK SES MODELİ EĞİT (Ön İşle + Eğit + Stüdyoya Kur)",
            font=("Century Gothic", 16, "bold"),
            height=46,
            fg_color=PRIMARY_COLOR,
            hover_color=HOVER_COLOR,
            command=lambda: self.start_voice_training(run_preprocess=True, run_extract=True, run_index=True, run_train=True)
        )
        self.btn_oneclick_train.pack(fill="x", padx=20, pady=(12, 6))

        self.btn_stop_train = ctk.CTkButton(
            action_card, text="⏹️ Eğitimi Güvenle Durdur ve Modeli Kaydet",
            fg_color="#8b0000", hover_color="#5a0000", command=self.stop_voice_training
        )
        self.btn_stop_train.pack(pady=4)

    def run_training_loss_plot_gui(self):
        mod = self.tr_model_name_var.get().strip() or self.model_var.get()
        print(f"[Eğitim Grafiği] '{mod}' için kayıp değerleri taranıyor...")
        try:
            p = generate_training_loss_plot(mod)
            if p and os.path.exists(p):
                print(f"[Eğitim Grafiği] Grafik başarıyla kaydedildi: {p}")
                os.system(f'explorer /select,"{os.path.normpath(p)}"')
            else:
                print(f"[Eğitim Grafiği] Bu model için kayıtlı eğitim checkpointi bulunamadı.")
        except Exception as e:
            print(f"[Eğitim Grafik Hatası] {e}")

    def run_quick_audio_test_gui(self):
        mod = self.tr_model_name_var.get().strip() or self.model_var.get()
        print(f"[Hızlı Test] '{mod}' için 5 saniyelik test vokali üretiliyor...")

        def _worker():
            try:
                out = run_quick_audio_test(mod, progress_cb=lambda msg, p: print(f"[Hızlı Test] {msg}"))
                self.dispatch_to_ui(lambda: self.audio_player.load_audio(out, autoplay=True))
                print(f"[Hızlı Test] Test vokali hazır ve oynatıcıya yüklendi! 🎧")
            except Exception as e:
                print(f"[Hızlı Test Hatası] {e}")

        threading.Thread(target=_worker, daemon=True).start()

    # =========================================================================
    # CORE HANDLERS & WRAPPERS
    # =========================================================================
    def run_smart_pitch_recommendation(self):
        s = self.song_input_var.get().strip()
        m = self.model_var.get()
        if not s:
            print("[Hata] Lütfen önce bir şarkı dosyası seçin.")
            return

        print(f"[Akıllı Pitch] '{os.path.basename(s)}' taranıyor...")

        def _worker():
            res = analyze_pitch_and_recommend(s, target_model_name=m)

            def _ui():
                self.pitch_var.set(res["recommended_pitch"])
                txt = f"🎯 Şarkı Tonu: {res['detected_key']} | {res['source_gender_est']} ➔ {res['explanation']}"
                self.lbl_pitch_rec_status.configure(text=txt, text_color="#69F0AE")
                print(f"[Akıllı Pitch] {txt}")

            self.dispatch_to_ui(_ui)

        threading.Thread(target=_worker, daemon=True).start()

    def run_style_advisor(self):
        s = self.song_input_var.get().strip()
        if not s:
            print("[Hata] Lütfen önce bir şarkı seçin.")
            return

        def _worker():
            res = advise_style_parameters(s)

            def _ui():
                self.f0_method_var.set(res["recommended_algo"])
                self.index_rate_var.set(res["recommended_index"])
                self.filter_radius_var.set(res["recommended_filter_radius"])
                self.protect_var.set(res["recommended_protect"])
                self.reverb_wet_var.set(res["recommended_reverb_wet"])
                txt = f"💡 Stil: {res['genre']} ({res['bpm']} BPM) | {res['explanation']}"
                self.lbl_pitch_rec_status.configure(text=txt, text_color="#00d2d3")
                print(f"[Stil Danışmanı] {txt}")

            self.dispatch_to_ui(_ui)

        threading.Thread(target=_worker, daemon=True).start()

    def toggle_lyrics_panel(self):
        if self.lyrics_panel.winfo_ismapped():
            self.lyrics_panel.pack_forget()
        else:
            self.lyrics_panel.pack(fill="x", padx=10, pady=6, after=self.generate_btn)
            # Fetch lyrics
            s = os.path.basename(self.song_input_var.get()).split(".")[0]
            if s:
                self.lyrics_text.delete("0.0", "end")
                self.lyrics_text.insert("end", "Şarkı sözleri aranıyor...\n")

                def _w():
                    data = fetch_lyrics(s)
                    if data.get("error"):
                        self.dispatch_to_ui(lambda: (self.lyrics_text.delete("0.0", "end"), self.lyrics_text.insert("end", f"Sözler bulunamadı: {data['error']}")))
                    else:
                        tr_txt = translate_lyrics_text(data["plain_lyrics"], "tr")
                        full = f"=== {data['title']} ({data['artist']}) ===\n\n[TÜRKÇE ÇEVİRİ]\n{tr_txt}\n\n[ORİJİNAL SÖZLER]\n{data['plain_lyrics']}"
                        self.dispatch_to_ui(lambda: (self.lyrics_text.delete("0.0", "end"), self.lyrics_text.insert("end", full)))

                threading.Thread(target=_w, daemon=True).start()

    def select_dataset_local_files(self):
        files = filedialog.askopenfilenames(
            title="Dataset İçin Ses / Video Dosyaları Seç",
            filetypes=[("Tüm Desteklenen Medyalar", "*.wav *.mp3 *.flac *.ogg *.m4a *.mp4 *.mkv *.avi *.mov *.webm *.aac"), ("Tüm Dosyalar", "*.*")],
        )
        if files:
            for f in files:
                if f not in self.ds_local_files:
                    self.ds_local_files.append(f)
            self.lbl_ds_files_summary.configure(text=f"Seçilen Yerel Dosya: {len(self.ds_local_files)} adet", text_color="#00E676")

    def clear_dataset_local_files(self):
        self.ds_local_files.clear()
        self.lbl_ds_files_summary.configure(text="Seçilen Yerel Dosya: 0 adet", text_color="#DDD")

    def start_dataset_builder(self):
        name = self.ds_name_var.get().strip()
        if not name:
            print("[Hata] Lütfen bir dataset / model adı belirleyin.")
            return

        yt_links = [l.strip() for l in self.ds_yt_textbox.get("1.0", "end").split("\n") if l.strip() and l.strip().startswith("http")]
        if not self.ds_local_files and not yt_links:
            print("[Hata] Lütfen en az bir yerel ses/video dosyası veya YouTube linki ekleyin.")
            return

        self.btn_build_dataset.configure(state="disabled", text="⏳ Dataset Hazırlanıyor... Lütfen Bekleyin")

        def _worker():
            try:
                res = build_automatic_dataset(
                    name=name,
                    local_files=self.ds_local_files,
                    youtube_urls=yt_links,
                    clean_mode=self.ds_clean_mode_var.get(),
                    remove_backups=self.ds_remove_backups_var.get(),
                    apply_dereverb=self.ds_apply_dereverb_var.get(),
                    sample_rate=int(self.ds_sample_rate_var.get()),
                    silence_db=self.ds_silence_db_var.get(),
                    max_duration_minutes=self.ds_max_minutes_var.get(),
                    progress_callback=print,
                )
                print(f"[Dataset Builder] Dataset tamamlandı: {res['clean_dataset_dir']}")
                send_windows_notification("Ripleytia AI Studio", f"'{name}' dataseti başarıyla hazırlandı! 📦")
                self.dispatch_to_ui(self.refresh_datasets_dropdown)
            except Exception as e:
                print(f"[Dataset Hatası] {e}")
            finally:
                self.dispatch_to_ui(lambda: self.btn_build_dataset.configure(state="normal", text="🚀 TAM OTOMATİK DATASETİ HAZIRLA VE TEMİZLE"))

        threading.Thread(target=_worker, daemon=True).start()

    def apply_training_preset(self, preset_name: str):
        if preset_name in TRAINING_PRESETS:
            cfg = TRAINING_PRESETS[preset_name]
            self.tr_preset_var.set(preset_name)
            self.tr_sample_rate_var.set(cfg["sample_rate"])
            self.tr_vocoder_var.set(cfg["vocoder"])
            self.tr_f0_method_var.set(cfg["f0_method"])
            self.tr_embedder_var.set(cfg["embedder_model"])
            self.tr_pretrain_var.set(cfg["pretrain_type"])
            self.tr_batch_size_var.set(cfg["batch_size"])
            self.tr_total_epoch_var.set(cfg["total_epoch"])
            self.tr_save_every_var.set(cfg["save_every_epoch"])
            print(f"[Preset] '{preset_name}' ayarları uygulandı!")

    def refresh_datasets_dropdown(self):
        ds = get_available_datasets()
        if ds:
            self.tr_ds_dropdown.configure(values=[os.path.basename(d) for d in ds])
            self.tr_dataset_path_var.set(ds[0])
            self.tr_ds_dropdown.set(os.path.basename(ds[0]))

    def browse_training_dataset(self):
        d = filedialog.askdirectory(initialdir=DATASETS_ROOT)
        if d:
            self.tr_dataset_path_var.set(d)
            self.tr_ds_dropdown.set(os.path.basename(d))

    def smart_calculate_epochs_for_selected_dataset(self):
        target = self.tr_dataset_path_var.get()
        if not target or not os.path.exists(target):
            ds = get_available_datasets()
            target = ds[0] if ds else ""
        if not target:
            return
        rec = calculate_recommended_epochs(target)
        self.tr_total_epoch_var.set(rec)
        print(f"[Smart Epoch] '{os.path.basename(target)}' için ideal epoch hesaplandı: {rec}")

    def start_voice_training(self, run_preprocess=True, run_extract=True, run_index=True, run_train=True):
        mname = self.tr_model_name_var.get().strip()
        dpath = self.tr_dataset_path_var.get()
        if not mname or not dpath:
            print("[Hata] Model adı veya dataset eksik.")
            return

        self.btn_oneclick_train.configure(state="disabled", text="⏳ Model Eğitiliyor...")

        def _worker():
            try:
                res = run_full_voice_training_pipeline(
                    model_name=mname,
                    dataset_path=dpath,
                    sample_rate=int(self.tr_sample_rate_var.get()),
                    total_epoch=self.tr_total_epoch_var.get(),
                    batch_size=self.tr_batch_size_var.get(),
                    save_every_epoch=self.tr_save_every_var.get(),
                    progress_cb=print
                )
                print(f"[Eğitim] Model eğitimi tamamlandı: {res['model_name']}")
                send_windows_notification("Ripleytia AI Studio", f"'{mname}' ses modeli eğitimi tamamlandı! 🎙️")
                self.dispatch_to_ui(self.refresh_models)
            except Exception as e:
                print(f"[Eğitim Hatası] {e}")
            finally:
                self.dispatch_to_ui(lambda: self.btn_oneclick_train.configure(state="normal", text="🚀 TEK TIKLA TAM OTOMATİK SES MODELİ EĞİT"))

        threading.Thread(target=_worker, daemon=True).start()

    def stop_voice_training(self):
        stop_active_training()
        print("[Eğitim] Durdurma sinyali gönderildi.")

    def update_texts(self):
        self.title(translations[self.lang]["title"])
        self.btn_save_settings.configure(text=translations[self.lang]["save_settings"])
        self.mode_btn_cover.configure(text=translations[self.lang]["mode_cover"])
        self.mode_btn_batch.configure(text=translations[self.lang]["mode_batch"])
        self.mode_btn_history.configure(text=translations[self.lang]["mode_history"])
        self.mode_btn_analysis.configure(text=translations[self.lang]["mode_analysis"])
        self.mode_btn_models.configure(text=translations[self.lang]["mode_models"])
        self.mode_btn_dataset.configure(text=translations[self.lang]["mode_dataset"])
        self.mode_btn_train.configure(text=translations[self.lang]["mode_train"])
        self.lbl_model_select.configure(text=translations[self.lang]["model_select"])
        self.btn_refresh.configure(text=translations[self.lang]["refresh"])
        self.btn_upload.configure(text=translations[self.lang]["upload_model"])
        self.lbl_pipeline_select.configure(text=translations[self.lang]["pipeline_select"])
        self.lbl_input.configure(text=translations[self.lang]["input_select"])
        self.btn_browse.configure(text=translations[self.lang]["browse"])
        if hasattr(self, "btn_yt_toggle"):
            self.btn_yt_toggle.configure(text=translations[self.lang]["yt_search_toggle"])
        if hasattr(self, "lbl_yt_title"):
            self.lbl_yt_title.configure(text=translations[self.lang]["yt_panel_title"])
        if hasattr(self, "yt_search_entry"):
            self.yt_search_entry.configure(placeholder_text=translations[self.lang]["yt_search_placeholder"])
        if hasattr(self, "btn_yt_search"):
            self.btn_yt_search.configure(text=translations[self.lang]["yt_btn_search"])
        self.lbl_output_dir.configure(text=translations[self.lang]["output_select"])
        self.btn_browse_dir.configure(text=translations[self.lang]["browse_dir"])
        self.lbl_vocal_pitch.configure(text=translations[self.lang]["vocal_pitch"])
        self.lbl_overall_pitch.configure(text=translations[self.lang]["overall_pitch"])
        self.sw_auto_mode.configure(text=translations[self.lang]["auto_mode"])
        self.generate_btn.configure(text=translations[self.lang]["generate"])
        self.btn_open_folder.configure(text=translations[self.lang]["open_folder"])
        self.play_btn.configure(text=translations[self.lang]["play_last"])
        self.btn_clear_log.configure(text=translations[self.lang]["clear_log"])
        self.lbl_logs.configure(text=translations[self.lang]["logs"])

    def save_settings(self):
        settings = {
            "lang": self.lang,
            "theme": self.theme_name_var.get(),
            "model": self.model_var.get(),
            "pipeline": self.pipeline_var.get(),
            "pitch": self.pitch_var.get(),
            "pitch_all": self.pitch_all_var.get(),
            "main_gain": self.main_gain_var.get(),
            "backup_gain": self.backup_gain_var.get(),
            "inst_gain": self.inst_gain_var.get(),
            "mute_backup": self.mute_backup_var.get(),
            "reverb_size": self.reverb_size_var.get(),
            "reverb_wet": self.reverb_wet_var.get(),
            "reverb_dry": self.reverb_dry_var.get(),
            "reverb_damp": self.reverb_damp_var.get(),
            "index_rate": self.index_rate_var.get(),
            "filter_radius": self.filter_radius_var.get(),
            "rms_mix": self.rms_mix_var.get(),
            "protect": self.protect_var.get(),
            "f0_method": self.f0_method_var.get(),
            "crepe_hop": self.crepe_hop_var.get(),
            "noise_gate": self.noise_gate_var.get(),
            "noise_gate_thresh": self.noise_gate_thresh_var.get(),
            "agg_dereverb": self.aggressive_dereverb_var.get(),
            "keep_files": self.keep_files_var.get(),
            "output_format": self.output_format_var.get(),
            "output_dir": self.output_dir_var.get(),
            "deesser": self.deesser_var.get(),
            "studio_eq": self.studio_eq_var.get(),
            "sidechain": self.sidechain_var.get(),
            "lufs_master": self.lufs_master_var.get(),
            "breath_pass": self.breath_passthrough_var.get(),
            "doubler": self.doubler_var.get(),
            "harmonies": self.harmonies_var.get(),
            "harmony_mode": self.harmony_mode_var.get(),
            "formant_shift": self.formant_shift_var.get(),
            "dyn_pitch": self.dynamic_pitch_rules_var.get(),
        }
        with open(settings_path, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=4)
        print(translations[self.lang]["settings_saved"])

    def load_settings(self):
        if not os.path.exists(settings_path):
            return
        try:
            with open(settings_path, 'r', encoding='utf-8') as f:
                s = json.load(f)
            self.lang = s.get("lang", "tr")
            th = s.get("theme", "Gothic Purple")
            self.theme_name_var.set(th)
            self.change_theme(th)
            if s.get("model") in self.models:
                self.model_var.set(s["model"])
            if s.get("pipeline") in self.pipelines:
                self.pipeline_var.set(s["pipeline"])
            self.pitch_var.set(float(s.get("pitch", 0.0)))
            self.pitch_all_var.set(float(s.get("pitch_all", 0.0)))
            self.main_gain_var.set(s.get("main_gain", 0))
            self.backup_gain_var.set(s.get("backup_gain", 0))
            self.inst_gain_var.set(s.get("inst_gain", 0))
            self.mute_backup_var.set(s.get("mute_backup", True))
            self.reverb_size_var.set(s.get("reverb_size", 0.15))
            self.reverb_wet_var.set(s.get("reverb_wet", 0.2))
            self.reverb_dry_var.set(s.get("reverb_dry", 0.8))
            self.reverb_damp_var.set(s.get("reverb_damp", 0.7))
            self.index_rate_var.set(s.get("index_rate", 0.5))
            self.filter_radius_var.set(s.get("filter_radius", 3))
            self.rms_mix_var.set(s.get("rms_mix", 0.25))
            self.protect_var.set(s.get("protect", 0.33))
            self.f0_method_var.set(s.get("f0_method", "rmvpe"))
            self.crepe_hop_var.set(s.get("crepe_hop", 128))
            self.noise_gate_var.set(s.get("noise_gate", False))
            self.noise_gate_thresh_var.set(s.get("noise_gate_thresh", -35))
            self.aggressive_dereverb_var.set(s.get("agg_dereverb", True))
            self.keep_files_var.set(s.get("keep_files", False))
            self.output_format_var.set(s.get("output_format", "mp3"))
            self.output_dir_var.set(s.get("output_dir", output_dir))
            self.deesser_var.set(s.get("deesser", True))
            self.studio_eq_var.set(s.get("studio_eq", True))
            self.sidechain_var.set(s.get("sidechain", True))
            self.lufs_master_var.set(s.get("lufs_master", True))
            self.breath_passthrough_var.set(s.get("breath_pass", True))
            self.doubler_var.set(s.get("doubler", False))
            self.harmonies_var.set(s.get("harmonies", False))
            self.harmony_mode_var.set(s.get("harmony_mode", "3rd_and_5th (+Maj 3rd & 5th)"))
            self.formant_shift_var.set(s.get("formant_shift", 0.0))
            self.dynamic_pitch_rules_var.set(s.get("dyn_pitch", ""))
        except Exception as e:
            print(f"[Settings] Error loading settings: {e}")

    def refresh_models(self):
        self.models = get_current_models(rvc_models_dir)
        self.model_dropdown.configure(values=self.models if self.models else [""])
        self.compare_dropdown.configure(values=self.models if self.models else [""])
        self.batch_model_dropdown.configure(values=self.models if self.models else [""])
        if self.models and not self.model_var.get():
            self.model_var.set(self.models[0])

    def upload_model_gui(self):
        f = filedialog.askopenfilename(filetypes=[("Zip Arşivi", "*.zip"), ("PTH Dosyası", "*.pth")])
        if not f:
            return
        mname = os.path.splitext(os.path.basename(f))[0]
        tdir = os.path.join(rvc_models_dir, mname)
        os.makedirs(tdir, exist_ok=True)
        if f.endswith(".zip"):
            import zipfile
            with zipfile.ZipFile(f, 'r') as z:
                z.extractall(tdir)
        else:
            shutil.copy2(f, tdir)
        self.refresh_models()
        print(f"[Model] '{mname}' başarıyla stüdyoya eklendi!")

    def check_file_cache(self, *args):
        pass

    def _check_ui_queue(self):
        try:
            while not self.ui_queue.empty():
                fn, args, kwargs = self.ui_queue.get_nowait()
                try:
                    fn(*args, **kwargs)
                except Exception as e:
                    print(f"[UI Queue Error] {e}")
        finally:
            self.after(50, self._check_ui_queue)

    def dispatch_to_ui(self, fn, *args, **kwargs):
        self.ui_queue.put((fn, args, kwargs))

    def toggle_yt_panel(self):
        if self.yt_panel.winfo_ismapped():
            self.yt_panel.pack_forget()
        else:
            self.yt_panel.pack(fill="x", padx=10, pady=(5, 10), after=self.input_frame.winfo_children()[0])
            self.yt_search_entry.focus_set()

    def clear_yt_search(self):
        self.yt_search_var.set("")
        for w in self.yt_results_frame.winfo_children():
            w.destroy()
        self.lbl_yt_status.configure(text=translations[self.lang]["yt_search_hint"], text_color="#AAAAAA")

    def do_yt_search(self):
        q = self.yt_search_var.get().strip()
        if not q:
            return
        self.btn_yt_search.configure(state="disabled")
        self.lbl_yt_status.configure(text=translations[self.lang]["yt_searching"], text_color="#00E5FF")

        def _w():
            err = None
            res = []
            try:
                res = search_youtube(q, max_results=8)
            except Exception as e:
                err = str(e)
            self.dispatch_to_ui(self._render_yt_results, res, err)

        threading.Thread(target=_w, daemon=True).start()

    def _render_yt_results(self, results, error):
        self.btn_yt_search.configure(state="normal")
        for w in self.yt_results_frame.winfo_children():
            w.destroy()

        if error:
            self.lbl_yt_status.configure(text=f"❌ Arama hatası: {error}", text_color="#FF5252")
            return
        if not results:
            self.lbl_yt_status.configure(text=translations[self.lang]["yt_no_results"], text_color="#FFB300")
            return

        self.lbl_yt_status.configure(text=f"✅ {len(results)} sonuç bulundu:", text_color="#69F0AE")

        for video in results:
            card = ctk.CTkFrame(self.yt_results_frame, fg_color="#181824", corner_radius=8, border_color="#2A2A3E", border_width=1)
            card.pack(fill="x", padx=4, pady=3)

            # Thumbnail Preview
            thumb_p = get_cached_thumbnail(video.get("id"), video.get("thumbnail"))
            if thumb_p and os.path.exists(thumb_p):
                try:
                    pim = Image.open(thumb_p)
                    ctk_pim = ctk.CTkImage(light_image=pim, dark_image=pim, size=(80, 45))
                    lbl_th = ctk.CTkLabel(card, image=ctk_pim, text="")
                    lbl_th.pack(side="left", padx=(6, 2), pady=4)
                except Exception:
                    pass

            info_col = ctk.CTkFrame(card, fg_color="transparent")
            info_col.pack(side="left", fill="both", expand=True, padx=8, pady=4)

            lbl_t = ctk.CTkLabel(info_col, text=f"🎵 {video['title'][:70]}", font=("Arial", 12, "bold"), text_color="#FFF", anchor="w")
            lbl_t.pack(fill="x", anchor="w")
            lbl_m = ctk.CTkLabel(info_col, text=f"👤 {video['channel']}  •  ⏱️ {video['duration_string']}  •  👁️ {video.get('views', '')}", font=("Arial", 11), text_color="#A390E4", anchor="w")
            lbl_m.pack(fill="x", anchor="w")

            btn_col = ctk.CTkFrame(card, fg_color="transparent")
            btn_col.pack(side="right", padx=6, pady=4)

            ctk.CTkButton(
                btn_col, text="📥 İndir", width=80, height=28, fg_color="#333",
                command=lambda v=video: self.start_yt_download(v, auto_cover=False)
            ).pack(side="left", padx=2)

            ctk.CTkButton(
                btn_col, text="⚡ Cover Yap", width=100, height=28, fg_color=PRIMARY_COLOR, hover_color=HOVER_COLOR,
                command=lambda v=video: self.start_yt_download(v, auto_cover=True)
            ).pack(side="left", padx=2)

            ctk.CTkButton(
                btn_col, text="➕ Kuyruğa Ekle", width=95, height=28, fg_color="#00695C",
                command=lambda v=video: (self.batch_manager.add_item(v["url"], v["title"], self.model_var.get(), self.pitch_var.get()), print(f"[Kuyruk] '{v['title']}' eklendi!"))
            ).pack(side="left", padx=2)

    def start_yt_download(self, video, auto_cover=False):
        ddir = os.path.join(output_dir, "youtube_downloads")
        os.makedirs(ddir, exist_ok=True)
        title_s = video["title"][:35]
        self.lbl_yt_status.configure(text=f"📥 İndiriliyor: {title_s}...", text_color="#00E5FF")

        def _w():
            try:
                def pcb(msg):
                    self.dispatch_to_ui(lambda m=msg: self.lbl_yt_status.configure(text=f"📥 [{title_s}...] {m}", text_color="#00E5FF"))

                mp3_p = download_youtube_audio(video["url"], ddir, progress_cb=pcb)

                def _done():
                    self.song_input_var.set(mp3_p)
                    self.lbl_yt_status.configure(text=f"✅ İndirildi: {os.path.basename(mp3_p)}", text_color="#69F0AE")
                    if auto_cover:
                        self.start_generation()

                self.dispatch_to_ui(_done)
            except Exception as e:
                self.dispatch_to_ui(lambda err=str(e): self.lbl_yt_status.configure(text=f"❌ İndirme hatası: {err}", text_color="#FF5252"))

        threading.Thread(target=_w, daemon=True).start()

    def browse_file(self):
        f = filedialog.askopenfilename(filetypes=[("Audio Files", "*.mp3 *.wav *.flac *.m4a")])
        if f:
            self.song_input_var.set(f)

    def browse_dir(self):
        d = filedialog.askdirectory()
        if d:
            self.output_dir_var.set(d)

    def open_output_folder(self):
        target_file = self.last_output_path
        if target_file and os.path.exists(target_file):
            try:
                import subprocess
                subprocess.Popen(f'explorer /select,"{os.path.normpath(target_file)}"')
                return
            except Exception as ee:
                print(f"[Explorer Uyarısı] {ee}")
        odir = self.output_dir_var.get().strip() if hasattr(self, "output_dir_var") else ""
        if not odir or not os.path.exists(odir):
            odir = output_dir
        if os.path.exists(odir):
            os.startfile(odir)

    def play_output(self):
        if self.last_output_path and os.path.exists(self.last_output_path):
            self.audio_player.load_audio(self.last_output_path, autoplay=True)

    def clear_logs(self):
        self.log_textbox.delete("1.0", "end")

    # =========================================================================
    # PIPELINE EXECUTION (GENERATION)
    # =========================================================================
    def start_generation(self):
        song = self.song_input_var.get()
        model = self.model_var.get()
        if not song or not model:
            print("[Hata] Lütfen bir ses modeli ve şarkı seçin.")
            return

        # Unload audio player to release Windows file locks on previous outputs
        if hasattr(self, "audio_player") and self.audio_player:
            try:
                self.audio_player.unload()
            except Exception as pe:
                print(f"[Player Uyarısı] {pe}")

        self.generate_btn.configure(state="disabled", text=translations[self.lang]["generating"])

        def run_pipeline():
            try:
                if hasattr(self, "audio_player") and self.audio_player:
                    self.dispatch_to_ui(self.audio_player.unload)
                print("=" * 60)
                print(f"[Başlat] Ripleytia AI Cover Boru Hattı Başlatılıyor...")
                print(f"[Şarkı] {song}")
                print(f"[Model] {model}")
                print(f"[UVR Modu] {self.pipeline_var.get()}")
                print("=" * 60)

                bgain = -100 if self.mute_backup_var.get() else self.backup_gain_var.get()

                out_path = song_cover_pipeline(
                    song_input=song,
                    voice_model=model,
                    pitch_change=self.pitch_var.get(),
                    keep_files=self.keep_files_var.get(),
                    is_webui=0,
                    main_gain=self.main_gain_var.get(),
                    backup_gain=bgain,
                    inst_gain=self.inst_gain_var.get(),
                    index_rate=self.index_rate_var.get(),
                    filter_radius=self.filter_radius_var.get(),
                    rms_mix_rate=self.rms_mix_var.get(),
                    f0_method=self.f0_method_var.get(),
                    crepe_hop_length=self.crepe_hop_var.get(),
                    protect=self.protect_var.get(),
                    force_reprocess=self.force_reprocess_var.get(),
                    auto_mode=self.auto_mode_var.get(),
                    pitch_change_all=self.pitch_all_var.get(),
                    reverb_rm_size=self.reverb_size_var.get(),
                    reverb_wet=self.reverb_wet_var.get(),
                    reverb_dry=self.reverb_dry_var.get(),
                    reverb_damping=self.reverb_damp_var.get(),
                    output_format=self.output_format_var.get(),
                    aggressive_dereverb=self.aggressive_dereverb_var.get(),
                    uvr_model=self.pipeline_var.get(),
                    vocal_noise_gate=self.noise_gate_var.get(),
                    noise_gate_threshold=self.noise_gate_thresh_var.get(),
                    enable_deesser=self.deesser_var.get(),
                    enable_studio_eq=self.studio_eq_var.get(),
                    enable_doubler=self.doubler_var.get(),
                    enable_sidechain=self.sidechain_var.get(),
                    enable_lufs_mastering=self.lufs_master_var.get(),
                    enable_breath_passthrough=self.breath_passthrough_var.get(),
                    formant_shift=self.formant_shift_var.get(),
                    generate_harmonies=self.harmonies_var.get(),
                    harmony_mode=self.harmony_mode_var.get().split()[0],
                    dynamic_pitch_rules=self.dynamic_pitch_rules_var.get(),
                )

                # Ensure the final output file is placed directly in the designated output folder
                target_out_dir = self.output_dir_var.get().strip() if hasattr(self, "output_dir_var") else ""
                if not target_out_dir:
                    target_out_dir = output_dir

                os.makedirs(target_out_dir, exist_ok=True)
                dest_file = os.path.join(target_out_dir, os.path.basename(out_path))
                if os.path.abspath(out_path) != os.path.abspath(dest_file):
                    try:
                        import shutil
                        shutil.copyfile(out_path, dest_file)
                        print(f"[+] Sarki secilen cikti klasorune aktarildi: {dest_file}")
                        out_path = dest_file
                    except Exception as ce:
                        print(f"[Uyari] Cikti klasorune kopyalama hatasi: {ce}")

                self.last_output_path = out_path

                # 1. Save to Cover History
                add_history_entry(
                    song_name=os.path.basename(song),
                    model_name=model,
                    pitch=self.pitch_var.get(),
                    algo=self.f0_method_var.get(),
                    uvr_model=self.pipeline_var.get(),
                    output_path=out_path
                )

                # 2. Evaluate AI Clone Quality
                q_res = evaluate_clone_quality(song, out_path)

                def _ui_done():
                    self.play_btn.configure(state="normal")
                    self.audio_player.load_audio(out_path, autoplay=True)
                    # Update Quality Scorecard
                    self.lbl_quality_stars.configure(
                        text=f"{q_res['stars']} AI Kalite Puani: %{q_res['overall_score']} ({q_res['verdict']})"
                    )
                    self.lbl_quality_metrics.configure(
                        text=f"Dogallik: %{q_res['naturalness']}  |  Ton Sadakati: %{q_res['pitch_accuracy']}  |  Tini Uyumu: %{q_res['timbre_match']}"
                    )
                    # Windows Notification
                    send_windows_notification("Ripleytia AI Studio", f"'{os.path.basename(song)}' ({model}) coveri hazir! 🎧")

                self.dispatch_to_ui(_ui_done)

                # 3. Model Comparison A/B Test (if enabled)
                if self.enable_compare_var.get() and self.compare_model_var.get():
                    comp_model = self.compare_model_var.get()
                    print(f"\n[A/B Test] Karsilastirma modeliyle ({comp_model}) ikinci cover uretiliyor...")
                    out_path_b = song_cover_pipeline(
                        song_input=song,
                        voice_model=comp_model,
                        pitch_change=self.pitch_var.get(),
                        f0_method=self.f0_method_var.get(),
                        uvr_model=self.pipeline_var.get(),
                        enable_deesser=self.deesser_var.get(),
                        enable_studio_eq=self.studio_eq_var.get(),
                        enable_doubler=self.doubler_var.get(),
                        enable_sidechain=self.sidechain_var.get(),
                        enable_lufs_mastering=self.lufs_master_var.get(),
                        enable_breath_passthrough=self.breath_passthrough_var.get(),
                        formant_shift=self.formant_shift_var.get(),
                        generate_harmonies=self.harmonies_var.get(),
                        harmony_mode=self.harmony_mode_var.get().split()[0],
                        dynamic_pitch_rules=self.dynamic_pitch_rules_var.get(),
                    )
                    dest_file_b = os.path.join(target_out_dir, os.path.basename(out_path_b))
                    if os.path.abspath(out_path_b) != os.path.abspath(dest_file_b):
                        try:
                            import shutil
                            shutil.copyfile(out_path_b, dest_file_b)
                            out_path_b = dest_file_b
                        except Exception:
                            pass
                    print(f"[A/B Test] Model B tamamlandi: {out_path_b}")

            except Exception as e:
                print(f"\n[Error] Pipeline failed: {str(e)}")
            finally:
                self.dispatch_to_ui(lambda: self.generate_btn.configure(state="normal", text=translations[self.lang]["generate"]))

        threading.Thread(target=run_pipeline, daemon=True).start()


if __name__ == "__main__":
    app = App()
    app.mainloop()
