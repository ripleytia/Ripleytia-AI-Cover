import os
import sys
import threading
import json
import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog
import subprocess
import shutil
from PIL import Image, ImageTk
import numpy as np

# Add src path explicitly just in case
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from main import song_cover_pipeline
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
)

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


translations = {
    "en": {
        "title": "Ripleytia AI Sound Studio - Vocal, Dataset & Voice Training Suite",
        "lang_switch": "🇹🇷 Türkçe",
        "refresh": "Refresh",
        "save_settings": "Save Settings",
        "mode_cover": "🎵 AI Cover Studio",
        "mode_dataset": "📦 Auto Dataset Builder",
        "mode_train": "🧠 Voice Model Training",
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
        "vocal_pitch": "Vocal Pitch Change (Octaves)",
        "overall_pitch": "Vocal Pitch Change (Semitones)",
        "main_gain": "Main Vocals Gain (dB)",
        "backup_gain": "Backup Vocals Gain (dB)",
        "inst_gain": "Instrumentals Gain (dB)",
        "mute_backup": "Mute Backup Vocals (Fixes original voice bleeding)",
        "room_size": "Room Size",
        "wetness": "Wetness (Reverb Level)",
        "dryness": "Dryness (Dry Vocal Level)",
        "damping": "Damping",
        "index_rate": "Index Rate (Timbre vs Original)",
        "filter_radius": "Filter Radius (Breathiness reduction)",
        "rms_mix": "RMS Mix Rate (Loudness Mimicry)",
        "protect": "Protect Rate (Voiceless Consonants)",
        "f0_method": "F0 Method (Pitch detection)",
        "crepe_hop": "Crepe Hop Length",
        "noise_gate": "Vocal Noise Gate (Echo Tail Cutter)",
        "noise_gate_thresh": "Noise Gate Threshold (dB)",
        "agg_dereverb": "Aggressive DeReverb (Fix heavy echo)",
        "keep_files": "Keep Intermediate Files",
        "force_reprocess": "Force Re-Extract Vocals (Ignore Cache)",
        "no_cache": " (No Cache Found)",
        "output_format": "Output Format",
        "generate": "Generate AI Cover",
        "auto_mode": "Auto Audio Mode (AI adjusts settings)",
        "open_folder": "Open Output Folder",
        "play_last": "Play Last Output",
        "clear_log": "Clear Console",
        "logs": "Studio Process Console:",
        "generating": "Generating... Please wait",
        "settings_saved": "[System] Settings saved successfully!"
    },
    "tr": {
        "title": "Ripleytia AI Sound Studio - Kapsamlı Ses, Dataset ve Modelleme Stüdyosu",
        "lang_switch": "🇬🇧 English",
        "refresh": "Yenile",
        "save_settings": "Ayarları Kaydet",
        "mode_cover": "🎵 AI Cover Stüdyosu",
        "mode_dataset": "📦 Otomatik Dataset Hazırlayıcı",
        "mode_train": "🧠 Ses Modelleme (Eğitim)",
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
        "vocal_pitch": "Vokal Ton (Pitch) Değişimi (Oktav)",
        "overall_pitch": "Vokal Ton Değişimi (Yarım Sesler)",
        "main_gain": "Ana Vokal Ses Seviyesi (dB)",
        "backup_gain": "Arka Vokal Ses Seviyesi (dB)",
        "inst_gain": "Enstrüman Ses Seviyesi (dB)",
        "mute_backup": "Arka Vokalleri Tamamen Sustur (Orijinal ses karışmasını önler)",
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
        "noise_gate": "Vokal Noise Gate (Yankı Kuyruğu Kesici - Aşırı yankı için)",
        "noise_gate_thresh": "Noise Gate Eşiği (Kaç dB altı sesler kesilsin?)",
        "agg_dereverb": "Agresif Yankı Temizleme (Çok yankılı sesleri onarır)",
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
    }
}

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")
PRIMARY_COLOR = "#5D3FD3"  # Deep purple
HOVER_COLOR = "#4A0E4E"
ACCENT_TEAL = "#00695C"
ACCENT_TEAL_HOVER = "#004D40"


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.lang = "tr"  # Default Turkish
        self.title(translations[self.lang]["title"])
        self.geometry("1040x980")

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

        # --- Cover Variables ---
        self.model_var = ctk.StringVar()
        self.pipeline_var = ctk.StringVar(value="UVR5 Special 2")
        self.song_input_var = ctk.StringVar()
        self.output_dir_var = ctk.StringVar(value=output_dir)
        self.pitch_var = tk.IntVar(value=0)
        self.pitch_all_var = tk.IntVar(value=0)
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
        self.header_frame = ctk.CTkFrame(self.main_frame, fg_color="#1A1A1A", corner_radius=15)
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
            text="Ripleytia Gothic AI Sound Studio",
            font=("Century Gothic", 26, "bold"),
            text_color=PRIMARY_COLOR,
        )
        self.header_title.pack(anchor="w")
        self.header_subtitle = ctk.CTkLabel(
            title_col,
            text="AI Cover Üretimi  •  Otomatik Dataset Hazırlayıcı  •  Profesyonel Ses Modelleme",
            font=("Arial", 12),
            text_color="#AAA",
        )
        self.header_subtitle.pack(anchor="w")

        self.lang_btn = ctk.CTkButton(
            self.header_frame,
            text=translations[self.lang]["lang_switch"],
            width=100,
            fg_color="#333",
            hover_color="#555",
            command=self.switch_lang,
        )
        self.lang_btn.pack(side="right", padx=10)

        self.btn_save_settings = ctk.CTkButton(
            self.header_frame,
            text=translations[self.lang]["save_settings"],
            width=110,
            fg_color=PRIMARY_COLOR,
            hover_color=HOVER_COLOR,
            command=self.save_settings,
        )
        self.btn_save_settings.pack(side="right", padx=10)

        # --- Top Studio Mode Navigation Bar ---
        self.mode_bar_frame = ctk.CTkFrame(self.main_frame, fg_color="#1A1A1A", corner_radius=12)
        self.mode_bar_frame.pack(fill="x", pady=5, padx=10)

        self.studio_mode_var = ctk.StringVar(value="cover")
        self.mode_btn_cover = ctk.CTkButton(
            self.mode_bar_frame,
            text=translations[self.lang]["mode_cover"],
            font=("Century Gothic", 15, "bold"),
            height=42,
            fg_color=PRIMARY_COLOR,
            hover_color=HOVER_COLOR,
            command=lambda: self.switch_studio_mode("cover"),
        )
        self.mode_btn_cover.pack(side="left", expand=True, fill="x", padx=6, pady=6)

        self.mode_btn_dataset = ctk.CTkButton(
            self.mode_bar_frame,
            text=translations[self.lang]["mode_dataset"],
            font=("Century Gothic", 15, "bold"),
            height=42,
            fg_color="#2B2B2B",
            hover_color=HOVER_COLOR,
            command=lambda: self.switch_studio_mode("dataset"),
        )
        self.mode_btn_dataset.pack(side="left", expand=True, fill="x", padx=6, pady=6)

        self.mode_btn_train = ctk.CTkButton(
            self.mode_bar_frame,
            text=translations[self.lang]["mode_train"],
            font=("Century Gothic", 15, "bold"),
            height=42,
            fg_color="#2B2B2B",
            hover_color=HOVER_COLOR,
            command=lambda: self.switch_studio_mode("train"),
        )
        self.mode_btn_train.pack(side="left", expand=True, fill="x", padx=6, pady=6)

        # --- Container Frames for the 3 Studio Modes ---
        self.cover_view_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.dataset_view_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.train_view_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")

        self._build_cover_view()
        self._build_dataset_view()
        self._build_train_view()

        # Show Cover View by default
        self.cover_view_frame.pack(fill="x")

        # Load Settings after defining widgets
        self.load_settings()

        # --- Shared Tools & Log Console at Bottom ---
        self.tools_frame = ctk.CTkFrame(self.main_frame, fg_color="#1A1A1A", corner_radius=10)
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
            self.main_frame, height=210, font=("Consolas", 12), fg_color="#0A0A0A", text_color="#A982FA"
        )
        self.log_textbox.pack(fill="x", padx=10, pady=5)

        # Redirect stdout and stderr
        sys.stdout = RedirectText(self.log_textbox)
        sys.stderr = RedirectText(self.log_textbox)

        # Initial updates
        self.update_texts()
        self.refresh_datasets_dropdown()
        print(f"[{translations[self.lang]['title']}] Hazır!")
        print(f"Çıktı Klasörü: {self.output_dir_var.get()}")

    def switch_studio_mode(self, mode: str):
        self.studio_mode_var.set(mode)
        self.cover_view_frame.pack_forget()
        self.dataset_view_frame.pack_forget()
        self.train_view_frame.pack_forget()

        self.mode_btn_cover.configure(fg_color=PRIMARY_COLOR if mode == "cover" else "#2B2B2B")
        self.mode_btn_dataset.configure(fg_color=PRIMARY_COLOR if mode == "dataset" else "#2B2B2B")
        self.mode_btn_train.configure(fg_color=PRIMARY_COLOR if mode == "train" else "#2B2B2B")

        if mode == "cover":
            self.cover_view_frame.pack(fill="x", before=self.tools_frame)
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
        self.model_frame = ctk.CTkFrame(self.cover_view_frame, fg_color="#1A1A1A", corner_radius=10)
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
            hover_color="#222",
            command=self.refresh_models,
        )
        self.btn_refresh.pack(side="left", padx=5)
        self.btn_upload = ctk.CTkButton(
            self.model_btn_frame,
            text=translations[self.lang]["upload_model"],
            fg_color=PRIMARY_COLOR,
            hover_color=HOVER_COLOR,
            command=self.upload_model_gui,
        )
        self.btn_upload.pack(side="left", padx=5)

        # --- Pipeline / UVR5 Models ---
        self.lbl_pipeline_select = ctk.CTkLabel(
            self.model_frame, text=translations[self.lang]["pipeline_select"], font=("Arial", 16, "bold"), text_color="#DDD"
        )
        self.lbl_pipeline_select.pack(pady=(15, 5))

        self.mdx_models = ["Default RVC", "UVR5 Special", "UVR5 Special 2"] + get_mdx_models()
        self.pipeline_dropdown = ctk.CTkOptionMenu(
            self.model_frame,
            values=self.mdx_models,
            variable=self.pipeline_var,
            width=320,
            fg_color="#005b96",
            button_color="#003f5c",
        )
        self.pipeline_dropdown.pack(pady=5)
        self.pipeline_var.trace_add("write", self.check_file_cache)

        # --- Input Selection ---
        self.input_frame = ctk.CTkFrame(self.cover_view_frame, fg_color="#1A1A1A", corner_radius=10)
        self.input_frame.pack(fill="x", pady=5, padx=10)

        input_inner = ctk.CTkFrame(self.input_frame, fg_color="transparent")
        input_inner.pack(fill="x", pady=5)

        self.lbl_input = ctk.CTkLabel(
            input_inner, text=translations[self.lang]["input_select"], font=("Arial", 14, "bold"), text_color="#DDD"
        )
        self.lbl_input.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.song_input_entry = ctk.CTkEntry(input_inner, textvariable=self.song_input_var, width=500)
        self.song_input_entry.grid(row=0, column=1, padx=10, pady=5)
        self.btn_browse = ctk.CTkButton(
            input_inner,
            text=translations[self.lang]["browse"],
            fg_color=PRIMARY_COLOR,
            hover_color=HOVER_COLOR,
            width=100,
            command=self.browse_file,
        )
        self.btn_browse.grid(row=0, column=2, padx=10, pady=5)

        self.lbl_output_dir = ctk.CTkLabel(
            input_inner, text=translations[self.lang]["output_select"], font=("Arial", 14, "bold"), text_color="#DDD"
        )
        self.lbl_output_dir.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.output_dir_entry = ctk.CTkEntry(input_inner, textvariable=self.output_dir_var, width=500)
        self.output_dir_entry.grid(row=1, column=1, padx=10, pady=5)
        self.btn_browse_dir = ctk.CTkButton(
            input_inner,
            text=translations[self.lang]["browse_dir"],
            fg_color=PRIMARY_COLOR,
            hover_color=HOVER_COLOR,
            width=100,
            command=self.browse_dir,
        )
        self.btn_browse_dir.grid(row=1, column=2, padx=10, pady=5)

        # --- Advanced Options TabView ---
        self.tabview = ctk.CTkTabview(
            self.cover_view_frame,
            width=850,
            fg_color="#1A1A1A",
            segmented_button_selected_color=PRIMARY_COLOR,
            segmented_button_selected_hover_color=HOVER_COLOR,
        )
        self.tabview.pack(fill="x", pady=10, padx=10)

        self.tab_names = ["tab_tuning", "tab_mix", "tab_fx", "tab_adv"]
        self.tabs = {}
        self.current_tab_titles = {}
        for t in self.tab_names:
            title = translations[self.lang][t]
            self.tabs[t] = self.tabview.add(title)
            self.current_tab_titles[t] = title

        # - Pitch & Tuning
        tab_pitch = self.tabs["tab_tuning"]
        self.lbl_vocal_pitch = ctk.CTkLabel(tab_pitch, text=translations[self.lang]["vocal_pitch"])
        self.lbl_vocal_pitch.pack(pady=(10, 0))
        ctk.CTkSlider(
            tab_pitch, from_=-12, to=12, number_of_steps=24, variable=self.pitch_var, button_color=PRIMARY_COLOR, button_hover_color=HOVER_COLOR
        ).pack()
        self.pitch_label = ctk.CTkLabel(tab_pitch, text=str(self.pitch_var.get()))
        self.pitch_label.pack()
        self.pitch_var.trace_add("write", lambda *args: self.pitch_label.configure(text=str(self.pitch_var.get())))

        self.lbl_overall_pitch = ctk.CTkLabel(tab_pitch, text=translations[self.lang]["overall_pitch"])
        self.lbl_overall_pitch.pack(pady=(10, 0))
        ctk.CTkSlider(
            tab_pitch, from_=-12, to=12, number_of_steps=24, variable=self.pitch_all_var, button_color=PRIMARY_COLOR, button_hover_color=HOVER_COLOR
        ).pack()
        self.pitch_all_label = ctk.CTkLabel(tab_pitch, text=str(self.pitch_all_var.get()))
        self.pitch_all_label.pack()
        self.pitch_all_var.trace_add("write", lambda *args: self.pitch_all_label.configure(text=str(self.pitch_all_var.get())))

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
        ctk.CTkSlider(tab_fx, from_=0, to=1, variable=self.reverb_size_var, button_color=PRIMARY_COLOR, button_hover_color=HOVER_COLOR).pack()
        self.reverb_size_label = ctk.CTkLabel(tab_fx, text=str(round(self.reverb_size_var.get(), 2)))
        self.reverb_size_label.pack()
        self.reverb_size_var.trace_add("write", lambda *args: self.reverb_size_label.configure(text=str(round(self.reverb_size_var.get(), 2))))

        self.lbl_wetness = ctk.CTkLabel(tab_fx, text=translations[self.lang]["wetness"])
        self.lbl_wetness.pack(pady=(5, 0))
        ctk.CTkSlider(tab_fx, from_=0, to=1, variable=self.reverb_wet_var, button_color=PRIMARY_COLOR, button_hover_color=HOVER_COLOR).pack()
        self.reverb_wet_label = ctk.CTkLabel(tab_fx, text=str(round(self.reverb_wet_var.get(), 2)))
        self.reverb_wet_label.pack()
        self.reverb_wet_var.trace_add("write", lambda *args: self.reverb_wet_label.configure(text=str(round(self.reverb_wet_var.get(), 2))))

        self.lbl_dryness = ctk.CTkLabel(tab_fx, text=translations[self.lang]["dryness"])
        self.lbl_dryness.pack(pady=(5, 0))
        ctk.CTkSlider(tab_fx, from_=0, to=1, variable=self.reverb_dry_var, button_color=PRIMARY_COLOR, button_hover_color=HOVER_COLOR).pack()
        self.reverb_dry_label = ctk.CTkLabel(tab_fx, text=str(round(self.reverb_dry_var.get(), 2)))
        self.reverb_dry_label.pack()
        self.reverb_dry_var.trace_add("write", lambda *args: self.reverb_dry_label.configure(text=str(round(self.reverb_dry_var.get(), 2))))

        self.lbl_damping = ctk.CTkLabel(tab_fx, text=translations[self.lang]["damping"])
        self.lbl_damping.pack(pady=(5, 0))
        ctk.CTkSlider(tab_fx, from_=0, to=1, variable=self.reverb_damp_var, button_color=PRIMARY_COLOR, button_hover_color=HOVER_COLOR).pack()
        self.reverb_damp_label = ctk.CTkLabel(tab_fx, text=str(round(self.reverb_damp_var.get(), 2)))
        self.reverb_damp_label.pack()
        self.reverb_damp_var.trace_add("write", lambda *args: self.reverb_damp_label.configure(text=str(round(self.reverb_damp_var.get(), 2))))

        # - Advanced Conversion
        tab_adv = self.tabs["tab_adv"]
        self.lbl_index_rate = ctk.CTkLabel(tab_adv, text=translations[self.lang]["index_rate"])
        self.lbl_index_rate.pack(pady=(5, 0))
        ctk.CTkSlider(tab_adv, from_=0, to=1, variable=self.index_rate_var, button_color=PRIMARY_COLOR, button_hover_color=HOVER_COLOR).pack()
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
        ctk.CTkSlider(tab_adv, from_=0, to=1, variable=self.rms_mix_var, button_color=PRIMARY_COLOR, button_hover_color=HOVER_COLOR).pack()
        self.rms_mix_label = ctk.CTkLabel(tab_adv, text=str(round(self.rms_mix_var.get(), 2)))
        self.rms_mix_label.pack()
        self.rms_mix_var.trace_add("write", lambda *args: self.rms_mix_label.configure(text=str(round(self.rms_mix_var.get(), 2))))

        self.lbl_protect = ctk.CTkLabel(tab_adv, text=translations[self.lang]["protect"])
        self.lbl_protect.pack(pady=(5, 0))
        ctk.CTkSlider(tab_adv, from_=0, to=0.5, variable=self.protect_var, button_color=PRIMARY_COLOR, button_hover_color=HOVER_COLOR).pack()
        self.protect_label = ctk.CTkLabel(tab_adv, text=str(round(self.protect_var.get(), 2)))
        self.protect_label.pack()
        self.protect_var.trace_add("write", lambda *args: self.protect_label.configure(text=str(round(self.protect_var.get(), 2))))

        self.lbl_f0_method = ctk.CTkLabel(tab_adv, text=translations[self.lang]["f0_method"])
        self.lbl_f0_method.pack(pady=(5, 0))
        f0_seg = ctk.CTkSegmentedButton(
            tab_adv, values=["rmvpe", "mangio-crepe"], variable=self.f0_method_var, selected_color=PRIMARY_COLOR, selected_hover_color=HOVER_COLOR
        )
        f0_seg.pack(pady=5)

        self.lbl_crepe_hop = ctk.CTkLabel(tab_adv, text=translations[self.lang]["crepe_hop"])
        self.crepe_slider = ctk.CTkSlider(
            tab_adv, from_=32, to=320, number_of_steps=288, variable=self.crepe_hop_var, button_color=PRIMARY_COLOR, button_hover_color=HOVER_COLOR
        )
        self.crepe_hop_label = ctk.CTkLabel(tab_adv, text=str(self.crepe_hop_var.get()))
        self.crepe_hop_var.trace_add("write", lambda *args: self.crepe_hop_label.configure(text=str(self.crepe_hop_var.get())))

        def toggle_crepe(*args):
            if self.f0_method_var.get() == "mangio-crepe":
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

        # --- Auto Mode Switch ---
        self.auto_mode_var = ctk.BooleanVar(value=False)
        self.lock_settings_var = ctk.BooleanVar(value=False)

        self.auto_mode_frame = ctk.CTkFrame(self.cover_view_frame, fg_color="transparent")
        self.auto_mode_frame.pack(pady=(15, 0))

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
        self.generate_btn.pack(pady=20, fill="x", padx=50)
        self._toggle_crepe_fn()

    # =========================================================================
    # VIEW 2: AUTOMATIC DATASET BUILDER (OTOMATİK DATASET HAZIRLAYICI)
    # =========================================================================
    def _build_dataset_view(self):
        banner = ctk.CTkFrame(self.dataset_view_frame, fg_color="#1A1A1A", corner_radius=12)
        banner.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(
            banner,
            text="📦 Otomatik Dataset Hazırlayıcı (Çoklu Dosya & Çoklu YouTube Stüdyosu)",
            font=("Century Gothic", 18, "bold"),
            text_color="#A982FA",
        ).pack(anchor="w", padx=15, pady=(10, 2))
        ctk.CTkLabel(
            banner,
            text="Bilgisayarınızdan birden fazla ses/video seçin veya YouTube linklerini alt alta yapıştırın.\n"
                 "Stüdyo arka plan müziğini, geri vokalleri, oda yankısını ve boşlukları otomatik temizleyip tek bir kusursuz eğitim datasetine dönüştürür.",
            font=("Arial", 12),
            text_color="#BBB",
            justify="left",
        ).pack(anchor="w", padx=15, pady=(0, 10))

        # Step 1: Dataset Name & Sources
        src_card = ctk.CTkFrame(self.dataset_view_frame, fg_color="#1A1A1A", corner_radius=12)
        src_card.pack(fill="x", padx=10, pady=5)

        row_name = ctk.CTkFrame(src_card, fg_color="transparent")
        row_name.pack(fill="x", padx=15, pady=10)
        ctk.CTkLabel(row_name, text="1. Dataset / Model Adı:", font=("Arial", 14, "bold"), text_color="#EEE").pack(side="left")
        ctk.CTkEntry(row_name, textvariable=self.ds_name_var, width=320, placeholder_text="Örn: Sanatci_Sesi_V1").pack(side="left", padx=15)

        # Local multi-file selector
        row_files = ctk.CTkFrame(src_card, fg_color="transparent")
        row_files.pack(fill="x", padx=15, pady=5)
        ctk.CTkLabel(row_files, text="2. Bilgisayardan Ses/Video Dosyaları:", font=("Arial", 14, "bold"), text_color="#EEE").pack(side="left")
        ctk.CTkButton(
            row_files,
            text="➕ Çoklu Dosya Seç (MP3/WAV/MP4/FLAC)",
            fg_color=PRIMARY_COLOR,
            hover_color=HOVER_COLOR,
            command=self.select_dataset_local_files,
        ).pack(side="left", padx=10)
        ctk.CTkButton(
            row_files,
            text="🗑️ Seçimi Temizle",
            fg_color="#444",
            hover_color="#222",
            width=120,
            command=self.clear_dataset_local_files,
        ).pack(side="left", padx=5)
        self.lbl_ds_files_count = ctk.CTkLabel(row_files, text="Seçilen: 0 dosya", font=("Arial", 12, "bold"), text_color="#A982FA")
        self.lbl_ds_files_count.pack(side="left", padx=10)

        # Multi-YouTube URLs box
        yt_frame = ctk.CTkFrame(src_card, fg_color="transparent")
        yt_frame.pack(fill="x", padx=15, pady=(8, 12))
        ctk.CTkLabel(
            yt_frame,
            text="3. Çoklu YouTube Video Linkleri (İsteğe Bağlı - Her satıra 1 YouTube linki yapıştırın):",
            font=("Arial", 14, "bold"),
            text_color="#EEE",
        ).pack(anchor="w", pady=(0, 5))
        self.ds_yt_textbox = ctk.CTkTextbox(yt_frame, height=95, font=("Consolas", 12), fg_color="#0E0E0E", text_color="#DDD")
        self.ds_yt_textbox.pack(fill="x")

        # Step 2: Cleaning & Duration Settings
        clean_card = ctk.CTkFrame(self.dataset_view_frame, fg_color="#1A1A1A", corner_radius=12)
        clean_card.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(
            clean_card,
            text="4. Arka Plan Müziği, Yankı ve Sessizlik Temizleme Ayarları",
            font=("Arial", 15, "bold"),
            text_color="#DDD",
        ).pack(anchor="w", padx=15, pady=(10, 5))

        grid_clean = ctk.CTkFrame(clean_card, fg_color="transparent")
        grid_clean.pack(fill="x", padx=15, pady=5)

        ctk.CTkLabel(grid_clean, text="Yapay Zeka Temizleme Motoru:", font=("Arial", 13, "bold")).grid(row=0, column=0, sticky="w", padx=5, pady=6)
        ctk.CTkOptionMenu(
            grid_clean,
            values=[
                "UVR5 Special 2 (Ultra Kalite)",
                "UVR5 Special (Hızlı Roformer)",
                "Sadece Yankı & Dip Ses Temizle (Konuşma)",
                "Temizleme Yapma (Zaten Temiz Kayıt)",
            ],
            variable=self.ds_clean_mode_var,
            width=320,
            fg_color="#005b96",
            button_color="#003f5c",
        ).grid(row=0, column=1, sticky="w", padx=10, pady=6)

        ctk.CTkLabel(grid_clean, text="Dataset Örnekleme Hızı (Hz):", font=("Arial", 13, "bold")).grid(row=0, column=2, sticky="w", padx=15, pady=6)
        ctk.CTkSegmentedButton(
            grid_clean,
            values=["32000", "40000", "48000"],
            variable=self.ds_sample_rate_var,
            selected_color=PRIMARY_COLOR,
            selected_hover_color=HOVER_COLOR,
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
        chain_frame = ctk.CTkFrame(self.dataset_view_frame, fg_color="#1A1A1A", corner_radius=12)
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
            text="⚙️ OTOMATİK DATASET HAZIRLA, TEMİZLE VE BİRLEŞTİR",
            fg_color=PRIMARY_COLOR,
            hover_color=HOVER_COLOR,
            font=("Century Gothic", 18, "bold"),
            height=52,
            command=self.start_dataset_builder,
        )
        self.btn_build_dataset.pack(pady=12, fill="x", padx=50)

    def select_dataset_local_files(self):
        files = filedialog.askopenfilenames(
            title="Dataset İçin Ses veya Video Dosyalarını Seçin",
            filetypes=[("Audio & Video Files", "*.wav;*.mp3;*.flac;*.ogg;*.m4a;*.mp4;*.mkv;*.webm;*.aac")],
        )
        if files:
            for f in files:
                norm_f = os.path.normpath(f)
                if norm_f not in self.ds_local_files:
                    self.ds_local_files.append(norm_f)
            self.lbl_ds_files_count.configure(text=f"Seçilen: {len(self.ds_local_files)} dosya")
            print(f"[Dataset] Toplam {len(self.ds_local_files)} yerel dosya seçildi.")

    def clear_dataset_local_files(self):
        self.ds_local_files = []
        self.lbl_ds_files_count.configure(text="Seçilen: 0 dosya")
        print("[Dataset] Yerel dosya listesi temizlendi.")

    def start_dataset_builder(self):
        ds_name = sanitize_name(self.ds_name_var.get())
        self.ds_name_var.set(ds_name)

        yt_raw = self.ds_yt_textbox.get("1.0", "end").strip()
        yt_urls = [line.strip() for line in yt_raw.splitlines() if line.strip()]

        if not self.ds_local_files and not yt_urls:
            print("[Hata] Lütfen en az bir yerel dosya seçin veya YouTube linki yapıştırın!")
            return

        self.btn_build_dataset.configure(state="disabled", text="⏳ Dataset Hazırlanıyor & Temizleniyor... Lütfen Bekleyin")

        def _worker():
            try:
                manifest = build_automatic_dataset(
                    dataset_name=ds_name,
                    local_files=self.ds_local_files,
                    youtube_urls=yt_urls,
                    clean_mode=self.ds_clean_mode_var.get(),
                    remove_backups=self.ds_remove_backups_var.get(),
                    apply_dereverb=self.ds_apply_dereverb_var.get(),
                    target_sr=int(self.ds_sample_rate_var.get()),
                    silence_top_db=self.ds_silence_db_var.get(),
                    max_duration_minutes=float(self.ds_max_minutes_var.get()),
                    progress_cb=lambda msg: print(msg),
                )

                def _on_done():
                    summary_txt = (
                        f"✅ Dataset Hazır: {manifest['dataset_name']} | "
                        f"Ham Süre: {manifest['raw_duration_str']} -> Net Vokal: {manifest['clean_duration_str']} | "
                        f"Önerilen: {manifest['recommended_epochs']} Epoch"
                    )
                    self.lbl_ds_summary.configure(text=summary_txt)
                    self.refresh_datasets_dropdown()

                    if self.ds_auto_send_train_var.get():
                        self.tr_model_name_var.set(manifest["dataset_name"])
                        self.tr_dataset_path_var.set(manifest["local_dataset_dir"])
                        self.tr_total_epoch_var.set(manifest["recommended_epochs"])
                        self.tr_save_every_var.set(manifest["recommended_save_every"])
                        print(
                            f"[Akıllı Aktarım] '{manifest['dataset_name']}' eğitim sekmesine aktarıldı "
                            f"(Otomatik Epoch: {manifest['recommended_epochs']})."
                        )

                    if self.ds_auto_start_train_var.get():
                        self.switch_studio_mode("train")
                        self.start_voice_training(
                            run_preprocess=True, run_extract=True, run_index=True, run_train=True
                        )

                self.after(0, _on_done)
            except Exception as e:
                print(f"[Hata] Dataset hazırlama başarısız: {e}")
            finally:
                self.after(
                    0,
                    lambda: self.btn_build_dataset.configure(
                        state="normal", text="⚙️ OTOMATİK DATASET HAZIRLA, TEMİZLE VE BİRLEŞTİR"
                    ),
                )

        threading.Thread(target=_worker, daemon=True).start()

    # =========================================================================
    # VIEW 3: VOICE MODEL TRAINING STUDIO (SES MODELLEME & EĞİTİM)
    # =========================================================================
    def _build_train_view(self):
        # Top Info & Preset Selector Card
        preset_card = ctk.CTkFrame(self.train_view_frame, fg_color="#1A1A1A", corner_radius=12)
        preset_card.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(
            preset_card,
            text="🧠 Profesyonel Ses Modelleme Stüdyosu (Applio Entegre & Gelişmiş Ön Eğitimli)",
            font=("Century Gothic", 18, "bold"),
            text_color="#A982FA",
        ).pack(anchor="w", padx=15, pady=(10, 4))

        preset_row = ctk.CTkFrame(preset_card, fg_color="transparent")
        preset_row.pack(fill="x", padx=15, pady=5)
        ctk.CTkLabel(preset_row, text="🎯 Hazır Eğitim Preseti Seçin:", font=("Arial", 14, "bold"), text_color="#EEE").pack(side="left")
        self.preset_menu = ctk.CTkOptionMenu(
            preset_row,
            values=list(TRAINING_PRESETS.keys()),
            variable=self.tr_preset_var,
            width=360,
            fg_color=PRIMARY_COLOR,
            button_color=HOVER_COLOR,
            command=self.apply_training_preset,
        )
        self.preset_menu.pack(side="left", padx=12)

        ctk.CTkButton(
            preset_row,
            text="🧮 Dataset Süresine Göre Akıllı Epoch Hesapla",
            fg_color=ACCENT_TEAL,
            hover_color=ACCENT_TEAL_HOVER,
            command=self.smart_calculate_epochs_for_selected_dataset,
        ).pack(side="left", padx=5)

        self.lbl_preset_desc = ctk.CTkLabel(
            preset_card,
            text=TRAINING_PRESETS[self.tr_preset_var.get()]["description"],
            font=("Arial", 12, "italic"),
            text_color="#00E676",
            wraplength=920,
            justify="left",
        )
        self.lbl_preset_desc.pack(anchor="w", padx=15, pady=(2, 10))

        # Step 1: Model Name & Dataset Folder
        ds_select_card = ctk.CTkFrame(self.train_view_frame, fg_color="#1A1A1A", corner_radius=12)
        ds_select_card.pack(fill="x", padx=10, pady=5)

        ds_grid = ctk.CTkFrame(ds_select_card, fg_color="transparent")
        ds_grid.pack(fill="x", padx=15, pady=10)

        ctk.CTkLabel(ds_grid, text="1. Eğitilecek Model Adı:", font=("Arial", 14, "bold")).grid(row=0, column=0, sticky="w", padx=5, pady=5)
        ctk.CTkEntry(ds_grid, textvariable=self.tr_model_name_var, width=280).grid(row=0, column=1, sticky="w", padx=10, pady=5)

        ctk.CTkLabel(ds_grid, text="2. Dataset Klasörü:", font=("Arial", 14, "bold")).grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.tr_dataset_dropdown = ctk.CTkComboBox(
            ds_grid,
            values=get_available_datasets() or [""],
            variable=self.tr_dataset_path_var,
            width=480,
            button_color=PRIMARY_COLOR,
        )
        self.tr_dataset_dropdown.grid(row=1, column=1, sticky="w", padx=10, pady=5)

        ds_btns = ctk.CTkFrame(ds_grid, fg_color="transparent")
        ds_btns.grid(row=1, column=2, padx=5, pady=5)
        ctk.CTkButton(
            ds_btns, text="Klasör Seç", width=95, fg_color=PRIMARY_COLOR, hover_color=HOVER_COLOR, command=self.browse_training_dataset
        ).pack(side="left", padx=3)
        ctk.CTkButton(
            ds_btns, text="Yenile", width=75, fg_color="#444", hover_color="#222", command=self.refresh_datasets_dropdown
        ).pack(side="left", padx=3)

        # Step 2: Training Settings Tabview (Easy Mode vs Pro Extra Quality Mode)
        self.train_tabs = ctk.CTkTabview(
            self.train_view_frame,
            width=880,
            height=300,
            fg_color="#1A1A1A",
            segmented_button_selected_color=PRIMARY_COLOR,
            segmented_button_selected_hover_color=HOVER_COLOR,
        )
        self.train_tabs.pack(fill="x", padx=10, pady=5)

        tab_easy = self.train_tabs.add("⚡ Temel Eğitim Ayarları (Kolay Mod)")
        tab_pro = self.train_tabs.add("🔬 Ekstra Kalite & Gelişmiş Eğitim Modelleri (Pro Mod)")

        # --- TAB EASY ---
        easy_grid = ctk.CTkFrame(tab_easy, fg_color="transparent")
        easy_grid.pack(fill="x", padx=15, pady=10)

        # Total Epoch slider + direct entry (safe against empty text while typing)
        self.tr_total_epoch_str = ctk.StringVar(value=str(self.tr_total_epoch_var.get()))
        self.tr_save_every_str = ctk.StringVar(value=str(self.tr_save_every_var.get()))

        def _on_epoch_var_change(*args):
            try:
                val = str(self.tr_total_epoch_var.get())
                if self.tr_total_epoch_str.get() != val:
                    self.tr_total_epoch_str.set(val)
            except Exception:
                pass

        def _on_epoch_str_change(*args):
            txt = self.tr_total_epoch_str.get().strip()
            if txt.isdigit():
                val = max(1, min(10000, int(txt)))
                if self.tr_total_epoch_var.get() != val:
                    self.tr_total_epoch_var.set(val)

        def _on_save_var_change(*args):
            try:
                val = str(self.tr_save_every_var.get())
                if self.tr_save_every_str.get() != val:
                    self.tr_save_every_str.set(val)
            except Exception:
                pass

        def _on_save_str_change(*args):
            txt = self.tr_save_every_str.get().strip()
            if txt.isdigit():
                val = max(1, min(1000, int(txt)))
                if self.tr_save_every_var.get() != val:
                    self.tr_save_every_var.set(val)

        self.tr_total_epoch_var.trace_add("write", _on_epoch_var_change)
        self.tr_total_epoch_str.trace_add("write", _on_epoch_str_change)
        self.tr_save_every_var.trace_add("write", _on_save_var_change)
        self.tr_save_every_str.trace_add("write", _on_save_str_change)

        ctk.CTkLabel(easy_grid, text="Toplam Eğitim Sayısı (Total Epoch):", font=("Arial", 13, "bold")).grid(row=0, column=0, sticky="w", padx=5, pady=8)
        ctk.CTkSlider(
            easy_grid, from_=10, to=1000, number_of_steps=99, variable=self.tr_total_epoch_var, width=280, button_color=PRIMARY_COLOR, button_hover_color=HOVER_COLOR
        ).grid(row=0, column=1, padx=10, pady=8)
        ctk.CTkEntry(easy_grid, textvariable=self.tr_total_epoch_str, width=70).grid(row=0, column=2, padx=5, pady=8)

        # Save Every Epoch slider + entry
        ctk.CTkLabel(easy_grid, text="Kaç Epoch'ta Bir Kaydedilsin:", font=("Arial", 13, "bold")).grid(row=1, column=0, sticky="w", padx=5, pady=8)
        ctk.CTkSlider(
            easy_grid, from_=5, to=100, number_of_steps=19, variable=self.tr_save_every_var, width=280, button_color=PRIMARY_COLOR, button_hover_color=HOVER_COLOR
        ).grid(row=1, column=1, padx=10, pady=8)
        ctk.CTkEntry(easy_grid, textvariable=self.tr_save_every_str, width=70).grid(row=1, column=2, padx=5, pady=8)

        # Batch Size slider
        ctk.CTkLabel(easy_grid, text="Batch Size (RTX 4060 için 6 önerilir):", font=("Arial", 13, "bold")).grid(row=2, column=0, sticky="w", padx=5, pady=8)
        ctk.CTkSlider(
            easy_grid, from_=1, to=16, number_of_steps=15, variable=self.tr_batch_size_var, width=280, button_color=PRIMARY_COLOR, button_hover_color=HOVER_COLOR
        ).grid(row=2, column=1, padx=10, pady=8)
        self.lbl_tr_batch = ctk.CTkLabel(easy_grid, text=str(self.tr_batch_size_var.get()), font=("Arial", 13, "bold"))
        self.lbl_tr_batch.grid(row=2, column=2, padx=5, pady=8)
        self.tr_batch_size_var.trace_add("write", lambda *a: self.lbl_tr_batch.configure(text=str(self.tr_batch_size_var.get())))

        # Sample Rate
        ctk.CTkLabel(easy_grid, text="Model Örnekleme Kalitesi (Sample Rate):", font=("Arial", 13, "bold")).grid(row=3, column=0, sticky="w", padx=5, pady=8)
        ctk.CTkSegmentedButton(
            easy_grid,
            values=["32000", "40000", "48000"],
            variable=self.tr_sample_rate_var,
            selected_color=PRIMARY_COLOR,
            selected_hover_color=HOVER_COLOR,
        ).grid(row=3, column=1, sticky="w", padx=10, pady=8)

        # Auto Deploy & Cleanup Checkboxes
        easy_chks = ctk.CTkFrame(tab_easy, fg_color="transparent")
        easy_chks.pack(fill="x", padx=15, pady=5)
        ctk.CTkCheckBox(
            easy_chks,
            text="✅ Eğitim Bitince Modeli Otomatik 'AI Cover Stüdyosu'na Kur (Anında Kullanıma Hazır)",
            variable=self.tr_auto_deploy_var,
            fg_color=PRIMARY_COLOR,
            hover_color=HOVER_COLOR,
            font=("Arial", 13, "bold"),
        ).pack(anchor="w", pady=4)
        ctk.CTkCheckBox(
            easy_chks,
            text="🧹 Sıfırdan Temiz Eğitim (Önceki Eski Ağırlıkları Temizle - Fresh Training)",
            variable=self.tr_cleanup_fresh_var,
            fg_color=PRIMARY_COLOR,
            hover_color=HOVER_COLOR,
        ).pack(anchor="w", pady=4)

        # --- TAB PRO (BEYOND APPLIO) ---
        pro_grid = ctk.CTkFrame(tab_pro, fg_color="transparent")
        pro_grid.pack(fill="x", padx=15, pady=8)

        ctk.CTkLabel(pro_grid, text="Özel Ön Eğitim Modeli (Pretrain):", font=("Arial", 13, "bold")).grid(row=0, column=0, sticky="w", padx=5, pady=6)
        ctk.CTkOptionMenu(
            pro_grid,
            values=list(PRETRAIN_MAP.keys()),
            variable=self.tr_pretrain_var,
            width=290,
            fg_color="#005b96",
            button_color="#003f5c",
        ).grid(row=0, column=1, sticky="w", padx=8, pady=6)

        ctk.CTkLabel(pro_grid, text="Ses Sentezleyici Mimari (Vocoder):", font=("Arial", 13, "bold")).grid(row=0, column=2, sticky="w", padx=12, pady=6)
        ctk.CTkOptionMenu(
            pro_grid,
            values=["HiFi-GAN", "RefineGAN"],
            variable=self.tr_vocoder_var,
            width=180,
            fg_color=PRIMARY_COLOR,
            button_color=HOVER_COLOR,
        ).grid(row=0, column=3, sticky="w", padx=5, pady=6)

        ctk.CTkLabel(pro_grid, text="Ses Kimliği Modeli (Embedder):", font=("Arial", 13, "bold")).grid(row=1, column=0, sticky="w", padx=5, pady=6)
        ctk.CTkOptionMenu(
            pro_grid,
            values=["contentvec", "spin-v2", "spin", "chinese-hubert-base", "japanese-hubert-base", "korean-hubert-base"],
            variable=self.tr_embedder_var,
            width=290,
            fg_color=PRIMARY_COLOR,
            button_color=HOVER_COLOR,
        ).grid(row=1, column=1, sticky="w", padx=8, pady=6)

        ctk.CTkLabel(pro_grid, text="Pitch (F0) Algoritması:", font=("Arial", 13, "bold")).grid(row=1, column=2, sticky="w", padx=12, pady=6)
        ctk.CTkOptionMenu(
            pro_grid,
            values=["rmvpe", "fcpe", "crepe", "crepe-tiny"],
            variable=self.tr_f0_method_var,
            width=180,
            fg_color=PRIMARY_COLOR,
            button_color=HOVER_COLOR,
        ).grid(row=1, column=3, sticky="w", padx=5, pady=6)

        ctk.CTkLabel(pro_grid, text="Dataset Kesim Modu:", font=("Arial", 13, "bold")).grid(row=2, column=0, sticky="w", padx=5, pady=6)
        ctk.CTkSegmentedButton(
            pro_grid,
            values=["Automatic", "Simple", "Skip"],
            variable=self.tr_cut_mode_var,
            selected_color=PRIMARY_COLOR,
            selected_hover_color=HOVER_COLOR,
        ).grid(row=2, column=1, sticky="w", padx=8, pady=6)

        self.lbl_tr_mutes_title = ctk.CTkLabel(
            pro_grid, text=f"Sessizlik Öğretimi (Mute Dosyası): {self.tr_include_mutes_var.get()}", font=("Arial", 13, "bold")
        )
        self.lbl_tr_mutes_title.grid(row=2, column=2, sticky="w", padx=12, pady=6)

        mute_slider_frame = ctk.CTkFrame(pro_grid, fg_color="transparent")
        mute_slider_frame.grid(row=2, column=3, sticky="w", padx=5, pady=6)
        ctk.CTkSlider(
            mute_slider_frame, from_=0, to=10, number_of_steps=10, variable=self.tr_include_mutes_var, width=135, button_color=PRIMARY_COLOR, button_hover_color=HOVER_COLOR
        ).pack(side="left")
        self.lbl_tr_mutes_val = ctk.CTkLabel(
            mute_slider_frame, text=str(self.tr_include_mutes_var.get()), font=("Arial", 14, "bold"), text_color="#A982FA", width=28
        )
        self.lbl_tr_mutes_val.pack(side="left", padx=(6, 0))

        def _update_mutes_lbl(*args):
            v = self.tr_include_mutes_var.get()
            self.lbl_tr_mutes_title.configure(text=f"Sessizlik Öğretimi (Mute Dosyası): {v}")
            self.lbl_tr_mutes_val.configure(text=str(v))

        self.tr_include_mutes_var.trace_add("write", _update_mutes_lbl)

        pro_chks = ctk.CTkFrame(tab_pro, fg_color="transparent")
        pro_chks.pack(fill="x", padx=15, pady=6)
        ctk.CTkCheckBox(
            pro_chks, text="Preprocess Ses Filtresi (Highpass)", variable=self.tr_process_fx_var, fg_color=PRIMARY_COLOR, hover_color=HOVER_COLOR
        ).pack(side="left", padx=8)
        ctk.CTkCheckBox(
            pro_chks, text="Spektral Gürültü Azaltma", variable=self.tr_noise_red_var, fg_color=PRIMARY_COLOR, hover_color=HOVER_COLOR
        ).pack(side="left", padx=8)
        ctk.CTkCheckBox(
            pro_chks, text="VRAM Tasarrufu (Checkpointing)", variable=self.tr_checkpointing_var, fg_color=PRIMARY_COLOR, hover_color=HOVER_COLOR
        ).pack(side="left", padx=8)
        ctk.CTkCheckBox(
            pro_chks, text="GPU Önbellekleme (Cache in GPU)", variable=self.tr_cache_gpu_var, fg_color=PRIMARY_COLOR, hover_color=HOVER_COLOR
        ).pack(side="left", padx=8)

        # Action Buttons (One-Click Full Training + Step-by-Step + Stop + Deploy)
        action_card = ctk.CTkFrame(self.train_view_frame, fg_color="#1A1A1A", corner_radius=12)
        action_card.pack(fill="x", padx=10, pady=8)

        self.btn_oneclick_train = ctk.CTkButton(
            action_card,
            text="🚀 TEK TIKLA TAM OTOMATİK SES MODELİ EĞİT (Ön İşle + Özellik Çıkar + Eğit + Stüdyoya Kur)",
            fg_color=PRIMARY_COLOR,
            hover_color=HOVER_COLOR,
            font=("Century Gothic", 16, "bold"),
            height=50,
            command=lambda: self.start_voice_training(run_preprocess=True, run_extract=True, run_index=True, run_train=True),
        )
        self.btn_oneclick_train.pack(fill="x", padx=20, pady=(12, 8))

        modular_row = ctk.CTkFrame(action_card, fg_color="transparent")
        modular_row.pack(fill="x", padx=20, pady=(0, 12))

        ctk.CTkButton(
            modular_row,
            text="1. Sadece Ön İşle + Özellik Çıkar",
            fg_color="#333",
            hover_color="#555",
            command=lambda: self.start_voice_training(run_preprocess=True, run_extract=True, run_index=True, run_train=False),
        ).pack(side="left", expand=True, fill="x", padx=4)

        ctk.CTkButton(
            modular_row,
            text="2. Sadece Eğitimi Başlat / Devam Et",
            fg_color="#005b96",
            hover_color="#003f5c",
            command=lambda: self.start_voice_training(run_preprocess=False, run_extract=False, run_index=False, run_train=True),
        ).pack(side="left", expand=True, fill="x", padx=4)

        ctk.CTkButton(
            modular_row,
            text="📥 Mevcut En İyi Epoch'u Stüdyoya Aktar",
            fg_color=ACCENT_TEAL,
            hover_color=ACCENT_TEAL_HOVER,
            command=self.manual_deploy_trained_model,
        ).pack(side="left", expand=True, fill="x", padx=4)

        ctk.CTkButton(
            modular_row,
            text="🛑 Eğitimi Durdur",
            fg_color="#C0392B",
            hover_color="#922B21",
            width=140,
            command=self.stop_voice_training,
        ).pack(side="left", padx=4)

    def apply_training_preset(self, preset_name: str):
        p = TRAINING_PRESETS.get(preset_name)
        if not p:
            return
        self.tr_sample_rate_var.set(p["sample_rate"])
        self.tr_vocoder_var.set(p["vocoder"])
        self.tr_f0_method_var.set(p["f0_method"])
        self.tr_embedder_var.set(p["embedder_model"])
        self.tr_pretrain_var.set(p["pretrain_type"])
        self.tr_batch_size_var.set(p["batch_size"])
        self.tr_total_epoch_var.set(p["total_epoch"])
        self.tr_save_every_var.set(p["save_every_epoch"])
        self.tr_cut_mode_var.set(p["cut_preprocess"])
        self.tr_chunk_len_var.set(p["chunk_len"])
        self.tr_overlap_len_var.set(p["overlap_len"])
        self.tr_norm_mode_var.set(p["normalization_mode"])
        self.tr_process_fx_var.set(p["process_effects"])
        self.tr_noise_red_var.set(p["noise_reduction"])
        self.tr_include_mutes_var.set(p["include_mutes"])
        self.tr_index_algo_var.set(p["index_algorithm"])
        self.tr_save_only_latest_var.set(p["save_only_latest"])
        self.tr_save_every_weights_var.set(p["save_every_weights"])
        self.tr_cache_gpu_var.set(p["cache_data_in_gpu"])
        self.tr_checkpointing_var.set(p["checkpointing"])
        self.lbl_preset_desc.configure(text=p["description"])
        print(f"[Preset Uygulandı] '{preset_name}' ayarları yüklendi!")

    def refresh_datasets_dropdown(self):
        ds_list = get_available_datasets()
        self.tr_dataset_dropdown.configure(values=ds_list if ds_list else [""])
        if ds_list and not self.tr_dataset_path_var.get():
            self.tr_dataset_path_var.set(ds_list[0])

    def browse_training_dataset(self):
        dirpath = filedialog.askdirectory(title="Eğitim İçin Ses Dataset Klasörünü Seçin")
        if dirpath:
            dirpath = os.path.normpath(dirpath)
            self.tr_dataset_path_var.set(dirpath)
            print(f"[Eğitim] Dataset klasörü seçildi: {dirpath}")

    def smart_calculate_epochs_for_selected_dataset(self):
        ds_path = self.tr_dataset_path_var.get().strip()
        if not ds_path or not os.path.exists(ds_path):
            print("[Uyarı] Lütfen önce geçerli bir Dataset klasörü seçin!")
            return

        import soundfile as sf
        total_sec = 0.0
        file_count = 0
        for root, _, files in os.walk(ds_path):
            if "_temp_work" in root:
                continue
            for f in files:
                if f.lower().endswith((".wav", ".flac", ".mp3", ".ogg")):
                    fp = os.path.join(root, f)
                    try:
                        info = sf.info(fp)
                        total_sec += info.duration
                        file_count += 1
                    except Exception:
                        pass

        if total_sec <= 0:
            print("[Uyarı] Seçilen klasörde okunabilir ses dosyası bulunamadı.")
            return

        preset_short = "Dengeli"
        cur_p = self.tr_preset_var.get()
        if "Hızlı" in cur_p:
            preset_short = "Hızlı Render"
        elif "Maksimum" in cur_p:
            preset_short = "Maksimum Kalite"
        elif "W-Okada" in cur_p:
            preset_short = "W-Okada (Canlı Konuşma)"

        advice = calculate_recommended_epochs(total_sec, preset=preset_short)
        self.tr_total_epoch_var.set(advice["recommended_epochs"])
        self.tr_save_every_var.set(advice["save_every_epoch"])
        print(
            f"[Akıllı Epoch Hesaplayıcı] Dataset Süresi: {format_duration(total_sec)} ({file_count} dosya) "
            f"-> Önerilen: {advice['recommended_epochs']} Epoch (Kayıt: Her {advice['save_every_epoch']} Epoch) | {advice['note']}"
        )

    def start_voice_training(self, run_preprocess=True, run_extract=True, run_index=True, run_train=True):
        model_name = sanitize_name(self.tr_model_name_var.get())
        self.tr_model_name_var.set(model_name)
        ds_path = self.tr_dataset_path_var.get().strip()

        if run_preprocess and (not ds_path or not os.path.exists(ds_path)):
            print("[Hata] Lütfen geçerli bir Dataset klasörü seçin!")
            return

        self.btn_oneclick_train.configure(state="disabled", text="⏳ Ses Modeli Eğitiliyor... (Konsoldan Epoch Takibi Yapabilirsiniz)")

        def _train_worker():
            try:
                res = run_full_voice_training_pipeline(
                    model_name=model_name,
                    dataset_path=ds_path,
                    sample_rate=int(self.tr_sample_rate_var.get()),
                    vocoder=self.tr_vocoder_var.get(),
                    f0_method=self.tr_f0_method_var.get(),
                    embedder_model=self.tr_embedder_var.get(),
                    pretrain_type=self.tr_pretrain_var.get(),
                    batch_size=int(self.tr_batch_size_var.get()),
                    total_epoch=int(self.tr_total_epoch_var.get()),
                    save_every_epoch=int(self.tr_save_every_var.get()),
                    cut_preprocess=self.tr_cut_mode_var.get(),
                    chunk_len=float(self.tr_chunk_len_var.get()),
                    overlap_len=float(self.tr_overlap_len_var.get()),
                    normalization_mode=self.tr_norm_mode_var.get(),
                    process_effects=self.tr_process_fx_var.get(),
                    noise_reduction=self.tr_noise_red_var.get(),
                    include_mutes=int(self.tr_include_mutes_var.get()),
                    index_algorithm=self.tr_index_algo_var.get(),
                    save_only_latest=self.tr_save_only_latest_var.get(),
                    save_every_weights=self.tr_save_every_weights_var.get(),
                    cache_data_in_gpu=self.tr_cache_gpu_var.get(),
                    checkpointing=self.tr_checkpointing_var.get(),
                    cleanup_fresh=self.tr_cleanup_fresh_var.get() if run_preprocess else False,
                    cpu_cores=int(self.tr_cpu_cores_var.get()),
                    run_preprocess=run_preprocess,
                    run_extract=run_extract,
                    run_train=run_train,
                    run_index=run_index,
                    auto_deploy_to_studio=self.tr_auto_deploy_var.get() and run_train,
                    progress_cb=lambda msg: print(msg),
                )
                if res.get("deployed_pth"):
                    self.after(0, lambda: self._select_newly_trained_model(model_name))
            except Exception as e:
                print(f"[Hata] Eğitim işlemi sırasında hata oluştu: {e}")
            finally:
                self.after(
                    0,
                    lambda: self.btn_oneclick_train.configure(
                        state="normal",
                        text="🚀 TEK TIKLA TAM OTOMATİK SES MODELİ EĞİT (Ön İşle + Özellik Çıkar + Eğit + Stüdyoya Kur)",
                    ),
                )

        threading.Thread(target=_train_worker, daemon=True).start()

    def _select_newly_trained_model(self, model_name: str):
        self.refresh_models()
        if model_name in self.models:
            self.model_var.set(model_name)

    def stop_voice_training(self):
        model_name = self.tr_model_name_var.get()
        stop_active_training(model_name, progress_cb=lambda msg: print(msg))

    def manual_deploy_trained_model(self):
        model_name = self.tr_model_name_var.get()
        try:
            deploy_trained_model_to_studio(model_name, progress_cb=lambda msg: print(msg))
            self._select_newly_trained_model(sanitize_name(model_name))
        except Exception as e:
            print(f"[Hata] Model aktarımı başarısız: {e}")

    # =========================================================================
    # SHARED METHODS & SETTINGS
    # =========================================================================
    def update_texts(self):
        self.title(translations[self.lang]["title"])
        self.lang_btn.configure(text=translations[self.lang]["lang_switch"])
        self.btn_save_settings.configure(text=translations[self.lang]["save_settings"])
        self.mode_btn_cover.configure(text=translations[self.lang]["mode_cover"])
        self.mode_btn_dataset.configure(text=translations[self.lang]["mode_dataset"])
        self.mode_btn_train.configure(text=translations[self.lang]["mode_train"])
        self.lbl_model_select.configure(text=translations[self.lang]["model_select"])
        self.btn_refresh.configure(text=translations[self.lang]["refresh"])
        self.btn_upload.configure(text=translations[self.lang]["upload_model"])
        self.lbl_pipeline_select.configure(text=translations[self.lang]["pipeline_select"])
        self.lbl_input.configure(text=translations[self.lang]["input_select"])
        self.btn_browse.configure(text=translations[self.lang]["browse"])
        self.lbl_output_dir.configure(text=translations[self.lang]["output_select"])
        self.btn_browse_dir.configure(text=translations[self.lang]["browse_dir"])
        self.lbl_vocal_pitch.configure(text=translations[self.lang]["vocal_pitch"])
        self.lbl_overall_pitch.configure(text=translations[self.lang]["overall_pitch"])
        self.lbl_main_gain.configure(text=translations[self.lang]["main_gain"])
        self.lbl_backup_gain.configure(text=translations[self.lang]["backup_gain"])
        self.cb_mute_backup.configure(text=translations[self.lang]["mute_backup"])
        self.lbl_inst_gain.configure(text=translations[self.lang]["inst_gain"])
        self.lbl_room_size.configure(text=translations[self.lang]["room_size"])
        self.lbl_wetness.configure(text=translations[self.lang]["wetness"])
        self.lbl_dryness.configure(text=translations[self.lang]["dryness"])
        self.lbl_damping.configure(text=translations[self.lang]["damping"])
        self.lbl_index_rate.configure(text=translations[self.lang]["index_rate"])
        self.lbl_filter_radius.configure(text=translations[self.lang]["filter_radius"])
        self.lbl_rms_mix.configure(text=translations[self.lang]["rms_mix"])
        self.lbl_protect.configure(text=translations[self.lang]["protect"])
        self.lbl_f0_method.configure(text=translations[self.lang]["f0_method"])
        self.lbl_crepe_hop.configure(text=translations[self.lang]["crepe_hop"])
        self.cb_agg_dereverb.configure(text=translations[self.lang]["agg_dereverb"])
        self.cb_keep_files.configure(text=translations[self.lang]["keep_files"])
        self.cb_force_reprocess.configure(text=translations[self.lang]["force_reprocess"])
        self.lbl_out_format.configure(text=translations[self.lang]["output_format"])
        self.sw_auto_mode.configure(text=translations[self.lang]["auto_mode"])
        self.generate_btn.configure(text=translations[self.lang]["generate"])
        self.btn_open_folder.configure(text=translations[self.lang]["open_folder"])
        self.play_btn.configure(text=translations[self.lang]["play_last"])
        self.btn_clear_log.configure(text=translations[self.lang]["clear_log"])
        self.lbl_logs.configure(text=translations[self.lang]["logs"])

        for name in self.tab_names:
            new_title = translations[self.lang][name]
            old_title = self.current_tab_titles[name]
            if old_title != new_title:
                try:
                    self.tabview.rename(old_title, new_title)
                    self.current_tab_titles[name] = new_title
                except ValueError:
                    pass

    def switch_lang(self):
        self.lang = "en" if self.lang == "tr" else "tr"
        self.update_texts()

    def save_settings(self):
        settings = {
            "lang": self.lang,
            "pipeline_var": self.pipeline_var.get(),
            "output_dir_var": self.output_dir_var.get(),
            "pitch_var": self.pitch_var.get(),
            "pitch_all_var": self.pitch_all_var.get(),
            "main_gain_var": self.main_gain_var.get(),
            "backup_gain_var": self.backup_gain_var.get(),
            "inst_gain_var": self.inst_gain_var.get(),
            "mute_backup_var": self.mute_backup_var.get(),
            "reverb_size_var": self.reverb_size_var.get(),
            "reverb_wet_var": self.reverb_wet_var.get(),
            "reverb_dry_var": self.reverb_dry_var.get(),
            "reverb_damp_var": self.reverb_damp_var.get(),
            "index_rate_var": self.index_rate_var.get(),
            "filter_radius_var": self.filter_radius_var.get(),
            "rms_mix_var": self.rms_mix_var.get(),
            "protect_var": self.protect_var.get(),
            "f0_method_var": self.f0_method_var.get(),
            "crepe_hop_var": self.crepe_hop_var.get(),
            "noise_gate_var": self.noise_gate_var.get(),
            "noise_gate_thresh_var": self.noise_gate_thresh_var.get(),
            "aggressive_dereverb_var": self.aggressive_dereverb_var.get(),
            "keep_files_var": self.keep_files_var.get(),
            "force_reprocess_var": self.force_reprocess_var.get(),
            "auto_mode_var": self.auto_mode_var.get(),
            "output_format_var": self.output_format_var.get(),
            "tr_preset_var": self.tr_preset_var.get(),
        }
        with open(settings_path, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=4)
        print(translations[self.lang]["settings_saved"])

    def load_settings(self):
        if os.path.exists(settings_path):
            try:
                with open(settings_path, 'r', encoding='utf-8') as f:
                    s = json.load(f)
                self.lang = s.get("lang", "tr")
                self.pipeline_var.set(s.get("pipeline_var", "UVR5 Special 2"))
                self.output_dir_var.set(s.get("output_dir_var", output_dir))
                self.pitch_var.set(s.get("pitch_var", 0))
                self.pitch_all_var.set(s.get("pitch_all_var", 0))
                self.main_gain_var.set(s.get("main_gain_var", 0))
                self.backup_gain_var.set(s.get("backup_gain_var", 0))
                self.inst_gain_var.set(s.get("inst_gain_var", 0))
                self.mute_backup_var.set(s.get("mute_backup_var", True))
                self.reverb_size_var.set(s.get("reverb_size_var", 0.15))
                self.reverb_wet_var.set(s.get("reverb_wet_var", 0.2))
                self.reverb_dry_var.set(s.get("reverb_dry_var", 0.8))
                self.reverb_damp_var.set(s.get("reverb_damp_var", 0.7))
                self.index_rate_var.set(s.get("index_rate_var", 0.5))
                self.filter_radius_var.set(s.get("filter_radius_var", 3))
                self.rms_mix_var.set(s.get("rms_mix_var", 0.25))
                self.protect_var.set(s.get("protect_var", 0.33))
                self.f0_method_var.set(s.get("f0_method_var", "rmvpe"))
                self.crepe_hop_var.set(s.get("crepe_hop_var", 128))
                self.noise_gate_var.set(s.get("noise_gate_var", False))
                self.noise_gate_thresh_var.set(s.get("noise_gate_thresh_var", -35))
                self.aggressive_dereverb_var.set(s.get("aggressive_dereverb_var", True))
                self.keep_files_var.set(s.get("keep_files_var", False))
                self.force_reprocess_var.set(s.get("force_reprocess_var", False))
                self.output_format_var.set(s.get("output_format_var", "mp3"))
                if "tr_preset_var" in s and s["tr_preset_var"] in TRAINING_PRESETS:
                    self.tr_preset_var.set(s["tr_preset_var"])
                    self.apply_training_preset(s["tr_preset_var"])
            except Exception as e:
                print(f"[System] Failed to load settings: {str(e)}")

    def refresh_models(self):
        self.models = get_current_models(rvc_models_dir)
        self.model_dropdown.configure(values=self.models if self.models else [""])
        if self.models and self.model_var.get() not in self.models:
            self.model_var.set(self.models[0])

        self.mdx_models = ["Default RVC", "UVR5 Special", "UVR5 Special 2"] + get_mdx_models()
        self.pipeline_dropdown.configure(values=self.mdx_models)
        print("[System] Ses ve UVR5 modelleri yenilendi.")

    def upload_model_gui(self):
        pth_path = filedialog.askopenfilename(title="Select .pth Model File", filetypes=[("PTH Files", "*.pth")])
        if not pth_path:
            return

        index_path = filedialog.askopenfilename(title="Select .index File (Optional)", filetypes=[("Index Files", "*.index")])

        dialog = ctk.CTkInputDialog(text="Enter a unique name for this model (No spaces/special chars):", title="Model Name")
        model_name = dialog.get_input()
        if not model_name:
            return

        import re
        model_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', model_name)

        target_dir = os.path.join(rvc_models_dir, model_name)
        try:
            os.makedirs(target_dir, exist_ok=True)
            shutil.copy2(pth_path, os.path.join(target_dir, f"{model_name}.pth"))
            if index_path:
                shutil.copy2(index_path, os.path.join(target_dir, f"{model_name}.index"))
            print(f"[System] Model '{model_name}' uploaded successfully!")
            self.refresh_models()
            self.model_var.set(model_name)
        except Exception as e:
            print(f"[Error] Failed to upload model: {e}")

    def check_file_cache(self, *args):
        filepath = self.song_input_var.get()
        if not filepath or not os.path.exists(filepath):
            self.cb_force_reprocess.configure(state="normal", text=translations[self.lang]["force_reprocess"])
            return

        try:
            from main import get_hash, get_audio_paths
            song_id = get_hash(filepath)
            song_dir = os.path.join(output_dir, song_id)
            if os.path.exists(song_dir):
                paths = get_audio_paths(song_dir, uvr_model=self.pipeline_var.get())
                if not any(path is None for path in paths):
                    self.cb_force_reprocess.configure(state="normal", text=translations[self.lang]["force_reprocess"])
                    return
        except Exception:
            pass

        self.force_reprocess_var.set(True)
        self.cb_force_reprocess.configure(
            state="disabled", text=translations[self.lang]["force_reprocess"] + translations[self.lang]["no_cache"]
        )

    def browse_file(self):
        filepath = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav;*.mp3;*.flac;*.ogg;*.m4a")])
        if filepath:
            filepath = os.path.normpath(filepath)
            self.song_input_var.set(filepath)
            print(f"[System] Selected local file: {filepath}")

    def browse_dir(self):
        dirpath = filedialog.askdirectory()
        if dirpath:
            dirpath = os.path.normpath(dirpath)
            self.output_dir_var.set(dirpath)
            print(f"[System] Output directory set to: {dirpath}")

    def open_output_folder(self):
        curr_out = self.output_dir_var.get()
        if not curr_out or not os.path.exists(curr_out):
            curr_out = output_dir
        if sys.platform == 'win32':
            os.startfile(curr_out)
        elif sys.platform == 'darwin':
            subprocess.Popen(['open', curr_out])
        else:
            subprocess.Popen(['xdg-open', curr_out])

    def play_output(self):
        if self.last_output_path and os.path.exists(self.last_output_path):
            if sys.platform == 'win32':
                os.startfile(self.last_output_path)
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', self.last_output_path])
            else:
                subprocess.Popen(['xdg-open', self.last_output_path])
        else:
            print("[System] Output file not found.")

    def clear_logs(self):
        self.log_textbox.delete("1.0", "end")

    def update_sliders_from_auto_mode(self, new_settings):
        def update():
            if 'rms_mix_rate' in new_settings:
                self.rms_mix_var.set(new_settings['rms_mix_rate'])
            if 'protect' in new_settings:
                self.protect_var.set(new_settings['protect'])
            if 'index_rate' in new_settings:
                self.index_rate_var.set(new_settings['index_rate'])
            if 'filter_radius' in new_settings:
                self.filter_radius_var.set(new_settings['filter_radius'])
            if 'reverb_rm_size' in new_settings:
                self.reverb_size_var.set(new_settings['reverb_rm_size'])
            if 'reverb_wet' in new_settings:
                self.reverb_wet_var.set(new_settings['reverb_wet'])
            if 'reverb_dry' in new_settings:
                self.reverb_dry_var.set(new_settings['reverb_dry'])
            print("[UI] Ayarlar Otomatik Ses Modu tarafından tespit edilip uygulandı!")
        self.after(0, update)

    def start_generation(self):
        song = self.song_input_var.get()
        model = self.model_var.get()
        if not song or not model or model == "Select a model" or model == "":
            print("[Error] Please select a model and input a song.")
            return

        self.generate_btn.configure(state="disabled", text=translations[self.lang]["generating"])

        def run_pipeline():
            try:
                print("=" * 60)
                print(f"[Start] Initiating Ripleytia Cover Pipeline...")
                print(f"[Start] Song: {song}")
                print(f"[Start] Model: {model}")
                print(f"[Start] UVR5 Mode: {self.pipeline_var.get()}")
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
                    auto_mode_callback=self.update_sliders_from_auto_mode,
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
                )

                final_out_dir = self.output_dir_var.get()
                if final_out_dir and os.path.exists(final_out_dir) and os.path.abspath(final_out_dir) != os.path.abspath(output_dir):
                    new_path = os.path.join(final_out_dir, os.path.basename(out_path))
                    shutil.copy2(out_path, new_path)
                    print(f"[System] Copied final output to: {new_path}")
                    out_path = new_path

                self.last_output_path = out_path
                self.play_btn.configure(state="normal")
            except Exception as e:
                print(f"\n[Error] Pipeline failed: {str(e)}")
            finally:
                self.generate_btn.configure(state="normal", text=translations[self.lang]["generate"])
                self.check_file_cache()

        threading.Thread(target=run_pipeline, daemon=True).start()


if __name__ == "__main__":
    app = App()
    app.mainloop()
