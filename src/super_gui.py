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
        self.text_widget.insert("end", string)
        self.text_widget.see("end")
    def flush(self): pass

def get_current_models(models_dir):
    if not os.path.exists(models_dir): return []
    return [item for item in os.listdir(models_dir) if item not in ['hubert_base.pt', 'MODELS.txt', 'public_models.json', 'rmvpe.pt']]

def get_mdx_models():
    if not os.path.exists(mdxnet_models_dir): return []
    return [item for item in os.listdir(mdxnet_models_dir) if item.endswith(".onnx") and item not in ['UVR_MDXNET_KARA_2.onnx', 'Reverb_HQ_By_FoxJoy.onnx']]

translations = {
    "en": {
        "title": "Ripleytia AI Studio - Ultimate Vocal & Cover Generator",
        "lang_switch": "🇹🇷 Türkçe",
        "refresh": "Refresh",
        "save_settings": "Save Settings",
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
        "noise_gate": "Vocal Noise Gate (Yankı Kuyruğu Kesici - For extreme echo)",
        "noise_gate_thresh": "Noise Gate Threshold (dB)",
        "agg_dereverb": "Aggressive DeReverb (Fix heavy echo)",
        "keep_files": "Keep Intermediate Files",
        "force_reprocess": "Force Re-Extract Vocals (Ignore Cache)",
        "no_cache": " (No Cache Found)",
        "output_format": "Output Format",
        "generate": "Generate AI Cover",
        "open_folder": "Open Output Folder",
        "play_last": "Play Last Output",
        "clear_log": "Clear Console",
        "logs": "Process Logs Console:",
        "generating": "Generating... Please wait",
        "settings_saved": "[System] Settings saved successfully!"
    },
    "tr": {
        "title": "Ripleytia AI Studio - Gelişmiş Vokal ve Cover Üretici",
        "lang_switch": "🇬🇧 English",
        "refresh": "Yenile",
        "save_settings": "Ayarları Kaydet",
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
        "open_folder": "Çıktı Klasörünü Aç",
        "play_last": "Son Çıktıyı Çal",
        "clear_log": "Konsolu Temizle",
        "logs": "İşlem Kayıtları (Log) Konsolu:",
        "generating": "İşleniyor... Lütfen bekleyin",
        "settings_saved": "[Sistem] Ayarlar başarıyla kaydedildi!"
    }
}

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")
PRIMARY_COLOR = "#5D3FD3" # Deep purple
HOVER_COLOR = "#4A0E4E"

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.lang = "tr" # Default Turkish
        self.title(translations[self.lang]["title"])
        self.geometry("980x950")
        
        # Configure layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Set Window and Taskbar Icon
        ico_path = os.path.join(BASE_DIR, 'logo.ico')
        if os.path.exists(ico_path):
            try:
                self.iconbitmap(ico_path)
            except: pass
            
        self.main_frame = ctk.CTkScrollableFrame(self, fg_color="#121212")
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Initialize Variables early for setting loading
        self.model_var = ctk.StringVar()
        self.pipeline_var = ctk.StringVar(value="Default RVC")
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
        
        # --- Header ---
        self.header_frame = ctk.CTkFrame(self.main_frame, fg_color="#1A1A1A", corner_radius=15)
        self.header_frame.pack(fill="x", pady=10, padx=10)
        
        # Logo
        if os.path.exists(logo_path):
            try:
                img = Image.open(logo_path)
                self.logo_image = ctk.CTkImage(light_image=img, dark_image=img, size=(80, 80))
                ctk.CTkLabel(self.header_frame, image=self.logo_image, text="").pack(side="left", padx=15, pady=10)
            except: pass
            
        self.header_title = ctk.CTkLabel(self.header_frame, text="Ripleytia Gothic AI", font=("Century Gothic", 28, "bold"), text_color=PRIMARY_COLOR)
        self.header_title.pack(side="left", padx=10, pady=20)
        
        self.lang_btn = ctk.CTkButton(self.header_frame, text=translations[self.lang]["lang_switch"], width=100, fg_color="#333", hover_color="#555", command=self.switch_lang)
        self.lang_btn.pack(side="right", padx=10)
        
        self.btn_save_settings = ctk.CTkButton(self.header_frame, text=translations[self.lang]["save_settings"], width=100, fg_color=PRIMARY_COLOR, hover_color=HOVER_COLOR, command=self.save_settings)
        self.btn_save_settings.pack(side="right", padx=10)
        
        # --- Models ---
        self.model_frame = ctk.CTkFrame(self.main_frame, fg_color="#1A1A1A", corner_radius=10)
        self.model_frame.pack(fill="x", pady=5, padx=10)
        
        self.lbl_model_select = ctk.CTkLabel(self.model_frame, text=translations[self.lang]["model_select"], font=("Arial", 16, "bold"), text_color="#DDD")
        self.lbl_model_select.pack(pady=5)
        
        self.models = get_current_models(rvc_models_dir)
        if not self.model_var.get() and self.models: self.model_var.set(self.models[0])
        self.model_dropdown = ctk.CTkOptionMenu(self.model_frame, values=self.models if self.models else [""], variable=self.model_var, width=300, fg_color=PRIMARY_COLOR, button_color=HOVER_COLOR)
        self.model_dropdown.pack(pady=5)
        
        self.model_btn_frame = ctk.CTkFrame(self.model_frame, fg_color="transparent")
        self.model_btn_frame.pack(pady=5)
        self.btn_refresh = ctk.CTkButton(self.model_btn_frame, text=translations[self.lang]["refresh"], fg_color="#444", hover_color="#222", command=self.refresh_models)
        self.btn_refresh.pack(side="left", padx=5)
        self.btn_upload = ctk.CTkButton(self.model_btn_frame, text=translations[self.lang]["upload_model"], fg_color=PRIMARY_COLOR, hover_color=HOVER_COLOR, command=self.upload_model_gui)
        self.btn_upload.pack(side="left", padx=5)

        # --- Pipeline / UVR5 Models ---
        self.lbl_pipeline_select = ctk.CTkLabel(self.model_frame, text=translations[self.lang]["pipeline_select"], font=("Arial", 16, "bold"), text_color="#DDD")
        self.lbl_pipeline_select.pack(pady=(15,5))
        
        self.mdx_models = ["Default RVC", "UVR5 Special"] + get_mdx_models()
        self.pipeline_dropdown = ctk.CTkOptionMenu(self.model_frame, values=self.mdx_models, variable=self.pipeline_var, width=300, fg_color="#005b96", button_color="#003f5c")
        self.pipeline_dropdown.pack(pady=5)
        
        # --- Input Selection ---
        self.input_frame = ctk.CTkFrame(self.main_frame, fg_color="#1A1A1A", corner_radius=10)
        self.input_frame.pack(fill="x", pady=5, padx=10)
        
        # Inputs & Outputs logic inline
        input_inner = ctk.CTkFrame(self.input_frame, fg_color="transparent")
        input_inner.pack(fill="x", pady=5)
        
        self.lbl_input = ctk.CTkLabel(input_inner, text=translations[self.lang]["input_select"], font=("Arial", 14, "bold"), text_color="#DDD")
        self.lbl_input.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.song_input_entry = ctk.CTkEntry(input_inner, textvariable=self.song_input_var, width=500)
        self.song_input_entry.grid(row=0, column=1, padx=10, pady=5)
        self.btn_browse = ctk.CTkButton(input_inner, text=translations[self.lang]["browse"], fg_color=PRIMARY_COLOR, hover_color=HOVER_COLOR, width=100, command=self.browse_file)
        self.btn_browse.grid(row=0, column=2, padx=10, pady=5)
        
        self.lbl_output_dir = ctk.CTkLabel(input_inner, text=translations[self.lang]["output_select"], font=("Arial", 14, "bold"), text_color="#DDD")
        self.lbl_output_dir.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.output_dir_entry = ctk.CTkEntry(input_inner, textvariable=self.output_dir_var, width=500)
        self.output_dir_entry.grid(row=1, column=1, padx=10, pady=5)
        self.btn_browse_dir = ctk.CTkButton(input_inner, text=translations[self.lang]["browse_dir"], fg_color=PRIMARY_COLOR, hover_color=HOVER_COLOR, width=100, command=self.browse_dir)
        self.btn_browse_dir.grid(row=1, column=2, padx=10, pady=5)
        
        # --- Advanced Options TabView ---
        self.tabview = ctk.CTkTabview(self.main_frame, width=850, fg_color="#1A1A1A", segmented_button_selected_color=PRIMARY_COLOR, segmented_button_selected_hover_color=HOVER_COLOR)
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
        self.lbl_vocal_pitch.pack(pady=(10,0))
        ctk.CTkSlider(tab_pitch, from_=-12, to=12, number_of_steps=24, variable=self.pitch_var, button_color=PRIMARY_COLOR, button_hover_color=HOVER_COLOR).pack()
        self.pitch_label = ctk.CTkLabel(tab_pitch, text=str(self.pitch_var.get()))
        self.pitch_label.pack()
        self.pitch_var.trace_add("write", lambda *args: self.pitch_label.configure(text=str(self.pitch_var.get())))
        
        self.lbl_overall_pitch = ctk.CTkLabel(tab_pitch, text=translations[self.lang]["overall_pitch"])
        self.lbl_overall_pitch.pack(pady=(10,0))
        ctk.CTkSlider(tab_pitch, from_=-12, to=12, number_of_steps=24, variable=self.pitch_all_var, button_color=PRIMARY_COLOR, button_hover_color=HOVER_COLOR).pack()
        self.pitch_all_label = ctk.CTkLabel(tab_pitch, text=str(self.pitch_all_var.get()))
        self.pitch_all_label.pack()
        self.pitch_all_var.trace_add("write", lambda *args: self.pitch_all_label.configure(text=str(self.pitch_all_var.get())))
        
        # - Audio Mixing
        tab_mix = self.tabs["tab_mix"]
        self.lbl_main_gain = ctk.CTkLabel(tab_mix, text=translations[self.lang]["main_gain"])
        self.lbl_main_gain.pack(pady=(5,0))
        ctk.CTkSlider(tab_mix, from_=-20, to=20, number_of_steps=40, variable=self.main_gain_var, button_color=PRIMARY_COLOR, button_hover_color=HOVER_COLOR).pack()
        self.main_gain_label = ctk.CTkLabel(tab_mix, text=str(self.main_gain_var.get()))
        self.main_gain_label.pack()
        self.main_gain_var.trace_add("write", lambda *args: self.main_gain_label.configure(text=str(self.main_gain_var.get())))

        self.lbl_backup_gain = ctk.CTkLabel(tab_mix, text=translations[self.lang]["backup_gain"])
        self.lbl_backup_gain.pack(pady=(5,0))
        ctk.CTkSlider(tab_mix, from_=-20, to=20, number_of_steps=40, variable=self.backup_gain_var, button_color=PRIMARY_COLOR, button_hover_color=HOVER_COLOR).pack()
        self.backup_gain_label = ctk.CTkLabel(tab_mix, text=str(self.backup_gain_var.get()))
        self.backup_gain_label.pack()
        self.backup_gain_var.trace_add("write", lambda *args: self.backup_gain_label.configure(text=str(self.backup_gain_var.get())))

        self.cb_mute_backup = ctk.CTkCheckBox(tab_mix, text=translations[self.lang]["mute_backup"], variable=self.mute_backup_var, fg_color=PRIMARY_COLOR, hover_color=HOVER_COLOR)
        self.cb_mute_backup.pack(pady=10)

        self.lbl_inst_gain = ctk.CTkLabel(tab_mix, text=translations[self.lang]["inst_gain"])
        self.lbl_inst_gain.pack(pady=(5,0))
        ctk.CTkSlider(tab_mix, from_=-20, to=20, number_of_steps=40, variable=self.inst_gain_var, button_color=PRIMARY_COLOR, button_hover_color=HOVER_COLOR).pack()
        self.inst_gain_label = ctk.CTkLabel(tab_mix, text=str(self.inst_gain_var.get()))
        self.inst_gain_label.pack()
        self.inst_gain_var.trace_add("write", lambda *args: self.inst_gain_label.configure(text=str(self.inst_gain_var.get())))

        # - Effects (Reverb)
        tab_fx = self.tabs["tab_fx"]
        self.lbl_room_size = ctk.CTkLabel(tab_fx, text=translations[self.lang]["room_size"])
        self.lbl_room_size.pack(pady=(5,0))
        ctk.CTkSlider(tab_fx, from_=0, to=1, variable=self.reverb_size_var, button_color=PRIMARY_COLOR, button_hover_color=HOVER_COLOR).pack()
        self.reverb_size_label = ctk.CTkLabel(tab_fx, text=str(round(self.reverb_size_var.get(), 2)))
        self.reverb_size_label.pack()
        self.reverb_size_var.trace_add("write", lambda *args: self.reverb_size_label.configure(text=str(round(self.reverb_size_var.get(), 2))))
        
        self.lbl_wetness = ctk.CTkLabel(tab_fx, text=translations[self.lang]["wetness"])
        self.lbl_wetness.pack(pady=(5,0))
        ctk.CTkSlider(tab_fx, from_=0, to=1, variable=self.reverb_wet_var, button_color=PRIMARY_COLOR, button_hover_color=HOVER_COLOR).pack()
        self.reverb_wet_label = ctk.CTkLabel(tab_fx, text=str(round(self.reverb_wet_var.get(), 2)))
        self.reverb_wet_label.pack()
        self.reverb_wet_var.trace_add("write", lambda *args: self.reverb_wet_label.configure(text=str(round(self.reverb_wet_var.get(), 2))))
        
        self.lbl_dryness = ctk.CTkLabel(tab_fx, text=translations[self.lang]["dryness"])
        self.lbl_dryness.pack(pady=(5,0))
        ctk.CTkSlider(tab_fx, from_=0, to=1, variable=self.reverb_dry_var, button_color=PRIMARY_COLOR, button_hover_color=HOVER_COLOR).pack()
        self.reverb_dry_label = ctk.CTkLabel(tab_fx, text=str(round(self.reverb_dry_var.get(), 2)))
        self.reverb_dry_label.pack()
        self.reverb_dry_var.trace_add("write", lambda *args: self.reverb_dry_label.configure(text=str(round(self.reverb_dry_var.get(), 2))))
        
        self.lbl_damping = ctk.CTkLabel(tab_fx, text=translations[self.lang]["damping"])
        self.lbl_damping.pack(pady=(5,0))
        ctk.CTkSlider(tab_fx, from_=0, to=1, variable=self.reverb_damp_var, button_color=PRIMARY_COLOR, button_hover_color=HOVER_COLOR).pack()
        self.reverb_damp_label = ctk.CTkLabel(tab_fx, text=str(round(self.reverb_damp_var.get(), 2)))
        self.reverb_damp_label.pack()
        self.reverb_damp_var.trace_add("write", lambda *args: self.reverb_damp_label.configure(text=str(round(self.reverb_damp_var.get(), 2))))
        
        # - Advanced Conversion
        tab_adv = self.tabs["tab_adv"]
        self.lbl_index_rate = ctk.CTkLabel(tab_adv, text=translations[self.lang]["index_rate"])
        self.lbl_index_rate.pack(pady=(5,0))
        ctk.CTkSlider(tab_adv, from_=0, to=1, variable=self.index_rate_var, button_color=PRIMARY_COLOR, button_hover_color=HOVER_COLOR).pack()
        self.index_rate_label = ctk.CTkLabel(tab_adv, text=str(round(self.index_rate_var.get(), 2)))
        self.index_rate_label.pack()
        self.index_rate_var.trace_add("write", lambda *args: self.index_rate_label.configure(text=str(round(self.index_rate_var.get(), 2))))
        
        self.lbl_filter_radius = ctk.CTkLabel(tab_adv, text=translations[self.lang]["filter_radius"])
        self.lbl_filter_radius.pack(pady=(5,0))
        ctk.CTkSlider(tab_adv, from_=0, to=7, number_of_steps=7, variable=self.filter_radius_var, button_color=PRIMARY_COLOR, button_hover_color=HOVER_COLOR).pack()
        self.filter_radius_label = ctk.CTkLabel(tab_adv, text=str(self.filter_radius_var.get()))
        self.filter_radius_label.pack()
        self.filter_radius_var.trace_add("write", lambda *args: self.filter_radius_label.configure(text=str(self.filter_radius_var.get())))
        
        self.lbl_rms_mix = ctk.CTkLabel(tab_adv, text=translations[self.lang]["rms_mix"])
        self.lbl_rms_mix.pack(pady=(5,0))
        ctk.CTkSlider(tab_adv, from_=0, to=1, variable=self.rms_mix_var, button_color=PRIMARY_COLOR, button_hover_color=HOVER_COLOR).pack()
        self.rms_mix_label = ctk.CTkLabel(tab_adv, text=str(round(self.rms_mix_var.get(), 2)))
        self.rms_mix_label.pack()
        self.rms_mix_var.trace_add("write", lambda *args: self.rms_mix_label.configure(text=str(round(self.rms_mix_var.get(), 2))))
        
        self.lbl_protect = ctk.CTkLabel(tab_adv, text=translations[self.lang]["protect"])
        self.lbl_protect.pack(pady=(5,0))
        ctk.CTkSlider(tab_adv, from_=0, to=0.5, variable=self.protect_var, button_color=PRIMARY_COLOR, button_hover_color=HOVER_COLOR).pack()
        self.protect_label = ctk.CTkLabel(tab_adv, text=str(round(self.protect_var.get(), 2)))
        self.protect_label.pack()
        self.protect_var.trace_add("write", lambda *args: self.protect_label.configure(text=str(round(self.protect_var.get(), 2))))
        
        self.lbl_f0_method = ctk.CTkLabel(tab_adv, text=translations[self.lang]["f0_method"])
        self.lbl_f0_method.pack(pady=(5,0))
        f0_seg = ctk.CTkSegmentedButton(tab_adv, values=["rmvpe", "mangio-crepe"], variable=self.f0_method_var, selected_color=PRIMARY_COLOR, selected_hover_color=HOVER_COLOR)
        f0_seg.pack(pady=5)
        
        self.lbl_crepe_hop = ctk.CTkLabel(tab_adv, text=translations[self.lang]["crepe_hop"])
        self.crepe_slider = ctk.CTkSlider(tab_adv, from_=32, to=320, number_of_steps=288, variable=self.crepe_hop_var, button_color=PRIMARY_COLOR, button_hover_color=HOVER_COLOR)
        self.crepe_hop_label = ctk.CTkLabel(tab_adv, text=str(self.crepe_hop_var.get()))
        self.crepe_hop_var.trace_add("write", lambda *args: self.crepe_hop_label.configure(text=str(self.crepe_hop_var.get())))
        
        def toggle_crepe(*args):
            if self.f0_method_var.get() == "mangio-crepe":
                self.lbl_crepe_hop.pack(pady=(5,0))
                self.crepe_slider.pack()
                self.crepe_hop_label.pack()
            else:
                self.lbl_crepe_hop.pack_forget()
                self.crepe_slider.pack_forget()
                self.crepe_hop_label.pack_forget()
                
        self.f0_method_var.trace_add("write", toggle_crepe)
        
        self.noise_gate_var = ctk.BooleanVar(value=False)
        self.cb_noise_gate = ctk.CTkCheckBox(tab_adv, text=translations[self.lang]["noise_gate"], variable=self.noise_gate_var, fg_color=PRIMARY_COLOR, hover_color=HOVER_COLOR)
        self.cb_noise_gate.pack(pady=10)
        
        self.lbl_noise_gate_thresh = ctk.CTkLabel(tab_adv, text=translations[self.lang]["noise_gate_thresh"])
        self.ng_slider = ctk.CTkSlider(tab_adv, from_=-60, to=-10, number_of_steps=50, variable=self.noise_gate_thresh_var, button_color=PRIMARY_COLOR, button_hover_color=HOVER_COLOR)
        self.ng_thresh_label = ctk.CTkLabel(tab_adv, text=str(self.noise_gate_thresh_var.get()))
        self.noise_gate_thresh_var.trace_add("write", lambda *args: self.ng_thresh_label.configure(text=str(self.noise_gate_thresh_var.get())))
        
        def toggle_ng(*args):
            if self.noise_gate_var.get():
                self.lbl_noise_gate_thresh.pack(pady=(5,0))
                self.ng_slider.pack()
                self.ng_thresh_label.pack()
            else:
                self.lbl_noise_gate_thresh.pack_forget()
                self.ng_slider.pack_forget()
                self.ng_thresh_label.pack_forget()
                
        self.noise_gate_var.trace_add("write", toggle_ng)
        
        self.aggressive_dereverb_var = ctk.BooleanVar(value=True) # Default to true for better results
        self.cb_agg_dereverb = ctk.CTkCheckBox(tab_adv, text=translations[self.lang]["agg_dereverb"], variable=self.aggressive_dereverb_var, fg_color=PRIMARY_COLOR, hover_color=HOVER_COLOR)
        self.cb_agg_dereverb.pack(pady=10)

        self.cb_keep_files = ctk.CTkCheckBox(tab_adv, text=translations[self.lang]["keep_files"], variable=self.keep_files_var, fg_color=PRIMARY_COLOR, hover_color=HOVER_COLOR)
        self.cb_keep_files.pack(pady=10)

        self.cb_force_reprocess = ctk.CTkCheckBox(tab_adv, text=translations[self.lang]["force_reprocess"], variable=self.force_reprocess_var, fg_color=PRIMARY_COLOR, hover_color=HOVER_COLOR)
        self.cb_force_reprocess.pack(pady=10)
        self.song_input_var.trace_add("write", self.check_file_cache)
        
        self.lbl_out_format = ctk.CTkLabel(tab_adv, text=translations[self.lang]["output_format"])
        self.lbl_out_format.pack(pady=(5,0))
        ctk.CTkSegmentedButton(tab_adv, values=["mp3", "wav"], variable=self.output_format_var, selected_color=PRIMARY_COLOR, selected_hover_color=HOVER_COLOR).pack(pady=5)

        # Load Settings after defining widgets
        self.load_settings()

        # --- Generate Button ---
        self.generate_btn = ctk.CTkButton(self.main_frame, text=translations[self.lang]["generate"], fg_color=PRIMARY_COLOR, hover_color=HOVER_COLOR, font=("Century Gothic", 18, "bold"), command=self.start_generation, height=50)
        self.generate_btn.pack(pady=20, fill="x", padx=50)
        
        # --- Tools ---
        self.tools_frame = ctk.CTkFrame(self.main_frame, fg_color="#1A1A1A", corner_radius=10)
        self.tools_frame.pack(fill="x", pady=5, padx=10)
        
        self.btn_open_folder = ctk.CTkButton(self.tools_frame, text=translations[self.lang]["open_folder"], command=self.open_output_folder, fg_color="#333", hover_color="#555")
        self.btn_open_folder.pack(side="left", padx=10, pady=10)
        self.play_btn = ctk.CTkButton(self.tools_frame, text=translations[self.lang]["play_last"], command=self.play_output, state="disabled", fg_color="#333", hover_color="#555")
        self.play_btn.pack(side="left", padx=10, pady=10)
        self.btn_clear_log = ctk.CTkButton(self.tools_frame, text=translations[self.lang]["clear_log"], command=self.clear_logs, fg_color="#8b0000", hover_color="#5a0000")
        self.btn_clear_log.pack(side="right", padx=10, pady=10)
        
        self.last_output_path = None
        
        # --- Log Console ---
        self.lbl_logs = ctk.CTkLabel(self.main_frame, text=translations[self.lang]["logs"], font=("Arial", 12, "bold"))
        self.lbl_logs.pack(anchor="w", padx=10, pady=(10,0))
        self.log_textbox = ctk.CTkTextbox(self.main_frame, height=200, font=("Consolas", 12), fg_color="#0A0A0A", text_color="#A982FA") # Purple-ish logs
        self.log_textbox.pack(fill="x", padx=10, pady=5)
        
        # Redirect stdout and stderr
        sys.stdout = RedirectText(self.log_textbox)
        sys.stderr = RedirectText(self.log_textbox)
        
        # Initial updates
        self.update_texts()
        toggle_crepe()
        print(f"[{translations[self.lang]['title']}] Initialized!")
        print(f"Output Directory: {self.output_dir_var.get()}")

    def update_texts(self):
        self.title(translations[self.lang]["title"])
        self.lang_btn.configure(text=translations[self.lang]["lang_switch"])
        self.btn_save_settings.configure(text=translations[self.lang]["save_settings"])
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
            "output_format_var": self.output_format_var.get()
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
                self.pipeline_var.set(s.get("pipeline_var", "Default RVC"))
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
            except Exception as e:
                print(f"[System] Failed to load settings: {str(e)}")

    def refresh_models(self):
        self.models = get_current_models(rvc_models_dir)
        self.model_dropdown.configure(values=self.models)
        if self.models and not self.model_var.get() in self.models: 
            self.model_var.set(self.models[0])
        
        self.mdx_models = ["Default RVC", "UVR5 Special"] + get_mdx_models()
        self.pipeline_dropdown.configure(values=self.mdx_models)
        print("[System] Voice & UVR5 models refreshed.")
        
    def upload_model_gui(self):
        pth_path = filedialog.askopenfilename(title="Select .pth Model File", filetypes=[("PTH Files", "*.pth")])
        if not pth_path: return
        
        index_path = filedialog.askopenfilename(title="Select .index File (Optional)", filetypes=[("Index Files", "*.index")])
        
        dialog = ctk.CTkInputDialog(text="Enter a unique name for this model (No spaces/special chars):", title="Model Name")
        model_name = dialog.get_input()
        if not model_name: return
        
        import re
        import shutil
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
                paths = get_audio_paths(song_dir)
                if not any(path is None for path in paths):
                    self.cb_force_reprocess.configure(state="normal", text=translations[self.lang]["force_reprocess"])
                    return
        except Exception:
            pass

        self.force_reprocess_var.set(True)
        self.cb_force_reprocess.configure(state="disabled", text=translations[self.lang]["force_reprocess"] + translations[self.lang]["no_cache"])

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
        if not curr_out or not os.path.exists(curr_out): curr_out = output_dir
        if sys.platform == 'win32': os.startfile(curr_out)
        elif sys.platform == 'darwin': subprocess.Popen(['open', curr_out])
        else: subprocess.Popen(['xdg-open', curr_out])

    def play_output(self):
        if self.last_output_path and os.path.exists(self.last_output_path):
            if sys.platform == 'win32': os.startfile(self.last_output_path)
            elif sys.platform == 'darwin': subprocess.Popen(['open', self.last_output_path])
            else: subprocess.Popen(['xdg-open', self.last_output_path])
        else:
            print("[System] Output file not found.")

    def clear_logs(self):
        self.log_textbox.delete("1.0", "end")

    def start_generation(self):
        song = self.song_input_var.get()
        model = self.model_var.get()
        if not song or not model or model == "Select a model" or model == "":
            print("[Error] Please select a model and input a song.")
            return
            
        self.generate_btn.configure(state="disabled", text=translations[self.lang]["generating"])
        
        def run_pipeline():
            try:
                print("="*60)
                print(f"[Start] Initiating Ripleytia Cover Pipeline...")
                print(f"[Start] Song: {song}")
                print(f"[Start] Model: {model}")
                print(f"[Start] UVR5 Mode: {self.pipeline_var.get()}")
                print("="*60)
                
                # If they choose to mute backup vocals, set gain to -100
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
                    pitch_change_all=self.pitch_all_var.get(),
                    reverb_rm_size=self.reverb_size_var.get(),
                    reverb_wet=self.reverb_wet_var.get(),
                    reverb_dry=self.reverb_dry_var.get(),
                    reverb_damping=self.reverb_damp_var.get(),
                    output_format=self.output_format_var.get(),
                    aggressive_dereverb=self.aggressive_dereverb_var.get(),
                    uvr_model=self.pipeline_var.get(),
                    vocal_noise_gate=self.noise_gate_var.get(),
                    noise_gate_threshold=self.noise_gate_thresh_var.get()
                )
                
                # Move to custom output dir if set
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













