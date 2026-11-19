"""
Custom Waveform Audio Player Widget for Ripleytia AI Studio.
Provides in-app playback with real-time waveform visualization, scrubbing slider, volume control, and play/pause.
Powered by pygame.mixer.
"""

import os
import time
import threading
import tkinter as tk
import customtkinter as ctk
import soundfile as sf
import numpy as np

# Suppress pygame welcome message
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import pygame


class AudioPlayerWidget(ctk.CTkFrame):
    """
    Embeddable in-app Waveform Audio Player.
    """
    def __init__(self, master, fg_color="#181824", border_color="#5D3FD3", border_width=1, **kwargs):
        super().__init__(master, fg_color=fg_color, border_color=border_color, border_width=border_width, corner_radius=12, **kwargs)

        self.current_file = None
        self.duration_seconds = 0.0
        self.is_playing = False
        self.is_paused = False
        self.start_time = 0.0
        self.pause_offset = 0.0
        self.waveform_peaks = []
        self._user_seeking = False

        # Initialize pygame mixer once
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=44100)
        except Exception as e:
            print(f"[Player Uyarısı] pygame mixer başlatılamadı: {e}")

        # Top title & track name
        self.top_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.top_frame.pack(fill="x", padx=14, pady=(10, 4))

        self.lbl_title = ctk.CTkLabel(
            self.top_frame,
            text="🎧 Dahili Oynatıcı: Ses dosyası bekleniyor...",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            text_color="#A390E4",
            anchor="w"
        )
        self.lbl_title.pack(side="left", fill="x", expand=True)

        self.lbl_time = ctk.CTkLabel(
            self.top_frame,
            text="00:00 / 00:00",
            font=ctk.CTkFont(family="Consolas", size=12),
            text_color="#B0B0C0"
        )
        self.lbl_time.pack(side="right")

        # Waveform Canvas
        self.canvas_frame = ctk.CTkFrame(self, fg_color="#0F0F17", corner_radius=8)
        self.canvas_frame.pack(fill="x", padx=14, pady=4)

        self.canvas = tk.Canvas(
            self.canvas_frame,
            height=54,
            bg="#0F0F17",
            highlightthickness=0
        )
        self.canvas.pack(fill="x", padx=6, pady=4, expand=True)
        self.canvas.bind("<Configure>", lambda e: self._draw_waveform())

        # Progress timeline slider
        self.slider = ctk.CTkSlider(
            self,
            from_=0.0,
            to=100.0,
            number_of_steps=1000,
            button_color="#5D3FD3",
            button_hover_color="#7B68EE",
            progress_color="#5D3FD3",
            fg_color="#2A2A3E",
            command=self._on_slider_change
        )
        self.slider.pack(fill="x", padx=14, pady=(4, 6))
        self.slider.set(0.0)
        self.slider.bind("<ButtonPress-1>", lambda e: self._on_slider_press())
        self.slider.bind("<ButtonRelease-1>", lambda e: self._on_slider_release())

        # Bottom controls (Play/Pause, Stop, Volume)
        self.ctrl_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.ctrl_frame.pack(fill="x", padx=14, pady=(0, 10))

        self.btn_play = ctk.CTkButton(
            self.ctrl_frame,
            text="▶️ Oynat",
            width=90,
            height=30,
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            fg_color="#5D3FD3",
            hover_color="#7B68EE",
            command=self.toggle_play
        )
        self.btn_play.pack(side="left", padx=(0, 8))

        self.btn_stop = ctk.CTkButton(
            self.ctrl_frame,
            text="⏹️ Durdur",
            width=80,
            height=30,
            font=ctk.CTkFont(family="Segoe UI", size=12),
            fg_color="#2A2A3E",
            hover_color="#3D3D58",
            command=self.stop
        )
        self.btn_stop.pack(side="left", padx=4)

        # Volume control
        self.lbl_vol = ctk.CTkLabel(self.ctrl_frame, text="🔊", font=ctk.CTkFont(size=14))
        self.lbl_vol.pack(side="right", padx=(4, 0))

        self.slider_vol = ctk.CTkSlider(
            self.ctrl_frame,
            from_=0.0,
            to=1.0,
            width=90,
            button_color="#7B68EE",
            progress_color="#5D3FD3",
            command=self._set_volume
        )
        self.slider_vol.pack(side="right", padx=6)
        self.slider_vol.set(0.85)

        # Periodic loop to update UI playback position
        self._update_loop()

    def load_audio(self, file_path: str, autoplay: bool = False):
        """
        Loads an audio file into the player, calculates waveform, and optionally autoplays.
        """
        if not file_path or not os.path.exists(file_path):
            return

        self.stop()
        self.current_file = file_path
        file_name = os.path.basename(file_path)
        self.lbl_title.configure(text=f"🎧 {file_name}")

        # Compute duration & waveform in background thread to avoid freezing UI
        def _calc():
            try:
                # Read audio data with soundfile for waveform
                data, sr = sf.read(file_path, dtype="float32", always_2d=True)
                mono = np.mean(data, axis=1)
                self.duration_seconds = len(mono) / float(sr)

                # Downsample to 100 peaks for waveform canvas
                num_bars = 100
                chunk_size = max(1, len(mono) // num_bars)
                peaks = []
                for i in range(num_bars):
                    chunk = mono[i * chunk_size : (i + 1) * chunk_size]
                    if len(chunk) > 0:
                        peaks.append(float(np.max(np.abs(chunk))))
                    else:
                        peaks.append(0.0)

                # Normalize peaks to [0.05, 1.0]
                max_peak = max(peaks) if max(peaks) > 0 else 1.0
                self.waveform_peaks = [max(0.08, p / max_peak) for p in peaks]

                # Update UI on main thread
                self.after(0, self._on_audio_loaded, autoplay)
            except Exception as e:
                print(f"[Player Hatası] Dalga formu çıkarılamadı: {e}")
                self.duration_seconds = 180.0
                self.waveform_peaks = [0.3] * 100
                self.after(0, self._on_audio_loaded, autoplay)

        threading.Thread(target=_calc, daemon=True).start()

    def _on_audio_loaded(self, autoplay: bool):
        total_str = self._format_time(self.duration_seconds)
        self.lbl_time.configure(text=f"00:00 / {total_str}")
        self.slider.configure(to=max(1.0, self.duration_seconds))
        self.slider.set(0.0)
        self._draw_waveform()
        if autoplay:
            self.play()

    def toggle_play(self):
        if not self.current_file or not os.path.exists(self.current_file):
            return
        if self.is_playing:
            self.pause()
        else:
            self.play()

    def play(self):
        if not self.current_file or not os.path.exists(self.current_file):
            return
        try:
            if self.is_paused:
                pygame.mixer.music.unpause()
                self.is_paused = False
                self.is_playing = True
                self.start_time = time.time() - self.pause_offset
            else:
                pygame.mixer.music.load(self.current_file)
                pygame.mixer.music.set_volume(self.slider_vol.get())
                start_pos = self.slider.get()
                pygame.mixer.music.play(start=start_pos)
                self.is_playing = True
                self.is_paused = False
                self.start_time = time.time() - start_pos

            self.btn_play.configure(text="⏸️ Duraklat", fg_color="#3D3D58")
        except Exception as e:
            print(f"[Player Hatası] Oynatma başarısız: {e}")

    def pause(self):
        if self.is_playing:
            try:
                pygame.mixer.music.pause()
                self.is_paused = True
                self.is_playing = False
                self.pause_offset = time.time() - self.start_time
                self.btn_play.configure(text="▶️ Sürdür", fg_color="#5D3FD3")
            except Exception as e:
                print(f"[Player Hatası] Duraklatma hatası: {e}")

    def stop(self):
        try:
            pygame.mixer.music.stop()
        except Exception:
            pass
        self.is_playing = False
        self.is_paused = False
        self.pause_offset = 0.0
        self.slider.set(0.0)
        self.btn_play.configure(text="▶️ Oynat", fg_color="#5D3FD3")
        total_str = self._format_time(self.duration_seconds)
        self.lbl_time.configure(text=f"00:00 / {total_str}")
        self._draw_waveform(progress_ratio=0.0)

    def _set_volume(self, val):
        try:
            pygame.mixer.music.set_volume(float(val))
        except Exception:
            pass

    def _on_slider_press(self):
        self._user_seeking = True

    def _on_slider_release(self):
        self._user_seeking = False
        if self.current_file:
            seek_pos = self.slider.get()
            self.pause_offset = seek_pos
            if self.is_playing:
                try:
                    pygame.mixer.music.stop()
                    pygame.mixer.music.load(self.current_file)
                    pygame.mixer.music.play(start=seek_pos)
                    self.start_time = time.time() - seek_pos
                except Exception as e:
                    print(f"[Player Seek Hatası] {e}")

    def _on_slider_change(self, val):
        curr_str = self._format_time(float(val))
        total_str = self._format_time(self.duration_seconds)
        self.lbl_time.configure(text=f"{curr_str} / {total_str}")
        if self.duration_seconds > 0:
            self._draw_waveform(progress_ratio=float(val) / self.duration_seconds)

    def _update_loop(self):
        if self.is_playing and not self._user_seeking and self.duration_seconds > 0:
            elapsed = time.time() - self.start_time
            if elapsed >= self.duration_seconds or not pygame.mixer.music.get_busy():
                self.stop()
            else:
                self.slider.set(elapsed)
                curr_str = self._format_time(elapsed)
                total_str = self._format_time(self.duration_seconds)
                self.lbl_time.configure(text=f"{curr_str} / {total_str}")
                self._draw_waveform(progress_ratio=elapsed / self.duration_seconds)

        self.after(100, self._update_loop)

    def _draw_waveform(self, progress_ratio: float = None):
        if not self.waveform_peaks:
            return

        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w <= 10 or h <= 10:
            return

        self.canvas.delete("all")
        num_bars = len(self.waveform_peaks)
        bar_width = max(2, (w / num_bars) - 1.5)

        if progress_ratio is None:
            if self.duration_seconds > 0:
                progress_ratio = self.slider.get() / self.duration_seconds
            else:
                progress_ratio = 0.0

        for i, peak in enumerate(self.waveform_peaks):
            x0 = i * (w / num_bars)
            x1 = x0 + bar_width
            bar_h = max(4, peak * (h - 8))
            y0 = (h - bar_h) / 2
            y1 = y0 + bar_h

            ratio = i / float(num_bars)
            if ratio <= progress_ratio:
                # Active / Played color (Electric Purple / Pink)
                color = "#9966FF"
            else:
                # Inactive / Remaining color (Dark Navy / Gray)
                color = "#363654"

            self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="")

    @staticmethod
    def _format_time(sec: float) -> str:
        s = int(max(0, sec))
        m = s // 60
        s = s % 60
        return f"{m:02d}:{s:02d}"
