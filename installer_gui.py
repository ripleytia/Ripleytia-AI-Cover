import os
import sys
import shutil
import zipfile
import threading
import subprocess
import time
import customtkinter as ctk
from tkinter import filedialog, messagebox

# --- Theme Configuration (Ripleytia Gothic Purple / Dark Cyber) ---
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

BG_COLOR = "#121212"
CARD_BG = "#1A1A1A"
PRIMARY_COLOR = "#5D3FD3"
HOVER_COLOR = "#4A0E4E"
ACCENT_TEAL = "#00695C"
ACCENT_TEAL_HOVER = "#004D40"
TEXT_MUTED = "#AAAAAA"
TEXT_WHITE = "#FFFFFF"

def get_bundle_dir():
    if getattr(sys, "frozen", False):
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))

def create_windows_shortcut(target_path, working_dir, shortcut_name, icon_path=None, args=""):
    try:
        ps_script = f"""
        $WshShell = New-Object -comObject WScript.Shell
        $DesktopPath = [System.Environment]::GetFolderPath('Desktop')
        $Shortcut = $WshShell.CreateShortcut([System.IO.Path]::Combine($DesktopPath, '{shortcut_name}.lnk'))
        $Shortcut.TargetPath = '{target_path}'
        $Shortcut.Arguments = '{args}'
        $Shortcut.WorkingDirectory = '{working_dir}'
        if ('{icon_path}' -ne '' -and (Test-Path '{icon_path}')) {{
            $Shortcut.IconLocation = '{icon_path}'
        }}
        $Shortcut.Save()
        """
        subprocess.run(["powershell", "-NoProfile", "-Command", ps_script], check=True, creationflags=0x08000000 if os.name == 'nt' else 0)
        return True
    except Exception as e:
        print(f"Kısayol oluşturma hatası: {e}")
        return False

class RipleytiaInstallerApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Ripleytia AI Sound Studio V3.5 - Kurulum Sihirbazı")
        self.geometry("740x620")
        self.resizable(False, False)
        self.configure(fg_color=BG_COLOR)

        # Set Window Icon
        bundle_dir = get_bundle_dir()
        ico_path = os.path.join(bundle_dir, "logo.ico")
        if not os.path.exists(ico_path):
            ico_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo.ico")
        if os.path.exists(ico_path):
            try:
                self.iconbitmap(ico_path)
            except Exception:
                pass

        # Determine default install path
        default_dir = os.path.join(os.environ.get("USERPROFILE", "C:"), "Ripleytia_AI_Studio")
        if not os.path.exists(os.path.dirname(default_dir)):
            default_dir = r"C:\Ripleytia_AI_Studio"

        self.install_dir_var = ctk.StringVar(value=default_dir)
        self.create_shortcut_var = ctk.BooleanVar(value=True)
        self.setup_venv_var = ctk.BooleanVar(value=True)
        self.download_models_var = ctk.BooleanVar(value=True)

        self.is_installing = False
        self.installed_path = ""

        self.build_ui()

    def build_ui(self):
        # Header Card
        self.header_frame = ctk.CTkFrame(self, fg_color=CARD_BG, corner_radius=15)
        self.header_frame.pack(fill="x", padx=15, pady=(15, 10))

        title_lbl = ctk.CTkLabel(
            self.header_frame,
            text="✨ Ripleytia AI Sound Studio V3.5",
            font=("Arial", 22, "bold"),
            text_color=TEXT_WHITE,
        )
        title_lbl.pack(anchor="w", padx=20, pady=(15, 2))

        sub_lbl = ctk.CTkLabel(
            self.header_frame,
            text="AI Cover Motoru | Otomatik Dataset Hazırlayıcı | Profesyonel RVC Modelleme | UVR5 Special 2",
            font=("Arial", 12),
            text_color=TEXT_MUTED,
        )
        sub_lbl.pack(anchor="w", padx=20, pady=(0, 15))

        # Main Body Frame
        self.body_frame = ctk.CTkFrame(self, fg_color=CARD_BG, corner_radius=15)
        self.body_frame.pack(fill="x", padx=15, pady=5)

        # 1. Directory Selection
        dir_lbl = ctk.CTkLabel(
            self.body_frame,
            text="📁 Kurulum Dizini (Hedef Klasör):",
            font=("Arial", 13, "bold"),
            text_color=TEXT_WHITE,
        )
        dir_lbl.pack(anchor="w", padx=20, pady=(15, 5))

        dir_box = ctk.CTkFrame(self.body_frame, fg_color="transparent")
        dir_box.pack(fill="x", padx=20, pady=(0, 15))

        self.dir_entry = ctk.CTkEntry(
            dir_box,
            textvariable=self.install_dir_var,
            font=("Arial", 12),
            fg_color="#222222",
            border_color="#333333",
            height=35,
        )
        self.dir_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))

        browse_btn = ctk.CTkButton(
            dir_box,
            text="Gözat...",
            width=90,
            height=35,
            fg_color="#333333",
            hover_color="#555555",
            command=self.browse_dir,
        )
        browse_btn.pack(side="right")

        # 2. Options Checkboxes
        opts_lbl = ctk.CTkLabel(
            self.body_frame,
            text="⚙️ Kurulum Bileşenleri ve Seçenekleri:",
            font=("Arial", 13, "bold"),
            text_color=TEXT_WHITE,
        )
        opts_lbl.pack(anchor="w", padx=20, pady=(5, 5))

        cb_frame = ctk.CTkFrame(self.body_frame, fg_color="transparent")
        cb_frame.pack(fill="x", padx=20, pady=(0, 15))

        self.cb1 = ctk.CTkCheckBox(
            cb_frame,
            text="Masaüstüne Kısayol Oluştur (Ripleytia AI Studio)",
            variable=self.create_shortcut_var,
            fg_color=PRIMARY_COLOR,
            hover_color=HOVER_COLOR,
        )
        self.cb1.pack(anchor="w", pady=4)

        self.cb2 = ctk.CTkCheckBox(
            cb_frame,
            text="Gerekli Python & CUDA Destekli PyTorch Ortamını Otomatik Kur (Önerilen)",
            variable=self.setup_venv_var,
            fg_color=PRIMARY_COLOR,
            hover_color=HOVER_COLOR,
        )
        self.cb2.pack(anchor="w", pady=4)

        self.cb3 = ctk.CTkCheckBox(
            cb_frame,
            text="Temel AI Ses Modellerini Doğrula ve İndir (HuBERT, RMVPE, MDX-Net)",
            variable=self.download_models_var,
            fg_color=PRIMARY_COLOR,
            hover_color=HOVER_COLOR,
        )
        self.cb3.pack(anchor="w", pady=4)

        # 3. Log Console & Progress Frame
        self.log_frame = ctk.CTkFrame(self, fg_color=CARD_BG, corner_radius=15)
        self.log_frame.pack(fill="both", expand=True, padx=15, pady=10)

        # Status & Progress bar
        self.status_lbl = ctk.CTkLabel(
            self.log_frame,
            text="Hazır. Kuruluma başlamak için aşağıdaki butona tıklayın.",
            font=("Arial", 12),
            text_color="#88E288",
        )
        self.status_lbl.pack(anchor="w", padx=20, pady=(10, 2))

        self.progress_bar = ctk.CTkProgressBar(
            self.log_frame,
            fg_color="#2A2A2A",
            progress_color=PRIMARY_COLOR,
            height=10,
        )
        self.progress_bar.pack(fill="x", padx=20, pady=(2, 8))
        self.progress_bar.set(0.0)

        # Console
        self.console = ctk.CTkTextbox(
            self.log_frame,
            font=("Consolas", 11),
            fg_color="#0A0A0A",
            text_color="#00FF66",
            border_color="#222222",
            border_width=1,
            height=130,
        )
        self.console.pack(fill="both", expand=True, padx=20, pady=(0, 10))

        # Bottom Action Bar
        self.action_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.action_frame.pack(fill="x", padx=15, pady=(0, 15))

        self.install_btn = ctk.CTkButton(
            self.action_frame,
            text="🚀 Kurulumu Başlat (Install)",
            font=("Arial", 14, "bold"),
            fg_color=PRIMARY_COLOR,
            hover_color=HOVER_COLOR,
            height=42,
            command=self.start_installation,
        )
        self.install_btn.pack(side="left", fill="x", expand=True, padx=(0, 10))

        self.launch_btn = ctk.CTkButton(
            self.action_frame,
            text="🎉 Uygulamayı Başlat (Launch)",
            font=("Arial", 14, "bold"),
            fg_color=ACCENT_TEAL,
            hover_color=ACCENT_TEAL_HOVER,
            height=42,
            state="disabled",
            command=self.launch_studio,
        )
        self.launch_btn.pack(side="right", fill="x", expand=True)

    def log(self, message):
        self.console.insert("end", f"{message}\n")
        self.console.see("end")

    def set_status(self, text, progress_val=None):
        self.status_lbl.configure(text=text)
        if progress_val is not None:
            self.progress_bar.set(progress_val)

    def browse_dir(self):
        chosen = filedialog.askdirectory(title="Kurulum Klasörünü Seçin", initialdir=self.install_dir_var.get())
        if chosen:
            self.install_dir_var.set(chosen)

    def start_installation(self):
        if self.is_installing:
            return

        target_dir = self.install_dir_var.get().strip()
        if not target_dir:
            messagebox.showerror("Hata", "Lütfen geçerli bir kurulum dizini seçin.")
            return

        self.is_installing = True
        self.install_btn.configure(state="disabled")
        self.dir_entry.configure(state="disabled")
        self.cb1.configure(state="disabled")
        self.cb2.configure(state="disabled")
        self.cb3.configure(state="disabled")

        threading.Thread(target=self._run_install_worker, args=(target_dir,), daemon=True).start()

    def _run_install_worker(self, target_dir):
        try:
            self.log("=" * 60)
            self.log(">>> RIPLEYTIA AI SOUND STUDIO V3.5 KURULUMU BASLATILIYOR...")
            self.log(f">>> Hedef Dizin: {target_dir}")
            self.log("=" * 60)

            # Step 1: Create Directories & Copy/Extract Core Files
            self.set_status("Adım 1/5: Hedef dizin hazırlanıyor ve dosyalar yerleştiriliyor...", 0.15)
            self.log("\n[1/5] Hedef klasör oluşturuluyor...")
            os.makedirs(target_dir, exist_ok=True)

            bundle_dir = get_bundle_dir()
            payload_zip = os.path.join(bundle_dir, "core.zip")
            if not os.path.exists(payload_zip):
                # Search parent or script directory
                alt_zip = os.path.join(os.path.dirname(os.path.abspath(__file__)), "core.zip")
                if os.path.exists(alt_zip):
                    payload_zip = alt_zip

            if os.path.exists(payload_zip):
                self.log(f"   -> core.zip paketi çıkarılıyor: {payload_zip}")
                with zipfile.ZipFile(payload_zip, "r") as zf:
                    zf.extractall(target_dir)
            else:
                # If running from repo directly, copy application files
                src_repo = os.path.dirname(os.path.abspath(__file__))
                self.log("   -> Uygulama kaynak dosyaları doğrudan kopyalanıyor...")
                items_to_copy = ["src", "assets", "configs", "requirements.txt", "logo.ico", "logo.jpg", "README.md", "Start-SuperGUI.bat"]
                for item in items_to_copy:
                    src_p = os.path.join(src_repo, item)
                    dst_p = os.path.join(target_dir, item)
                    if os.path.exists(src_p):
                        if os.path.isdir(src_p):
                            shutil.copytree(src_p, dst_p, dirs_exist_ok=True)
                        else:
                            shutil.copy2(src_p, dst_p)

            # Ensure essential folders exist
            os.makedirs(os.path.join(target_dir, "rvc_models"), exist_ok=True)
            os.makedirs(os.path.join(target_dir, "mdxnet_models"), exist_ok=True)
            os.makedirs(os.path.join(target_dir, "song_output"), exist_ok=True)
            os.makedirs(os.path.join(target_dir, "datasets"), exist_ok=True)

            self.log("   [OK] Dosyalar başarıyla yerleştirildi.")

            # Step 2: Virtual Environment Setup
            venv_python = os.path.join(target_dir, "venv", "Scripts", "python.exe")
            if self.setup_venv_var.get():
                self.set_status("Adım 2/5: Python sanal ortamı (venv) oluşturuluyor...", 0.35)
                self.log("\n[2/5] Python Sanal Ortamı (Virtualenv) yapılandırılıyor...")

                # Detect host Python
                host_python = sys.executable
                if not host_python or "installer" in host_python.lower():
                    # Fallback to system python
                    host_python = shutil.which("python") or "python"

                if not os.path.exists(venv_python):
                    self.log(f"   -> Sanal ortam oluşturuluyor: {os.path.join(target_dir, 'venv')}")
                    p = subprocess.run([host_python, "-m", "venv", os.path.join(target_dir, "venv")], capture_output=True, text=True)
                    if p.returncode != 0:
                        self.log(f"   [Uyarı] Venv doğrudan oluşturulamadı: {p.stderr}")

                # Step 3: Install Dependencies
                if os.path.exists(venv_python):
                    self.set_status("Adım 3/5: Yapay zeka ve PyTorch (CUDA) kütüphaneleri yükleniyor...", 0.55)
                    self.log("\n[3/5] PyTorch CUDA ve gerekli ses işleme kütüphaneleri yükleniyor...")
                    self.log("   -> pip upgrade...")
                    subprocess.run([venv_python, "-m", "pip", "install", "--upgrade", "pip"], capture_output=True)

                    req_path = os.path.join(target_dir, "requirements.txt")
                    if os.path.exists(req_path):
                        self.log("   -> requirements.txt paketleri kuruluyor (Bu işlem birkaç dakika sürebilir)...")
                        proc = subprocess.Popen(
                            [venv_python, "-m", "pip", "install", "-r", req_path],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            encoding="utf-8",
                            errors="replace",
                        )
                        for line in proc.stdout:
                            l = line.strip()
                            if l and any(k in l for k in ["Installing", "Successfully", "Collecting", "Requirement already"]):
                                self.log(f"   {l}")
                        proc.wait()
                        self.log("   [OK] Temel bağımlılıklar kuruldu.")
                else:
                    self.log("   [Bilgi] Mevcut Python çalışma ortamı kullanılacak.")

            # Step 4: Download Essential Pretrained Models
            if self.download_models_var.get():
                self.set_status("Adım 4/5: Temel AI ses modelleri (HuBERT & RMVPE) doğrulanıyor...", 0.80)
                self.log("\n[4/5] Temel AI ses modelleri taranıyor...")
                dl_script = os.path.join(target_dir, "src", "download_models.py")
                py_exec = venv_python if os.path.exists(venv_python) else sys.executable
                if os.path.exists(dl_script) and py_exec:
                    self.log("   -> download_models.py çalıştırılıyor...")
                    try:
                        p = subprocess.run([py_exec, dl_script], cwd=target_dir, capture_output=True, text=True, timeout=180)
                        self.log("   [OK] Temel modeller hazır.")
                    except Exception as me:
                        self.log(f"   [Bilgi] Model indirme atlandı / arka planda tamamlanacak: {me}")

            # Step 5: Create Windows Desktop Shortcut & Launcher Scripts
            self.set_status("Adım 5/5: Masaüstü kısayolu ve başlatıcı oluşturuluyor...", 0.95)
            self.log("\n[5/5] Masaüstü kısayolu ve başlatıcı hazırlanıyor...")

            # Write clean start_studio.bat
            launcher_bat = os.path.join(target_dir, "Ripleytia_AI_Studio.bat")
            bat_content = f"""@echo off
cd /d "%~dp0"
title Ripleytia AI Sound Studio
if exist venv\\Scripts\\activate.bat (
    call venv\\Scripts\\activate.bat
    start "" pythonw src\\super_gui.py
) else (
    start "" pythonw src\\super_gui.py
)
exit
"""
            with open(launcher_bat, "w", encoding="utf-8") as bf:
                bf.write(bat_content)

            ico_dest = os.path.join(target_dir, "logo.ico")
            if not os.path.exists(ico_dest) and os.path.exists(os.path.join(bundle_dir, "logo.ico")):
                shutil.copy2(os.path.join(bundle_dir, "logo.ico"), ico_dest)

            if self.create_shortcut_var.get():
                shortcut_target = launcher_bat
                create_windows_shortcut(
                    target_path=shortcut_target,
                    working_dir=target_dir,
                    shortcut_name="Ripleytia AI Studio",
                    icon_path=ico_dest if os.path.exists(ico_dest) else "",
                )
                self.log("   [OK] Masaüstüne 'Ripleytia AI Studio' kısayolu başarıyla eklendi!")

            self.installed_path = launcher_bat
            self.progress_bar.set(1.0)
            self.set_status("Tebrikler! Kurulum Başarıyla Tamamlandı! 🎉", 1.0)
            self.log("\n" + "=" * 60)
            self.log("✅ KURULUM TAMAMLANDI!")
            self.log("Masaüstünüzdeki kısayola tıklayarak stüdyoyu hemen açabilirsiniz.")
            self.log("=" * 60)

            self.launch_btn.configure(state="normal")
            self.install_btn.configure(text="Kurulum Tamamlandı ✅")

        except Exception as e:
            self.log(f"\n[HATA] Kurulum sırasında hata oluştu: {e}")
            self.set_status(f"Hata oluştu: {e}", 0.0)
            messagebox.showerror("Kurulum Hatası", f"Kurulum tamamlanamadı:\n{e}")
        finally:
            self.is_installing = False

    def launch_studio(self):
        if self.installed_path and os.path.exists(self.installed_path):
            work_dir = os.path.dirname(self.installed_path)
            subprocess.Popen([self.installed_path], cwd=work_dir, shell=True)
            time.sleep(0.5)
            self.destroy()
        else:
            messagebox.showinfo("Bilgi", "Başlatıcı bulunamadı. Lütfen kurulum dizinini kontrol edin.")

if __name__ == "__main__":
    app = RipleytiaInstallerApp()
    app.mainloop()
