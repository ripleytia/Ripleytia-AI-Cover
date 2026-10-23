import os
import sys
import shutil
import zipfile
import subprocess

BASE_DIR = r"C:\Users\Ripleytia\.gemini\antigravity\scratch\AICoverGen"
DIST_DIR = os.path.join(BASE_DIR, "release_dist")
CORE_ZIP = os.path.join(BASE_DIR, "core.zip")

print(">>> [1/4] core.zip olusturuluyor...")
if os.path.exists(CORE_ZIP):
    os.remove(CORE_ZIP)

# Items to include in core.zip
include_dirs = ["src", "assets", "configs"]
include_files = [
    "requirements.txt",
    "logo.ico",
    "logo.jpg",
    "README.md",
    "LICENSE",
    "Start-SuperGUI.bat",
]

with zipfile.ZipFile(CORE_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
    for d in include_dirs:
        dp = os.path.join(BASE_DIR, d)
        if os.path.exists(dp):
            for root, _, files in os.walk(dp):
                if "__pycache__" in root or ".git" in root:
                    continue
                for f in files:
                    fp = os.path.join(root, f)
                    arcname = os.path.relpath(fp, BASE_DIR)
                    zf.write(fp, arcname)

    for f in include_files:
        fp = os.path.join(BASE_DIR, f)
        if os.path.exists(fp):
            zf.write(fp, f)

    # Include placeholder files
    for ph in [
        ("mdxnet_models/model_data.json", os.path.join(BASE_DIR, "mdxnet_models", "model_data.json")),
        ("rvc_models/public_models.json", os.path.join(BASE_DIR, "rvc_models", "public_models.json")),
    ]:
        if os.path.exists(ph[1]):
            zf.write(ph[1], ph[0])

print(f"   [OK] core.zip hazir! Boyut: {os.path.getsize(CORE_ZIP) / (1024*1024):.2f} MB")

print(">>> [2/4] Setup.exe PyInstaller ile derleniyor...")
os.makedirs(DIST_DIR, exist_ok=True)

pyinstaller_exe = r"C:\Users\Ripleytia\AppData\Local\Programs\Python\Python314\Scripts\pyinstaller.exe"
icon_path = os.path.join(BASE_DIR, "logo.ico")

cmd = [
    pyinstaller_exe,
    "--noconfirm",
    "--clean",
    "--onefile",
    "--windowed",
    "--name", "Setup",
    "--icon", icon_path,
    "--add-data", f"{CORE_ZIP};.",
    "--add-data", f"{icon_path};.",
    "--distpath", DIST_DIR,
    "--workpath", os.path.join(BASE_DIR, "build_installer"),
    os.path.join(BASE_DIR, "installer_gui.py"),
]

subprocess.run(cmd, check=True)
setup_exe = os.path.join(DIST_DIR, "Setup.exe")
print(f"   [OK] Setup.exe basariyla derlendi! Boyut: {os.path.getsize(setup_exe) / (1024*1024):.2f} MB")

print(">>> [3/4] Release Zip paketi olusturuluyor...")
RELEASE_ZIP_NAME = "Ripleytia_AI_Sound_Studio_V3.5_Setup.zip"
RELEASE_ZIP_PATH = os.path.join(DIST_DIR, RELEASE_ZIP_NAME)
if os.path.exists(RELEASE_ZIP_PATH):
    os.remove(RELEASE_ZIP_PATH)

readme_txt = """================================================================================
  RIPLEYTIA AI SOUND STUDIO V3.5 - SETUP & KURULUM KILAVUZU
================================================================================

1. KURULUM:
   - "Setup.exe" dosyasini cift tiklayarak calistirin.
   - Kurulum yapmak istediginiz klasoru secin (Varsayilan: C:\\Ripleytia_AI_Studio).
   - "Kurulumu Baslat" butonuna tiklayin.
   - Kurulum sihirbazi gerekli ortami, Python bagimliliklarini ve modelleri
     otomatik olarak kuracaktir.

2. MASAUSTU KISAYOLU:
   - Kurulum bittiginde masaustunuze "Ripleytia AI Studio" logosuyla
     otomatik kisayol eklenecektir.
   - Bu kisayola cift tiklayarak uygulamayi tek tikla acabilirsiniz!

Yenilikler ve Destek:
https://github.com/ripleytia/Ripleytia-AI-Cover
================================================================================
"""

readme_path = os.path.join(DIST_DIR, "README_KURULUM.txt")
with open(readme_path, "w", encoding="utf-8") as f:
    f.write(readme_txt)

with zipfile.ZipFile(RELEASE_ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as rz:
    rz.write(setup_exe, "Setup.exe")
    rz.write(readme_path, "README_KURULUM.txt")
    rz.write(icon_path, "logo.ico")

print(f"   [OK] Release Zip olusturuldu: {RELEASE_ZIP_PATH}")
print(f"   Boyut: {os.path.getsize(RELEASE_ZIP_PATH) / (1024*1024):.2f} MB")
print(">>> [4/4] Tum paketleme islemleri basariyla tamamlandi!")
