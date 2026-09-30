import os
import subprocess
import logging
import platform

def create_desktop_shortcut(exe_path, icon_path):
    """
    Windows üzerinde kullanıcının masaüstüne otomatik kısayol oluşturur.
    Herhangi bir ekstra pip paketi gerektirmemesi için VBScript fallback kullanır.
    """
    desktop = os.path.join(os.environ['USERPROFILE'], 'Desktop')
    shortcut_path = os.path.join(desktop, "Ripleytia AI Cover.lnk")
    
    # Mutlak (Absolute) yolları al
    abs_exe = os.path.abspath(exe_path)
    abs_dir = os.path.abspath(os.path.dirname(exe_path))
    abs_icon = os.path.abspath(icon_path) if os.path.exists(icon_path) else abs_exe
    
    vbs_content = f"""
Set oWS = WScript.CreateObject("WScript.Shell")
sLinkFile = "{shortcut_path}"
Set oLink = oWS.CreateShortcut(sLinkFile)
oLink.TargetPath = "{abs_exe}"
oLink.WorkingDirectory = "{abs_dir}"
oLink.IconLocation = "{abs_icon}"
oLink.Save
"""
    vbs_path = os.path.abspath("create_shortcut.vbs")
    with open(vbs_path, "w", encoding="utf-8") as f:
        f.write(vbs_content)
        
    try:
        subprocess.run(["cscript", "//nologo", vbs_path], check=True)
        logging.info(f"[BAŞARILI] Masaüstü kısayolu oluşturuldu: {shortcut_path}")
    except Exception as e:
        logging.error(f"[HATA] Kısayol oluşturulamadı: {str(e)}")
    finally:
        if os.path.exists(vbs_path):
            os.remove(vbs_path)

def build_project():
    """
    Ripleytia AI Automated Cover - PyInstaller Universal Setup Generator
    """
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logging.info("=== Ripleytia AI Cover - PyInstaller Dağıtım Motoru Başlatılıyor ===")
    
    os.makedirs("assets", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    icon_path = os.path.join("assets", "ripleytia_icon.ico")
    
    cmd = [
        "pyinstaller",
        "--name=Ripleytia_AI_Cover",
        "--windowed",     
        "--noconsole", 
        "--clean",        
        "--onedir",       
        "--add-data=assets;assets",
        "--add-data=models;models",
        "--hidden-import=PyQt6",
        "--hidden-import=pydub",
        "--hidden-import=torch",
        "--hidden-import=torchaudio",
        "--hidden-import=demucs",
        "--hidden-import=rvc_python",
        "main.py"
    ]
    
    if os.path.exists(icon_path):
        cmd.append(f"--icon={icon_path}")
        
    logging.info(f"Çalıştırılan derleme komutu:\n{' '.join(cmd)}")
    result = subprocess.run(cmd, text=True)
    
    if result.returncode == 0:
        logging.info("=== DERLEME BAŞARILI! ===")
        exe_path = os.path.join("dist", "Ripleytia_AI_Cover", "Ripleytia_AI_Cover.exe")
        
        # Eğer EXE başarıyla üretildiyse kısayol oluştur
        if os.path.exists(exe_path):
            create_desktop_shortcut(exe_path, icon_path)
        else:
            logging.warning("EXE dosyası bulunamadı, kısayol atlanıyor.")
            
        logging.info("Çıktı dizini: dist/Ripleytia_AI_Cover/")
    else:
        logging.error("=== DERLEME HATASI OLUŞTU ===")

if __name__ == "__main__":
    build_project()
