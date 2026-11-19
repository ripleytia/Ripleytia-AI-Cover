"""
Automatic RVC Model Downloader for Ripleytia AI Studio.
Allows searching and installing community voice models from Hugging Face and direct URLs.
"""

import os
import sys
import json
import zipfile
import urllib.request
import threading
from typing import List, Dict, Callable, Optional


RVC_MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "rvc_models"
)

# Curated catalog of community models
CURATED_MODELS: List[Dict] = [
    {
        "id": "turkce_populer_nesil",
        "name": "Nesil (Stüdyo Pop & R&B)",
        "creator": "Ripleytia AI",
        "gender": "Kadın",
        "sample_url": "",
        "download_url": "https://huggingface.co/QuickW/VoiceModels/resolve/main/Nesil_RVC.zip",
        "description": "Temiz stüdyo vokali, yüksek dinamik aralık."
    },
    {
        "id": "pqueen_studio",
        "name": "Pqueen (Ultra HD Konuşma & Şarkı)",
        "creator": "Ripleytia AI",
        "gender": "Kadın",
        "download_url": "",
        "description": "Geniş vokal aralığı, enerjik ve tiz tonlar."
    },
    {
        "id": "baris_manco",
        "name": "Barış Manço (Efsanevi Rock & Bariton)",
        "creator": "Türkçe RVC Topluluğu",
        "gender": "Erkek",
        "download_url": "https://huggingface.co/datasets/models/resolve/main/BarisManco.zip",
        "description": "Derin bariton, tok ve nostaljik Anadolu Rock tonu."
    },
    {
        "id": "sagopa_kajmer",
        "name": "Sagopa Kajmer (Rap & Melodram)",
        "creator": "Türkçe RVC Topluluğu",
        "gender": "Erkek",
        "download_url": "https://huggingface.co/datasets/models/resolve/main/SagopaKajmer.zip",
        "description": "Karakteristik pes tonlar ve hızlı heceleme yeteneği."
    },
    {
        "id": "sebnem_ferah",
        "name": "Şebnem Ferah (Rock & Yüksek Tiz)",
        "creator": "Türkçe RVC Topluluğu",
        "gender": "Kadın",
        "download_url": "https://huggingface.co/datasets/models/resolve/main/SebnemFerah.zip",
        "description": "Güçlü rock kafa sesi ve vibrato."
    },
    {
        "id": "miku_v4",
        "name": "Hatsune Miku (Cyber Pop / J-Pop)",
        "creator": "Community",
        "gender": "Kadın",
        "download_url": "https://huggingface.co/datasets/models/resolve/main/HatsuneMiku.zip",
        "description": "J-Pop ve yüksek oktav elektronik vokal tınısı."
    }
]


def search_curated_models(query: str = "") -> List[Dict]:
    """Returns curated models matching query."""
    q = query.strip().lower()
    if not q:
        return CURATED_MODELS
    results = []
    for m in CURATED_MODELS:
        if (q in m["name"].lower() or 
            q in m["description"].lower() or 
            q in m["gender"].lower()):
            results.append(m)
    return results


def download_and_extract_model(
    url: str,
    custom_model_name: str = "",
    progress_cb: Optional[Callable[[str, float], None]] = None
) -> str:
    """
    Downloads a zip or direct pth/index from URL, extracts into rvc_models/<Model_Name>/
    Returns the installed model folder name.
    """
    if not url or not url.startswith("http"):
        raise ValueError("Geçerli bir indirme bağlantısı (URL) giriniz.")

    # Deduce model name from URL if not given
    if not custom_model_name:
        base_file = os.path.basename(url.split("?")[0])
        model_name = os.path.splitext(base_file)[0]
    else:
        model_name = custom_model_name

    model_name = "".join(c for c in model_name if c.isalnum() or c in ("_", "-", " ")).strip()
    if not model_name:
        model_name = "Indirilen_Model"

    target_dir = os.path.join(RVC_MODELS_DIR, model_name)
    os.makedirs(target_dir, exist_ok=True)
    temp_zip = os.path.join(target_dir, "model_archive.download")

    if progress_cb:
        progress_cb(f"Model indiriliyor: {model_name}...", 0.1)

    # Download with progress
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            total_size = resp.getheader("Content-Length")
            total_size = int(total_size) if total_size else 0
            downloaded = 0
            chunk_size = 1024 * 1024  # 1MB

            with open(temp_zip, "wb") as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0 and progress_cb:
                        pct = 0.1 + (downloaded / total_size) * 0.75
                        mb_done = downloaded / (1024 * 1024)
                        mb_tot = total_size / (1024 * 1024)
                        progress_cb(f"İndiriliyor: {mb_done:.1f}MB / {mb_tot:.1f}MB (%{int(pct*100)})", pct)

        if progress_cb:
            progress_cb("Dosyalar ayıklanıyor...", 0.9)

        # Check if downloaded file is a ZIP archive
        if zipfile.is_zipfile(temp_zip):
            with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
                # Extract all files directly or flatten
                for file_info in zip_ref.infolist():
                    filename = os.path.basename(file_info.filename)
                    if not filename:
                        continue
                    if filename.lower().endswith((".pth", ".index", ".txt", ".json")):
                        source = zip_ref.open(file_info)
                        target_file = open(os.path.join(target_dir, filename), "wb")
                        with source, target_file:
                            target_file.write(source.read())
            try:
                os.remove(temp_zip)
            except Exception:
                pass
        else:
            # If direct .pth was downloaded
            if url.lower().endswith(".pth"):
                final_pth = os.path.join(target_dir, f"{model_name}.pth")
                if os.path.exists(final_pth):
                    os.remove(final_pth)
                os.rename(temp_zip, final_pth)
            else:
                # Keep as is
                pass

        if progress_cb:
            progress_cb("Model başarıyla yüklendi! ✅", 1.0)

        return model_name

    except Exception as e:
        if os.path.exists(temp_zip):
            try:
                os.remove(temp_zip)
            except Exception:
                pass
        raise RuntimeError(f"Model indirme hatası: {e}")
