import os
import sys
import json
import subprocess
import time
import urllib.request
from urllib.parse import urlparse, parse_qs
from PIL import Image

# Ensure dataset_builder is accessible for modern yt-dlp locator
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from dataset_builder import get_modern_ytdlp_cmd
except ImportError:
    def get_modern_ytdlp_cmd():
        return ["yt-dlp"]


THUMB_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "song_output",
    "yt_thumbnails"
)
os.makedirs(THUMB_CACHE_DIR, exist_ok=True)


def format_views(view_count):
    if not view_count or not isinstance(view_count, (int, float)):
        return ""
    try:
        vc = int(view_count)
        if vc >= 1_000_000_000:
            return f"{vc / 1_000_000_000:.1f}B"
        if vc >= 1_000_000:
            return f"{vc / 1_000_000:.1f}M"
        if vc >= 1_000:
            return f"{vc / 1_000:.1f}K"
        return str(vc)
    except Exception:
        return ""


def get_cached_thumbnail(video_id: str, thumb_url: str = None, target_size=(96, 54)) -> str:
    """
    Downloads, resizes and caches YouTube thumbnail on disk. Returns local image file path.
    """
    if not video_id:
        return ""
    out_file = os.path.join(THUMB_CACHE_DIR, f"{video_id}.jpg")
    if os.path.exists(out_file) and os.path.getsize(out_file) > 100:
        return out_file

    if not thumb_url:
        thumb_url = f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"

    try:
        req = urllib.request.Request(
            thumb_url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )
        with urllib.request.urlopen(req, timeout=4) as response:
            with open(out_file, "wb") as f:
                f.write(response.read())

        # Resize to compact card thumbnail
        try:
            with Image.open(out_file) as im:
                im = im.convert("RGB")
                im.thumbnail(target_size, Image.Resampling.LANCZOS)
                im.save(out_file, "JPEG", quality=85)
        except Exception:
            pass

        return out_file
    except Exception:
        return ""


def is_playlist_url(url: str) -> bool:
    """Checks if a URL is a YouTube playlist."""
    parsed = urlparse(url)
    if "youtube.com" in parsed.netloc or "youtu.be" in parsed.netloc:
        query_params = parse_qs(parsed.query)
        if "list" in query_params and query_params["list"]:
            return True
    return False


def get_playlist_videos(playlist_url: str, max_items: int = 25) -> list:
    """
    Extracts video list from a YouTube playlist using modern yt-dlp.
    Returns list of video dicts.
    """
    ytdlp_base = get_modern_ytdlp_cmd()
    cmd = ytdlp_base + [
        playlist_url,
        "--flat-playlist",
        "--dump-json",
        "--playlist-end", str(max_items),
        "--no-warnings",
        "--no-check-certificate"
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30
        )
    except Exception as e:
        raise RuntimeError(f"Playlist çekilemedi: {e}")

    videos = []
    lines = (proc.stdout or "").strip().split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except Exception:
            continue

        vid_id = data.get("id") or ""
        if not vid_id:
            continue
        title = data.get("title") or "Bilinmeyen Başlık"
        channel = data.get("channel") or data.get("uploader") or "Bilinmeyen Kanal"
        dur_str = data.get("duration_string") or "--:--"
        url = data.get("url") or f"https://www.youtube.com/watch?v={vid_id}"
        if not url.startswith("http"):
            url = f"https://www.youtube.com/watch?v={vid_id}"

        thumb = data.get("thumbnail") or f"https://i.ytimg.com/vi/{vid_id}/hqdefault.jpg"

        videos.append({
            "id": vid_id,
            "title": title,
            "channel": channel,
            "duration_string": dur_str,
            "views": format_views(data.get("view_count")),
            "url": url,
            "thumbnail": thumb
        })
    return videos


def search_youtube(query: str, max_results: int = 6) -> list:
    """
    Searches YouTube or fetches a single video info using modern yt-dlp.
    Returns a list of dicts with thumbnail, title, channel, views, url, etc.
    """
    query = query.strip()
    if not query:
        return []

    ytdlp_base = get_modern_ytdlp_cmd()

    # Check if query is a playlist
    if is_playlist_url(query):
        return get_playlist_videos(query, max_items=max_results * 2)

    # Check if query is a direct YouTube link
    parsed = urlparse(query)
    is_direct_url = (parsed.scheme in ("http", "https")) and any(
        host in parsed.netloc for host in ("youtube.com", "youtu.be", "music.youtube.com")
    )

    if is_direct_url:
        cmd = ytdlp_base + [
            query,
            "--no-playlist",
            "--dump-json",
            "--no-warnings",
            "--no-check-certificate",
        ]
    else:
        cmd = ytdlp_base + [
            f"ytsearch{max_results}:{query}",
            "--flat-playlist",
            "--dump-json",
            "--no-warnings",
            "--no-check-certificate",
            "--default-search", "ytsearch",
        ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=25,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("YouTube araması zaman aşımına uğradı. Lütfen internet bağlantınızı kontrol edin.")
    except Exception as e:
        raise RuntimeError(f"YouTube arama hatası: {e}")

    results = []
    lines = (proc.stdout or "").strip().split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except Exception:
            continue

        vid_id = data.get("id") or ""
        if not vid_id:
            continue

        title = data.get("title") or "Bilinmeyen Başlık"
        channel = data.get("channel") or data.get("uploader") or "Bilinmeyen Sanatçı"
        duration_str = data.get("duration_string")
        if not duration_str and data.get("duration"):
            dur = int(data.get("duration"))
            duration_str = f"{dur // 60}:{dur % 60:02d}"
        if not duration_str:
            duration_str = "--:--"

        raw_views = data.get("view_count")
        views_formatted = format_views(raw_views)

        video_url = data.get("url") or f"https://www.youtube.com/watch?v={vid_id}"
        if not video_url.startswith("http"):
            video_url = f"https://www.youtube.com/watch?v={vid_id}"

        # Thumbnail URL
        thumb = data.get("thumbnail")
        if not thumb and data.get("thumbnails"):
            thumb = data.get("thumbnails")[-1].get("url")
        if not thumb:
            thumb = f"https://i.ytimg.com/vi/{vid_id}/hqdefault.jpg"

        results.append({
            "id": vid_id,
            "title": title,
            "channel": channel,
            "duration_string": duration_str,
            "views": views_formatted,
            "url": video_url,
            "thumbnail": thumb,
        })

    return results


def download_youtube_audio(video_url: str, download_dir: str, progress_cb=None) -> str:
    """
    Downloads high-quality audio (MP3 320kbps) from YouTube using modern yt-dlp.
    Returns the absolute filepath of the downloaded audio.
    """
    os.makedirs(download_dir, exist_ok=True)
    ytdlp_base = get_modern_ytdlp_cmd()

    out_template = os.path.join(download_dir, "%(title)s.%(ext)s")
    cmd = ytdlp_base + [
        "--no-playlist",
        "--no-check-certificate",
        "--no-warnings",
        "-f", "bestaudio/best",
        "-x",
        "--audio-format", "mp3",
        "--audio-quality", "0",
        "-o", out_template,
        "--print", "after_move:filepath",
        video_url,
    ]

    if progress_cb:
        progress_cb("İndirme başlatılıyor...")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    downloaded_path = None
    output_lines = []

    for line in iter(proc.stdout.readline, ""):
        line_clean = line.strip()
        if not line_clean:
            continue
        output_lines.append(line_clean)
        
        # Check if line is the printed filepath from --print after_move:filepath
        if os.path.exists(line_clean) and line_clean.lower().endswith((".mp3", ".wav", ".m4a")):
            downloaded_path = line_clean

        if progress_cb and any(k in line_clean for k in ["[download]", "[ExtractAudio]", "Destination:"]):
            if "[download]" in line_clean and "%" in line_clean:
                try:
                    pct = line_clean.split("%")[0].split()[-1]
                    progress_cb(f"İndiriliyor: %{pct}")
                except Exception:
                    progress_cb(line_clean)
            elif "[ExtractAudio]" in line_clean:
                progress_cb("Ses dönüştürülüyor (MP3 320k)...")

    proc.stdout.close()
    return_code = proc.wait()

    if return_code != 0 and not downloaded_path:
        err_msg = "\n".join(output_lines[-5:]) if output_lines else f"Çıkış kodu: {return_code}"
        raise RuntimeError(f"YouTube indirme hatası: {err_msg}")

    # Verify or find downloaded file
    if downloaded_path and os.path.exists(downloaded_path):
        return os.path.normpath(downloaded_path)

    # Fallback: find the latest modified mp3 in download_dir
    candidates = [
        os.path.join(download_dir, f)
        for f in os.listdir(download_dir)
        if f.lower().endswith((".mp3", ".wav", ".m4a"))
    ]
    if candidates:
        latest = max(candidates, key=os.path.getmtime)
        return os.path.normpath(latest)

    raise RuntimeError("İndirilen ses dosyası hedef klasörde bulunamadı.")
