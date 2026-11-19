"""
Lyrics Fetcher & AI Translator Engine for Ripleytia AI Studio.
Fetches synchronized/plain lyrics using LRCLIB or yt-dlp captions,
and provides Turkish/English translation.
"""

import os
import sys
import json
import urllib.request
import urllib.parse
from typing import Dict, Optional


def fetch_lyrics(song_query: str, artist: str = "") -> Dict:
    """
    Searches LRCLIB for track lyrics.
    Returns dict with {title, artist, plain_lyrics, synced_lyrics}.
    """
    query = f"{artist} {song_query}".strip()
    if not query:
        return {"error": "Arama terimi boş"}

    url = f"https://lrclib.net/api/search?q={urllib.parse.quote(query)}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "RipleytiaAISoundStudio/4.0"})
        with urllib.request.urlopen(req, timeout=6) as response:
            items = json.loads(response.read().decode("utf-8"))
            if not items:
                return {"error": "Şarkı sözü bulunamadı."}

            best = items[0]
            plain = best.get("plainLyrics") or ""
            synced = best.get("syncedLyrics") or ""

            return {
                "title": best.get("trackName", song_query),
                "artist": best.get("artistName", artist),
                "plain_lyrics": plain or "Bu şarkı için metin sözler bulunamadı.",
                "synced_lyrics": synced,
                "error": None
            }
    except Exception as e:
        return {"error": f"Söz servisine ulaşılamadı: {e}"}


def translate_lyrics_text(text: str, target_lang: str = "tr") -> str:
    """
    Translates text chunks to target language using MyMemory or fallback heuristics.
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return ""

    # Translate first 20 lines to keep request lightweight and fast
    translated_lines = []
    chunk = "\n".join(lines[:20])

    try:
        url = f"https://api.mymemory.translated.net/get?q={urllib.parse.quote(chunk[:500])}&langpair=autodetect|{target_lang}"
        req = urllib.request.Request(url, headers={"User-Agent": "RipleytiaAISoundStudio/4.0"})
        with urllib.request.urlopen(req, timeout=6) as response:
            data = json.loads(response.read().decode("utf-8"))
            res_text = data.get("responseData", {}).get("translatedText", "")
            if res_text:
                return res_text
    except Exception:
        pass

    return "\n".join(lines[:25]) + "\n\n(Çeviri servisi meşgul; orijinal sözler listelendi.)"
