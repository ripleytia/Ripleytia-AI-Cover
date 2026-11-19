"""
Cover History Manager for Ripleytia AI Studio.
Maintains persistent JSON log of all completed covers with quick playback and parameter reload support.
"""

import os
import json
import time
from typing import List, Dict, Optional


HISTORY_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "song_output",
    "cover_history.json"
)


def _load_history_raw() -> List[Dict]:
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            return []
    except Exception as e:
        print(f"[Cover History Hatası] Okunamadı: {e}")
        return []


def _save_history_raw(items: List[Dict]):
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[Cover History Hatası] Yazılamadı: {e}")


def add_history_entry(
    song_name: str,
    model_name: str,
    pitch: int,
    algo: str,
    uvr_model: str,
    output_path: str,
    vocals_path: str = "",
    instrumentals_path: str = "",
    duration_str: str = ""
) -> Dict:
    """
    Appends a new cover entry to the top of the history list.
    """
    items = _load_history_raw()
    entry = {
        "id": f"cov_{int(time.time() * 1000)}",
        "song_name": song_name,
        "model_name": model_name,
        "pitch": pitch,
        "algo": algo,
        "uvr_model": uvr_model,
        "output_path": output_path,
        "vocals_path": vocals_path,
        "instrumentals_path": instrumentals_path,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_str": duration_str
    }
    # Keep up to 200 items, most recent first
    items.insert(0, entry)
    items = items[:200]
    _save_history_raw(items)
    return entry


def get_history_entries(limit: int = 50) -> List[Dict]:
    """
    Returns latest history entries, verifying file existence.
    """
    items = _load_history_raw()
    valid_items = []
    for item in items:
        # Check if output file still exists
        item["file_exists"] = os.path.exists(item.get("output_path", ""))
        valid_items.append(item)
    return valid_items[:limit]


def delete_history_entry(entry_id: str) -> bool:
    items = _load_history_raw()
    new_items = [it for it in items if it.get("id") != entry_id]
    if len(new_items) != len(items):
        _save_history_raw(new_items)
        return True
    return False


def clear_history():
    _save_history_raw([])
