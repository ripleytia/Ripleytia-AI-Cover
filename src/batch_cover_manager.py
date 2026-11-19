"""
Batch Cover Queue Manager for Ripleytia AI Studio.
Allows scheduling multiple songs / YouTube URLs for sequential unattended or overnight AI Cover processing.
"""

import os
import json
import time
import threading
from typing import List, Dict, Callable, Optional


class BatchQueueManager:
    """
    Manages a list of pending AI Cover jobs and runs them sequentially.
    """
    def __init__(self):
        self.queue: List[Dict] = []
        self.is_running = False
        self._stop_requested = False
        self.current_item_id = None
        self.worker_thread = None

    def add_item(
        self,
        song_input: str,
        song_name: str,
        model_name: str,
        pitch: int = 0,
        algo: str = "rmvpe",
        uvr_model: str = "UVR5 Special 2",
        reverb_wet: float = 0.2,
        index_rate: float = 0.75
    ) -> Dict:
        item = {
            "id": f"batch_{int(time.time() * 1000)}_{len(self.queue)}",
            "song_input": song_input,
            "song_name": song_name or os.path.basename(song_input),
            "model_name": model_name,
            "pitch": pitch,
            "algo": algo,
            "uvr_model": uvr_model,
            "reverb_wet": reverb_wet,
            "index_rate": index_rate,
            "status": "Bekliyor",  # Bekliyor, İşleniyor, Tamamlandı, Hata
            "progress_pct": 0,
            "output_path": "",
            "error_msg": ""
        }
        self.queue.append(item)
        return item

    def remove_item(self, item_id: str):
        if self.current_item_id == item_id and self.is_running:
            return False  # Cannot remove active item
        self.queue = [it for it in self.queue if it["id"] != item_id]
        return True

    def clear_queue(self):
        if not self.is_running:
            self.queue.clear()

    def get_items(self) -> List[Dict]:
        return list(self.queue)

    def start_processing(
        self,
        process_fn: Callable[[Dict, Callable[[str, float], None]], str],
        on_item_update: Optional[Callable[[Dict], None]] = None,
        on_queue_finished: Optional[Callable[[], None]] = None
    ):
        """
        Starts processing pending items in a background worker thread.
        `process_fn(item, progress_callback)` must do the actual song cover generation and return output_path.
        """
        if self.is_running:
            return

        self.is_running = True
        self._stop_requested = False

        def _worker():
            for item in self.queue:
                if self._stop_requested:
                    break
                if item["status"] == "Tamamlandı":
                    continue

                self.current_item_id = item["id"]
                item["status"] = "İşleniyor"
                item["progress_pct"] = 5
                if on_item_update:
                    on_item_update(item)

                def _prog(msg, pct):
                    item["progress_pct"] = int(pct * 100)
                    if on_item_update:
                        on_item_update(item)

                try:
                    out_path = process_fn(item, _prog)
                    item["status"] = "Tamamlandı"
                    item["progress_pct"] = 100
                    item["output_path"] = out_path
                except Exception as e:
                    item["status"] = "Hata"
                    item["error_msg"] = str(e)

                if on_item_update:
                    on_item_update(item)

            self.is_running = False
            self.current_item_id = None
            if on_queue_finished:
                on_queue_finished()

        self.worker_thread = threading.Thread(target=_worker, daemon=True)
        self.worker_thread.start()

    def stop_processing(self):
        self._stop_requested = True
        self.is_running = False
