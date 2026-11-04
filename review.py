"""Persistent manual selections for original dataset episodes."""

from __future__ import annotations

import copy
import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path

REVIEW_FILE = "review_annotations.json"
_lock = threading.RLock()


class ReviewConflict(ValueError):
    pass


def fingerprint(path: Path) -> dict:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def read_store(directory: Path) -> dict:
    path = directory / REVIEW_FILE
    if not path.exists():
        return {
            "schema_version": 1,
            "interval_convention": "[start_frame,end_frame)",
            "trajectories": {},
        }
    with path.open(encoding="utf-8") as handle:
        store = json.load(handle)
    if store.get("schema_version") != 1 or not isinstance(
        store.get("trajectories"), dict
    ):
        raise ValueError("不支持的标注文件格式，请检查 review_annotations.json")
    return store


def entry_for(
    directory: Path, name: str, source_fingerprint: dict, store: dict | None = None
) -> dict:
    store = read_store(directory) if store is None else store
    entry = copy.deepcopy(
        store["trajectories"].get(
            name,
            {
                "revision": 0,
                "reviewed": False,
                "segments": [],
                "fingerprint": source_fingerprint,
            },
        )
    )
    entry["stale"] = entry["fingerprint"] != source_fingerprint
    return entry


def update_entry(
    directory: Path, name: str, source_fingerprint: dict, revision: int, mutate
) -> dict:
    with _lock:
        store = read_store(directory)
        entry = entry_for(directory, name, source_fingerprint, store)
        if entry["stale"]:
            raise ReviewConflict(
                "源 episode 已改变，旧标注不能应用到新数据。请核对源文件。"
            )
        if type(revision) is not int or entry["revision"] != revision:
            raise ReviewConflict("标注已被其他页面更新，请重新载入后操作。")
        mutate(entry)
        entry.pop("stale", None)
        entry["revision"] += 1
        entry["updated_at"] = datetime.now(timezone.utc).isoformat()
        store["trajectories"][name] = entry
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=directory,
                prefix=".review-",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                json.dump(store, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(directory / REVIEW_FILE)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return {**entry, "stale": False}


def add_segment(entry: dict, segment: dict, replace_id: str | None = None) -> None:
    if replace_id is not None and not any(
        item["id"] == replace_id for item in entry["segments"]
    ):
        raise ValueError("待修改片段不存在，请重新载入。")
    for previous in entry["segments"]:
        if previous["id"] == replace_id:
            continue
        if any(
            a < d and c < b
            for a, b in segment["source_ranges"]
            for c, d in previous["source_ranges"]
        ):
            raise ValueError("选区与已保留片段重叠，请调整起止帧或先删除旧标注。")
    if replace_id is not None:
        entry["segments"] = [
            item for item in entry["segments"] if item["id"] != replace_id
        ]
    entry["segments"].append(segment)
    entry["segments"].sort(key=lambda item: item["start_frame"])
    entry["reviewed"] = True
