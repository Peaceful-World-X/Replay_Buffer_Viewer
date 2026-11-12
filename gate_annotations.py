"""Human actor-gate boundaries and full-episode training sidecars."""

from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import os
import shutil
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import av
import numpy as np

import lerobot_source
from review import ReviewConflict

STORE = "actor_gate_annotations.json"
DIRECTORY = "actor_gate_training"
FORMAT = "replay_buffer_viewer.actor_gate.v1"
_lock = threading.RLock()


def write_json(path: Path, value: dict) -> None:
    encoded = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=".gate-", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_store(root: Path) -> dict:
    path = root / STORE
    if not path.exists():
        return {"format": FORMAT, "revision": 0, "episodes": {}}
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("format") != FORMAT or not isinstance(value.get("episodes"), dict):
        raise ValueError("无法识别 actor_gate_annotations.json，未覆盖此文件。")
    return value


def labels_for(length: int, start: int, end: int) -> np.ndarray:
    """The start observation is ON; the confirmed-passed end observation is OFF."""
    if type(start) is not int or type(end) is not int or not 0 <= start < end < length:
        raise ValueError("门控要求 0 ≤ 开启帧 < 关闭帧 < 总帧数；关闭帧本身标为0。")
    labels = np.zeros(length, np.uint8)
    labels[start:end] = 1
    return labels


def state_for(root: Path, episode: dict, fingerprint: dict) -> dict:
    with _lock:
        store = read_store(root)
    entry = copy.deepcopy(store["episodes"].get(str(episode["episode_index"])))
    if entry is not None:
        entry["stale"] = entry["fingerprint"] != fingerprint
    marker = root / DIRECTORY / "gate_export.json"
    exported = json.loads(marker.read_text()) if marker.is_file() else None
    if exported and exported.get("format") != FORMAT:
        exported = None
    return {
        "entry": entry,
        "revision": store["revision"],
        "reviewedEpisodes": sorted(map(int, store["episodes"])),
        "annotationPath": str(root / STORE),
        "exportDirectory": str(root / DIRECTORY),
        "exportCurrent": bool(exported and exported["revision"] == store["revision"]),
        "lastExport": exported,
    }


def update(
    root: Path, episode: dict, fingerprint: dict, body: dict, remove=False
) -> None:
    with _lock:
        store = read_store(root)
        if (
            type(body.get("revision")) is not int
            or body["revision"] != store["revision"]
        ):
            raise ReviewConflict("门控标注已被其他页面修改，请重新载入。")
        identifier = str(episode["episode_index"])
        previous = store["episodes"].get(identifier)
        if previous and previous["fingerprint"] != fingerprint:
            raise ReviewConflict("原始数据已变化，请先核对旧门控标注。")
        if remove:
            store["episodes"].pop(identifier, None)
        else:
            mode = body.get("mode", "interval")
            if mode not in {"interval", "all_off"}:
                raise ValueError("未知门控标注类型。")
            start, end = body.get("startFrame"), body.get("endFrame")
            if mode == "interval":
                labels_for(episode["length"], start, end)
            else:
                start = end = None
            note = body.get("note", "")
            if not isinstance(note, str) or len(note) > 1000:
                raise ValueError("备注最多1000字。")
            store["episodes"][identifier] = {
                "episode_index": episode["episode_index"],
                "length": episode["length"],
                "mode": mode,
                "start_frame": start,
                "end_frame": end,
                "note": note.strip(),
                "fingerprint": fingerprint,
                "annotator": "human",
                "updated_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017 - Python 3.10 support
            }
        store["revision"] += 1
        write_json(root / STORE, store)


def image_at(media: dict, frame: int) -> bytes:
    """Decode an exact episode-local frame using its original video offset."""
    wanted = int(media["indices"][frame])
    with av.open(str(media["path"])) as container:
        stream = container.streams.video[0]
        origin = stream.start_time or 0
        target = origin + int(wanted / media["fps"] / stream.time_base)
        container.seek(target, stream=stream, backward=True)
        for decoded in container.decode(stream):
            number = round(
                float((decoded.pts - origin) * stream.time_base) * media["fps"]
            )
            if number < wanted:
                continue
            if number != wanted:
                raise ValueError(f"视频缺少原始帧 {wanted}")
            output = io.BytesIO()
            decoded.to_image().save(output, format="JPEG", quality=93)
            return output.getvalue()
    raise ValueError(f"视频未找到原始帧 {wanted}")


def check_target(root: Path) -> Path:
    output = root / DIRECTORY
    if output.is_symlink():
        raise ValueError("门控导出目录为符号链接，不能覆盖。")
    if output.exists():
        marker = output / "gate_export.json"
        value = json.loads(marker.read_text()) if marker.is_file() else {}
        if value.get("format") != FORMAT or value.get("source_dataset") != str(root):
            raise ValueError(f"{output} 不属于此数据集的门控导出，不能覆盖。")
    return output


def export(root: Path, progress=lambda text: None) -> dict:
    """Publish all explicitly reviewed full episodes; unknown episodes are excluded."""
    root = root.resolve()
    check_target(root)
    with _lock:
        store = read_store(root)
    info, episodes = lerobot_source.dataset(root)
    by_id = {ep["episode_index"]: ep for ep in episodes}
    staging = Path(tempfile.mkdtemp(prefix=".actor-gate-export-", dir=root))
    try:
        for name in ("labels", "data", "images"):
            (staging / name).mkdir()
        manifest = {
            "format": FORMAT,
            "revision": store["revision"],
            "source_dataset": str(root),
            "fps": info["fps"],
            "interval_convention": "[start_frame,end_frame)",
            "label_definition": "1 from approach/actor ON until confirmed passage; OFF boundary is 0",
            "unreviewed_policy": "excluded, never assumed negative",
            "episodes": [],
            "videos_copied": False,
        }
        total = positive = 0
        with (staging / "frame_labels.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "episode_index",
                    "frame_index",
                    "source_index",
                    "timestamp",
                    "actor_gate",
                ]
            )
            for identifier, entry in sorted(
                store["episodes"].items(), key=lambda item: int(item[0])
            ):
                ep = by_id.get(int(identifier))
                if (
                    ep is None
                    or lerobot_source.source_fingerprint(root, info, ep)
                    != entry["fingerprint"]
                ):
                    raise ReviewConflict(
                        f"Episode {identifier} 源数据已改变，旧导出保留。"
                    )
                progress(f"导出完整 Episode {identifier}（{ep['length']} 帧）…")
                table = lerobot_source.table_for(root, info, ep)
                labels = (
                    labels_for(len(table), entry["start_frame"], entry["end_frame"])
                    if entry["mode"] == "interval"
                    else np.zeros(len(table), np.uint8)
                )
                frames = lerobot_source.column_array(table, "frame_index").reshape(-1)
                indices = lerobot_source.column_array(table, "index").reshape(-1)
                stamps = lerobot_source.column_array(table, "timestamp").reshape(-1)
                states = lerobot_source.column_array(table, "observation.state").astype(
                    np.float32
                )
                media = lerobot_source.cameras(root, info, ep)
                labels_file = f"labels/episode_{int(identifier):06d}.npz"
                camera_order = list(media)
                video_indices = (
                    np.column_stack([media[k]["indices"].numpy() for k in camera_order])
                    if media
                    else np.empty((len(table), 0), np.int64)
                )
                np.savez_compressed(
                    staging / labels_file,
                    labels=labels,
                    state=states,
                    frame_index=frames,
                    source_index=indices,
                    timestamp=stamps,
                    video_frame_indices=video_indices,
                )
                pa, pq = lerobot_source._arrow()
                if "annotation.actor_gate" in table.column_names:
                    raise ValueError("源数据已有 annotation.actor_gate，拒绝覆盖。")
                data_file = f"data/episode_{int(identifier):06d}.parquet"
                pq.write_table(
                    table.append_column("annotation.actor_gate", pa.array(labels)),
                    staging / data_file,
                )
                writer.writerows(
                    (int(identifier), int(f), int(i), float(t), int(y))
                    for f, i, t, y in zip(frames, indices, stamps, labels, strict=True)
                )
                pictures = {}
                if entry["mode"] == "interval":
                    for boundary in ("start", "end"):
                        pictures[boundary] = {}
                        for number, (camera, item) in enumerate(media.items()):
                            filename = f"images/episode_{int(identifier):06d}_{boundary}_{number}.jpg"
                            (staging / filename).write_bytes(
                                image_at(item, entry[f"{boundary}_frame"])
                            )
                            pictures[boundary][camera] = filename
                manifest["episodes"].append(
                    {
                        "episode_index": int(identifier),
                        "parent_episode_index": int(identifier),
                        "length": len(table),
                        "positive_frames": int(labels.sum()),
                        "start_frame": entry["start_frame"],
                        "end_frame": entry["end_frame"],
                        "mode": entry["mode"],
                        "note": entry["note"],
                        "labels_file": labels_file,
                        "labels_sha256": hashlib.sha256(
                            (staging / labels_file).read_bytes()
                        ).hexdigest(),
                        "data_file": data_file,
                        "boundary_images": pictures,
                        "camera_order": camera_order,
                        "media": {
                            k: {
                                "path": str(v["path"].resolve()),
                                "fps": v["fps"],
                                "first_video_frame": int(v["indices"][0]),
                            }
                            for k, v in media.items()
                        },
                        "source_fingerprint": entry["fingerprint"],
                    }
                )
                total += len(table)
                positive += int(labels.sum())
        manifest.update(
            total_frames=total,
            positive_frames=positive,
            negative_frames=total - positive,
        )
        write_json(staging / "labels_manifest.json", manifest)
        write_json(staging / STORE, store)
        result = {
            "format": FORMAT,
            "source_dataset": str(root),
            "revision": store["revision"],
            "directory": str(root / DIRECTORY),
            "episodes": len(manifest["episodes"]),
            "frames": total,
            "positive_frames": positive,
            "negative_frames": total - positive,
            "generation": uuid.uuid4().hex,
        }
        write_json(staging / "gate_export.json", result)
        with _lock:
            if read_store(root) != store:
                raise ReviewConflict("导出期间标注发生变化，请重新导出。")
            current_info, current_episodes = lerobot_source.dataset(root)
            current = {e["episode_index"]: e for e in current_episodes}
            for item in manifest["episodes"]:
                ep = current.get(item["episode_index"])
                if (
                    ep is None
                    or lerobot_source.source_fingerprint(root, current_info, ep)
                    != item["source_fingerprint"]
                ):
                    raise ReviewConflict("导出期间原始数据发生变化，请重新载入。")
            output = check_target(root)
            backup = root / f".actor-gate-backup-{result['generation']}"
            if output.exists():
                output.rename(backup)
            try:
                staging.rename(output)
            except Exception:
                if backup.exists():
                    backup.rename(output)
                raise
            if backup.exists():
                shutil.rmtree(backup)
        return result
    finally:
        if staging.exists():
            shutil.rmtree(staging)
