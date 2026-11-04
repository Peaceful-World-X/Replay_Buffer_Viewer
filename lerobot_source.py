"""Read and crop local LeRobot v3 episodes with their original camera offsets."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import av
import numpy as np
import torch

import review
import training_export


def _arrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ValueError(
            "读取原始 LeRobot 数据需要 pyarrow，请安装：python -m pip install pyarrow"
        ) from exc
    return pa, pq


def dataset(root: Path) -> tuple[dict, list[dict]]:
    _, pq = _arrow()
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    if info.get("codebase_version") != "v3.0":
        raise ValueError("原始数据裁剪目前支持 LeRobot v3.0 数据集。")
    episodes = []
    for path in sorted((root / "meta/episodes").rglob("*.parquet")):
        columns = [
            key for key in pq.read_schema(path).names if not key.startswith("stats/")
        ]
        episodes.extend(pq.read_table(path, columns=columns).to_pylist())
    if not episodes:
        raise ValueError("数据集没有 episode 元数据。")
    return info, sorted(episodes, key=lambda item: item["episode_index"])


def data_path(root: Path, info: dict, episode: dict) -> Path:
    return root / info["data_path"].format(
        chunk_index=episode["data/chunk_index"], file_index=episode["data/file_index"]
    )


def cameras(root: Path, info: dict, episode: dict) -> dict:
    result = {}
    for key, spec in info["features"].items():
        if spec["dtype"] != "video":
            continue
        prefix = f"videos/{key}/"
        path = root / info["video_path"].format(
            video_key=key,
            chunk_index=episode[prefix + "chunk_index"],
            file_index=episode[prefix + "file_index"],
        )
        fps = float(spec.get("info", {}).get("video.fps", info["fps"]))
        first = round(float(episode[prefix + "from_timestamp"]) * fps)
        indices = (
            torch.round(
                torch.arange(episode["length"], dtype=torch.float64) / info["fps"] * fps
            ).long()
            + first
        )
        result[key] = {"path": path, "fps": fps, "indices": indices}
    return result


def source_fingerprint(root: Path, info: dict, episode: dict) -> dict:
    paths = [root / "meta/info.json", data_path(root, info, episode)]
    paths += [item["path"] for item in cameras(root, info, episode).values()]
    files = {
        str(path.relative_to(root)): review.fingerprint(path)
        if path.is_file()
        else None
        for path in paths
    }
    return {"files": files, "episode": episode}


def table_for(root: Path, info: dict, episode: dict):
    _, pq = _arrow()
    table = pq.read_table(
        data_path(root, info, episode),
        filters=[("episode_index", "=", episode["episode_index"])],
    )
    table = table.sort_by([("frame_index", "ascending")])
    if len(table) != episode["length"] or table["frame_index"].to_pylist() != list(
        range(episode["length"])
    ):
        raise ValueError("episode 的帧数或 frame_index 不连续，不能正确映射视频。")
    return table


def column_array(table, key: str) -> np.ndarray:
    return np.asarray(table[key].to_pylist())


def load_episode(root: Path, episode_id: int | None) -> tuple[dict, dict, dict]:
    info, episodes = dataset(root)
    episode = (
        next((row for row in episodes if row["episode_index"] == episode_id), None)
        if episode_id is not None
        else episodes[0]
    )
    if episode is None:
        raise ValueError("找不到指定 episode。")
    table = table_for(root, info, episode)
    n = len(table)
    numeric = {}
    for key in table.column_names:
        values = column_array(table, key)
        if values.dtype.kind in "biuf":
            spec = info["features"].get(key, {})
            dtype = spec.get("dtype")
            if dtype in ("float32", "float64", "int64", "int32", "bool"):
                values = values.astype(dtype)
            numeric[key] = torch.from_numpy(values.copy()).reshape(n, 1, -1)
    data = {
        "actions": numeric["action"],
        "curr_obs": {"states": numeric["observation.state"]},
        "forward_inputs": {
            "episode_index": numeric["episode_index"],
            "frame_index": numeric["frame_index"],
        },
        "source_fields": numeric,
    }
    if "complementary_info.is_intervention" in numeric:
        data["intervene_flags"] = numeric["complementary_info.is_intervention"].bool()
    all_media = cameras(root, info, episode)
    roles = {
        "main": "cam_left_high",
        "left": "cam_left_wrist",
        "right": "cam_right_wrist",
    }
    media, used = {}, set()
    for role, suffix in roles.items():
        key = next((key for key in all_media if key.rsplit(".", 1)[-1] == suffix), None)
        if key:
            media[role] = all_media[key]
            used.add(key)
    for key in all_media:
        if key not in used:
            role = next((role for role in roles if role not in media), None)
            if role:
                media[role] = all_media[key]
    missing = [
        str(item["path"]) for item in media.values() if not item["path"].is_file()
    ]
    media = {key: item for key, item in media.items() if item["path"].is_file()}
    fingerprint = source_fingerprint(root, info, episode)
    digest = hashlib.sha256(
        (str(root) + json.dumps(fingerprint, sort_keys=True)).encode()
    ).hexdigest()[:20]
    context = {
        "episode": episode,
        "fingerprint": fingerprint,
        "entryName": f"episode_{episode['episode_index']}",
        "key": digest,
        "fps": info["fps"],
        "collectorPolicyCodebook": info["features"]
        .get("complementary_info.collector_policy_id", {})
        .get("info", {})
        .get("codebook", {}),
        "mediaMessage": ("缺少视频：" + ", ".join(missing)) if missing else "",
    }
    return data, media, context


def episode_result(episode: dict) -> dict:
    value = episode.get("episode_success")
    success = {"success": True, "failure": False}.get(value)
    return {"isSuccess": success,
            "outcomeSource": "依据原始 episode 的成功/失败标注。" if success is not None
            else "原始 episode 未标注明确的成功/失败结果。"}


def browse(root: Path) -> dict:
    info, episodes = dataset(root)
    store = review.read_store(root)
    items = []
    for ep in episodes:
        entry = review.entry_for(
            root,
            f"episode_{ep['episode_index']}",
            source_fingerprint(root, info, ep),
            store,
        )
        items.append(
            {
                "id": ep["episode_index"],
                "episodeId": ep["episode_index"],
                "path": str(root),
                "filename": f"Episode {ep['episode_index']}",
                "numSamples": ep["length"],
                "fileBytes": data_path(root, info, ep).stat().st_size,
                "modelWeightsId": "LeRobot v3 · 完整原始数据",
                "hasIntervention": bool(ep.get("human_intervention_intervals")),
                **episode_result(ep),
                "reviewCount": len(entry["segments"]),
                "reviewed": entry["reviewed"],
                "reviewStale": entry["stale"],
            }
        )
    return {
        "directory": str(root),
        "sourceType": "lerobot",
        "metadata": {"total_samples": info["total_frames"]},
        "trajectories": items,
    }


def _stats(values: np.ndarray, image=False) -> dict:
    array = values.reshape(-1, 3) if image else values.reshape(len(values), -1)
    shape = (3, 1, 1) if image else (array.shape[1],)
    output = {
        name: value.reshape(shape).tolist()
        for name, value in (
            ("min", array.min(0)),
            ("max", array.max(0)),
            ("mean", array.mean(0)),
            ("std", array.std(0)),
        )
    }
    for q in (0.01, 0.1, 0.5, 0.9, 0.99):
        output[f"q{round(q * 100):02d}"] = (
            np.quantile(array, q, axis=0).reshape(shape).tolist()
        )
    output["count"] = [len(values)]
    return output


def _video_samples(path: Path, length: int) -> np.ndarray:
    picks = set(np.linspace(0, length - 1, min(length, 32), dtype=int).tolist())
    samples = []
    with av.open(str(path)) as video:
        for index, frame in enumerate(video.decode(video=0)):
            if index in picks:
                samples.append(
                    frame.reformat(width=64, height=64)
                    .to_ndarray(format="rgb24")
                    .astype(np.float32)
                    / 255
                )
    if len(samples) != len(picks):
        raise ValueError("裁剪视频实际帧数不足。")
    return np.stack(samples)


def _clip_intervals(intervals: list, start: int, end: int, outcomes=False) -> list:
    clipped = []
    for interval in intervals or []:
        a, b = max(start, interval["start_frame"]), min(end, interval["end_frame"])
        if a >= b:
            continue
        row = dict(interval, start_frame=a - start, end_frame=b - start)
        if outcomes and end < interval["end_frame"]:
            row["outcome"] = "unknown"
            row["autonomous_success"] = False
        clipped.append(row)
    return clipped


def export_dataset(root: Path, encode_video, progress=lambda text: None) -> dict:
    """Create one LeRobot episode per selected interval, rebuilding indexes and stats."""
    pa, pq = _arrow()
    info, episodes = dataset(root)
    store = review.read_store(root)
    selected = [
        (ep, store["trajectories"][f"episode_{ep['episode_index']}"])
        for ep in episodes
        if store["trajectories"]
        .get(f"episode_{ep['episode_index']}", {})
        .get("segments")
    ]
    output = training_export.check_target(root)
    if not selected and not output.exists():
        raise ValueError("还没有保存保留片段，请先设置起止点并保存。")
    staging = Path(tempfile.mkdtemp(prefix=".training-export-", dir=root))
    replay_trajectories = []
    rows, all_numeric, all_images, provenance, total = [], {}, {}, [], 0
    try:
        for ep, entry in selected:
            if source_fingerprint(root, info, ep) != entry["fingerprint"]:
                raise review.ReviewConflict(
                    f"Episode {ep['episode_index']} 已变化，请核对旧标注。"
                )
            table = table_for(root, info, ep)
            media = cameras(root, info, ep)
            used = set()
            for segment in entry["segments"]:
                start, end = segment["start_frame"], segment["end_frame"]
                if (
                    type(start) is not int
                    or type(end) is not int
                    or not 0 <= start < end <= len(table)
                ):
                    raise ValueError("标注帧范围超出 episode。")
                if used.intersection(range(start, end)):
                    raise ValueError("保留片段存在重叠。")
                used.update(range(start, end))
                replay_trajectories.append(
                    training_export.build_trajectory(table, ep, segment, len(media))
                )
                n, eid = end - start, len(rows)
                chunk, file = divmod(eid, int(info.get("chunks_size", 1000)))
                clipped = table.slice(start, n)
                for key, values in {
                    "index": np.arange(total, total + n),
                    "frame_index": np.arange(n),
                    "episode_index": np.full(n, eid),
                    "timestamp": np.arange(n) / info["fps"],
                }.items():
                    i = clipped.schema.get_field_index(key)
                    clipped = clipped.set_column(
                        i,
                        clipped.schema.field(i),
                        pa.array(values, type=clipped.schema.field(i).type),
                    )
                target = staging / info["data_path"].format(
                    chunk_index=chunk, file_index=file
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(clipped, target)
                row = {
                    k: v
                    for k, v in ep.items()
                    if not k.startswith(("stats/", "videos/"))
                }
                row.update(
                    {
                        "episode_index": eid,
                        "length": n,
                        "data/chunk_index": chunk,
                        "data/file_index": file,
                        "dataset_from_index": total,
                        "dataset_to_index": total + n,
                        "meta/episodes/chunk_index": 0,
                        "meta/episodes/file_index": 0,
                        "episode_success": ep.get("episode_success", "unknown")
                        if start == 0 and end == len(table)
                        else "unknown",
                        "source_episode_index": ep["episode_index"],
                        "source_start_frame": start,
                        "source_end_frame": end,
                        "review_note": segment["note"],
                        "review_label": segment["label"],
                    }
                )
                row["rl_intervals"] = _clip_intervals(
                    ep.get("rl_intervals", []), start, end, outcomes=True
                )
                row["human_intervention_intervals"] = _clip_intervals(
                    ep.get("human_intervention_intervals", []), start, end
                )
                if "complementary_info.is_intervention" in clipped.column_names:
                    flags = (
                        column_array(clipped, "complementary_info.is_intervention")
                        .reshape(n, -1)
                        .any(1)
                    )
                    for interval in row["rl_intervals"]:
                        human = bool(
                            flags[interval["start_frame"] : interval["end_frame"]].any()
                        )
                        interval["has_intervention"] = human
                        interval["autonomous_success"] = (
                            interval.get("outcome") == "success" and not human
                        )
                stats = {}
                for key in clipped.column_names:
                    values = column_array(clipped, key)
                    if values.dtype.kind in "biuf":
                        stats[key] = _stats(values)
                        all_numeric.setdefault(key, []).append(values)
                for key, source in media.items():
                    progress(
                        f"正在导出片段 {eid + 1} · Episode {ep['episode_index']} · {key.rsplit('.', 1)[-1]}"
                    )
                    target = staging / info["video_path"].format(
                        video_key=key, chunk_index=chunk, file_index=file
                    )
                    target.parent.mkdir(parents=True, exist_ok=True)
                    encode_video(source, info["fps"], target, torch.arange(start, end))
                    samples = _video_samples(target, n)
                    stats[key] = _stats(samples, image=True)
                    all_images.setdefault(key, []).append(samples)
                    prefix = f"videos/{key}/"
                    row.update(
                        {
                            prefix + "chunk_index": chunk,
                            prefix + "file_index": file,
                            prefix + "from_timestamp": 0.0,
                            prefix + "to_timestamp": n / info["fps"],
                        }
                    )
                for key, metrics in stats.items():
                    row.update(
                        {
                            f"stats/{key}/{name}": value
                            for name, value in metrics.items()
                        }
                    )
                rows.append(row)
                provenance.append(
                    {
                        "episode_index": eid,
                        "source_episode_index": ep["episode_index"],
                        "source_range": [start, end],
                        "source_fingerprint": entry["fingerprint"],
                        "annotation": segment,
                    }
                )
                total += n
            if source_fingerprint(root, info, ep) != entry["fingerprint"]:
                raise review.ReviewConflict("导出期间原始数据发生变化，请重试。")
        new_info = copy.deepcopy(info)
        new_info.update(
            total_episodes=len(rows),
            total_frames=total,
            splits={"train": f"0:{len(rows)}"},
        )
        for key, spec in new_info["features"].items():
            if spec["dtype"] == "video":
                spec.setdefault("info", {}).update(
                    {
                        "video.fps": info["fps"],
                        "video.codec": "h264",
                        "video.pix_fmt": "yuv420p",
                    }
                )
        new_info["manual_review"] = {
            "source_dataset": str(root),
            "interval_convention": "[start_frame,end_frame)",
            "image_stats": "Up to 32 evenly spaced decoded frames per clip, resized to 64x64; RGB in [0,1].",
        }
        (staging / "meta/episodes/chunk-000").mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.Table.from_pylist(rows),
            staging / "meta/episodes/chunk-000/file-000.parquet",
        )
        shutil.copyfile(root / "meta/tasks.parquet", staging / "meta/tasks.parquet")
        (staging / "meta/info.json").write_text(
            json.dumps(new_info, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        stats = {
            key: _stats(np.concatenate(values)) for key, values in all_numeric.items()
        }
        stats.update(
            {
                key: _stats(np.concatenate(values), image=True)
                for key, values in all_images.items()
            }
        )
        (staging / "meta/stats.json").write_text(
            json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        counts = training_export.write_replay(
            staging, info, rows, provenance, replay_trajectories
        )
        (staging / "review_selection.json").write_text(
            json.dumps(
                {
                    "source_dataset": str(root),
                    "interval_convention": "[start_frame,end_frame)",
                    "segments": provenance,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        for ep, entry in selected:
            if source_fingerprint(root, info, ep) != entry["fingerprint"]:
                raise review.ReviewConflict(
                    "导出期间源数据发生变化，旧训练轨迹仍保留。"
                )
        return training_export.publish(root, staging, store, total, counts)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
