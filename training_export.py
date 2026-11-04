"""Build and publish the RLinf replay view of reviewed source intervals."""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import numpy as np
import torch

import review

DIRECTORY = "training_trajectories"
MARKER = "training_export.json"
FORMAT = "replay_buffer_viewer.training.v1"
WEIGHTS_ID = "reviewed_raw_v1"


def read_marker(directory: Path) -> dict | None:
    path = directory / MARKER
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("format") != FORMAT:
        raise ValueError("无法识别的训练轨迹导出标记。")
    return data


def navigation(directory: Path) -> dict:
    marker = read_marker(directory)
    if marker is None:
        return {}
    return {
        "sourceType": "training_replay",
        "sourceDirectory": str((directory / marker["source_dataset"]).resolve()),
        "trainingDirectory": str(directory),
        "exportGeneration": marker["generation"],
        "selectedFrames": marker["frames"],
        "exportedTransitions": marker["transitions"],
    }


def check_target(root: Path) -> Path:
    if read_marker(root) is not None:
        raise ValueError("请返回原始数据集裁剪，训练目录用于预览。")
    output = root / DIRECTORY
    if output.is_symlink():
        raise ValueError("训练输出路径是符号链接，不能覆盖。")
    if output.exists():
        marker = read_marker(output) if output.is_dir() else None
        if (
            marker is None
            or (output / marker["source_dataset"]).resolve() != root.resolve()
        ):
            raise ValueError(f"{output} 不是此数据集的查看器导出目录，不能覆盖。")
    return output


def build_trajectory(table, episode: dict, segment: dict, camera_count: int) -> dict:
    """N selected observations form N-1 real one-step transitions within the selection."""
    start, end = segment["start_frame"], segment["end_frame"]
    selected = table.slice(start, end - start)
    count = len(selected) - 1
    if count < 1:
        raise ValueError("训练片段至少需要 2 帧：最后一帧用于 next_obs。")

    def tensor(key, dtype=None):
        value = torch.from_numpy(np.asarray(selected[key].to_pylist()).copy()).reshape(
            count + 1, 1, -1
        )
        return value.to(dtype) if dtype is not None else value

    actions = tensor("action", torch.float32)[:-1].clone()
    states = tensor("observation.state", torch.float32)
    source_index = tensor("index", torch.int64)
    frame_index = tensor("frame_index", torch.int64)
    flags = (
        tensor("complementary_info.is_intervention", torch.bool)[:-1]
        if "complementary_info.is_intervention" in selected.column_names
        else torch.zeros(count, 1, 1, dtype=torch.bool)
    )
    rewards = torch.zeros(count, 1, 1, dtype=torch.float32)
    terminations = torch.zeros(count, 1, 1, dtype=torch.bool)
    truncations = torch.zeros_like(terminations)
    known_reward = torch.zeros_like(terminations)
    # These are subtask labels, not rewards measured by a simulator.
    for interval in episode.get("rl_intervals", []) or []:
        a, b = int(interval["start_frame"]), int(interval["end_frame"])
        if interval.get("outcome") not in {"success", "failure"}:
            continue
        selected_indices = frame_index[:-1]
        known_reward |= (selected_indices >= a) & (selected_indices < b)
        terminal = selected_indices == b - 1
        terminations |= terminal
        if interval["outcome"] == "success":
            rewards[terminal] = 1
    # Preserve recorded feedback when the original dataset supplies it.
    for names, target in [
        (("rewards", "reward"), rewards),
        (("terminations", "terminated"), terminations),
        (("truncations", "truncated"), truncations),
    ]:
        name = next((name for name in names if name in selected.column_names), None)
        if name:
            target.copy_(tensor(name, target.dtype)[:-1])
            if target is rewards:
                known_reward[:] = True
    outcome = segment.get("outcome", "source")
    if outcome not in {"source", "success", "failure"}:
        raise ValueError("未知片段末端结果。")
    if outcome != "source":
        rewards[-1] = float(outcome == "success")
        terminations[-1] = True
        truncations[-1] = False
        known_reward[-1] = True
    forward = {
        "episode_index": tensor("episode_index", torch.int64)[:-1].clone(),
        "frame_index": frame_index[:-1].clone(),
        "reward_is_annotated": known_reward,
    }
    for source, target in [
        ("complementary_info.policy_action", "recorded_policy_action"),
        ("complementary_info.collector_policy_id", "collector_policy_id"),
    ]:
        if source in selected.column_names:
            forward[target] = tensor(source)[:-1].clone()
    video_indices = (
        torch.arange(count + 1).reshape(-1, 1, 1).expand(-1, 1, camera_count).clone()
    )
    return {
        "max_episode_length": count,
        "model_weights_id": WEIGHTS_ID,
        "actions": actions,
        "intervene_flags": flags.expand_as(actions).clone(),
        "rewards": rewards,
        "terminations": terminations,
        "truncations": truncations,
        "dones": terminations | truncations,
        "forward_inputs": forward,
        "curr_obs": {
            "states": states[:-1].clone(),
            "source_index": source_index[:-1].clone(),
            "video_frame_indices": video_indices[:-1].clone(),
        },
        "next_obs": {
            "states": states[1:].clone(),
            "source_index": source_index[1:].clone(),
            "video_frame_indices": video_indices[1:].clone(),
        },
    }


def write_replay(
    staging: Path, info: dict, rows: list, provenance: list, trajectories: list
) -> dict:
    index, media, transitions = {}, {}, 0
    camera_order = [
        key for key, spec in info["features"].items() if spec["dtype"] == "video"
    ]
    for identifier, (row, origin, trajectory) in enumerate(
        zip(rows, provenance, trajectories)
    ):
        count = len(trajectory["actions"])
        filename = f"trajectory_{identifier}_{WEIGHTS_ID}.pt"
        torch.save(trajectory, staging / filename)
        index[str(identifier)] = {
            "trajectory_id": identifier,
            "num_samples": count,
            "max_episode_length": count,
            "shape": [count, 1, 1],
            "model_weights_id": WEIGHTS_ID,
            "source_episode_index": origin["source_episode_index"],
            "source_range": origin["source_range"],
            "segment_id": origin["annotation"]["id"],
        }
        media[filename] = {
            "media": [
                {
                    "camera": key,
                    "path": info["video_path"].format(
                        video_key=key,
                        chunk_index=row[f"videos/{key}/chunk_index"],
                        file_index=row[f"videos/{key}/file_index"],
                    ),
                    "fps": info["fps"],
                    "num_frames": row["length"],
                }
                for key in camera_order
            ]
        }
        origin["trajectory"] = filename
        transitions += count
    (staging / "metadata.json").write_text(
        json.dumps(
            {
                "trajectory_format": "pt",
                "size": len(index),
                "total_samples": transitions,
                "trajectory_counter": len(index),
                "seed": 42,
            }
        ),
        encoding="utf-8",
    )
    (staging / "trajectory_index.json").write_text(
        json.dumps(
            {"trajectory_index": index, "trajectory_id_list": list(range(len(index)))}
        ),
        encoding="utf-8",
    )
    (staging / "source_manifest.json").write_text(
        json.dumps(
            {
                "fps": info["fps"],
                "camera_order": camera_order,
                "trajectories": media,
                "rlt_features_ready": False,
                "reward_definition": "Recorded feedback when available; otherwise source rl_intervals labels. Unannotated transitions have reward 0 with reward_is_annotated=False. Explicit reviewed endpoint labels override the last transition.",
                "transition_definition": "N selected frames produce N-1 transitions. Final selected frame is next_obs only; no transition crosses a clip boundary.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"transitions": transitions, "trajectories": len(index)}


def publish(root: Path, staging: Path, store: dict, frames: int, counts: dict) -> dict:
    output = check_target(root)
    generation = uuid.uuid4().hex
    marker = {
        "format": FORMAT,
        "source_dataset": "..",
        "generation": generation,
        "frames": frames,
        **counts,
    }
    (staging / MARKER).write_text(json.dumps(marker, indent=2), encoding="utf-8")
    # Serialize publishing against review edits; never publish an obsolete selection.
    with review._lock:
        if review.read_store(root) != store:
            raise review.ReviewConflict(
                "导出期间标注已修改，请重新导出；旧训练轨迹仍保留。"
            )
        backup = root / f".training-backup-{generation}"
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
    return {
        "directory": str(output),
        "episodes": counts["trajectories"],
        "frames": frames,
        "generation": generation,
        **counts,
    }
