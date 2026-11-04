#!/usr/bin/env python3
"""Web viewer for RLinf replay-buffer trajectory files."""

from __future__ import annotations

import hashlib
import io
import json
import os
import threading
import uuid
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import numpy as np
import torch
from flask import Flask, Response, jsonify, render_template, request, send_file

import lerobot_source
import review
import training_export

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

_lock = threading.RLock()
_trajectory: dict[str, Any] | None = None
_trajectory_path: Path | None = None
_trajectory_key: str | None = None
_trajectory_info: dict[str, Any] | None = None
_exports: dict[str, dict[str, Any]] = {}


def _layout(data: dict[str, Any]) -> tuple[int, int, float]:
    """Infer the timeline independently of whether images are embedded."""
    inputs = data["forward_inputs"]
    substeps = inputs.get("substep_main_images")
    if _is_h264(substeps):
        items = _encoded_items(substeps)
        return len(items), int(items[0]["shape"][1]), float(items[0]["fps"])
    if _is_raw_image(substeps) and substeps.ndim == 6:
        return int(substeps.shape[0]), int(substeps.shape[2]), 30.0
    if substeps is not None:
        raise ValueError("Unsupported embedded replay image format")
    for tensor in (inputs.get("chunk_valid_mask"), data.get("rewards")):
        if isinstance(tensor, torch.Tensor) and tensor.ndim == 3 and tensor.shape[1] == 1:
            return int(tensor.shape[0]), int(tensor.shape[2]), 30.0
    actions = data.get("actions")
    if isinstance(actions, torch.Tensor) and actions.ndim >= 2:
        return int(actions.shape[0]), 1, 30.0
    raise ValueError("Cannot determine the trajectory timeline from images, rewards or actions")


def _packaged_media(data: dict, manifest: dict, manifest_path: Path, indices: torch.Tensor) -> dict:
    """Resolve merged LeRobot videos using packaged episode IDs, not source IDs."""
    timeline = {}
    for key in ("episode_index", "frame_index"):
        value = data.get("forward_inputs", {}).get(key)
        if (
            not isinstance(value, torch.Tensor)
            or list(value.shape) != [len(indices), 1, 1]
            or value.is_floating_point()
            or value.dtype == torch.bool
        ):
            raise ValueError(f"Packaged video lookup requires integer forward_inputs.{key} shaped [frames, 1, 1]")
        timeline[key] = value.reshape(-1).long()
    episode_ids = timeline["episode_index"]
    if not (episode_ids == episode_ids[0]).all():
        raise ValueError("Packaged video lookup requires a single episode per trajectory")
    root = (manifest_path.parent / manifest["packaged_source_dataset"]).resolve()
    info, episodes = lerobot_source.dataset(root)
    matches = [row for row in episodes if row["episode_index"] == int(episode_ids[0])]
    if len(matches) != 1:
        raise ValueError(f"Packaged dataset has no unique episode {int(episode_ids[0])}")
    episode = matches[0]
    frame_indices = timeline["frame_index"]
    if ((frame_indices < 0) | (frame_indices >= episode["length"])).any():
        raise ValueError("Replay frame_index exceeds the packaged episode length")
    if not np.isclose(float(info["fps"]), float(manifest.get("fps", 30))):
        raise ValueError("Packaged dataset FPS does not match the source manifest")
    media = lerobot_source.cameras(root, info, episode)
    for column, key in enumerate(manifest["camera_order"]):
        if key not in media:
            raise ValueError(f"Packaged dataset has no camera {key}")
        expected = media[key]["indices"].index_select(0, frame_indices)
        if not torch.equal(indices[:, 0, column].long(), expected):
            raise ValueError(f"Replay video_frame_indices do not match packaged episode offsets: {key}")
    return media


def _external_media(data: dict[str, Any], path: Path, frames: int) -> tuple[dict, float, str]:
    """Resolve the portable raw-source cache's relative media references."""
    indices = data.get("curr_obs", {}).get("video_frame_indices")
    if not isinstance(indices, torch.Tensor):
        return {}, 30.0, "该轨迹未保存相机图像或视频帧索引，可查看数值数据。"
    manifest_path = next((p for p in (
        path.parent / "source_manifest.json", path.parent.parent / "source_manifest.json"
    ) if p.is_file()), None)
    if manifest_path is None:
        return {}, 30.0, "未找到 source_manifest.json，请保留 raw_replay 与原缓存目录的相对位置；当前可查看数值数据。"
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    fps = float(manifest.get("fps", 30))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid source video FPS")
    order = manifest.get("camera_order", [])
    if list(indices.shape) != [frames, 1, len(order)] or not order:
        raise ValueError("Expected video_frame_indices shaped [frames, 1, cameras] matching camera_order")
    if indices.is_floating_point() or indices.dtype == torch.bool or (indices < 0).any():
        raise ValueError("Video frame indices must be nonnegative integers")
    media_contract = manifest.get("trajectories", {}).get(path.name, manifest)
    media_by_camera = {item["camera"]: item for item in media_contract.get("media", [])}
    if "media" not in media_contract and manifest.get("packaged_source_dataset"):
        media_by_camera = _packaged_media(data, manifest, manifest_path, indices)
    # These names define the G1 source-cache camera roles, independent of column order.
    roles = {"main": "cam_left_high", "left": "cam_left_wrist", "right": "cam_right_wrist"}
    media, missing = {}, []
    for camera, name in roles.items():
        source_name = next((key for key in order if key.rsplit(".", 1)[-1] == name), None)
        item = media_by_camera.get(source_name)
        if item is None:
            missing.append(camera)
            continue
        source = (manifest_path.parent / item["path"]).resolve()
        if not source.is_file():
            missing.append(str(source))
            continue
        selected = indices[:, 0, order.index(source_name)].to(torch.int64)
        if (selected[1:] < selected[:-1]).any():
            raise ValueError("Video frame indices must be ordered within a trajectory")
        if item.get("num_frames") is not None and int(selected.max()) >= int(item["num_frames"]):
            raise ValueError(f"Video frame index exceeds source length: {source}")
        media[camera] = {"path": source, "indices": selected, "fps": float(item.get("fps", fps))}
        if not np.isfinite(media[camera]["fps"]) or media[camera]["fps"] <= 0:
            raise ValueError(f"Invalid source video FPS: {source}")
    warning = "缺少视频：" + ", ".join(missing) + "；仍可查看数值数据。" if missing else ""
    return media, fps, warning


def _is_h264(value: Any) -> bool:
    return isinstance(value, dict) and (
        value.get("__rlinf_h264__") == 1
        or value.get("__rlinf_h264_stack__") == 1
    )


def _is_raw_image(value: Any) -> bool:
    return (
        isinstance(value, torch.Tensor)
        and value.dtype == torch.uint8
        and value.ndim >= 3
        and tuple(value.shape[-1:]) == (3,)
    )


def _encoded_items(value: dict[str, Any]) -> list[dict[str, Any]]:
    if value.get("__rlinf_h264_stack__") == 1:
        return value["items"]
    return [value]


def _tensor_summary(value: torch.Tensor) -> dict[str, Any]:
    result: dict[str, Any] = {
        "type": "tensor",
        "shape": list(value.shape),
        "dtype": str(value.dtype).replace("torch.", ""),
        "bytes": value.numel() * value.element_size(),
    }
    if value.numel() and value.dtype != torch.bool:
        sample = value.detach().cpu().flatten()
        if sample.numel() > 200_000:
            sample = sample[:: max(1, sample.numel() // 200_000)]
        finite = sample[torch.isfinite(sample)] if sample.is_floating_point() else sample
        if finite.numel():
            result.update(min=float(finite.min()), max=float(finite.max()))
    return result


def _describe(value: Any) -> dict[str, Any]:
    if isinstance(value, torch.Tensor):
        return _tensor_summary(value)
    if _is_h264(value):
        items = _encoded_items(value)
        encoded_bytes = sum(
            int(stream.numel()) for item in items for stream in item["data"]
        )
        first = items[0]
        logical_shape = list(first["shape"])
        if value.get("__rlinf_h264_stack__") == 1:
            logical_shape.insert(int(value["dim"]), len(items))
        return {
            "type": "h264",
            "shape": logical_shape,
            "dtype": "uint8 (decoded)",
            "bytes": encoded_bytes,
            "chunks": len(items),
            "streams": len(first["data"]),
            "fps": int(first["fps"]),
            "crf": int(first["crf"]),
            "gop": int(first["gop"]),
        }
    if isinstance(value, dict):
        return {"type": "dict", "children": {key: _describe(item) for key, item in value.items()}}
    if isinstance(value, (list, tuple)):
        return {"type": type(value).__name__, "length": len(value)}
    return {"type": type(value).__name__, "value": value}


def _flatten_tensors(value: Any, prefix: str = "") -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    if isinstance(value, torch.Tensor):
        result[prefix] = value
    elif isinstance(value, dict) and not _is_h264(value):
        for key, item in value.items():
            result.update(_flatten_tensors(item, f"{prefix}.{key}" if prefix else key))
    return result


def _timeline_tensor(tensor: torch.Tensor, chunks: int, steps: int) -> torch.Tensor | None:
    value = tensor.detach().cpu()
    if value.ndim >= 3 and value.shape[0] == chunks and value.shape[1] == 1:
        value = value[:, 0]
        if value.shape[1] == steps:
            return value.reshape(chunks * steps, *value.shape[2:])
        if value.shape[1] % steps == 0:
            width = value.shape[1] // steps
            return value.reshape(chunks, steps, width, *value.shape[2:]).reshape(
                chunks * steps, width, *value.shape[2:]
            )
        return value.repeat_interleave(steps, dim=0)
    if value.ndim >= 1 and value.shape[0] == chunks:
        return value.repeat_interleave(steps, dim=0)
    return None


def _series_catalog(data: dict[str, Any], chunks: int, steps: int) -> list[dict[str, Any]]:
    catalog = []
    for path, tensor in _flatten_tensors(data).items():
        if _is_raw_image(tensor):
            continue
        timeline = _timeline_tensor(tensor, chunks, steps)
        if timeline is None or timeline.numel() == 0:
            continue
        component_shape = list(timeline.shape[1:])
        components = int(np.prod(component_shape)) if component_shape else 1
        catalog.append({
            "path": path,
            "dtype": str(tensor.dtype).replace("torch.", ""),
            "sourceShape": list(tensor.shape),
            "componentShape": component_shape,
            "components": components,
        })
    return catalog


def _replay_result(data: dict[str, Any]) -> dict[str, Any]:
    """Report sparse terminal feedback without treating an unfinished clip as failure."""
    unknown = {"isSuccess": None, "outcomeSource": "未记录明确的末端结果，或片段尚未到达结束点。"}
    rewards, terminations = data.get("rewards"), data.get("terminations")
    if not isinstance(rewards, torch.Tensor) or not isinstance(terminations, torch.Tensor):
        return unknown
    chunks, steps, _ = _layout(data)
    frames = chunks * steps
    if rewards.numel() != frames or terminations.shape != rewards.shape:
        return unknown
    valid = _valid_indices(data)
    if not len(valid):
        return unknown
    last = int(valid[-1])
    if terminations.reshape(-1)[last].item() != 1:
        return unknown
    for field, expected in ((data.get("truncations"), 0),
                            (data["forward_inputs"].get("reward_is_annotated"), 1)):
        if field is not None and (
            not isinstance(field, torch.Tensor)
            or field.shape != rewards.shape
            or field.reshape(-1)[last].item() != expected
        ):
            return unknown
    value = rewards.reshape(-1)[last].item()
    if value not in (0, 1):
        return {**unknown, "outcomeSource": "末端奖励不是 0/1，无法按稀疏成功奖励判断结果。"}
    return {"isSuccess": bool(value),
            "outcomeSource": f"依据末个有效帧的终止奖励：{value:g}（1=成功，0=失败）。"}


def _current() -> tuple[dict[str, Any], Path, str, dict[str, Any]]:
    with _lock:
        if _trajectory is None or _trajectory_path is None or _trajectory_key is None or _trajectory_info is None:
            raise RuntimeError("No trajectory is loaded")
        if request.args.get("key", _trajectory_key) != _trajectory_key:
            raise ValueError("Trajectory changed; reload the selected trajectory")
        if _trajectory_info.get("exportGeneration"):
            marker = training_export.read_marker(_trajectory_path.parent)
            if not marker or marker["generation"] != _trajectory_info["exportGeneration"]:
                raise ValueError("训练轨迹已更新，请重新打开训练目录。")
        return _trajectory, _trajectory_path, _trajectory_key, _trajectory_info


def _load(path_text: str, episode_id: int | None = None) -> dict[str, Any]:
    global _trajectory, _trajectory_path, _trajectory_key, _trajectory_info
    path = Path(path_text).expanduser().resolve()
    if path.is_dir() and training_export.read_marker(path) is not None:
        collection = _browse_directory(str(path))
        if not collection["trajectories"]:
            raise ValueError("训练目录没有已保留轨迹，请返回原始数据裁剪。")
        return _load(collection["trajectories"][0]["path"])
    if path.is_dir() and (path / "meta/info.json").is_file():
        return _load_original(path, episode_id)
    if not path.is_file() or path.suffix != ".pt":
        raise ValueError("Path must point to an existing .pt trajectory file")
    data = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(data, dict) or "forward_inputs" not in data:
        raise ValueError("The file is not a supported RLinf trajectory dictionary")
    chunks, steps, fps = _layout(data)
    if chunks < 1 or steps < 1:
        raise ValueError("Trajectory has no frames")
    media, warning = {}, ""
    embedded = data["forward_inputs"].get("substep_main_images") is not None
    if embedded:
        cameras = ["main"]
        extra = data["forward_inputs"].get("substep_extra_view_images")
        count = len(_encoded_items(extra)[0]["data"]) if _is_h264(extra) else (
            int(extra.shape[3]) if _is_raw_image(extra) and extra.ndim == 7 else 0
        )
        cameras.extend(["left", "right"][:count])
    else:
        try:
            media, fps, warning = _external_media(data, path, chunks * steps)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            warning = f"外部视频无法关联：{exc}；当前可查看数值数据。"
        cameras = list(media)
    navigation = training_export.navigation(path.parent)
    key_source = f"v2:{path}:{path.stat().st_size}:{path.stat().st_mtime_ns}:{fps}:{navigation.get('exportGeneration', '')}"
    for camera, item in media.items():
        stat = item["path"].stat()
        key_source += f":{camera}:{item['path']}:{stat.st_size}:{stat.st_mtime_ns}:{item['fps']}"
        key_source += hashlib.sha256(item["indices"].numpy().tobytes()).hexdigest()
    key = hashlib.sha256(key_source.encode()).hexdigest()[:20]
    mask = data["forward_inputs"].get("chunk_valid_mask")
    has_mask = isinstance(mask, torch.Tensor)
    valid = len(_valid_indices(data)) if has_mask else chunks * steps
    info = {
        "key": key,
        "path": str(path),
        "filename": path.name,
        "fileBytes": path.stat().st_size,
        "chunks": chunks,
        "stepsPerChunk": steps,
        "totalFrames": chunks * steps,
        "validFrames": valid,
        "fps": fps,
        "duration": chunks * steps / fps,
        "hasValidMask": has_mask,
        "cameras": cameras,
        "videoSource": "embedded" if embedded else "external" if media else "none",
        "mediaMessage": warning,
        "fpsAssumed": not embedded and not media,
        "modelWeightsId": str(data.get("model_weights_id", "")),
        "tree": _describe(data),
        "series": _series_catalog(data, chunks, steps),
        **_replay_result(data),
    }
    info.update(navigation)
    if info.get("sourceType") == "training_replay":
        index = json.loads((path.parent / "trajectory_index.json").read_text())["trajectory_index"]
        item = next((item for item in index.values() if path.name == f"trajectory_{item['trajectory_id']}_{item['model_weights_id']}.pt"), {})
        info.update(sourceEpisodeId=item.get("source_episode_index"), sourceRange=item.get("source_range"), segmentId=item.get("segment_id"))
    with _lock:
        _trajectory, _trajectory_path, _trajectory_key = data, path, key
        _trajectory_info = {**info, "media": media}
    return info


def _load_original(path: Path, episode_id: int | None) -> dict[str, Any]:
    global _trajectory, _trajectory_path, _trajectory_key, _trajectory_info
    data, media, context = lerobot_source.load_episode(path, episode_id)
    episode, fps = context["episode"], context["fps"]
    n = episode["length"]
    info = {"key": context["key"], "path": str(path), "sourceType": "lerobot",
        "episodeId": episode["episode_index"], "filename": f"Episode {episode['episode_index']}",
        "fileBytes": sum(item["size"] for item in context["fingerprint"]["files"].values() if item),
        "chunks": n, "stepsPerChunk": 1, "totalFrames": n, "validFrames": n,
        "hasValidMask": False, "fps": fps, "duration": n / fps, "fpsAssumed": False,
        "cameras": list(media), "videoSource": "external" if media else "none",
        "mediaMessage": context["mediaMessage"], "modelWeightsId": "LeRobot v3 · 完整原始数据",
        "tree": _describe(data), "series": _series_catalog(data, n, 1),
        **lerobot_source.episode_result(episode)}
    info.update(sourceDirectory=str(path), trainingDirectory=str(path / training_export.DIRECTORY),
                trainingExists=(path / training_export.DIRECTORY / training_export.MARKER).is_file())
    with _lock:
        _trajectory, _trajectory_path, _trajectory_key = data, path, info["key"]
        _trajectory_info = {**info, "media": media, "reviewContext": context}
    return info


def _browse_directory(path_text: str) -> dict[str, Any]:
    directory = Path(path_text).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError("Path must point to an existing replay-buffer directory")
    navigation = training_export.navigation(directory)
    if not navigation and (directory / "meta/info.json").is_file():
        return lerobot_source.browse(directory)
    index_path = directory / "trajectory_index.json"
    if not index_path.is_file():
        raise ValueError("Directory has no trajectory_index.json")
    with index_path.open(encoding="utf-8") as handle:
        index_data = json.load(handle)
    metadata_path = directory / "metadata.json"
    metadata = {}
    if metadata_path.is_file():
        with metadata_path.open(encoding="utf-8") as handle:
            metadata = json.load(handle)

    index = index_data.get("trajectory_index", {})
    ordered_ids = index_data.get("trajectory_id_list", list(index))
    trajectories = []
    for raw_id in ordered_ids:
        item = index.get(str(raw_id), {})
        trajectory_id = int(item.get("trajectory_id", raw_id))
        weights_id = str(item.get("model_weights_id", ""))
        file_path = directory / f"trajectory_{trajectory_id}_{weights_id}.pt"
        if not file_path.is_file():
            matches = sorted(directory.glob(f"trajectory_{trajectory_id}_*.pt"))
            if not matches:
                continue
            file_path = matches[0]
        trajectory = torch.load(file_path, map_location="cpu", weights_only=False, mmap=True)
        intervene_flags = trajectory.get("intervene_flags")
        trajectories.append(
            {
                "id": trajectory_id,
                "path": str(file_path),
                "filename": file_path.name,
                "fileBytes": file_path.stat().st_size,
                "numSamples": int(item.get("num_samples", 0)),
                "shape": item.get("shape", []),
                "modelWeightsId": weights_id,
                **_replay_result(trajectory),
                "hasIntervention": bool(
                    isinstance(intervene_flags, torch.Tensor) and intervene_flags.any()
                ),
                "sourceEpisodeId": item.get("source_episode_index"),
                "sourceRange": item.get("source_range"),
            }
        )
    if not trajectories and not navigation:
        raise ValueError("Directory index contains no available trajectory files")
    return {
        "directory": str(directory),
        "metadata": metadata,
        "trajectories": trajectories,
        **navigation,
    }


def _camera_payload(data: dict[str, Any], camera: str) -> tuple[dict[str, Any], int]:
    if camera == "main":
        return data["forward_inputs"]["substep_main_images"], 0
    if camera == "left":
        return data["forward_inputs"]["substep_extra_view_images"], 0
    if camera == "right":
        return data["forward_inputs"]["substep_extra_view_images"], 1
    raise ValueError("Unknown camera")


def _valid_indices(data: dict[str, Any]) -> torch.Tensor:
    mask = data["forward_inputs"].get("chunk_valid_mask")
    chunks, steps, _ = _layout(data)
    if not isinstance(mask, torch.Tensor):
        return torch.arange(chunks * steps)
    if list(mask.shape) != [chunks, 1, steps]:
        raise ValueError("chunk_valid_mask does not match the trajectory timeline")
    return torch.nonzero(mask.reshape(-1).to(torch.bool), as_tuple=False).flatten()


def _encode_external_camera(media: dict, fps: float, output_path: Path, keep: torch.Tensor) -> None:
    indices = media["indices"].index_select(0, keep).tolist()
    if not indices:
        raise ValueError("Trajectory has no valid video frames")
    temporary = output_path.with_suffix(".tmp.mp4")
    try:
        with av.open(str(media["path"])) as source, av.open(str(temporary), "w", format="mp4") as output:
            incoming = source.streams.video[0]
            origin = incoming.start_time or 0
            source.seek(origin + int(indices[0] / media["fps"] / incoming.time_base), stream=incoming)
            stream = output.add_stream("libx264", rate=Fraction(str(fps)))
            stream.width, stream.height = incoming.width, incoming.height
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "18", "preset": "veryfast", "g": str(max(1, round(fps)))}
            position = 0
            for frame in source.decode(video=0):
                number = round(float((frame.pts - origin) * incoming.time_base) * media["fps"])
                if number > indices[position]:
                    raise ValueError(f"Source video is missing requested frame {indices[position]}")
                while position < len(indices) and number == indices[position]:
                    encoded_frame = av.VideoFrame.from_ndarray(frame.to_ndarray(format="rgb24"), format="rgb24")
                    encoded_frame.pts = position
                    for packet in stream.encode(encoded_frame):
                        output.mux(packet)
                    position += 1
                if position == len(indices):
                    break
            if position != len(indices):
                raise ValueError(f"Source video ended before requested frame {indices[position]}")
            for packet in stream.encode():
                output.mux(packet)
        temporary.replace(output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _encode_filtered_camera(
    data: dict[str, Any], camera: str, output_path: Path
) -> None:
    payload, stream_index = _camera_payload(data, camera)
    chunks, steps, fps = _layout(data)
    mask = torch.zeros(chunks * steps, dtype=torch.bool)
    mask[_valid_indices(data)] = True
    mask = mask.reshape(chunks, steps)
    temporary = output_path.with_suffix(".tmp.mp4")
    output = av.open(str(temporary), mode="w", format="mp4")
    stream = None
    frame_index = 0

    def encode_frame(array: np.ndarray) -> None:
        nonlocal stream, frame_index
        if stream is None:
            stream = output.add_stream("libx264", rate=Fraction(str(fps)))
            stream.width, stream.height = int(array.shape[1]), int(array.shape[0])
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "18", "preset": "veryfast", "g": "30"}
        frame = av.VideoFrame.from_ndarray(array, format="rgb24")
        frame.pts = frame_index
        frame_index += 1
        for packet in stream.encode(frame):
            output.mux(packet)

    if _is_raw_image(payload):
        frames = payload[:, 0] if camera == "main" else payload[:, 0, :, stream_index]
        for chunk, chunk_mask in zip(frames, mask):
            for frame in chunk[chunk_mask]:
                encode_frame(frame.numpy())
    else:
        for item, chunk_mask in zip(_encoded_items(payload), mask):
            encoded = item["data"][stream_index].detach().cpu().contiguous().numpy().tobytes()
            source = av.open(io.BytesIO(encoded), mode="r", format="mp4")
            keep = chunk_mask.tolist()
            for index, frame in enumerate(source.decode(video=0)):
                if index < len(keep) and keep[index]:
                    encode_frame(frame.to_ndarray(format="rgb24"))
            source.close()
    if stream is None:
        output.close()
        temporary.unlink(missing_ok=True)
        raise ValueError("Trajectory has no valid video frames")
    for packet in stream.encode():
        output.mux(packet)
    output.close()
    temporary.replace(output_path)


def _remux_camera(data: dict[str, Any], camera: str, output_path: Path) -> None:
    payload, stream_index = _camera_payload(data, camera)
    if _is_raw_image(payload):
        frames = payload[:, 0]
        if camera != "main":
            frames = frames[:, :, stream_index]
        frames = frames.reshape(-1, *frames.shape[-3:])
        temporary = output_path.with_suffix(".tmp.mp4")
        output = av.open(str(temporary), mode="w", format="mp4")
        stream = output.add_stream("libx264", rate=30)
        stream.width, stream.height = int(frames.shape[-2]), int(frames.shape[-3])
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "preset": "veryfast", "g": "30"}
        for index, frame in enumerate(frames):
            video_frame = av.VideoFrame.from_ndarray(frame.numpy(), format="rgb24")
            video_frame.pts = index
            for packet in stream.encode(video_frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
        output.close()
        temporary.replace(output_path)
        return
    items = _encoded_items(payload)
    temporary = output_path.with_suffix(".tmp.mp4")
    output = av.open(str(temporary), mode="w", format="mp4")
    out_stream = None
    frames_written = 0
    for item in items:
        encoded = item["data"][stream_index].detach().cpu().contiguous().numpy().tobytes()
        source = av.open(io.BytesIO(encoded), mode="r", format="mp4")
        in_stream = source.streams.video[0]
        if out_stream is None:
            out_stream = output.add_stream_from_template(in_stream)
        pts_offset = round(frames_written / int(item["fps"]) / in_stream.time_base)
        for packet in source.demux(in_stream):
            if packet.dts is None:
                continue
            packet.dts += pts_offset
            if packet.pts is not None:
                packet.pts += pts_offset
            packet.stream = out_stream
            output.mux(packet)
        time_dim = item["time_dim"]
        frames_written += int(item["shape"][int(time_dim)]) if time_dim is not None else 1
        source.close()
    output.close()
    temporary.replace(output_path)


@app.get("/")
def index() -> str:
    asset_version = max(
        (ROOT / "static" / "app.js").stat().st_mtime_ns,
        (ROOT / "static" / "styles.css").stat().st_mtime_ns,
        (ROOT / "static" / "review.js").stat().st_mtime_ns,
        (ROOT / "static" / "control.js").stat().st_mtime_ns,
    )
    return render_template("index.html", asset_version=asset_version)


@app.post("/api/load")
def load_trajectory() -> Response:
    try:
        body = request.get_json(force=True)
        return jsonify(_load(str(body.get("path", "")), body.get("episodeId")))
    except Exception as exc:
        return jsonify(error=str(exc)), 400


@app.post("/api/browse")
def browse_directory() -> Response:
    try:
        body = request.get_json(force=True)
        return jsonify(_browse_directory(str(body.get("path", ""))))
    except Exception as exc:
        return jsonify(error=str(exc)), 400


@app.get("/api/series")
def series() -> Response:
    try:
        data, _, _, info = _current()
        path = request.args.get("path", "")
        tensors = _flatten_tensors(data)
        if path not in tensors:
            raise ValueError("Unknown numeric field")
        chunks, steps = info["chunks"], info["stepsPerChunk"]
        timeline = _timeline_tensor(tensors[path], chunks, steps)
        if timeline is None:
            raise ValueError("Field cannot be aligned to the frame timeline")
        if request.args.get("valid") == "1":
            source_indices = _valid_indices(data)
            timeline = timeline.index_select(0, source_indices)
        else:
            source_indices = torch.arange(timeline.shape[0])
        components = int(np.prod(timeline.shape[1:]))
        flat = timeline.reshape(timeline.shape[0], components)
        if flat.dtype == torch.bool:
            flat = flat.to(torch.uint8)
        return jsonify(
            path=path,
            componentShape=list(timeline.shape[1:]),
            components=int(flat.shape[1]),
            values=flat.tolist(),
            sourceIndices=source_indices.tolist(),
        )
    except Exception as exc:
        return jsonify(error=str(exc)), 400


def _control_phases(data, info):
    """Resolve per-frame controller labels, independently of rewards/task phases."""
    n = info["totalFrames"]
    phases = torch.full((n,), -1, dtype=torch.int8)
    tensors = _flatten_tensors(data)

    def rows(name):
        tensor = tensors.get(name)
        if tensor is None:
            return None
        timeline = _timeline_tensor(tensor, info["chunks"], info["stepsPerChunk"])
        return timeline.reshape(n, -1) if timeline is not None else None

    # A per-frame flag takes priority over episode-level interval metadata.
    flags = rows("source_fields.complementary_info.is_intervention")
    if flags is None:
        flags = rows("intervene_flags")
    if flags is not None:
        known = torch.isfinite(flags).all(1)
        phases[known] = flags[known].ne(0).any(1).to(torch.int8)
        return phases

    context = info.get("reviewContext", {})
    policy = rows("source_fields.complementary_info.collector_policy_id")
    codebook = context.get("collectorPolicyCodebook", {})
    if policy is not None and codebook:
        for code, label in codebook.items():
            if label in {"human", "pretrained_model"}:
                phases[policy.eq(int(code)).all(1)] = int(label == "human")
        return phases
    episode = context.get("episode", {})
    if "human_intervention_intervals" in episode:
        phases[:] = 0
        for interval in episode["human_intervention_intervals"] or []:
            start, end = max(0, int(interval["start_frame"])), min(n, int(interval["end_frame"]))
            if start < end:
                phases[start:end] = 1
    return phases


@app.get("/api/control")
def control_phases() -> Response:
    try:
        data, _, _, info = _current()
        phases = _control_phases(data, info)
        source_indices = torch.arange(len(phases))
        if request.args.get("valid") == "1" and info["hasValidMask"]:
            source_indices = _valid_indices(data)
            phases = phases[source_indices]
        names = {-1: "unknown", 0: "inference", 1: "human"}
        segments = []
        for index, code in enumerate(phases.tolist()):
            name = names[code]
            if segments and segments[-1]["phase"] == name:
                segments[-1]["endFrame"] = index + 1
            else:
                segments.append({"startFrame": index, "endFrame": index + 1, "phase": name})
        return jsonify(
            segments=segments, totalFrames=len(phases), sourceIndices=source_indices.tolist(),
            counts={name: int(phases.eq(code).sum()) for code, name in names.items()},
        )
    except Exception as exc:
        return jsonify(error=str(exc)), 400


@app.get("/api/video/<camera>.mp4")
def video(camera: str) -> Response:
    try:
        data, _, key, info = _current()
        if camera not in info["cameras"]:
            raise ValueError("This trajectory has no video for the requested camera")
        valid_only = request.args.get("valid") == "1" and info["hasValidMask"]
        output_path = CACHE_DIR / f"{key}-{camera}{'-valid' if valid_only else ''}.mp4"
        with _lock:
            if not output_path.exists():
                if info["videoSource"] == "external":
                    keep = _valid_indices(data) if valid_only else torch.arange(info["totalFrames"])
                    _encode_external_camera(info["media"][camera], info["fps"], output_path, keep)
                elif valid_only:
                    _encode_filtered_camera(data, camera, output_path)
                else:
                    _remux_camera(data, camera, output_path)
        return send_file(output_path, mimetype="video/mp4", conditional=True)
    except Exception as exc:
        return jsonify(error=str(exc)), 400


@app.get("/api/frame")
def frame_values() -> Response:
    try:
        data, _, _, info = _current()
        index = max(0, int(request.args.get("index", "0")))
        chunks, steps = info["chunks"], info["stepsPerChunk"]
        if request.args.get("valid") == "1":
            indices = _valid_indices(data)
            if not len(indices):
                raise ValueError("Trajectory has no valid frames")
            display_index = min(index, len(indices) - 1)
            index = int(indices[display_index])
        else:
            display_index = min(index, chunks * steps - 1)
            index = display_index
        result = {}
        preferred = [
            "forward_inputs.substep_states", "forward_inputs.chunk_valid_mask",
            "rewards", "dones", "intervene_flags", "versions",
            "forward_inputs.rollout_control_source", "forward_inputs.mc_return",
            "curr_obs.states", "actions", "next_obs.states",
            "forward_inputs.episode_index", "forward_inputs.frame_index",
            "curr_obs.source_index", "curr_obs.video_frame_indices",
            "forward_inputs.collector_policy_id", "terminations", "truncations",
        ]
        tensors = _flatten_tensors(data)
        for path in preferred:
            if path not in tensors:
                continue
            timeline = _timeline_tensor(tensors[path], chunks, steps)
            if timeline is None:
                continue
            value = timeline[index]
            result[path] = value.item() if value.numel() == 1 else value.flatten().tolist()
        return jsonify(index=display_index, sourceIndex=index, chunk=index // steps, step=index % steps, values=result)
    except Exception as exc:
        return jsonify(error=str(exc)), 400


def _review_current():
    data, path, _, info = _current()
    if not request.args.get("key") or info.get("sourceType") != "lerobot":
        raise ValueError("请先打开原始 LeRobot 数据集并选择 episode。")
    context = info["reviewContext"]
    current_info, episodes = lerobot_source.dataset(path)
    episode = next((ep for ep in episodes if ep["episode_index"] == info["episodeId"]), None)
    if episode is None or lerobot_source.source_fingerprint(path, current_info, episode) != context["fingerprint"]:
        raise review.ReviewConflict("源 episode 已改变，请重新载入。")
    return data, path, info, context


def _review_response(path, info, context):
    entry = review.entry_for(path, context["entryName"], context["fingerprint"])
    return {"entry": entry, "annotationPath": str(path / review.REVIEW_FILE),
            "interventions": context["episode"].get("human_intervention_intervals", []),
            "totalFrames": info["totalFrames"], "trainingDirectory": str(path / training_export.DIRECTORY),
            "trainingExists": (path / training_export.DIRECTORY / training_export.MARKER).is_file()}


@app.get("/api/review")
def get_review():
    try:
        _, path, info, context = _review_current()
        return jsonify(_review_response(path, info, context))
    except Exception as exc:
        return jsonify(error=str(exc)), 409 if isinstance(exc, review.ReviewConflict) else 400


@app.post("/api/review/segments")
@app.put("/api/review/segments/<segment_id>")
def save_review_segment(segment_id=None):
    try:
        with _lock:
            data, path, info, context = _review_current()
            body = request.get_json(force=True)
            start, end = body.get("startFrame"), body.get("endFrameInclusive")
            if type(start) is not int or type(end) is not int or not 0 <= start < end < info["totalFrames"]:
                raise ValueError("请输入至少 2 帧的选区：0 ≤ 开始 < 结束 < episode 总帧数。")
            label = body.get("label", "usable_intervention")
            if label not in {"usable_intervention", "usable_motion", "recovery"}:
                raise ValueError("未知片段标签。")
            note = body.get("note", "")
            if not isinstance(note, str) or len(note) > 1000:
                raise ValueError("备注最多 1000 字。")
            outcome = body.get("outcome", "source")
            if outcome not in {"source", "success", "failure"}:
                raise ValueError("未知末端结果。")
            flags = data.get("intervene_flags")
            human = int(flags[start:end + 1].reshape(end - start + 1, -1).any(1).sum()) if flags is not None else 0
            segment = {"id": segment_id or uuid.uuid4().hex, "start_frame": start, "end_frame": end + 1,
                "source_ranges": [[start, end + 1]], "label": label, "note": note.strip(),
                "outcome": outcome,
                "frame_count": end - start + 1, "intervention_frames": human}
            review.update_entry(path, context["entryName"], context["fingerprint"], body.get("revision"),
                                lambda entry: review.add_segment(entry, segment, segment_id))
            return jsonify(_review_response(path, info, context))
    except Exception as exc:
        return jsonify(error=str(exc)), 409 if isinstance(exc, review.ReviewConflict) else 400


@app.delete("/api/review/segments/<segment_id>")
def delete_review_segment(segment_id):
    try:
        with _lock:
            _, path, info, context = _review_current()
            body = request.get_json(force=True)
            def remove(entry):
                segment = next((item for item in entry["segments"] if item["id"] == segment_id), None)
                if segment is None:
                    raise ValueError("片段不存在，请重新载入。")
                entry["segments"].remove(segment)
            review.update_entry(path, context["entryName"], context["fingerprint"], body.get("revision"), remove)
            return jsonify(_review_response(path, info, context))
    except Exception as exc:
        return jsonify(error=str(exc)), 409 if isinstance(exc, review.ReviewConflict) else 400


@app.post("/api/review/complete")
def complete_review():
    try:
        with _lock:
            _, path, info, context = _review_current()
            body = request.get_json(force=True)
            review.update_entry(path, context["entryName"], context["fingerprint"], body.get("revision"),
                                lambda entry: entry.update(reviewed=True))
            return jsonify(_review_response(path, info, context))
    except Exception as exc:
        return jsonify(error=str(exc)), 409 if isinstance(exc, review.ReviewConflict) else 400


@app.get("/api/review/manifest")
def download_review():
    try:
        _, path, _, _ = _review_current()
        contents = json.dumps(review.read_store(path), ensure_ascii=False, indent=2).encode()
        return send_file(io.BytesIO(contents), mimetype="application/json", as_attachment=True,
                         download_name=f"{path.name}_review_annotations.json")
    except Exception as exc:
        return jsonify(error=str(exc)), 400


@app.post("/api/review/export")
def export_review():
    try:
        _, path, _, _ = _review_current()
        with _lock:
            if any(job["status"] == "running" and job["source"] == str(path) for job in _exports.values()):
                raise ValueError("此数据集正在导出，请等待当前任务完成。")
            job_id = uuid.uuid4().hex
            job = {"status": "running", "source": str(path), "message": "准备导出已保存片段…"}
            _exports[job_id] = job
        def run():
            try:
                def progress(message):
                    with _lock:
                        job["message"] = message
                result = lerobot_source.export_dataset(path, _encode_external_camera, progress)
                with _lock:
                    job.update(status="complete", result=result, message="导出完成")
            except Exception as exc:
                with _lock:
                    job.update(status="failed", message=str(exc))
        threading.Thread(target=run, daemon=True).start()
        return jsonify(jobId=job_id)
    except Exception as exc:
        return jsonify(error=str(exc)), 400


@app.get("/api/review/export/<job_id>")
def export_review_status(job_id):
    with _lock:
        if job_id not in _exports:
            return jsonify(error="导出任务不存在，请检查服务是否重启。"), 404
        return jsonify(_exports[job_id])


if __name__ == "__main__":
    host = os.environ.get("REPLAY_VIEWER_HOST", "0.0.0.0")
    port = int(os.environ.get("REPLAY_VIEWER_PORT", "8088"))
    app.run(host=host, port=port, threaded=True)
