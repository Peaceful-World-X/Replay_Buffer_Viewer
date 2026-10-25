#!/usr/bin/env python3
"""Web viewer for RLinf replay-buffer trajectory files."""

from __future__ import annotations

import hashlib
import io
import json
import os
import threading
from pathlib import Path
from typing import Any

import av
import numpy as np
import torch
from flask import Flask, Response, jsonify, render_template, request, send_file


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
    if value.ndim >= 2 and value.shape[0] == chunks:
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


def _current() -> tuple[dict[str, Any], Path, str]:
    with _lock:
        if _trajectory is None or _trajectory_path is None or _trajectory_key is None:
            raise RuntimeError("No trajectory is loaded")
        return _trajectory, _trajectory_path, _trajectory_key


def _load(path_text: str) -> dict[str, Any]:
    global _trajectory, _trajectory_path, _trajectory_key
    path = Path(path_text).expanduser().resolve()
    if not path.is_file() or path.suffix != ".pt":
        raise ValueError("Path must point to an existing .pt trajectory file")
    data = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(data, dict) or "forward_inputs" not in data:
        raise ValueError("The file is not a supported RLinf trajectory dictionary")
    substeps = data["forward_inputs"].get("substep_main_images")
    if _is_h264(substeps):
        items = _encoded_items(substeps)
        chunks, steps, fps = len(items), int(items[0]["shape"][1]), int(items[0]["fps"])
    elif _is_raw_image(substeps) and substeps.ndim == 6:
        chunks, steps, fps = int(substeps.shape[0]), int(substeps.shape[2]), 30
    else:
        raise ValueError("Unsupported replay image format; expected raw RGB or embedded H.264")
    key_source = f"{path}:{path.stat().st_size}:{path.stat().st_mtime_ns}"
    key = hashlib.sha256(key_source.encode()).hexdigest()[:20]
    with _lock:
        _trajectory, _trajectory_path, _trajectory_key = data, path, key
    mask = data["forward_inputs"].get("chunk_valid_mask")
    valid = int(mask.sum()) if isinstance(mask, torch.Tensor) else chunks * steps
    return {
        "path": str(path),
        "filename": path.name,
        "fileBytes": path.stat().st_size,
        "chunks": chunks,
        "stepsPerChunk": steps,
        "totalFrames": chunks * steps,
        "validFrames": valid,
        "fps": fps,
        "duration": chunks * steps / fps,
        "modelWeightsId": str(data.get("model_weights_id", "")),
        "tree": _describe(data),
        "series": _series_catalog(data, chunks, steps),
    }


def _browse_directory(path_text: str) -> dict[str, Any]:
    directory = Path(path_text).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError("Path must point to an existing replay-buffer directory")
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
        rewards = trajectory.get("rewards")
        terminations = trajectory.get("terminations")
        intervene_flags = trajectory.get("intervene_flags")
        is_success = None
        if isinstance(rewards, torch.Tensor) and isinstance(terminations, torch.Tensor):
            terminal_rewards = rewards[terminations.to(torch.bool)]
            if terminal_rewards.numel():
                is_success = bool((terminal_rewards > 0).any())
        trajectories.append(
            {
                "id": trajectory_id,
                "path": str(file_path),
                "filename": file_path.name,
                "fileBytes": file_path.stat().st_size,
                "numSamples": int(item.get("num_samples", 0)),
                "shape": item.get("shape", []),
                "modelWeightsId": weights_id,
                "isSuccess": is_success,
                "hasIntervention": bool(
                    isinstance(intervene_flags, torch.Tensor) and intervene_flags.any()
                ),
            }
        )
    if not trajectories:
        raise ValueError("Directory index contains no available trajectory files")
    return {
        "directory": str(directory),
        "metadata": metadata,
        "trajectories": trajectories,
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
    if not isinstance(mask, torch.Tensor):
        raise ValueError("Trajectory has no chunk_valid_mask")
    return torch.nonzero(mask[:, 0].reshape(-1).to(torch.bool), as_tuple=False).flatten()


def _encode_filtered_camera(
    data: dict[str, Any], camera: str, output_path: Path
) -> None:
    payload, stream_index = _camera_payload(data, camera)
    mask = data["forward_inputs"]["chunk_valid_mask"][:, 0].to(torch.bool)
    temporary = output_path.with_suffix(".tmp.mp4")
    output = av.open(str(temporary), mode="w", format="mp4")
    stream = None
    frame_index = 0

    def encode_frame(array: np.ndarray) -> None:
        nonlocal stream, frame_index
        if stream is None:
            stream = output.add_stream("libx264", rate=30)
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
    )
    return render_template("index.html", asset_version=asset_version)


@app.post("/api/load")
def load_trajectory() -> Response:
    try:
        body = request.get_json(force=True)
        return jsonify(_load(str(body.get("path", ""))))
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
        data, _, _ = _current()
        path = request.args.get("path", "")
        tensors = _flatten_tensors(data)
        if path not in tensors:
            raise ValueError("Unknown numeric field")
        substeps = data["forward_inputs"]["substep_main_images"]
        if _is_h264(substeps):
            main_items = _encoded_items(substeps)
            chunks, steps = len(main_items), int(main_items[0]["shape"][1])
        else:
            chunks, steps = int(substeps.shape[0]), int(substeps.shape[2])
        timeline = _timeline_tensor(tensors[path], chunks, steps)
        if timeline is None:
            raise ValueError("Field cannot be aligned to the frame timeline")
        if request.args.get("valid") == "1":
            source_indices = _valid_indices(data)
            timeline = timeline.index_select(0, source_indices)
        else:
            source_indices = torch.arange(timeline.shape[0])
        flat = timeline.reshape(timeline.shape[0], -1)
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


@app.get("/api/video/<camera>.mp4")
def video(camera: str) -> Response:
    try:
        data, _, key = _current()
        if camera not in {"main", "left", "right"}:
            raise ValueError("Unknown camera")
        valid_only = request.args.get("valid") == "1"
        output_path = CACHE_DIR / f"{key}-{camera}{'-valid' if valid_only else ''}.mp4"
        with _lock:
            if not output_path.exists():
                if valid_only:
                    _encode_filtered_camera(data, camera, output_path)
                else:
                    _remux_camera(data, camera, output_path)
        return send_file(output_path, mimetype="video/mp4", conditional=True)
    except Exception as exc:
        return jsonify(error=str(exc)), 400


@app.get("/api/frame")
def frame_values() -> Response:
    try:
        data, _, _ = _current()
        index = max(0, int(request.args.get("index", "0")))
        substeps = data["forward_inputs"]["substep_main_images"]
        if _is_h264(substeps):
            main_items = _encoded_items(substeps)
            chunks, steps = len(main_items), int(main_items[0]["shape"][1])
        else:
            chunks, steps = int(substeps.shape[0]), int(substeps.shape[2])
        if request.args.get("valid") == "1":
            indices = _valid_indices(data)
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


if __name__ == "__main__":
    host = os.environ.get("REPLAY_VIEWER_HOST", "0.0.0.0")
    port = int(os.environ.get("REPLAY_VIEWER_PORT", "8088"))
    app.run(host=host, port=port, threaded=True)
