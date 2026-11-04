"""Exercise review persistence and native LeRobot cropping through the viewer API."""

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import torch

import app
import lerobot_source
import review
import training_export
from test_app import decode, video_bytes


class OriginalDatasetReviewTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.parent = Path(self.tmp.name)
        self.root = self.parent / "original"
        (self.root / "meta/episodes/chunk-000").mkdir(parents=True)
        (self.root / "data/chunk-000").mkdir(parents=True)
        self.camera_keys = [
            "observation.images.cam_left_high",
            "observation.images.cam_left_wrist",
            "observation.images.cam_right_wrist",
        ]
        features = {
            key: {"dtype": dtype, "shape": shape}
            for key, dtype, shape in [
                ("action", "float32", [2]),
                ("observation.state", "float32", [2]),
                ("complementary_info.is_intervention", "float32", [1]),
                ("timestamp", "float32", [1]),
                ("frame_index", "int64", [1]),
                ("episode_index", "int64", [1]),
                ("index", "int64", [1]),
                ("task_index", "int64", [1]),
            ]
        }
        for key in self.camera_keys:
            features[key] = {
                "dtype": "video",
                "shape": [32, 32, 3],
                "info": {"video.fps": 20},
            }
            path = self.root / f"videos/{key}/chunk-000/file-000.mp4"
            path.parent.mkdir(parents=True)
            path.write_bytes(
                video_bytes([10, 30, 50, 70, 90, 110, 130, 150, 170, 190], fps=20)
            )
        info = {
            "codebase_version": "v3.0",
            "fps": 20,
            "total_frames": 10,
            "total_episodes": 2,
            "total_tasks": 1,
            "robot_type": "test",
            "chunks_size": 1000,
            "features": features,
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        }
        (self.root / "meta/info.json").write_text(json.dumps(info))
        rows = []
        for eid in range(2):
            row = {
                "episode_index": eid,
                "length": 5,
                "tasks": ["task"],
                "data/chunk_index": 0,
                "data/file_index": 0,
                "dataset_from_index": eid * 5,
                "dataset_to_index": (eid + 1) * 5,
                "episode_success": "success",
                "human_intervention_intervals": [{"start_frame": 2, "end_frame": 5}],
                "rl_intervals": [
                    {
                        "start_frame": 1,
                        "end_frame": 5,
                        "outcome": "success",
                        "autonomous_success": False,
                    }
                ],
            }
            for key in self.camera_keys:
                row.update(
                    {
                        f"videos/{key}/chunk_index": 0,
                        f"videos/{key}/file_index": 0,
                        f"videos/{key}/from_timestamp": eid * 0.25,
                        f"videos/{key}/to_timestamp": (eid + 1) * 0.25,
                    }
                )
            rows.append(row)
        pq.write_table(
            pa.Table.from_pylist(rows),
            self.root / "meta/episodes/chunk-000/file-000.parquet",
        )
        table = pa.table(
            {
                "action": pa.array(
                    [[i, i + 1] for i in range(10)], type=pa.list_(pa.float32(), 2)
                ),
                "observation.state": pa.array(
                    [[i + 0.1, i + 0.2] for i in range(10)],
                    type=pa.list_(pa.float32(), 2),
                ),
                "complementary_info.is_intervention": pa.array(
                    [0, 0, 1, 1, 1] * 2, type=pa.float32()
                ),
                "timestamp": pa.array(
                    [i / 20 for i in range(5)] * 2, type=pa.float32()
                ),
                "frame_index": list(range(5)) * 2,
                "episode_index": [0] * 5 + [1] * 5,
                "index": list(range(10)),
                "task_index": [0] * 10,
            }
        )
        pq.write_table(table, self.root / "data/chunk-000/file-000.parquet")
        pq.write_table(
            pa.table({"task_index": [0], "task": ["task"]}),
            self.root / "meta/tasks.parquet",
        )
        self.source_hashes = {
            str(p.relative_to(self.root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in self.root.rglob("*")
            if p.is_file()
        }
        self.client = app.app.test_client()
        cache = self.parent / "cache"
        cache.mkdir()
        self.cache_patch = patch.object(app, "CACHE_DIR", cache)
        self.cache_patch.start()
        self.addCleanup(self.cache_patch.stop)
        self.load(1)

    def load(self, eid):
        result = self.client.post(
            "/api/load", json={"path": str(self.root), "episodeId": eid}
        )
        self.assertEqual(result.status_code, 200, result.json)
        self.meta = result.json
        self.query = {"key": self.meta["key"]}
        self.state = self.client.get("/api/review", query_string=self.query).json
        return result.json

    def save(self, start, end, **options):
        result = self.client.post(
            "/api/review/segments",
            query_string=self.query,
            json={
                "startFrame": start,
                "endFrameInclusive": end,
                "revision": self.state["entry"]["revision"],
                **options,
            },
        )
        if result.status_code == 200:
            self.state = result.json
        return result

    def test_browse_full_episodes_and_video_offsets(self):
        result = self.client.post("/api/browse", json={"path": str(self.root)})
        self.assertEqual([t["numSamples"] for t in result.json["trajectories"]], [5, 5])
        self.assertEqual(self.meta["totalFrames"], 5)
        self.assertNotIn("rewards", [s["path"] for s in self.meta["series"]])
        values = self.client.get(
            "/api/series", query_string={**self.query, "path": "actions"}
        ).json["values"]
        self.assertEqual(values[0], [5, 6])
        frames = decode(self.client.get("/api/video/main.mp4", query_string=self.query))
        self.assertEqual(len(frames), 5)
        self.assertAlmostEqual(float(frames[0].mean()), 110, delta=4)

    def test_original_episode_outcomes_match_list_and_selected_episode(self):
        self.assertIs(self.meta['isSuccess'], True)
        path = self.root / 'meta/episodes/chunk-000/file-000.parquet'
        rows = pq.read_table(path).to_pylist()
        rows[0]['episode_success'] = 'failure'
        rows[1]['episode_success'] = 'unknown'
        pq.write_table(pa.Table.from_pylist(rows), path)
        result = self.client.post('/api/browse', json={'path': str(self.root)})
        self.assertEqual(result.status_code, 200, result.json)
        self.assertEqual([row['isSuccess'] for row in result.json['trajectories']], [False, None])
        self.assertIs(self.load(0)['isSuccess'], False)
        self.assertIsNone(self.load(1)['isSuccess'])

    def test_control_uses_frame_flags_when_interval_metadata_is_empty(self):
        path = self.root / "meta/episodes/chunk-000/file-000.parquet"
        rows = pq.read_table(path).to_pylist()
        rows[1]["human_intervention_intervals"] = []
        pq.write_table(pa.Table.from_pylist(rows), path)
        self.load(1)
        result = self.client.get("/api/control", query_string=self.query)
        self.assertEqual(result.status_code, 200, result.json)
        self.assertEqual(
            result.json["segments"],
            [
                {"startFrame": 0, "endFrame": 2, "phase": "inference"},
                {"startFrame": 2, "endFrame": 5, "phase": "human"},
            ],
        )
        self.assertEqual(
            result.json["counts"], {"inference": 2, "human": 3, "unknown": 0}
        )
        self.assertNotIn("rewards", [s["path"] for s in self.meta["series"]])

    def test_persistence_validation_revision_and_episode_isolation(self):
        self.assertEqual(self.save(4, 2).status_code, 400)
        self.assertEqual(self.save(-1, 2).status_code, 400)
        self.assertEqual(self.save(1.5, 2).status_code, 400)
        self.assertEqual(self.save(0, 5).status_code, 400)
        self.assertEqual(self.save(1, 3, note="人工复核").status_code, 200)
        segment = self.state["entry"]["segments"][0]
        self.assertEqual(
            (
                segment["start_frame"],
                segment["end_frame"],
                segment["frame_count"],
                segment["intervention_frames"],
            ),
            (1, 4, 3, 2),
        )
        self.assertEqual(self.save(3, 4).status_code, 400)
        conflict = self.client.post(
            "/api/review/complete", query_string=self.query, json={"revision": 0}
        )
        self.assertEqual(conflict.status_code, 409)
        self.load(0)
        self.assertEqual(self.state["entry"]["segments"], [])
        self.load(1)
        self.assertEqual(self.state["entry"]["segments"][0]["note"], "人工复核")
        saved = self.client.get("/api/review/manifest", query_string=self.query)
        self.assertEqual(saved.status_code, 200)
        self.assertIn("episode_1", json.loads(saved.data)["trajectories"])
        stale_key = self.query.copy()
        self.load(0)
        self.assertEqual(
            self.client.post(
                "/api/review/segments", query_string=stale_key, json={}
            ).status_code,
            400,
        )
        self.load(1)
        delete = self.client.delete(
            "/api/review/segments/" + segment["id"],
            query_string=self.query,
            json={"revision": self.state["entry"]["revision"]},
        )
        self.assertEqual(delete.status_code, 200)
        self.assertEqual(delete.json["entry"]["segments"], [])
        for name, digest in self.source_hashes.items():
            self.assertEqual(
                hashlib.sha256((self.root / name).read_bytes()).hexdigest(), digest
            )

    def test_export_clips_preserves_fields_and_rebuilds_video_and_metadata(self):
        self.save(0, 1, note="保留前两帧")
        self.save(3, 4, label="recovery", outcome="success")
        result = lerobot_source.export_dataset(self.root, app._encode_external_camera)
        output = Path(result["directory"])
        self.assertEqual(output, self.root / "training_trajectories")
        self.assertEqual(
            (result["episodes"], result["frames"], result["transitions"]), (2, 4, 2)
        )
        info, episodes = lerobot_source.dataset(output)
        self.assertEqual(info["total_frames"], 4)
        self.assertEqual(
            [e["episode_success"] for e in episodes], ["unknown", "unknown"]
        )
        self.assertEqual(episodes[0]["rl_intervals"][0]["outcome"], "unknown")
        self.assertEqual(
            episodes[1]["human_intervention_intervals"],
            [{"start_frame": 0, "end_frame": 2}],
        )
        tables = [lerobot_source.table_for(output, info, ep) for ep in episodes]
        self.assertEqual(tables[0]["action"].to_pylist(), [[5, 6], [6, 7]])
        self.assertEqual(tables[1]["action"].to_pylist(), [[8, 9], [9, 10]])
        collection = self.client.post("/api/browse", json={"path": str(output)}).json
        self.assertEqual(collection["sourceType"], "training_replay")
        self.assertEqual(collection["sourceDirectory"], str(self.root))
        self.assertEqual(collection["metadata"]["total_samples"], 2)
        for identifier, value in [(0, 110), (1, 170)]:
            item = collection["trajectories"][identifier]
            saved = torch.load(item["path"], weights_only=False)
            offset = 5 if identifier == 0 else 8
            self.assertEqual(saved["actions"].tolist(), [[[offset, offset + 1]]])
            self.assertEqual(saved["curr_obs"]["source_index"].item(), offset)
            self.assertEqual(saved["next_obs"]["source_index"].item(), offset + 1)
            self.assertTrue(
                torch.allclose(
                    saved["next_obs"]["states"],
                    torch.tensor([[[offset + 1.1, offset + 1.2]]]),
                )
            )
            self.assertEqual(saved["rewards"].item(), identifier)
            self.assertEqual(saved["terminations"].item(), bool(identifier))
            response = self.client.post("/api/load", json={"path": item["path"]})
            self.assertEqual(response.status_code, 200, response.json)
            self.assertEqual(response.json["sourceType"], "training_replay")
            self.assertEqual(response.json["totalFrames"], 1)
            self.assertEqual(response.json["cameras"], ["main", "left", "right"])
            self.assertEqual(response.json["sourceEpisodeId"], 1)
            self.assertEqual(
                self.client.get(
                    "/api/review", query_string={"key": response.json["key"]}
                ).status_code,
                400,
            )
            for camera in ["main", "left", "right"]:
                frames = decode(
                    self.client.get(
                        f"/api/video/{camera}.mp4",
                        query_string={"key": response.json["key"]},
                    )
                )
                self.assertEqual(len(frames), 1)
                self.assertAlmostEqual(float(frames[0].mean()), value, delta=5)
        self.assertEqual(
            json.loads((output / "meta/stats.json").read_text())["action"]["count"], [4]
        )
        for name, digest in self.source_hashes.items():
            self.assertEqual(
                hashlib.sha256((self.root / name).read_bytes()).hexdigest(), digest
            )

    def test_edit_and_overwrite_then_remove_old_trajectories(self):
        self.save(0, 1)
        self.save(3, 4)
        initial = lerobot_source.export_dataset(self.root, app._encode_external_camera)
        output = Path(initial["directory"])
        original_key = self.meta["key"]
        preview = self.client.post("/api/load", json={"path": str(output)}).json
        self.load(1)
        first, second = self.state["entry"]["segments"]
        updated = self.client.put(
            "/api/review/segments/" + first["id"],
            query_string=self.query,
            json={
                "startFrame": 0,
                "endFrameInclusive": 2,
                "revision": self.state["entry"]["revision"],
                "outcome": "failure",
            },
        )
        self.assertEqual(updated.status_code, 200, updated.json)
        self.assertEqual(updated.json["entry"]["segments"][0]["id"], first["id"])
        removed = self.client.delete(
            "/api/review/segments/" + second["id"],
            query_string=self.query,
            json={"revision": updated.json["entry"]["revision"]},
        )
        self.assertEqual(removed.status_code, 200)
        self.client.post("/api/load", json={"path": str(output)})
        replaced = lerobot_source.export_dataset(self.root, app._encode_external_camera)
        self.assertEqual(replaced["directory"], initial["directory"])
        self.assertNotEqual(replaced["generation"], initial["generation"])
        self.assertEqual(len(list(output.glob("*.pt"))), 1)
        self.assertEqual(replaced["transitions"], 2)
        stale = self.client.get("/api/frame", query_string={"key": preview["key"]})
        self.assertEqual(stale.status_code, 400)
        self.assertIn("已更新", stale.json["error"])
        current = self.client.post("/api/load", json={"path": str(output)}).json
        self.assertEqual(current["totalFrames"], 2)
        self.load(1)
        self.assertEqual(self.meta["key"], original_key)
        self.assertEqual(self.state["entry"]["segments"][0]["end_frame"], 3)
        self.client.delete(
            "/api/review/segments/" + first["id"],
            query_string=self.query,
            json={"revision": self.state["entry"]["revision"]},
        )
        empty = lerobot_source.export_dataset(self.root, app._encode_external_camera)
        self.assertEqual(empty["transitions"], 0)
        self.assertEqual(list(output.glob("*.pt")), [])
        result = self.client.post("/api/browse", json={"path": str(output)})
        self.assertEqual(result.status_code, 200, result.json)
        self.assertEqual(result.json["trajectories"], [])
        self.assertEqual(result.json["sourceDirectory"], str(self.root))

    def test_reexport_invalidates_preview_with_identical_file_stats(self):
        self.save(0, 1)
        result = lerobot_source.export_dataset(self.root, app._encode_external_camera)
        output = Path(result["directory"])
        before = {
            p.relative_to(output): p.stat() for p in output.rglob("*") if p.is_file()
        }
        with patch.object(app, "_external_media", return_value=({}, 20, "")):
            preview = self.client.post("/api/load", json={"path": str(output)}).json
        self.load(1)
        segment = self.state["entry"]["segments"][0]
        self.client.put(
            "/api/review/segments/" + segment["id"],
            query_string=self.query,
            json={
                "startFrame": 1,
                "endFrameInclusive": 2,
                "revision": self.state["entry"]["revision"],
            },
        )
        lerobot_source.export_dataset(self.root, app._encode_external_camera)
        for name, stat in before.items():
            os.utime(output / name, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        # A numeric-only preview must refresh even when size and mtime are identical.
        for name, stat in before.items():
            if name.suffix == ".pt":
                self.assertEqual((output / name).stat().st_size, stat.st_size)
        with patch.object(app, "_external_media", return_value=({}, 20, "")):
            current = self.client.post("/api/load", json={"path": str(output)}).json
        self.assertNotEqual(current["key"], preview["key"])
        values = self.client.get(
            "/api/series", query_string={"key": current["key"], "path": "actions"}
        ).json["values"]
        self.assertEqual(values, [[6, 7]])

    def test_failed_replacement_keeps_previous_export_and_protects_unowned_directory(
        self,
    ):
        output = self.root / "training_trajectories"
        output.mkdir()
        (output / "user-file").write_text("must survive")
        self.save(1, 3)
        with self.assertRaises(ValueError):
            lerobot_source.export_dataset(self.root, app._encode_external_camera)
        self.assertEqual((output / "user-file").read_text(), "must survive")
        (output / "user-file").unlink()
        output.rmdir()
        initial = lerobot_source.export_dataset(self.root, app._encode_external_camera)
        before = {
            str(p.relative_to(output)): p.read_bytes()
            for p in output.rglob("*")
            if p.is_file()
        }
        rename = Path.rename

        def fail_publish(path, target):
            if path.name.startswith(".training-export-") and Path(target) == output:
                raise OSError("simulated publish failure")
            return rename(path, target)

        with patch.object(Path, "rename", fail_publish):
            with self.assertRaisesRegex(OSError, "publish failure"):
                lerobot_source.export_dataset(self.root, app._encode_external_camera)
        self.assertEqual(
            {
                str(p.relative_to(output)): p.read_bytes()
                for p in output.rglob("*")
                if p.is_file()
            },
            before,
        )
        self.assertEqual(
            training_export.read_marker(output)["generation"], initial["generation"]
        )

        def edit_during_export(_):
            segment = self.state["entry"]["segments"][0]
            self.client.put(
                "/api/review/segments/" + segment["id"],
                query_string=self.query,
                json={
                    "startFrame": 0,
                    "endFrameInclusive": 3,
                    "revision": self.state["entry"]["revision"],
                },
            )

        with self.assertRaises(review.ReviewConflict):
            lerobot_source.export_dataset(
                self.root, app._encode_external_camera, edit_during_export
            )
        self.assertEqual(
            training_export.read_marker(output)["generation"], initial["generation"]
        )
        self.assertFalse(list(self.root.glob(".training-export-*")))
        self.assertFalse(list(self.root.glob(".training-backup-*")))

    def test_changed_source_blocks_save_and_export(self):
        self.save(1, 2)
        path = self.root / "meta/info.json"
        path.write_text(path.read_text() + "\n")
        self.assertEqual(self.save(3, 4).status_code, 409)
        with self.assertRaises(review.ReviewConflict):
            lerobot_source.export_dataset(self.root, app._encode_external_camera)
        self.assertFalse(list(self.root.glob(".training-export-*")))


if __name__ == "__main__":
    unittest.main()
