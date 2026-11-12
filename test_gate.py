"""Check human gate boundaries, exact pictures, and full-source exports."""

import hashlib
import io
import json
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

import gate_annotations as gate
import test_review as fixtures


class GateAnnotationTest(unittest.TestCase):
    setUp = fixtures.OriginalDatasetReviewTest.setUp
    load = fixtures.OriginalDatasetReviewTest.load

    def get_gate(self):
        response = self.client.get("/api/gate", query_string=self.query)
        self.assertEqual(response.status_code, 200, response.json)
        return response.json

    def wait_job(self, identifier):
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            result = self.client.get("/api/review/export/" + identifier).json
            if result["status"] != "running":
                return result
            time.sleep(0.02)
        self.fail("Gate export did not finish")

    def save_gate(self, **body):
        result = self.client.post(
            "/api/gate",
            query_string=self.query,
            json={"revision": self.get_gate()["revision"], **body},
        )
        self.assertEqual(result.status_code, 200, result.json)
        job = self.wait_job(result.json["jobId"])
        return result, job

    def test_exclusive_off_frame_full_episode_and_boundary_images(self):
        self.assertIsNone(self.get_gate()["entry"])
        _, job = self.save_gate(startFrame=1, endFrame=3, note="接近到穿过")
        self.assertEqual(job["status"], "complete", job)
        self.assertEqual(
            (job["result"]["frames"], job["result"]["positive_frames"]), (5, 2)
        )
        root = Path(job["result"]["directory"])
        manifest = json.loads((root / "labels_manifest.json").read_text())
        self.assertEqual(
            len(manifest["episodes"]), 1
        )  # Unreviewed episode 0 is excluded.
        ep = manifest["episodes"][0]
        with np.load(root / ep["labels_file"]) as data:
            np.testing.assert_array_equal(data["labels"], [0, 1, 1, 0, 0])
            np.testing.assert_array_equal(data["source_index"], [5, 6, 7, 8, 9])
            np.testing.assert_array_equal(
                data["video_frame_indices"][:, 0], [5, 6, 7, 8, 9]
            )
            self.assertEqual(data["state"].shape, (5, 2))
        table = pq.read_table(root / ep["data_file"])
        self.assertEqual(
            table["action"].to_pylist(), [[i, i + 1] for i in range(5, 10)]
        )
        self.assertEqual(table["annotation.actor_gate"].to_pylist(), [0, 1, 1, 0, 0])
        for boundary, intensity in [("start", 130), ("end", 170)]:
            for path in ep["boundary_images"][boundary].values():
                with Image.open(root / path) as img:
                    self.assertAlmostEqual(
                        float(np.asarray(img).mean()), intensity, delta=5
                    )
        image = self.client.get(
            "/api/gate/image/main", query_string={**self.query, "frame": 3}
        )
        self.assertEqual(image.status_code, 200, image.json)
        self.assertAlmostEqual(
            float(np.asarray(Image.open(io.BytesIO(image.data))).mean()), 170, delta=5
        )
        self.load(0)
        self.assertIsNone(self.get_gate()["entry"])
        self.load(1)
        self.assertEqual(self.get_gate()["entry"]["start_frame"], 1)
        self.assertTrue(self.get_gate()["exportCurrent"])
        download = self.client.get("/api/gate/download/csv", query_string=self.query)
        self.assertEqual(download.status_code, 200)
        self.assertEqual(len(download.data.decode().splitlines()), 6)
        for path, digest in self.source_hashes.items():
            self.assertEqual(
                hashlib.sha256((self.root / path).read_bytes()).hexdigest(), digest
            )

    def test_validation_revision_stale_keys_and_no_200_frame_limit(self):
        self.assertEqual(int(gate.labels_for(500, 20, 400).sum()), 380)
        for start, end in [(0, 0), (3, 1), (-1, 2), (1, 5), (1.5, 3), (True, 3)]:
            response = self.client.post(
                "/api/gate",
                query_string=self.query,
                json={"revision": 0, "startFrame": start, "endFrame": end},
            )
            self.assertEqual(response.status_code, 400, response.json)
        _, job = self.save_gate(startFrame=0, endFrame=4)
        self.assertEqual(job["status"], "complete")
        stale = self.client.post(
            "/api/gate",
            query_string=self.query,
            json={"revision": 0, "startFrame": 1, "endFrame": 3},
        )
        self.assertEqual(stale.status_code, 409)
        old_key = self.query.copy()
        self.load(0)
        self.assertEqual(
            self.client.get(
                "/api/gate/image/main", query_string={**old_key, "frame": 0}
            ).status_code,
            400,
        )
        self.load(1)
        p = self.root / "meta/info.json"
        p.write_text(p.read_text() + "\n")
        self.assertEqual(
            self.client.get("/api/gate", query_string=self.query).status_code, 409
        )

    def test_explicit_negative_delete_and_existing_clip_annotations_are_separate(self):
        clip = self.client.post(
            "/api/review/segments",
            query_string=self.query,
            json={"revision": 0, "startFrame": 0, "endFrameInclusive": 1},
        )
        self.assertEqual(clip.status_code, 200)
        self.assertIsNone(self.get_gate()["entry"])
        self.save_gate(startFrame=1, endFrame=3)
        self.load(0)
        _, job = self.save_gate(mode="all_off")
        self.assertEqual(job["result"]["episodes"], 2)
        self.assertEqual(job["result"]["frames"], 10)
        with np.load(self.root / gate.DIRECTORY / "labels/episode_000000.npz") as data:
            self.assertFalse(data["labels"].any())
        response = self.client.delete(
            "/api/gate",
            query_string=self.query,
            json={"revision": self.get_gate()["revision"]},
        )
        job = self.wait_job(response.json["jobId"])
        self.assertEqual(job["result"]["episodes"], 1)
        self.assertFalse(
            (self.root / gate.DIRECTORY / "labels/episode_000000.npz").exists()
        )
        self.load(1)
        self.assertEqual(
            len(
                self.client.get("/api/review", query_string=self.query).json["entry"][
                    "segments"
                ]
            ),
            1,
        )

    def test_failed_export_preserves_previous_data_and_allows_retry(self):
        self.save_gate(startFrame=1, endFrame=3)
        folder = self.root / gate.DIRECTORY
        before = {
            str(p.relative_to(folder)): p.read_bytes()
            for p in folder.rglob("*")
            if p.is_file()
        }
        with patch.object(
            gate, "image_at", side_effect=ValueError("video decode failed")
        ):
            _, job = self.save_gate(startFrame=0, endFrame=4)
        self.assertEqual(job["status"], "failed")
        self.assertFalse(self.get_gate()["exportCurrent"])
        self.assertEqual(self.get_gate()["entry"]["start_frame"], 0)
        self.assertEqual(
            before,
            {
                str(p.relative_to(folder)): p.read_bytes()
                for p in folder.rglob("*")
                if p.is_file()
            },
        )
        response = self.client.post("/api/gate/export", query_string=self.query)
        self.assertEqual(self.wait_job(response.json["jobId"])["status"], "complete")
        self.assertTrue(self.get_gate()["exportCurrent"])
        self.assertFalse(list(self.root.glob(".actor-gate-export-*")))

    def test_unowned_output_is_never_overwritten(self):
        folder = self.root / gate.DIRECTORY
        folder.mkdir()
        (folder / "mine").write_text("keep")
        response = self.client.post(
            "/api/gate",
            query_string=self.query,
            json={"revision": 0, "startFrame": 1, "endFrame": 3},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual((folder / "mine").read_text(), "keep")
        self.assertFalse((self.root / gate.STORE).exists())


if __name__ == "__main__":
    unittest.main()
