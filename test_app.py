"""Regression checks for embedded and externally referenced replay media."""
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

import app as viewer


def video_bytes(values, fps=20):
    buffer = io.BytesIO()
    with av.open(buffer, 'w', format='mp4') as output:
        stream = output.add_stream('libx264', rate=fps)
        stream.width = stream.height = 32
        stream.pix_fmt = 'yuv420p'
        for i, value in enumerate(values):
            frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), value, np.uint8), format='rgb24')
            frame.pts = i
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    return buffer.getvalue()


def decode(response):
    contents = response.data
    response.close()
    with av.open(io.BytesIO(contents)) as source:
        return [frame.to_ndarray(format='rgb24') for frame in source.decode(video=0)]


class ReplayViewerTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / 'raw_replay').mkdir()
        cache = self.root / 'cache'
        cache.mkdir()
        self.cache_patch = patch.object(viewer, 'CACHE_DIR', cache)
        self.cache_patch.start()
        self.addCleanup(self.cache_patch.stop)
        self.client = viewer.app.test_client()
        self.data = {
            'forward_inputs': {},
            'actions': torch.arange(48, dtype=torch.float32).reshape(3, 1, 16),
            'rewards': torch.tensor([0., 0., 1.]).reshape(3, 1, 1),
            'intervene_flags': torch.zeros(3, 1, 16, dtype=torch.bool),
            'curr_obs': {'states': torch.arange(48, dtype=torch.float32).reshape(3, 1, 16)},
        }

    def load(self, filename='trajectory_0_test.pt'):
        path = self.root / 'raw_replay' / filename
        torch.save(self.data, path)
        response = self.client.post('/api/load', json={'path': str(path)})
        self.assertEqual(response.status_code, 200, response.json)
        return response.json

    def external(self):
        # Deliberately reorder camera columns and select nonzero, gapped, repeated frames.
        order = ['observation.images.cam_right_wrist', 'observation.images.cam_left_high', 'observation.images.cam_left_wrist']
        self.data['curr_obs']['video_frame_indices'] = torch.tensor([[[1, 2, 3]], [[4, 5, 6]], [[4, 7, 8]]])
        (self.root / 'source.mp4').write_bytes(video_bytes(range(0, 200, 20)))
        manifest = {'fps': 20, 'camera_order': order, 'media': [
            {'camera': name, 'path': 'source.mp4', 'fps': 20, 'num_frames': 10} for name in order
        ]}
        (self.root / 'source_manifest.json').write_text(json.dumps(manifest))

    def test_numeric_without_images_or_mask(self):
        meta = self.load()
        self.assertEqual((meta['totalFrames'], meta['validFrames']), (3, 3))
        self.assertFalse(meta['hasValidMask'])
        self.assertEqual(meta['cameras'], [])
        for mode in ['', '&valid=1']:
            result = self.client.get('/api/series?path=actions' + mode).json
            self.assertEqual(result['values'], self.data['actions'][:, 0].tolist())
            frame = self.client.get('/api/frame?index=2' + mode).json
            self.assertEqual(frame['values']['curr_obs.states'], list(range(32, 48)))
            self.assertEqual(frame['values']['rewards'], 1)

    def test_outcome_agrees_for_single_trajectory_and_directory(self):
        index = {}
        for i, (reward, terminal, expected) in enumerate([
            (1., True, True), (0., True, False),
            (1., False, None), (float('nan'), True, None), (3., True, None),
        ]):
            self.data['rewards'][-1] = reward
            self.data['terminations'] = torch.tensor([False, False, terminal]).reshape(3, 1, 1)
            meta = self.load(f'trajectory_{i}_test.pt')
            self.assertIs(meta['isSuccess'], expected)
            self.assertTrue(meta['outcomeSource'])
            index[str(i)] = {'trajectory_id': i, 'model_weights_id': 'test', 'num_samples': 3}
        (self.root / 'raw_replay/trajectory_index.json').write_text(json.dumps({'trajectory_index': index}))
        result = self.client.post('/api/browse', json={'path': str(self.root / 'raw_replay')})
        self.assertEqual(result.status_code, 200, result.json)
        self.assertEqual([row['isSuccess'] for row in result.json['trajectories']], [True, False, None, None, None])

    def test_outcome_respects_endpoint_validity_annotations_and_truncation(self):
        self.data['rewards'] = torch.tensor([1., 0., 0.]).reshape(3, 1, 1)
        self.data['terminations'] = torch.tensor([True, True, False]).reshape(3, 1, 1)
        self.assertIsNone(self.load()['isSuccess'])  # An earlier success does not label an unfinished ending.
        self.data['forward_inputs']['chunk_valid_mask'] = torch.tensor([True, True, False]).reshape(3, 1, 1)
        self.assertIs(self.load()['isSuccess'], False)
        self.data['forward_inputs']['reward_is_annotated'] = torch.tensor([True, False, False]).reshape(3, 1, 1)
        self.assertIsNone(self.load()['isSuccess'])
        self.data['forward_inputs']['reward_is_annotated'][:] = True
        self.data['truncations'] = torch.tensor([False, True, False]).reshape(3, 1, 1)
        self.assertIsNone(self.load()['isSuccess'])
        self.data['truncations'][:] = False
        self.data['forward_inputs']['chunk_valid_mask'][:] = False
        self.assertIsNone(self.load()['isSuccess'])

    def test_external_camera_mapping_and_frame_selection(self):
        self.external()
        meta = self.load()
        self.assertEqual(meta['videoSource'], 'external')
        self.assertEqual(meta['fps'], 20)
        for camera, expected in [('main', [40, 100, 140]), ('left', [60, 120, 160]), ('right', [20, 80, 80])]:
            response = self.client.get(f'/api/video/{camera}.mp4?valid=1')
            self.assertEqual(response.status_code, 200, response.json if response.is_json else '')
            frames = decode(response)
            self.assertEqual(len(frames), 3)
            for frame, value in zip(frames, expected):
                self.assertAlmostEqual(float(frame.mean()), value, delta=4)
        self.assertEqual(len(list((self.root / 'cache').glob('*.mp4'))), 3)
        self.client.get('/api/video/main.mp4').close()
        self.assertEqual(len(list((self.root / 'cache').glob('*.mp4'))), 3)

    def packaged(self):
        # Merged episode 5 was episode 0 in the second source. Camera files and
        # offsets differ, and the replay filename is deliberately unrelated.
        order = ['observation.images.cam_right_wrist', 'observation.images.cam_left_high', 'observation.images.cam_left_wrist']
        root = self.root / 'dataset'
        (root / 'meta/episodes/chunk-000').mkdir(parents=True)
        info = {'codebase_version': 'v3.0', 'fps': 20,
                'video_path': 'videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4',
                'features': {key: {'dtype': 'video', 'info': {'video.fps': 20}} for key in order}}
        rows = []
        for source_id, eid in enumerate([0, 5]):
            source = self.root / f'source-{source_id}.mp4'
            source.write_bytes(video_bytes([i * 20 + source_id * 30 for i in range(10)]))
            row = {'episode_index': eid, 'length': 5}
            for column, key in enumerate(order):
                path = root / f'videos/{key}/chunk-{source_id:03d}/file-{column:03d}.mp4'
                path.parent.mkdir(parents=True)
                path.symlink_to(source)
                row.update({f'videos/{key}/chunk_index': source_id,
                            f'videos/{key}/file_index': column,
                            f'videos/{key}/from_timestamp': (column + 1) / 20})
            rows.append(row)
        (root / 'meta/info.json').write_text(json.dumps(info))
        pq.write_table(pa.Table.from_pylist(rows), root / 'meta/episodes/chunk-000/file-000.parquet')
        (self.root / 'source_manifest.json').write_text(json.dumps({
            'fps': 20, 'camera_order': order, 'packaged_source_dataset': 'dataset',
            'source_dataset': '/not-the-packaged-source',
        }))
        self.data['forward_inputs'].update({
            'episode_index': torch.full((3, 1, 1), 5),
            'original_episode_index': torch.zeros(3, 1, 1, dtype=torch.int64),
            'frame_index': torch.tensor([0, 2, 3]).reshape(3, 1, 1),
        })
        self.data['curr_obs']['video_frame_indices'] = torch.tensor([[[1, 2, 3]], [[3, 4, 5]], [[4, 5, 6]]])

    def test_packaged_v3_camera_files_and_offsets(self):
        self.packaged()
        meta = self.load('trajectory_42_merged.pt')
        self.assertEqual(meta['videoSource'], 'external')
        self.assertEqual(meta['mediaMessage'], '')
        self.assertEqual(meta['fps'], 20)
        for camera, expected in [('main', [70, 110, 130]), ('left', [90, 130, 150]), ('right', [50, 90, 110])]:
            response = self.client.get(f'/api/video/{camera}.mp4')
            self.assertEqual(response.status_code, 200)
            frames = decode(response)
            self.assertEqual(len(frames), 3)
            for frame, value in zip(frames, expected):
                self.assertAlmostEqual(float(frame.mean()), value, delta=4)

    def test_packaged_v3_rejects_ambiguous_or_misaligned_frames(self):
        self.packaged()
        inputs = self.data['forward_inputs']
        for key, invalid in [
            ('episode_index', torch.tensor([5, 0, 5]).reshape(3, 1, 1)),
            ('episode_index', torch.full((3, 1, 1), 99)),
            ('frame_index', torch.tensor([0, 2, 5]).reshape(3, 1, 1)),
            ('frame_index', torch.tensor([0, 1, 3]).reshape(3, 1, 1)),
            ('frame_index', torch.tensor([0., 2., 3.]).reshape(3, 1, 1)),
            ('episode_index', None),
        ]:
            with self.subTest(key=key, invalid=invalid):
                original = inputs[key]
                inputs[key] = invalid
                meta = self.load()
                self.assertEqual(meta['cameras'], [])
                self.assertIn('外部视频无法关联', meta['mediaMessage'])
                self.assertEqual(self.client.get('/api/series?path=actions').status_code, 200)
                inputs[key] = original

    def test_packaged_v3_missing_video_still_shows_other_cameras(self):
        self.packaged()
        path = self.root / 'dataset/videos/observation.images.cam_left_high/chunk-001/file-001.mp4'
        path.unlink()
        meta = self.load()
        self.assertEqual(meta['cameras'], ['left', 'right'])
        self.assertIn('cam_left_high', meta['mediaMessage'])
        self.assertEqual(self.client.get('/api/series?path=actions').status_code, 200)

    def test_missing_source_keeps_numeric_data(self):
        self.external()
        (self.root / 'source.mp4').unlink()
        meta = self.load()
        self.assertEqual(meta['cameras'], [])
        self.assertTrue(meta['mediaMessage'])
        self.assertEqual(self.client.get('/api/series?path=rewards&valid=1').status_code, 200)
        self.assertEqual(self.client.get('/api/video/main.mp4').status_code, 400)

    def test_stale_requests_do_not_read_another_trajectory(self):
        previous = self.load()['key']
        self.data['actions'] += 1
        current = self.load('trajectory_1_test.pt')['key']
        self.assertNotEqual(previous, current)
        response = self.client.get('/api/series', query_string={'path': 'actions', 'key': previous})
        self.assertEqual(response.status_code, 400)
        self.assertIn('Trajectory changed', response.json['error'])

    def test_controller_phases_align_with_filtered_frames(self):
        self.data['intervene_flags'][1, 0, 5] = True
        self.data['forward_inputs']['chunk_valid_mask'] = torch.tensor([True, False, True]).reshape(3, 1, 1)
        meta = self.load()
        full = self.client.get('/api/control', query_string={'key': meta['key']}).json
        self.assertEqual([s['phase'] for s in full['segments']], ['inference', 'human', 'inference'])
        filtered = self.client.get('/api/control', query_string={'key': meta['key'], 'valid': 1}).json
        self.assertEqual(filtered['sourceIndices'], [0, 2])
        self.assertEqual(filtered['segments'], [{'startFrame': 0, 'endFrame': 2, 'phase': 'inference'}])
        self.assertEqual(filtered['counts']['human'], 0)

    def test_missing_controller_flags_are_unknown_not_inference(self):
        del self.data['intervene_flags']
        self.data['source_fields'] = {'complementary_info.phase': torch.ones(3, 1, 1)}
        meta = self.load()
        result = self.client.get('/api/control', query_string={'key': meta['key']})
        self.assertEqual(result.json['segments'], [{'startFrame': 0, 'endFrame': 3, 'phase': 'unknown'}])

    def test_controller_fallback_uses_declared_codebook_or_intervals(self):
        data = {'source_fields': {'complementary_info.collector_policy_id': torch.tensor([8, 2, 99]).reshape(3, 1, 1)}}
        info = {'totalFrames': 3, 'chunks': 3, 'stepsPerChunk': 1, 'reviewContext': {
            'collectorPolicyCodebook': {'8': 'pretrained_model', '2': 'human'},
        }}
        self.assertEqual(viewer._control_phases(data, info).tolist(), [0, 1, -1])
        info['reviewContext'] = {'episode': {'human_intervention_intervals': [{'start_frame': 1, 'end_frame': 3}]}}
        self.assertEqual(viewer._control_phases({}, info).tolist(), [0, 1, 1])
        self.data['intervene_flags'] = torch.tensor([0., float('nan'), 1.]).reshape(3, 1, 1)
        self.assertEqual(viewer._control_phases(self.data, info).tolist(), [0, -1, 1])

    def test_external_mask_and_invalid_indices(self):
        self.external()
        self.data['forward_inputs']['chunk_valid_mask'] = torch.tensor([True, False, True]).reshape(3, 1, 1)
        self.load()
        response = self.client.get('/api/video/main.mp4?valid=1')
        self.assertEqual(len(decode(response)), 2)
        self.assertEqual(self.client.get('/api/series?path=actions&valid=1').json['sourceIndices'], [0, 2])
        self.data['curr_obs']['video_frame_indices'][2, 0, 0] = 100
        meta = self.load()
        self.assertEqual(meta['cameras'], [])
        self.assertIn('exceeds source length', meta['mediaMessage'])

    def test_embedded_rgb_and_h264_with_filter(self):
        for h264 in [False, True]:
            with self.subTest(h264=h264):
                self.data['rewards'] = torch.tensor([0., 0., 0., 1.]).reshape(2, 1, 2)
                inputs = self.data['forward_inputs']
                inputs['chunk_valid_mask'] = torch.tensor([True, False, True, True]).reshape(2, 1, 2)
                inputs['substep_states'] = torch.arange(64, dtype=torch.float32).reshape(2, 1, 2, 16)
                if h264:
                    inputs['substep_main_images'] = {'__rlinf_h264_stack__': 1, 'dim': 0, 'items': [
                        {'__rlinf_h264__': 1, 'shape': [1, 2, 32, 32, 3], 'time_dim': 1,
                         'fps': 20, 'crf': 18, 'gop': 20,
                         'data': [torch.tensor(list(video_bytes([20, 40])), dtype=torch.uint8)]}
                        for _ in range(2)
                    ]}
                else:
                    inputs['substep_main_images'] = torch.zeros(2, 1, 2, 32, 32, 3, dtype=torch.uint8)
                meta = self.load()
                self.assertEqual((meta['totalFrames'], meta['validFrames']), (4, 3))
                self.assertEqual(meta['cameras'], ['main'])
                for query, count in [('', 4), ('?valid=1', 3)]:
                    response = self.client.get('/api/video/main.mp4' + query)
                    self.assertEqual(response.status_code, 200, response.json if response.is_json else '')
                    self.assertEqual(len(decode(response)), count)
                result = self.client.get('/api/series?path=forward_inputs.substep_states&valid=1').json
                self.assertEqual(result['sourceIndices'], [0, 2, 3])
                self.assertEqual(result['values'][1], list(range(32, 48)))


if __name__ == '__main__':
    unittest.main()
