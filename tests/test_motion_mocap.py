"""Mocap overlay track-building: a clip riding on a walk must not corrupt keys."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from previs.motion import build_character_track  # noqa: E402


def _resolve_point(*_args):
    return [0.0, 0.0, 0.0]


class TestMocapOverlay(unittest.TestCase):
    """mocap_clip overlapping walk_to is a limb overlay, not a root action."""

    def _walking_track(self):
        char = {
            "id": "walker",
            "start_position": [-3.0, 2.0, 0.0],
            "start_facing_deg": 0.0,
            "actions": [
                {
                    "type": "walk_to",
                    "start_t": 0.5,
                    "end_t": 4.5,
                    "position": [2.5, -1.0, 0.0],
                    
                },
                {
                    "type": "mocap_clip",
                    "start_t": 0.5,
                    "end_t": 4.5,
                    "clip_id": "SnapMoGen/renamed_bvhs/dummy",
                    "root_mode": "lock_xy",
                },
            ],
        }
        return build_character_track(char, _resolve_point, 5.166)

    def test_keys_stay_time_sorted(self):
        # The original bug: mocap_clip appended root keys *after* the walk's
        # later keys, breaking the sorted-by-time invariant sample() assumes.
        track = self._walking_track()
        ts = [k["t"] for k in track.keys]
        self.assertEqual(ts, sorted(ts))

    def test_root_still_travels_and_pose_is_walk(self):
        track = self._walking_track()
        position, _facing, pose = track.sample(2.5)
        self.assertEqual(pose, "stand")
        self.assertGreater(position[0], -1.0)  # well past the start

    def test_segment_registered_for_overlay(self):
        track = self._walking_track()
        segment = track.mocap_segment_at(2.5)
        self.assertIsNotNone(segment)
        self.assertEqual(segment["root_mode"], "lock_xy")

    def test_standalone_clip_still_holds_root_and_pose(self):
        char = {
            "id": "sitter",
            "start_position": [1.0, 1.0, 0.0],
            "actions": [
                {
                    "type": "mocap_clip",
                    "start_t": 1.0,
                    "end_t": 4.0,
                    "clip_id": "SnapMoGen/renamed_bvhs/dummy",
                    
                    "root_mode": "from_clip",
                }
            ],
        }
        track = build_character_track(char, _resolve_point, 5.0)
        ts = [k["t"] for k in track.keys]
        self.assertEqual(ts, sorted(ts))
        self.assertEqual(track.sample(2.5)[2], "stand")


if __name__ == "__main__":
    unittest.main()
