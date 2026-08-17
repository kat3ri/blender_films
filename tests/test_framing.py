"""P1.1: framing presets are pure math and compile to ordinary moves."""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from previs import framing as F  # noqa: E402


class FakeTrack:
    def __init__(self, position, facing_deg=0.0):
        self.position, self.facing_deg = list(position), facing_deg

    def sample(self, t):
        return list(self.position), self.facing_deg, "stand"


def shot_with(preset, extra_chars=()):
    characters = [{"id": "hero", "asset_id": "hero"}]
    characters.extend({"id": cid, "asset_id": cid} for cid in extra_chars)
    return {
        "shot_id": "T", "duration_seconds": 5.0,
        "characters": characters,
        "camera": {"moves": [dict(preset, type="preset")]},
    }


class FakeLib:
    missing = []
    def get(self, kind, asset_id):
        return {"display_name": asset_id, "height_m": 1.77}


def expand(preset, tracks, extra_chars=()):
    shot = shot_with(preset, extra_chars)
    n = F.expand_presets(shot, tracks, FakeLib())
    return shot["camera"]["moves"][0], n


class TestPresets(unittest.TestCase):
    def setUp(self):
        self.tracks = {"hero": FakeTrack([0, 0, 0]),
                       "foil": FakeTrack([2, 0, 0])}

    def test_expands_to_an_ordinary_move_type(self):
        move, n = expand({"name": "single_med", "subject_id": "hero",
                          "start_t": 0, "end_t": 5}, self.tracks)
        self.assertEqual(n, 1)
        self.assertEqual(move["type"], "static")
        self.assertNotEqual(move["type"], "preset")
        self.assertEqual(move["_preset"], "single_med")

    def test_distance_matches_the_named_size(self):
        move, _ = expand({"name": "single_cu", "subject_id": "hero",
                          "start_t": 0, "end_t": 5}, self.tracks)
        d = math.dist(move["position"][:2], [0, 0])
        self.assertAlmostEqual(d, F.SHOT_DISTANCES["cu"], places=6)

    def test_default_bearing_is_downstage(self):
        move, _ = expand({"name": "single_med", "subject_id": "hero",
                          "start_t": 0, "end_t": 5}, self.tracks)
        # -90deg bearing => camera at -Y of the subject
        self.assertAlmostEqual(move["position"][0], 0.0, places=6)
        self.assertLess(move["position"][1], 0.0)

    def test_low_angle_is_below_eye_high_is_above(self):
        low, _ = expand({"name": "low_angle", "subject_id": "hero",
                         "start_t": 0, "end_t": 5}, self.tracks)
        high, _ = expand({"name": "high_angle", "subject_id": "hero",
                          "start_t": 0, "end_t": 5}, self.tracks)
        self.assertLess(low["position"][2], high["position"][2])

    def test_dutch_sets_roll(self):
        move, _ = expand({"name": "dutch", "subject_id": "hero",
                          "start_t": 0, "end_t": 5, "roll_deg": 15}, self.tracks)
        self.assertEqual(move["roll_deg"], 15.0)

    def test_ots_aims_at_the_other_subject(self):
        move, _ = expand({"name": "ots", "subject_id": "hero", "other_id": "foil",
                          "start_t": 0, "end_t": 5}, self.tracks, ("foil",))
        self.assertEqual(move["target_id"], "foil")
        # hero at x=0, foil at x=2: an over-the-shoulder on hero looking at
        # foil puts the camera BEHIND hero, i.e. further from foil (x < 0)
        self.assertLess(move["position"][0], 0.0)

    def test_two_shot_aims_between_them(self):
        move, _ = expand({"name": "two_shot", "subject_id": "hero", "other_id": "foil",
                          "start_t": 0, "end_t": 5}, self.tracks, ("foil",))
        self.assertNotIn("target_id", move)
        self.assertAlmostEqual(move["target_position"][0], 1.0, places=6)

    def test_push_in_ends_closer_than_it_starts(self):
        move, _ = expand({"name": "push_in", "subject_id": "hero",
                          "start_t": 0, "end_t": 5}, self.tracks)
        self.assertEqual(move["type"], "dolly")
        start = math.dist(move["position"][:2], [0, 0])
        end = math.dist(move["end_position"][:2], [0, 0])
        self.assertLess(end, start)

    def test_pull_back_is_the_reverse(self):
        move, _ = expand({"name": "pull_back", "subject_id": "hero",
                          "start_t": 0, "end_t": 5}, self.tracks)
        start = math.dist(move["position"][:2], [0, 0])
        end = math.dist(move["end_position"][:2], [0, 0])
        self.assertGreater(end, start)

    def test_unknown_preset_raises(self):
        with self.assertRaises(ValueError):
            expand({"name": "crash_zoom", "subject_id": "hero",
                    "start_t": 0, "end_t": 5}, self.tracks)

    def test_unknown_subject_raises(self):
        with self.assertRaises(ValueError):
            expand({"name": "single_med", "subject_id": "ghost",
                    "start_t": 0, "end_t": 5}, self.tracks)

    def test_ots_requires_other_id(self):
        with self.assertRaises(ValueError):
            expand({"name": "ots", "subject_id": "hero",
                    "start_t": 0, "end_t": 5}, self.tracks)

    def test_expansion_is_deterministic(self):
        a, _ = expand({"name": "ots", "subject_id": "hero", "other_id": "foil",
                       "start_t": 0, "end_t": 5}, self.tracks, ("foil",))
        b, _ = expand({"name": "ots", "subject_id": "hero", "other_id": "foil",
                       "start_t": 0, "end_t": 5}, self.tracks, ("foil",))
        self.assertEqual(a, b)

    def test_non_preset_moves_pass_through_untouched(self):
        shot = {"shot_id": "T", "duration_seconds": 5.0, "characters": [],
                "camera": {"moves": [{"type": "static", "position": [0, -5, 1.6],
                                      "start_t": 0, "end_t": 5}]}}
        before = [dict(m) for m in shot["camera"]["moves"]]
        n = F.expand_presets(shot, self.tracks, FakeLib())
        self.assertEqual(n, 0)
        self.assertEqual(shot["camera"]["moves"], before)


class TestRollBaking(unittest.TestCase):
    """roll_deg must reach the camera euler -- look-at alone can't imply it."""

    def test_roll_reaches_the_key_rotation(self):
        from previs.motion import build_camera_keys
        shot = {
            "shot_id": "T", "duration_seconds": 1.0, "fps": 4,
            "characters": [], "stage": {"size_m": [10, 10]},
            "camera": {"moves": [{
                "type": "static", "position": [0, -4, 1.6],
                "target_position": [0, 0, 1.4],
                "roll_deg": 12.0, "start_t": 0.0, "end_t": 1.0}]},
        }
        keys = build_camera_keys(shot, {}, FakeLib(), 4)
        self.assertAlmostEqual(keys[0].rotation_euler[1], math.radians(12.0), places=6)

    def test_no_roll_stays_level(self):
        from previs.motion import build_camera_keys
        shot = {
            "shot_id": "T", "duration_seconds": 1.0, "fps": 4,
            "characters": [], "stage": {"size_m": [10, 10]},
            "camera": {"moves": [{
                "type": "static", "position": [0, -4, 1.6],
                "target_position": [0, 0, 1.4],
                "start_t": 0.0, "end_t": 1.0}]},
        }
        keys = build_camera_keys(shot, {}, FakeLib(), 4)
        self.assertEqual(keys[0].rotation_euler[1], 0.0)


if __name__ == "__main__":
    unittest.main()
