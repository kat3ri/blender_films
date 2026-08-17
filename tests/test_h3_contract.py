"""P0 verification: retention prompt, fragments, H3 grid math, contract shape.

Host-side only -- no bpy. Run with: python -m pytest tests/ (or unittest).
"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from previs import bundle as B  # noqa: E402


def fixture_shot(duration=5.0, resolution=(960, 544)):
    return {
        "shot_id": "TEST_SEG01",
        "status": "blocked",
        "duration_seconds": duration,
        "fps": 12,
        "set": {"asset_id": "stone_tower"},
        "render": {"resolution": list(resolution), "fps": 12},
        "characters": [
            {"id": "mauryl", "asset_id": "mauryl",
             "start_position": [0, 0, 0], "start_facing_deg": 0,
             "actions": [
                 {"type": "walk_to", "position": [3, 0, 0], "start_t": 0.0, "end_t": 3.0},
                 {"type": "interact", "target_id": "hearth", "pose": "crouch",
                  "start_t": 3.0, "end_t": 5.0},
             ]},
        ],
        "props": [],
        "camera": {"lens_mm": 35, "moves": [
            {"type": "static", "position": [0, -6, 1.6], "target_id": "mauryl",
             "start_t": 0.0, "end_t": duration},
        ]},
        "notes": "nightly routine, unhurried",
    }


class FakeLibrary:
    """Just enough of AssetLibrary for prompt building."""
    def __init__(self):
        self.data = {
            ("characters", "mauryl"): {
                "display_name": "Mauryl",
                "notes": "Elderly, thin and wiry, long grey hair and beard, "
                         "frayed brown robe, tall wooden staff.",
                "height_m": 1.77,
            },
            ("sets", "stone_tower"): {
                "display_name": "a decaying stone tower interior",
                "notes": "Crumbling dark granite, worn stone stairs, a hearth "
                         "with a low fire.",
            },
        }
        self.missing = []

    def get(self, kind, asset_id):
        return self.data.get((kind, asset_id), {})


class TestGridMath(unittest.TestCase):
    PROFILE = B.GENERATOR_PROFILES["minimax"]

    def test_five_seconds_warns_and_suggests(self):
        out = B.check_target_constraints(fixture_shot(5.0), self.PROFILE)
        self.assertEqual(out["resampled_frames"], 120)
        self.assertEqual(out["kept_frames"], 107)
        self.assertAlmostEqual(out["kept_duration_s"], 107 / 24, places=6)
        self.assertEqual(len(out["warnings"]), 1)
        self.assertIn("4.458s", out["warnings"][0])
        self.assertIn("5.167s", out["warnings"][0])

    def test_on_grid_duration_passes(self):
        out = B.check_target_constraints(fixture_shot(124 / 24), self.PROFILE)
        self.assertEqual(out["kept_frames"], out["resampled_frames"])
        self.assertEqual(out["warnings"], [])

    def test_eight_seconds_is_exact(self):
        out = B.check_target_constraints(fixture_shot(8.0), self.PROFILE)
        self.assertEqual(out["resampled_frames"], 192)  # 192 = 17*11 + 5
        self.assertEqual(out["kept_frames"], 192)
        self.assertEqual(out["warnings"], [])

    def test_bad_canvas_warns_with_suggestion(self):
        out = B.check_target_constraints(fixture_shot(8.0, resolution=(960, 540)),
                                         self.PROFILE)
        self.assertFalse(out["canvas_valid"])
        self.assertIn("960x544", out["warnings"][0])

    def test_good_canvas_passes(self):
        out = B.check_target_constraints(fixture_shot(8.0), self.PROFILE)
        self.assertTrue(out["canvas_valid"])


class TestRetentionPrompt(unittest.TestCase):
    def build(self, duration=5.0):
        shot = fixture_shot(duration)
        lib = FakeLibrary()
        return B.build_prompt_fragments(shot, {}, lib, [], "minimax"), shot

    def test_literal_tokens(self):
        frags, _ = self.build()
        text = B._build_retention_prompt(frags)
        self.assertIn("<Video 1>", text)
        self.assertIn("<Picture 1>", text)
        self.assertNotIn("Video1", text)          # the failed phrasing

    def test_replace_retain_imperatives(self):
        frags, _ = self.build()
        joined = " ".join(frags["retain_lines"] + frags["replace_lines"])
        self.assertIn("Retain the character poses and camera motion from <Video 1>", joined)
        self.assertIn("Replace the figure from <Video 1>", joined)
        self.assertIn("with the character Mauryl from <Picture 1>", joined)
        self.assertIn("Replace the grey geometry with a decaying stone tower interior",
                      joined)

    def test_one_replace_line_per_character_plus_set(self):
        frags, shot = self.build()
        self.assertEqual(len(frags["replace_lines"]),
                         len(shot["characters"]) + 1)

    def test_prompt_rendered_from_fragments(self):
        # the text file and the fragments can never drift: text is a pure
        # function of the fragments
        frags, _ = self.build()
        a = B._build_retention_prompt(frags)
        b = B._build_retention_prompt(json.loads(json.dumps(frags)))
        self.assertEqual(a, b)

    def test_kept_duration_reaches_shot_data(self):
        frags, _ = self.build(5.0)
        self.assertAlmostEqual(frags["shot_data"]["kept_duration_s"], 107 / 24, places=6)
        text = B._build_retention_prompt(frags)
        self.assertIn("keeps only the first 4.458s", text)

    def test_non_retention_profiles_unchanged(self):
        shot = fixture_shot(5.0)
        lib = FakeLibrary()
        text = B.build_prompt(shot, {}, lib, [], "seedance")
        self.assertIn("[REFERENCE USE]", text)
        self.assertNotIn("[RETENTION]", text)

    def test_minimax_duration_cap(self):
        self.assertEqual(B.GENERATOR_PROFILES["minimax"]["max_duration_s"], 15.0)


class TestDeterminism(unittest.TestCase):
    def test_fragments_are_deterministic(self):
        shot = fixture_shot()
        lib = FakeLibrary()
        a = B.build_prompt_fragments(shot, {}, lib, [], "minimax")
        b = B.build_prompt_fragments(shot, {}, lib, [], "minimax")
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))


if __name__ == "__main__":
    unittest.main()


class TestRadialRepeat(unittest.TestCase):
    """The one in-code TODO: radial copies could not face outward."""

    def test_linear_repeat_has_no_yaw(self):
        from previs.asset_library import _repeat_positions
        out = _repeat_positions([0, 0, 1], {"axis": "x", "count": 3, "spacing": 1.0})
        self.assertEqual([yaw for _, yaw in out], [0.0, 0.0, 0.0])
        self.assertEqual([p[0] for p, _ in out], [-1.0, 0.0, 1.0])

    def test_radial_defaults_to_no_rotation(self):
        from previs.asset_library import _repeat_positions
        out = _repeat_positions([0, 0, 2], {"mode": "radial", "count": 4, "radius": 2.0})
        self.assertEqual([yaw for _, yaw in out], [0.0] * 4)

    def test_face_outward_yaws_to_tangent(self):
        from previs.asset_library import _repeat_positions
        out = _repeat_positions([0, 0, 2], {"mode": "radial", "count": 4,
                                            "radius": 2.0, "face_outward": True})
        self.assertEqual([round(yaw, 3) for _, yaw in out], [0.0, 90.0, 180.0, 270.0])

    def test_face_outward_composes_onto_part_rotation(self):
        from previs.asset_library import _expand_part_repeats
        parts = _expand_part_repeats([{
            "shape": "box", "position": [0, 0, 2], "rotation_deg": [0, 0, 15],
            "repeat": {"mode": "radial", "count": 4, "radius": 2.0, "face_outward": True},
        }])
        self.assertEqual([p["rotation_deg"][2] for p in parts], [15.0, 105.0, 195.0, 285.0])
