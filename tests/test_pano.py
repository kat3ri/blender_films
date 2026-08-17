"""Camera-driven background plates: FOV, sampling convention, segmentation."""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from previs import pano as P  # noqa: E402


def track(yaws, tilts=None, fps=24):
    tilts = tilts or [0.0] * len(yaws)
    return [{"frame": i + 1, "t": i / fps, "pan_deg": y, "tilt_deg": t}
            for i, (y, t) in enumerate(zip(yaws, tilts))]


class TestFov(unittest.TestCase):
    def test_35mm_is_the_classic_54_degrees(self):
        fx, _ = P.fov_from_lens(35)
        self.assertAlmostEqual(math.degrees(fx), 54.43, places=1)

    def test_longer_lens_is_narrower(self):
        self.assertLess(P.fov_from_lens(85)[0], P.fov_from_lens(24)[0])


class TestSampling(unittest.TestCase):
    """The equirect convention must match the vendored MoGe helpers exactly:
    u = 1 - (atan2(y, x) / 2pi) % 1, world +X at u=1.0."""

    def setUp(self):
        import numpy as np
        w, h = 1024, 512
        u = np.linspace(0, 1, w)[None, :].repeat(h, 0)
        self.pano = np.stack([(u * 255).astype("uint8")] * 3, -1)
        self.fov, _ = P.fov_from_lens(35, 16 / 9)

    def test_plate_centre_lands_at_the_requested_yaw(self):
        for yaw in (0.0, 45.0, 90.0, -90.0, 179.0):
            img = P.render_plate(self.pano, yaw, 0.0, self.fov, 64, 36)
            got = img[18, 32, 0] / 255.0
            want = (1.0 - (math.radians(yaw) / (2 * math.pi)) % 1.0) % 1.0
            delta = min(abs(got - want), 1 - abs(got - want))  # wraps
            self.assertLess(delta, 0.02, f"yaw {yaw}: got u={got:.3f} want {want:.3f}")

    def test_output_shape_and_dtype(self):
        img = P.render_plate(self.pano, 0.0, 0.0, self.fov, 128, 72)
        self.assertEqual(img.shape, (72, 128, 3))
        self.assertEqual(img.dtype.name, "uint8")

    def test_straight_up_does_not_divide_by_zero(self):
        img = P.render_plate(self.pano, 0.0, 90.0, self.fov, 32, 18)
        self.assertEqual(img.shape, (18, 32, 3))


class TestSegmentation(unittest.TestCase):
    FOV = 54.4

    def test_locked_off_shot_is_one_plate(self):
        self.assertEqual(len(P.segment_frames(track([30.0] * 124), self.FOV)), 1)

    def test_pan_splits_and_tiles_the_duration(self):
        plates = P.segment_frames(track([i * 90 / 123 for i in range(124)]), self.FOV)
        self.assertGreater(len(plates), 1)
        self.assertEqual(plates[0]["frame_start"], 1)
        self.assertEqual(plates[-1]["frame_end"], 124)
        for a, b in zip(plates, plates[1:]):
            self.assertEqual(b["frame_start"] - a["frame_end"], 1)  # no gap

    def test_yaw_offset_shifts_every_plate(self):
        base = P.segment_frames(track([10.0] * 60), self.FOV)
        moved = P.segment_frames(track([10.0] * 60), self.FOV, yaw_offset_deg=25.0)
        self.assertAlmostEqual(moved[0]["yaw_deg"] - base[0]["yaw_deg"], 25.0, places=3)

    def test_wrap_uses_a_circular_mean(self):
        # frames straddling +/-180 must average near 180, not near 0
        plates = P.segment_frames(track([178.0, 179.0, -179.0, -178.0]), self.FOV)
        self.assertEqual(len(plates), 1)
        self.assertGreater(abs(plates[0]["yaw_deg"]), 170.0)

    def test_capped_at_max_plates(self):
        plates = P.segment_frames(track([i * 360 / 123 for i in range(124)]), self.FOV)
        self.assertLessEqual(len(plates), P.MAX_PLATES)

    def test_empty_track_is_empty(self):
        self.assertEqual(P.segment_frames([], self.FOV), [])


class TestEaWorldImport(unittest.TestCase):
    def test_matrix_decomposition_round_trips(self):
        from previs.importers.ea_world import _decompose
        # the obj_013 matrix: uniform scale 2.2261 x the gltf->Z-up swap
        m = [[2.2261, 0, 0, -2.137], [0, 0, -2.2261, -0.142],
             [0, 2.2261, 0, -0.586], [0, 0, 0, 1]]
        pos, rot, scale = _decompose(m)
        self.assertAlmostEqual(pos[0], -2.137, places=3)
        for s in scale:
            self.assertAlmostEqual(s, 2.2261, places=3)
        # (x,y,z)->(x,-z,y) is +90 about X, with no yaw
        self.assertAlmostEqual(rot[0], 90.0, places=2)
        self.assertAlmostEqual(rot[2], 0.0, places=2)


if __name__ == "__main__":
    unittest.main()


class TestHandedness(unittest.TestCase):
    """A mirrored plate looks like a valid room until text reads backwards.

    Centre-pixel checks cannot catch it -- a mirror fixes the centre -- so this
    asserts on an asymmetric feature instead. Caught in a real H3 render by a
    'wifi hotspot' sign reading in reverse.
    """

    def setUp(self):
        import numpy as np
        # equirect whose brightness rises with u, so left/right is unambiguous
        w, h = 1024, 512
        u = np.linspace(0, 1, w)[None, :].repeat(h, 0)
        self.pano = np.stack([(u * 255).astype("uint8")] * 3, -1)
        self.fov, _ = P.fov_from_lens(35, 16 / 9)

    def test_plate_is_not_mirrored(self):
        # u DEcreases as world yaw increases (u = 1 - atan2(y,x)/2pi), so
        # screen-right -- which is -Y, i.e. lower yaw -- must be BRIGHTER.
        img = P.render_plate(self.pano, 90.0, 0.0, self.fov, 128, 72)
        left = float(img[36, :20, 0].mean())
        right = float(img[36, -20:, 0].mean())
        self.assertGreater(right, left,
                           "plate is horizontally mirrored: screen-right must "
                           "sample the +right (-Y) direction")

    def test_vertical_is_not_flipped(self):
        import numpy as np
        h, w = 512, 1024
        v = np.linspace(0, 1, h)[:, None].repeat(w, 1)   # v=0 is the up pole
        pano = np.stack([(v * 255).astype("uint8")] * 3, -1)
        img = P.render_plate(pano, 0.0, 0.0, self.fov, 128, 72)
        top = float(img[:10, 64, 0].mean())
        bottom = float(img[-10:, 64, 0].mean())
        self.assertLess(top, bottom, "plate is vertically flipped: v=0 is up")


class TestParallax(unittest.TestCase):
    """Off-centre camera: the plate stops being an exact reprojection."""

    FOV = 54.4

    def frames(self, positions, aim):
        out = []
        for i, pos in enumerate(positions):
            dx, dy = aim[0] - pos[0], aim[1] - pos[1]
            out.append({"frame": i + 1, "t": i / 24.0, "position": list(pos),
                        "aim": list(aim),
                        "pan_deg": math.degrees(math.atan2(dy, dx)),
                        "tilt_deg": 0.0})
        return out

    def test_at_the_pano_point_there_is_no_parallax(self):
        frames = self.frames([[0, 0, 1.1]] * 30, [4, 0, 1.1])
        plates = P.segment_frames(frames, self.FOV, pano_origin=[0, 0, 1.1])
        self.assertEqual(plates[0]["camera_offset_m"], 0.0)
        self.assertEqual(plates[0]["parallax_deg"], 0.0)
        self.assertTrue(plates[0]["parallax_ok"])

    def test_offset_camera_reports_the_angle(self):
        # 1 m off the pano point, subject 4 m away -> atan(1/4) ~ 14 deg
        frames = self.frames([[0, 1.0, 1.1]] * 30, [4, 1.0, 1.1])
        plates = P.segment_frames(frames, self.FOV, pano_origin=[0, 0, 1.1])
        self.assertAlmostEqual(plates[0]["camera_offset_m"], 1.0, places=3)
        self.assertAlmostEqual(plates[0]["parallax_deg"], 14.04, places=1)
        self.assertFalse(plates[0]["parallax_ok"])

    def test_close_subject_is_worse_than_far(self):
        near = P.segment_frames(self.frames([[0, 1.0, 1.1]] * 10, [1.5, 1.0, 1.1]),
                                self.FOV, pano_origin=[0, 0, 1.1])
        far = P.segment_frames(self.frames([[0, 1.0, 1.1]] * 10, [8.0, 1.0, 1.1]),
                               self.FOV, pano_origin=[0, 0, 1.1])
        self.assertGreater(near[0]["parallax_deg"], far[0]["parallax_deg"])

    def test_plate_is_aimed_from_the_pano_point_not_the_camera(self):
        # camera stands well off to the side but looks at the same wall; the
        # plate must be cut along origin->aim, not along the camera's heading
        aim = [4.0, 0.0, 1.1]
        frames = self.frames([[0.0, 3.0, 1.1]] * 10, aim)
        plates = P.segment_frames(frames, self.FOV, pano_origin=[0, 0, 1.1])
        self.assertAlmostEqual(plates[0]["yaw_deg"], 0.0, places=1)   # origin->aim
        self.assertNotAlmostEqual(frames[0]["pan_deg"], 0.0, places=1)  # camera heading

    def test_missing_aim_falls_back_to_camera_heading(self):
        frames = [{"frame": 1, "t": 0.0, "pan_deg": 33.0, "tilt_deg": 0.0}]
        plates = P.segment_frames(frames, self.FOV)
        self.assertAlmostEqual(plates[0]["yaw_deg"], 33.0, places=3)
