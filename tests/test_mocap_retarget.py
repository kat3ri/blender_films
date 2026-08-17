"""World-space retarget: heading stripping and hierarchy transport.

Uses a synthetic BVH shaped like SnapMoGen's failure mode: ROOT never
rotates, and the pelvis and spine branches EACH carry the whole-body heading
as a local rotation. A per-joint local copy scrambles that skeleton (the
"silly walk"); the world-transport retarget must not.
"""

import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from previs import mocap, mocap_bvh  # noqa: E402

# Y-up, facing +Z. Left leg at +X. Two branches off ROOT, like SnapMoGen.
_BVH = """HIERARCHY
ROOT ROOT
{
  OFFSET 0 0 0
  CHANNELS 6 Xposition Yposition Zposition Xrotation Yrotation Zrotation
  JOINT C_pelvis0001_bind_JNT
  {
    OFFSET 0 0 0
    CHANNELS 3 Xrotation Yrotation Zrotation
    JOINT L_legUpper0001_bind_JNT
    {
      OFFSET 6 -6 0
      CHANNELS 3 Xrotation Yrotation Zrotation
      JOINT L_legLower0001_bind_JNT
      {
        OFFSET 0 -19 0
        CHANNELS 3 Xrotation Yrotation Zrotation
        End Site
        {
          OFFSET 0 -18 0
        }
      }
    }
  }
  JOINT C_spine0001_bind_JNT
  {
    OFFSET 0 8 0
    CHANNELS 3 Xrotation Yrotation Zrotation
    End Site
    {
      OFFSET 0 20 0
    }
  }
}
MOTION
Frames: 2
Frame Time: 0.033333
0 0 0 0 0 0 0 YAW 0 KNEEFLEX 0 0 KNEEFLEX 0 0 0 YAW 0
0 0 0 0 0 0 0 YAW 0 KNEEFLEX 0 0 KNEEFLEX 0 0 0 YAW 0
"""


def _load(yaw_deg, knee_flex_deg=30.0):
    text = _BVH.replace("YAW", str(yaw_deg)).replace("KNEEFLEX", str(knee_flex_deg))
    with tempfile.NamedTemporaryFile("w", suffix=".bvh", delete=False) as fh:
        fh.write(text)
        path = fh.name
    return mocap_bvh.load_bvh(path)


class TestRetargetFrame(unittest.TestCase):
    def _retarget(self, yaw_deg):
        clip = _load(yaw_deg)
        rot = clip.sample_joint_rotations(0.0)
        return mocap.retarget_frame(clip, rot, mocap.canonical_joint_map(), "y")

    def test_heading_is_stripped(self):
        # The clip's heading belongs to the authored track: whatever yaw the
        # pelvis/spine branches carry, the rig must see the same pose.
        base = self._retarget(0.0)
        turned = self._retarget(147.0)
        for joint in ("hips", "spine", "l_hip", "l_knee"):
            for a, b in zip(base[joint], turned[joint]):
                self.assertAlmostEqual(a, b, places=3, msg=joint)

    def test_knee_flexes_about_rig_left_axis(self):
        # Source hip flexes the thigh forward (source -X rotation, Y-up facing
        # +Z) and the knee flexes back; in rig frame both are Y rotations and
        # the knee's must be positive (heel toward butt), not hyperextension.
        mapped = self._retarget(0.0)
        knee = mapped["l_knee"]
        self.assertGreater(knee[1], 20.0)
        self.assertLess(abs(knee[0]), 1.0)
        self.assertLess(abs(knee[2]), 1.0)

    def test_spine_gets_no_phantom_twist(self):
        # The old per-joint copy put the branch heading straight into the
        # spine local -- a ~150 deg twist between torso and legs.
        mapped = self._retarget(147.0)
        self.assertLess(max(abs(a) for a in mapped["spine"]), 1.0)

    def test_matrix_euler_roundtrip_matches_blender(self):
        # _matrix_to_euler_xyz_deg must decompose in Blender's 'XYZ'
        # convention (M = Rz.Ry.Rx); recompose and compare.
        from previs.mocap import _euler_to_matrix

        original = [31.0, -47.0, 112.0]
        recovered = mocap_bvh._matrix_to_euler_xyz_deg(_euler_to_matrix(original))
        matrix = _euler_to_matrix(recovered)
        expected = _euler_to_matrix(original)
        for r in range(3):
            for c in range(3):
                self.assertAlmostEqual(matrix[r][c], expected[r][c], places=6)


if __name__ == "__main__":
    unittest.main()
