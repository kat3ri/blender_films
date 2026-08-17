"""Draw pose_landmarks.json as an OpenPose-style skeleton video.

For control-adapter models (Wan, LTX) that take a pose video rather than a
reference clip. MiniMax H3 cannot consume this — it takes the clay reference
and a prompt — so this is shelf inventory, not part of the H3 path.

The interesting property: every other tool in this space *estimates* pose from
pixels (MediaPipe, DWPose) and inherits detection error, missed frames and
left/right swaps. previs authored the skeleton, so `capture_pose_landmarks`
already wrote the exact joints and their exact camera projection. This module
only draws them.

Runs on the host — no bpy — so it works on any bundle after the fact::

    python -m previs.cli pose-render renders/bundles/<shot>/

Colours follow the OpenPose BODY_25 convention (per-limb hues, red-ish on the
subject's right, blue-ish on the left) because that is what ControlNet pose
encoders were trained to see.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

# previs joint -> BODY_25 index, for the joints previs actually has. The
# missing BODY_25 points are face/feet detail (eyes, ears, toes) that a proxy
# rig has no opinion about; leaving them out is more honest than inventing them.
BODY25_INDEX = {
    "neck": 1,
    "r_shoulder": 2, "r_elbow": 3, "r_hand": 4,
    "l_shoulder": 5, "l_elbow": 6, "l_hand": 7,
    "hips": 8,
    "r_hip": 9, "r_knee": 10, "r_foot": 11,
    "l_hip": 12, "l_knee": 13, "l_foot": 14,
    "head": 0,
}

# (joint_a, joint_b, RGB) -- OpenPose's limb palette.
LIMBS = (
    ("neck", "r_shoulder", (255, 0, 0)),
    ("r_shoulder", "r_elbow", (255, 85, 0)),
    ("r_elbow", "r_hand", (255, 170, 0)),
    ("neck", "l_shoulder", (255, 255, 0)),
    ("l_shoulder", "l_elbow", (170, 255, 0)),
    ("l_elbow", "l_hand", (85, 255, 0)),
    ("neck", "hips", (0, 255, 0)),
    ("hips", "r_hip", (0, 255, 85)),
    ("r_hip", "r_knee", (0, 255, 170)),
    ("r_knee", "r_foot", (0, 255, 255)),
    ("hips", "l_hip", (0, 170, 255)),
    ("l_hip", "l_knee", (0, 85, 255)),
    ("l_knee", "l_foot", (0, 0, 255)),
    ("neck", "head", (85, 0, 255)),
    # previs also has chest/spine, which BODY_25 lacks; drawn faintly so the
    # torso reads as solid without pretending to be canonical points.
    ("chest", "spine", (120, 60, 200)),
)

JOINT_COLOR = (255, 255, 255)


def _draw_frame(draw, people_joints, limb_width, joint_radius):
    for joints in people_joints:
        for a, b, color in LIMBS:
            ja, jb = joints.get(a), joints.get(b)
            if not ja or not jb:
                continue
            if not (ja.get("visible") or jb.get("visible")):
                continue
            draw.line([tuple(ja["image"]), tuple(jb["image"])],
                      fill=color, width=limb_width)
        for name, joint in joints.items():
            if name not in BODY25_INDEX or not joint.get("visible"):
                continue
            x, y = joint["image"]
            draw.ellipse([x - joint_radius, y - joint_radius,
                          x + joint_radius, y + joint_radius], fill=JOINT_COLOR)


def render_pose_video(pose, output_path, fps=None, limb_width=4, joint_radius=3,
                      background=(0, 0, 0)):
    """Draw every frame of ``pose`` and encode to ``output_path``.

    ``pose`` is a loaded pose_landmarks.json. Returns the output path.
    """
    from PIL import Image, ImageDraw

    width, height = (int(v) for v in pose["resolution"])
    fps = int(fps or pose.get("fps", 24))
    people = pose.get("people") or []
    if not people:
        raise ValueError("pose_landmarks.json contains no people")
    frame_count = pose.get("frame_count") or len(people[0]["frames"])

    # index frames per person so a missing person-frame can't shift the video
    by_frame = {}
    for person in people:
        for entry in person.get("frames", []):
            by_frame.setdefault(entry["frame"], []).append(entry.get("joints") or {})

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="previs_pose_"))
    try:
        for frame in range(1, frame_count + 1):
            image = Image.new("RGB", (width, height), background)
            draw = ImageDraw.Draw(image)
            _draw_frame(draw, by_frame.get(frame, []), limb_width, joint_radius)
            image.save(tmp / f"{frame:05d}.png")

        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg not found on PATH; cannot encode the pose video")
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-framerate", str(fps),
             "-i", str(tmp / "%05d.png"),
             # yuv420p + even dimensions so every player and every ComfyUI
             # video loader accepts it
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
             str(output_path)],
            check=True,
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return output_path


def render_from_bundle(bundle_dir, output_path=None, **kwargs):
    """Convenience: read a bundle's pose_landmarks.json and draw it."""
    bundle_dir = Path(bundle_dir)
    pose_path = bundle_dir / "pose_landmarks.json"
    if not pose_path.is_file():
        raise FileNotFoundError(
            f"{pose_path} not found -- the bundle has no rigged characters, or "
            f"was exported with --no-pose")
    with pose_path.open(encoding="utf-8") as handle:
        pose = json.load(handle)
    if output_path is None:
        output_path = bundle_dir / "openpose_pose.mp4"
    return render_pose_video(pose, output_path, **kwargs)
