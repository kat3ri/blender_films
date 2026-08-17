"""Camera-driven background plates cut from an equirectangular panorama.

The point of this module: previs *authored* the camera, so `camera_motion.json`
already holds the exact direction the lens points on every frame. That makes the
background reference a deterministic reprojection rather than a guess — cut the
region of the pano the shot actually sees, and hand it to the generator as a
photographic plate alongside the grey-box control video.

Compare the alternative already in the fork: `MiniMaxH3RenderPLYCutoutViews`
sprays a blind ring of views around the full 360. MiniMax H3 accepts nine image
slots *in total*, shared with the cast, so a blind ring is unusable — you pick a
few and hope they are the walls in frame. Driving the crop from the camera track
yields the one to three plates that actually appear.

Conventions, which must agree with the decomposition or the plate shows the
wrong wall:

* Directions are Z-up, matching Blender and MoGe's pano helpers.
* Equirect UV follows the same vendored MoGe convention used by
  `decompose/pano_helpers.py` and `midi3d-spike/pano_helpers.py`::

      u = 1 - (atan2(d.y, d.x) / 2pi) % 1        v = acos(d.z) / pi

  so world +X lands at u=1.0 and yaw increases leftward, v=0 is the up pole.
* `pan_deg` from camera_motion is `atan2(forward.y, forward.x)` — the same
  quantity, so the two compose with a single additive offset per set
  (`pano_yaw_offset_deg`), which is what `previs pano-check` calibrates.

Numpy + Pillow only; no bpy, so this runs on the host over an existing bundle.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

# Blender's default sensor width. previs never changes it (create_camera only
# sets `lens`), so horizontal FOV follows from lens_mm alone.
SENSOR_WIDTH_MM = 36.0

DEFAULT_PLATE_W = 1024
DEFAULT_PLATE_H = 576
# Start a new plate once the view centre has drifted this fraction of the FOV
# from the plate's own centre. Below ~0.5 a slow pan produces redundant plates;
# above ~0.8 the plate no longer covers what the camera sees at the ends.
DEFAULT_NEW_PLATE_AT = 0.6
# Widen each plate beyond the taking FOV so the generator has margin at the
# edges rather than having to invent it.
DEFAULT_MARGIN = 1.15
# H3 has nine image slots shared with the cast; three plates is already
# generous for one shot.
MAX_PLATES = 3


def fov_from_lens(lens_mm, aspect=16.0 / 9.0):
    """(horizontal, vertical) FOV in radians for a Blender camera."""
    fov_x = 2.0 * math.atan(SENSOR_WIDTH_MM / (2.0 * float(lens_mm)))
    fov_y = 2.0 * math.atan(math.tan(fov_x / 2.0) / aspect)
    return fov_x, fov_y


def _angular_distance_deg(yaw_a, pitch_a, yaw_b, pitch_b):
    """Great-circle angle between two view directions, in degrees."""
    ya, pa, yb, pb = (math.radians(v) for v in (yaw_a, pitch_a, yaw_b, pitch_b))
    dot = (math.cos(pa) * math.cos(pb) * math.cos(ya - yb)
           + math.sin(pa) * math.sin(pb))
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


def segment_frames(frames, fov_x_deg, yaw_offset_deg=0.0,
                   new_plate_at=DEFAULT_NEW_PLATE_AT, max_plates=MAX_PLATES):
    """Split a camera track into the fewest plates that cover what it sees.

    A locked-off shot yields one plate; a pan yields two or three. Each plate is
    centred on the mean of the frames it covers, so it sits in the middle of its
    own span rather than at whichever frame happened to open it.

    Returns a list of dicts with yaw/pitch, the frame range, and t_start/t_end.
    """
    if not frames:
        return []

    threshold = fov_x_deg * float(new_plate_at)
    groups = [[frames[0]]]
    anchor = (frames[0]["pan_deg"] + yaw_offset_deg, frames[0]["tilt_deg"])
    for frame in frames[1:]:
        yaw = frame["pan_deg"] + yaw_offset_deg
        drift = _angular_distance_deg(yaw, frame["tilt_deg"], anchor[0], anchor[1])
        if drift > threshold and len(groups) < max_plates:
            groups.append([frame])
            anchor = (yaw, frame["tilt_deg"])
        else:
            groups[-1].append(frame)

    plates = []
    for index, group in enumerate(groups):
        # Circular mean for yaw so a group straddling +/-180 does not average
        # to the opposite wall.
        sin_sum = sum(math.sin(math.radians(f["pan_deg"] + yaw_offset_deg)) for f in group)
        cos_sum = sum(math.cos(math.radians(f["pan_deg"] + yaw_offset_deg)) for f in group)
        yaw = math.degrees(math.atan2(sin_sum, cos_sum))
        pitch = sum(f["tilt_deg"] for f in group) / len(group)
        plates.append({
            "index": index + 1,
            "yaw_deg": round(yaw, 3),
            "pitch_deg": round(pitch, 3),
            "frame_start": group[0]["frame"],
            "frame_end": group[-1]["frame"],
            "t_start": round(group[0]["t"], 4),
            "t_end": round(group[-1]["t"], 4),
            "frame_count": len(group),
        })
    return plates


def _sample_equirect(pano, directions):
    """Bilinear-sample an equirect (H, W, 3) uint8 at unit `directions`."""
    import numpy as np

    height, width = pano.shape[:2]
    dx, dy, dz = directions[..., 0], directions[..., 1], directions[..., 2]
    # Exactly the vendored MoGe convention -- see the module docstring.
    u = 1.0 - np.mod(np.arctan2(dy, dx) / (2.0 * np.pi), 1.0)
    v = np.arccos(np.clip(dz, -1.0, 1.0)) / np.pi

    x = u * width - 0.5
    y = v * height - 0.5
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = (x - x0)[..., None]
    fy = (y - y0)[..., None]
    # wrap in longitude, clamp in latitude -- the poles do not wrap
    x0m, x1m = np.mod(x0, width), np.mod(x0 + 1, width)
    y0m = np.clip(y0, 0, height - 1)
    y1m = np.clip(y0 + 1, 0, height - 1)

    top = pano[y0m, x0m] * (1 - fx) + pano[y0m, x1m] * fx
    bottom = pano[y1m, x0m] * (1 - fx) + pano[y1m, x1m] * fx
    return (top * (1 - fy) + bottom * fy).astype("uint8")


def render_plate(pano, yaw_deg, pitch_deg, fov_x, width=DEFAULT_PLATE_W,
                 height=DEFAULT_PLATE_H):
    """Perspective crop of `pano` looking at (yaw, pitch) with horizontal `fov_x`."""
    import numpy as np

    yaw, pitch = math.radians(yaw_deg), math.radians(pitch_deg)
    forward = np.array([math.cos(pitch) * math.cos(yaw),
                        math.cos(pitch) * math.sin(yaw),
                        math.sin(pitch)])
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    norm = np.linalg.norm(right)
    # Looking straight up or down: any horizon direction will do, pick +X.
    right = np.array([1.0, 0.0, 0.0]) if norm < 1e-8 else right / norm
    up = np.cross(right, forward)

    aspect = width / float(height)
    half_x = math.tan(fov_x / 2.0)
    half_y = half_x / aspect
    xs = np.linspace(half_x, -half_x, width)      # +X of the image is screen-left
    ys = np.linspace(half_y, -half_y, height)     # +Y of the image is screen-up
    grid_x, grid_y = np.meshgrid(xs, ys)

    directions = (forward[None, None, :]
                  + grid_x[..., None] * right[None, None, :]
                  + grid_y[..., None] * up[None, None, :])
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    return _sample_equirect(pano, directions)


def plates_for_shot(pano_path, camera_motion, out_dir, yaw_offset_deg=0.0,
                    width=DEFAULT_PLATE_W, height=DEFAULT_PLATE_H,
                    margin=DEFAULT_MARGIN, new_plate_at=DEFAULT_NEW_PLATE_AT,
                    max_plates=MAX_PLATES, prefix="plate"):
    """Cut the plates a shot's camera actually sees, and write them out.

    `camera_motion` is a loaded camera_motion.json. Returns the timeline dict
    that also lands as plates.json.
    """
    import numpy as np
    from PIL import Image

    frames = camera_motion.get("frames") or []
    if not frames:
        raise ValueError("camera_motion has no frames")
    lens_mm = float((camera_motion.get("camera") or {}).get("lens_mm")
                    or camera_motion.get("lens_mm") or 35.0)

    fov_x, _ = fov_from_lens(lens_mm, aspect=width / float(height))
    fov_x_deg = math.degrees(fov_x)
    plates = segment_frames(frames, fov_x_deg, yaw_offset_deg,
                            new_plate_at=new_plate_at, max_plates=max_plates)

    pano = np.asarray(Image.open(pano_path).convert("RGB"))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for plate in plates:
        image = render_plate(pano, plate["yaw_deg"], plate["pitch_deg"],
                             fov_x * float(margin), width, height)
        name = f"{prefix}_{plate['index']:02d}.png"
        Image.fromarray(image).save(out_dir / name)
        plate["file"] = name

    return {
        "format": "previs.plates",
        "version": "1.0",
        "pano": str(pano_path),
        "pano_yaw_offset_deg": round(float(yaw_offset_deg), 3),
        "lens_mm": lens_mm,
        "fov_x_deg": round(fov_x_deg, 3),
        "plate_margin": float(margin),
        "plate_size": [int(width), int(height)],
        "plates": plates,
    }


def load_camera_motion(bundle_dir):
    path = Path(bundle_dir) / "camera_motion.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found -- bundle the shot first")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)
