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
# A plate is only a correct reprojection from the pano's own capture point.
# Move the camera off it and the plate disagrees with the render by roughly
# atan(offset / distance_to_subject) -- a 1 m step with the subject 4 m away is
# ~14 deg; the same step 1.5 m away is ~34 deg. Warn past this so the error is
# a number you can design shots against rather than something you notice later.
PARALLAX_WARN_DEG = 12.0


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


def _origin_relative_view(frame, pano_origin):
    """Yaw/pitch from the PANO ORIGIN toward what this frame looks at.

    Once the camera leaves the capture point, its own forward vector is the
    wrong thing to cut a plate along: the plate is the view from the pano
    point, so aiming it at the camera's heading shows the correct wall only by
    luck. Aiming from the origin toward the camera's aim point keeps the plate
    on the right part of the room; the residual disagreement is parallax, which
    `plate_parallax` reports.
    """
    aim = frame.get("aim")
    if aim is None:
        return frame["pan_deg"], frame["tilt_deg"], 0.0
    position = frame.get("position") or [0.0, 0.0, 0.0]
    offset = math.sqrt(sum((position[i] - pano_origin[i]) ** 2 for i in range(3)))
    vector = [aim[i] - pano_origin[i] for i in range(3)]
    horizontal = math.hypot(vector[0], vector[1])
    if horizontal < 1e-9 and abs(vector[2]) < 1e-9:
        return frame["pan_deg"], frame["tilt_deg"], offset
    yaw = math.degrees(math.atan2(vector[1], vector[0]))
    pitch = math.degrees(math.atan2(vector[2], horizontal))
    return yaw, pitch, offset


def plate_parallax(offset_m, subject_distance_m):
    """Angular disagreement between the plate and the render, in degrees."""
    if subject_distance_m <= 1e-6:
        return 90.0
    return math.degrees(math.atan(abs(offset_m) / subject_distance_m))


def segment_frames(frames, fov_x_deg, yaw_offset_deg=0.0,
                   new_plate_at=DEFAULT_NEW_PLATE_AT, max_plates=MAX_PLATES,
                   pano_origin=None):
    """Split a camera track into the fewest plates that cover what it sees.

    A locked-off shot yields one plate; a pan yields two or three. Each plate is
    centred on the mean of the frames it covers, so it sits in the middle of its
    own span rather than at whichever frame happened to open it.

    Returns a list of dicts with yaw/pitch, the frame range, and t_start/t_end.
    """
    if not frames:
        return []

    origin = list(pano_origin or [0.0, 0.0, 0.0])
    views = []
    for frame in frames:
        yaw, pitch, offset = _origin_relative_view(frame, origin)
        views.append({"frame": frame, "yaw": yaw + yaw_offset_deg,
                      "pitch": pitch, "offset": offset})

    threshold = fov_x_deg * float(new_plate_at)
    groups = [[views[0]]]
    anchor = (views[0]["yaw"], views[0]["pitch"])
    for view in views[1:]:
        yaw = view["yaw"]
        drift = _angular_distance_deg(yaw, view["pitch"], anchor[0], anchor[1])
        if drift > threshold and len(groups) < max_plates:
            groups.append([view])
            anchor = (yaw, view["pitch"])
        else:
            groups[-1].append(view)

    plates = []
    for index, group in enumerate(groups):
        # Circular mean for yaw so a group straddling +/-180 does not average
        # to the opposite wall.
        sin_sum = sum(math.sin(math.radians(v["yaw"])) for v in group)
        cos_sum = sum(math.cos(math.radians(v["yaw"])) for v in group)
        yaw = math.degrees(math.atan2(sin_sum, cos_sum))
        pitch = sum(v["pitch"] for v in group) / len(group)

        offsets = [v["offset"] for v in group]
        distances = []
        for v in group:
            aim = v["frame"].get("aim")
            position = v["frame"].get("position")
            if aim and position:
                distances.append(math.sqrt(sum((aim[i] - position[i]) ** 2
                                               for i in range(3))))
        max_offset = max(offsets)
        near = min(distances) if distances else 0.0
        parallax = plate_parallax(max_offset, near) if distances else 0.0

        plates.append({
            "index": index + 1,
            "yaw_deg": round(yaw, 3),
            "pitch_deg": round(pitch, 3),
            "frame_start": group[0]["frame"]["frame"],
            "frame_end": group[-1]["frame"]["frame"],
            "t_start": round(group[0]["frame"]["t"], 4),
            "t_end": round(group[-1]["frame"]["t"], 4),
            "frame_count": len(group),
            # How far the camera strays from the pano capture point over this
            # plate's span, and what that costs in angular agreement.
            "camera_offset_m": round(max_offset, 4),
            "nearest_subject_m": round(near, 4),
            "parallax_deg": round(parallax, 2),
            "parallax_ok": bool(parallax <= PARALLAX_WARN_DEG),
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
    # Handedness matters and is easy to get backwards: with forward=+X and
    # world up=+Z, `right` is -Y, so the image's LAST column must sample
    # +right. Getting this reversed mirrors the plate -- which looks like a
    # perfectly plausible room until you notice signage reads backwards, and
    # which centre-pixel tests cannot catch.
    xs = np.linspace(-half_x, half_x, width)      # image +x is screen-right
    ys = np.linspace(half_y, -half_y, height)     # image +y is screen-down
    grid_x, grid_y = np.meshgrid(xs, ys)

    directions = (forward[None, None, :]
                  + grid_x[..., None] * right[None, None, :]
                  + grid_y[..., None] * up[None, None, :])
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    return _sample_equirect(pano, directions)


def plates_for_shot(pano_path, camera_motion, out_dir, yaw_offset_deg=0.0,
                    width=DEFAULT_PLATE_W, height=DEFAULT_PLATE_H,
                    margin=DEFAULT_MARGIN, new_plate_at=DEFAULT_NEW_PLATE_AT,
                    max_plates=MAX_PLATES, prefix="plate", pano_origin=None):
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
                            new_plate_at=new_plate_at, max_plates=max_plates,
                            pano_origin=pano_origin)

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
        "pano_origin": [round(float(v), 4) for v in (pano_origin or [0.0, 0.0, 0.0])],
        "parallax_warn_deg": PARALLAX_WARN_DEG,
        "warnings": [
            f"plate {p['index']} ({p['t_start']:g}-{p['t_end']:g}s): the camera "
            f"strays {p['camera_offset_m']:.2f} m from the pano point with the "
            f"nearest subject {p['nearest_subject_m']:.2f} m away, so the plate "
            f"disagrees with the render by about {p['parallax_deg']:.0f} deg."
            for p in plates if not p["parallax_ok"]
        ],
        "plates": plates,
    }


def load_camera_motion(bundle_dir):
    path = Path(bundle_dir) / "camera_motion.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found -- bundle the shot first")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def attach_to_bundle(bundle_dir, shot, library, pano_override=None, verbose=True):
    """Add background plates to an existing bundle. HOST-SIDE ONLY.

    This cannot run inside Blender: plate rendering needs numpy and Pillow, and
    Blender's bundled Python has no pip (the repo's standing constraint). So
    `compile_bundle` writes camera_motion.json inside Blender, and the CLI calls
    this afterwards on the host, where those packages exist. Found the hard way
    -- the first end-to-end run reported `pano plates failed: No module named
    'PIL'` from inside the Blender process.

    Re-emits the prompts too, so the [BACKGROUND] block and its <Picture N>
    numbering reflect the plates, and updates the manifest. Returns the timeline
    or None when the set has no panorama.
    """
    from . import bundle as bundle_mod

    bundle_dir = Path(bundle_dir)
    set_asset = library.get("sets", (shot.get("set") or {}).get("asset_id", "")) if library else {}
    pano_path = pano_override or (set_asset or {}).get("pano")
    if not pano_path or not Path(pano_path).is_file():
        if verbose and pano_path:
            print(f"[previs] WARNING     set pano not found: {pano_path}")
        return None

    camera_motion = load_camera_motion(bundle_dir)
    timeline = plates_for_shot(
        pano_path, camera_motion, bundle_dir / "plates",
        yaw_offset_deg=float((set_asset or {}).get("pano_yaw_offset_deg", 0.0)),
        pano_origin=(set_asset or {}).get("pano_origin_m"),
    )
    bundle_mod._dump(bundle_dir / "plates.json", timeline)

    manifest_path = bundle_dir / "bundle_manifest.json"
    manifest = {}
    if manifest_path.is_file():
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
    entries = dict(manifest.get("files") or {})
    entries["plates"] = "plates/"
    entries["plates_timeline"] = "plates.json"

    # Prompts are pure stdlib, so they re-emit fine here; passing _plates is
    # what produces the [BACKGROUND] block.
    shot = dict(shot)
    shot["_plates"] = timeline
    fps = int(manifest.get("fps") or shot.get("fps", 12))
    from .motion import build_camera_keys, build_tracks
    tracks = build_tracks(shot, library)
    camera_keys = build_camera_keys(shot, tracks, library, fps)
    rewritten, metadata = bundle_mod.write_sidecars(
        bundle_dir, shot, tracks, camera_keys, library, fps,
        generators=tuple(manifest.get("generators") or ("minimax",)),
        render_settings=shot.get("render") or {},
    )
    entries.update(rewritten)
    metadata["plates"] = timeline
    bundle_mod._dump(bundle_dir / "metadata.json", metadata)

    manifest["files"] = dict(sorted(entries.items()))
    # plates/ + plates.json are contract 1.1 additions
    manifest["contract_version"] = bundle_mod.CONTRACT_VERSION
    manifest["plates"] = {"count": len(timeline["plates"]),
                          "pano": timeline["pano"],
                          "pano_yaw_offset_deg": timeline["pano_yaw_offset_deg"]}
    bundle_mod._dump(manifest_path, manifest)

    if verbose:
        spans = ", ".join(f"{p['t_start']:.2f}-{p['t_end']:.2f}s @{p['yaw_deg']:.0f}deg"
                          for p in timeline["plates"])
        print(f"[previs] plates      {len(timeline['plates'])}: {spans}")
        for warning in timeline["warnings"]:
            print(f"[previs] WARNING     {warning}")
    return timeline
