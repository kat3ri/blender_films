"""Control-layer bundle: package a shot for a downstream AI video generator.

This is the *producer* side of the same pipeline Blockout and Motion Previs
Studio serve. Where those tools *solve* camera and pose from an existing video
(optical flow, MediaPipe), this system authored the shot, so the camera track
and every joint position are already exact — the sidecar JSON here is ground
truth, not an estimate.

The bundle layout mirrors Blockout's shot package so anything already built to
read one (a ComfyUI depth graph, a downstream script) reads ours too:

    <shot_id>/
      <shot_id>_reference.mp4     the grey-box control render     (Blender)
      <shot_id>_depth.mp4         depth pass for ControlNet        (Blender)
      camera_motion.json          per-frame camera, exact          (here)
      pose_landmarks.json         per-frame 3D+2D joints, exact    (Blender)
      metadata.json               marks / lenses / timings         (here)
      prompt.txt                  default cinematic prompt         (here)
      prompt.<generator>.txt      per-target-generator prompt      (here)
      prompt_fragments.<gen>.json prompt as addressable pieces     (here)
      stills/                     frame at each mark + diagram      (Blender)
      bundle_manifest.json        what actually got written        (here)
      README.txt                                                    (here)

Design decisions borrowed from Blockout, deliberately:

* **Deterministic + machine-readable.** Every sidecar is stable-key-order,
  pretty-printed JSON — diffable and branchable.
* **Prompt generated from the blocking, not hand-typed.** Lens, staging, and
  each subject's moves become the prompt text, tailored per generator via a
  small config table rather than bespoke code.
* **A generator is a config row, not a code path.** Adding a target model is a
  dict entry in ``GENERATOR_PROFILES``.

previs does not submit to any model: the bundle is handed to the shot
orchestrator, which composes the full scene prompt around these fragments and
owns the generation call. ``docs/BUNDLE_CONTRACT.md`` is that interface.

Nothing here imports ``bpy``; it runs on the host and inside Blender alike.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from .asset_library import aim_height
from .motion import pad3

EPS = 1e-9
BUNDLE_FORMAT_VERSION = "1.0"


# ---------------------------------------------------------------------------
# generator profiles — adding a target model is a row here, not a code path
# ---------------------------------------------------------------------------

# Each profile shapes the exported prompt and records the target's practical
# constraints (duration cap, native aspect) so a downstream operator does not
# have to remember them. ``reference_note`` states, in the model's own idiom,
# that the control video supplies motion/staging only and not appearance —
# the single instruction that most improves a grey-box-conditioned result.
GENERATOR_PROFILES = {
    "generic": {
        "display_name": "Generic / unspecified",
        "max_duration_s": None,
        "aspect": "16:9",
        "reference_note": (
            "The attached video is a grey-box previs render. Use it for motion "
            "path, staging, timing, and camera move only — take no appearance "
            "from it. All look and detail comes from the text."
        ),
    },
    "seedance": {
        "display_name": "Seedance 2.0",
        "max_duration_s": 10.0,
        "aspect": "16:9",
        "reference_note": (
            "Reference video defines camera movement, blocking, and timing. Do "
            "not inherit the grey proxy look; render the described scene."
        ),
    },
    "minimax": {
        "display_name": "MiniMax H3 (Omni Reference)",
        "max_duration_s": 15.0,
        "aspect": "16:9",
        # H3's tokenizer emits <Video N>/<Picture N> literally, so the prompt
        # must use those exact tokens. Validated live (2026-08): explicit
        # replace/retain imperatives work where pure negation ("do not inherit
        # its appearance") failed -- translate the blockout, don't suppress it.
        "format": "retention",
        "video_token": "<Video 1>",
        "picture_token": "<Picture {n}>",
        # H3 resamples the guide to 24 fps, trims frames down to the 17k+5
        # grid, and requires multiple-of-32 canvases (see check_target_constraints).
        "frame_grid": {"k": 17, "offset": 5, "fps": 24},
        "canvas_multiple": 32,
        "reference_note": (
            "Retain the character poses and camera motion from <Video 1>. "
            "<Video 1> is a grey untextured previs blocking render: replace all "
            "of its appearance with the description below."
        ),
    },
    "kling": {
        "display_name": "Kling",
        "max_duration_s": 10.0,
        "aspect": "16:9",
        "reference_note": (
            "Use the reference clip for camera motion and subject blocking only. "
            "Replace all appearance with the description below."
        ),
    },
    "veo": {
        "display_name": "Veo 3.1",
        "max_duration_s": 8.0,
        "aspect": "16:9",
        "reference_note": (
            "The control video encodes the camera move and staging. Keep its "
            "motion and timing; render the scene as described, not as grey "
            "boxes."
        ),
    },
    "wan": {
        "display_name": "Wan 2.2",
        "max_duration_s": 5.0,
        "aspect": "16:9",
        "reference_note": (
            "Depth-conditioned: the depth pass drives structure and the "
            "reference drives motion. Appearance is text-only."
        ),
    },
    "ltx": {
        "display_name": "LTX 2.3",
        "max_duration_s": 10.0,
        "aspect": "16:9",
        "reference_note": (
            "Reference and depth passes supply camera and structure. Do not copy "
            "the previs look; synthesise the described scene."
        ),
    },
}

DEFAULT_GENERATORS = ("generic", "seedance", "minimax")


# ---------------------------------------------------------------------------
# camera maths — exact, from the authored per-frame keys (no solve)
# ---------------------------------------------------------------------------

def _matrix_from_euler_xyz(rx, ry, rz):
    """Blender 'XYZ' euler (radians) -> 3x3 rotation matrix (Rz @ Ry @ Rx)."""
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    # Rz @ Ry @ Rx, row-major.
    return [
        [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
        [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
        [-sy,      cy * sx,                cy * cx],
    ]


def _mat_vec(m, v):
    return [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _normalize(v):
    length = math.sqrt(_dot(v, v))
    if length < EPS:
        return [0.0, 0.0, 0.0]
    return [v[0] / length, v[1] / length, v[2] / length]


def camera_orientation(rotation_euler):
    """Human-meaningful orientation from a Blender camera euler.

    A Blender camera looks down local -Z with local +Y up. Returns the world
    forward/up/right basis plus pan (azimuth), tilt (elevation) and roll in
    degrees — the same quantities Motion Previs Studio *solves* from footage,
    except here they are exact because we authored them.
    """
    matrix = _matrix_from_euler_xyz(*rotation_euler)
    forward = _normalize(_mat_vec(matrix, [0.0, 0.0, -1.0]))
    up = _normalize(_mat_vec(matrix, [0.0, 1.0, 0.0]))
    right = _normalize(_mat_vec(matrix, [1.0, 0.0, 0.0]))

    pan_deg = math.degrees(math.atan2(forward[1], forward[0]))
    tilt_deg = math.degrees(math.asin(max(-1.0, min(1.0, forward[2]))))

    # Roll: signed angle between the actual up and the level (no-roll) up,
    # measured about the forward axis. Degenerates when looking near-vertical.
    world_up = [0.0, 0.0, 1.0]
    level_up = [world_up[i] - _dot(world_up, forward) * forward[i] for i in range(3)]
    if _dot(level_up, level_up) < 1e-6:
        roll_deg = 0.0
    else:
        level_up = _normalize(level_up)
        sin_roll = _dot(_cross(level_up, up), forward)
        cos_roll = _dot(level_up, up)
        roll_deg = math.degrees(math.atan2(sin_roll, cos_roll))

    return {
        "forward": [round(c, 6) for c in forward],
        "up": [round(c, 6) for c in up],
        "right": [round(c, 6) for c in right],
        "pan_deg": round(pan_deg, 3),
        "tilt_deg": round(tilt_deg, 3),
        "roll_deg": round(roll_deg, 3),
    }


def build_camera_motion(shot, camera_keys, fps):
    """Per-frame camera track: position, orientation, pan/tilt/roll — exact.

    This is the direct analogue of Blockout's ``camera_motion.json`` /
    ``metadata.json`` camera block and Motion Previs Studio's solved camera,
    but ground-truth rather than estimated.
    """
    camera_spec = shot.get("camera") or {}
    lens_mm = float(camera_spec.get("lens_mm", 35.0))

    frames = []
    for key in camera_keys:
        orient = camera_orientation(key.rotation_euler)
        frames.append({
            "frame": key.frame,
            "t": round(key.t, 5),
            "position": [round(c, 5) for c in key.position],
            "rotation_euler_deg": [round(math.degrees(a), 4) for a in key.rotation_euler],
            "pan_deg": orient["pan_deg"],
            "tilt_deg": orient["tilt_deg"],
            "roll_deg": orient["roll_deg"],
            "forward": orient["forward"],
        })

    # A compact move summary alongside the dense per-frame track: the authored
    # intent (dolly/orbit/track...) with its resolved endpoints.
    moves = []
    for move in sorted(
        (m for m in camera_spec.get("moves", []) if isinstance(m, dict)),
        key=lambda m: (m.get("start_t", 0.0), m.get("end_t", 0.0)),
    ):
        start_t, end_t = float(move.get("start_t", 0.0)), float(move.get("end_t", 0.0))
        moves.append({
            "type": move.get("type"),
            "start_t": round(start_t, 4),
            "end_t": round(end_t, 4),
            "start_frame": _frame_at(start_t, fps, camera_keys),
            "end_frame": _frame_at(end_t, fps, camera_keys),
            "target_id": move.get("target_id"),
            "easing": move.get("easing", "smooth"),
            "lens_mm": lens_mm,
        })

    return {
        "format": "previs.camera_motion",
        "version": BUNDLE_FORMAT_VERSION,
        "fps": int(fps),
        "frame_count": len(frames),
        "lens_mm": lens_mm,
        "units": "meters",
        "axis": "blender_z_up_right_handed",
        "notes": (
            "Ground-truth camera track exported from the previs authoring "
            "system. pan_deg is world azimuth (0=+X, CCW), tilt_deg is "
            "elevation (+up), roll_deg is rotation about the view axis. A "
            "Blender camera looks down local -Z with +Y up."
        ),
        "moves": moves,
        "frames": frames,
    }


def _frame_at(t, fps, camera_keys):
    frame = int(round(t * fps)) + 1
    if camera_keys:
        frame = max(camera_keys[0].frame, min(camera_keys[-1].frame, frame))
    return frame


# ---------------------------------------------------------------------------
# metadata — marks, lenses, timings (machine-readable shot digest)
# ---------------------------------------------------------------------------

def _sample_character(track, times):
    """Resolve a character's marks: position/facing/pose at each time."""
    marks = []
    for t in times:
        position, facing, pose = track.sample(t)
        marks.append({
            "t": round(t, 4),
            "position": [round(c, 4) for c in position],
            "facing_deg": round(float(facing), 2),
            "pose": pose,
            "speed_mps": round(track.speed_at(t), 3),
        })
    return marks


def build_metadata(shot, tracks, camera_keys, library, fps, render_settings=None):
    """A distilled, machine-readable digest of the shot: the marks a downstream
    tool (or a human diffing two takes) needs, without the authoring verbosity.
    """
    render_settings = render_settings or shot.get("render") or {}
    duration = float(shot["duration_seconds"])
    resolution = list(render_settings.get("resolution", [960, 544]))

    characters = []
    for character in shot.get("characters", []):
        if not isinstance(character, dict) or "id" not in character:
            continue
        track = tracks.get(character["id"])
        if track is None:
            continue
        # Mark times: every action boundary, deduped and sorted.
        times = {0.0, duration}
        for action in character.get("actions", []):
            if isinstance(action, dict):
                times.add(float(action.get("start_t", 0.0)))
                times.add(float(action.get("end_t", 0.0)))
        times = sorted(t for t in times if 0.0 <= t <= duration)
        asset = library.get("characters", character.get("asset_id", "")) if library else {}
        characters.append({
            "id": character["id"],
            "asset_id": character.get("asset_id"),
            "display_name": (asset or {}).get("display_name"),
            "aim_height_m": round(aim_height(asset), 3) if asset else None,
            "marks": _sample_character(track, times),
        })

    props = [
        {
            "id": p.get("id"),
            "asset_id": p.get("asset_id"),
            "position": [round(c, 4) for c in pad3(p.get("position", [0, 0, 0]))],
            "facing_deg": round(float(p.get("facing_deg", 0.0)), 2),
        }
        for p in shot.get("props", []) if isinstance(p, dict)
    ]

    camera_spec = shot.get("camera") or {}
    return {
        "format": "previs.metadata",
        "version": BUNDLE_FORMAT_VERSION,
        "shot_id": shot.get("shot_id"),
        "duration_s": duration,
        "fps": int(fps),
        "frame_count": max(1, int(round(duration * fps))),
        "resolution": resolution,
        "aspect_ratio": round(resolution[0] / resolution[1], 4) if resolution[1] else None,
        "lens_mm": float(camera_spec.get("lens_mm", 35.0)),
        "engine": render_settings.get("engine", "WORKBENCH"),
        "set": (shot.get("set") or {}).get("asset_id"),
        "stage": shot.get("stage") or {},
        "camera": {
            "lens_mm": float(camera_spec.get("lens_mm", 35.0)),
            "smoothing_s": float(camera_spec.get("smoothing_s", 0.0)),
            "move_count": len([m for m in camera_spec.get("moves", []) if isinstance(m, dict)]),
        },
        "characters": characters,
        "props": props,
        "continuity": shot.get("continuity"),
    }


# ---------------------------------------------------------------------------
# prompt — generated from the blocking, tailored per generator
# ---------------------------------------------------------------------------

_SHOT_SIZE_BY_DISTANCE = (
    (2.0, "close-up"),
    (3.0, "medium-close"),
    (4.5, "medium"),
    (7.0, "wide"),
    (float("inf"), "establishing wide"),
)

_MOVE_LANGUAGE = {
    "static": "The camera holds a locked-off {size} shot",
    "dolly": "The camera {dolly_dir} in a {size} shot",
    "track": "The camera tracks with the subject in a {size} shot",
    "orbit": "The camera arcs around the subject",
    "pan": "The camera pans across the space",
    "tilt": "The camera tilts",
}

_POSE_LANGUAGE = {
    "stand": "stands",
    "crouch": "crouches",
    "kneel": "kneels",
    "sit": "sits",
    "reach": "reaches out",
}


def _shot_size(distance_m):
    for limit, label in _SHOT_SIZE_BY_DISTANCE:
        if distance_m <= limit:
            return label
    return "wide"


def _first_camera_size(shot, tracks, library, camera_keys):
    """Approximate the opening shot size from camera-to-subject distance."""
    camera = shot.get("camera") or {}
    moves = [m for m in camera.get("moves", []) if isinstance(m, dict)]
    if not moves or not camera_keys:
        return "medium"
    first = min(moves, key=lambda m: m.get("start_t", 0.0))
    target_id = first.get("target_id")
    cam_pos = camera_keys[0].position
    if target_id and tracks.get(target_id) is not None:
        subject = tracks[target_id].sample(camera_keys[0].t)[0]
    elif "target_position" in first:
        subject = pad3(first["target_position"], 1.2)
    else:
        return "medium"
    return _shot_size(math.dist(cam_pos, subject))


def _describe_action(action, asset_name):
    kind = action.get("type")
    pose = action.get("pose")
    if kind == "walk_to":
        return f"{asset_name} walks across the space"
    if kind == "turn_to":
        return f"{asset_name} turns"
    if kind == "interact":
        verb = _POSE_LANGUAGE.get(pose, "reaches toward")
        return f"{asset_name} {verb} to interact"
    if kind == "mocap_clip":
        return f"{asset_name} performs a captured motion"
    if kind == "idle":
        verb = _POSE_LANGUAGE.get(pose, "holds still")
        return f"{asset_name} {verb}"
    return None


def _blocking_sentences(shot, library):
    """Prose describing what each character does, in beat order."""
    beats = []
    for character in shot.get("characters", []):
        if not isinstance(character, dict):
            continue
        asset = library.get("characters", character.get("asset_id", "")) if library else {}
        name = (asset or {}).get("display_name") or character.get("id", "the subject")
        for action in sorted(
            (a for a in character.get("actions", []) if isinstance(a, dict)),
            key=lambda a: a.get("start_t", 0.0),
        ):
            sentence = _describe_action(action, name)
            if sentence:
                beats.append((float(action.get("start_t", 0.0)), sentence))
    beats.sort(key=lambda b: b[0])
    # Collapse consecutive identical sentences (e.g. paused walk legs).
    out = []
    for _, sentence in beats:
        if not out or out[-1] != sentence:
            out.append(sentence)
    return out


def build_prompt_fragments(shot, tracks, library, camera_keys, generator="minimax"):
    """The prompt as addressable pieces, for the shot orchestrator.

    previs does not talk to the video model itself -- its output is handed to
    the orchestrator, which rolls these fragments into the full scene prompt
    (adding character/scene prose previs cannot know). prompt.<generator>.txt
    is rendered FROM these fragments, so the two can never drift.

    Phrasing is the live-validated replace/retain imperative form: name what
    each reference controls, then translate the grey blockout into the intended
    scene instead of asking the model to ignore it.
    """
    profile = GENERATOR_PROFILES.get(generator, GENERATOR_PROFILES["generic"])
    video_token = profile.get("video_token", "the reference video")
    picture_tmpl = profile.get("picture_token", "reference image {n}")
    camera = shot.get("camera") or {}
    duration = float(shot["duration_seconds"])

    # -- subjects: one line per reference, stating what it controls -----------
    subjects = [
        f"{video_token} controls only: motion path, body mechanics, performance "
        f"timing, staging, screen geography, camera position and framing, and "
        f"cut rhythm."
    ]
    replace_lines = []
    for n, character in enumerate(
        (c for c in shot.get("characters", []) if isinstance(c, dict)), start=1
    ):
        asset = library.get("characters", character.get("asset_id", "")) if library else {}
        name = (asset or {}).get("display_name") or character.get("id", f"character {n}")
        token = picture_tmpl.format(n=n)
        detail = (asset or {}).get("notes") or ""
        subjects.append(
            f"{token} controls identity, wardrobe and props: {name}."
            + (f" {detail}" if detail else "")
        )
        replace_lines.append(
            f"Replace the figure from {video_token} that marks {name}'s position "
            f"and timing with the character {name} from {token}."
        )

    # -- retention: explicit take/replace imperatives -------------------------
    retain_lines = [
        f"Retain the character poses and camera motion from {video_token}.",
        f"Retain the timing: every mark, turn and stop lands when it does in "
        f"{video_token}.",
    ]
    set_asset = library.get("sets", (shot.get("set") or {}).get("asset_id", "")) if library else {}
    set_name = (set_asset or {}).get("display_name") or (shot.get("set") or {}).get("asset_id")
    set_notes = (set_asset or {}).get("notes") or ""
    if set_name:
        scene_translation = (
            f"Replace the grey geometry with {set_name}."
            + (f" {set_notes}" if set_notes else "")
        )
    else:
        scene_translation = (
            "Replace the grey geometry with the scene described below."
        )
    replace_lines.append(scene_translation)

    # -- prose from the actual blocking ---------------------------------------
    size = _first_camera_size(shot, tracks, library, camera_keys)
    moves = sorted(
        (m for m in camera.get("moves", []) if isinstance(m, dict)),
        key=lambda m: m.get("start_t", 0.0),
    )
    camera_bits = []
    for move in moves:
        # A framing preset already names the shot ("an over-the-shoulder
        # shot"); prefer that over re-deriving generic move language.
        framing = move.get("_framing")
        if framing:
            camera_bits.append(f"The camera holds {framing}")
            continue
        template = _MOVE_LANGUAGE.get(move.get("type"))
        if not template:
            continue
        dolly_dir = "pushes in"
        if move.get("type") == "dolly" and "end_position" in move and "position" in move:
            start = pad3(move["position"], 1.6)
            end = pad3(move["end_position"], start[2])
            aim = pad3(move.get("target_position", [0, 0, 0]))
            dolly_dir = "pushes in" if math.dist(end, aim) < math.dist(start, aim) else "pulls back"
        camera_bits.append(template.format(size=size, dolly_dir=dolly_dir))

    shot_data = {
        "duration_s": duration,
        "lens_mm": camera.get("lens_mm", 35),
        "aspect": profile["aspect"],
        "max_duration_s": profile.get("max_duration_s"),
    }
    constraints = check_target_constraints(shot, profile)
    if constraints.get("kept_duration_s") is not None:
        shot_data["kept_duration_s"] = constraints["kept_duration_s"]

    return {
        "generator": generator,
        "subjects_block": subjects,
        "retain_lines": retain_lines,
        "replace_lines": replace_lines,
        "scene_translation": scene_translation,
        "camera_prose": "; then ".join(camera_bits) if camera_bits else f"A {size} shot",
        "action_prose": _blocking_sentences(shot, library),
        "shot_data": shot_data,
        "notes": shot.get("notes") or "",
    }


def _build_retention_prompt(fragments):
    """Render prompt text from fragments (the retention/imperative format)."""
    f = fragments
    lines = ["[SUBJECTS]"]
    lines.extend(f["subjects_block"])
    lines.append("")
    lines.append("[RETENTION]")
    lines.extend(f["retain_lines"])
    lines.extend(f["replace_lines"])
    lines.append("")
    lines.append("[SCENE]")
    lines.append(f["camera_prose"] + ".")
    if f["action_prose"]:
        lines.append("Action: " + "; ".join(f["action_prose"]) + ".")
    lines.append("")
    lines.append("[SHOT DATA]")
    sd = f["shot_data"]
    data_line = (f"Duration: {sd['duration_s']:g}s. Lens: {sd['lens_mm']:g}mm. "
                 f"Aspect: {sd['aspect']}.")
    if sd.get("kept_duration_s") is not None and sd["kept_duration_s"] < sd["duration_s"]:
        data_line += (f" The generator keeps only the first "
                      f"{sd['kept_duration_s']:.3f}s -- author the action to land "
                      f"inside that.")
    lines.append(data_line)
    cap = sd.get("max_duration_s")
    if cap and sd["duration_s"] > cap:
        lines.append(f"NOTE: target caps at {cap:g}s; trim or retime this "
                     f"{sd['duration_s']:g}s shot.")
    if f["notes"]:
        lines.append("")
        lines.append("[DIRECTOR NOTES]")
        lines.append(f["notes"])
    return "\n".join(lines).strip() + "\n"


def check_target_constraints(shot, profile, render_settings=None):
    """Pure math: what the target model actually keeps of this shot.

    MiniMax H3 resamples the guide video to ``frame_grid.fps``, then trims the
    frame count DOWN to the nearest ``k*n + offset`` grid point -- a 5.0s shot
    loses its final ~0.54s, which is usually the landing mark. Canvases must be
    multiples of ``canvas_multiple`` per axis or the workflow needs a resize
    node. Warns; never mutates the shot.
    """
    out = {"warnings": [], "kept_duration_s": None}
    duration = float(shot["duration_seconds"])
    grid = profile.get("frame_grid")
    if grid:
        k, offset, fps = grid["k"], grid["offset"], grid["fps"]
        frames = int(round(duration * fps))
        kept = frames
        while kept % k != offset:
            kept -= 1
        out["target_fps"] = fps
        out["resampled_frames"] = frames
        out["kept_frames"] = kept
        out["kept_duration_s"] = kept / fps
        if kept != frames:
            below = kept / fps
            above_frames = kept + k
            above = above_frames / fps
            out["warnings"].append(
                f"duration {duration:g}s -> {frames} frames @{fps}fps; the target "
                f"trims to {kept} ({below:.3f}s) -- the last "
                f"{(frames - kept) / fps:.3f}s of blocking is cut. Use "
                f"{below:.3f}s or {above:.3f}s."
            )
    multiple = profile.get("canvas_multiple")
    resolution = None
    if render_settings and render_settings.get("resolution"):
        resolution = render_settings["resolution"]
    elif (shot.get("render") or {}).get("resolution"):
        resolution = shot["render"]["resolution"]
    if multiple and resolution:
        w, h = int(resolution[0]), int(resolution[1])
        if w % multiple or h % multiple:
            sw = max(multiple, round(w / multiple) * multiple)
            sh = max(multiple, round(h / multiple) * multiple)
            out["canvas_valid"] = False
            out["warnings"].append(
                f"resolution {w}x{h} is not a multiple of {multiple}; the "
                f"workflow will need a resize. Use {sw}x{sh}."
            )
        else:
            out["canvas_valid"] = True
    return out


def build_prompt(shot, tracks, library, camera_keys, generator="generic"):
    """A cinematic prompt written from the actual blocking, per generator."""
    profile = GENERATOR_PROFILES.get(generator, GENERATOR_PROFILES["generic"])
    if profile.get("format") == "retention":
        return _build_retention_prompt(
            build_prompt_fragments(shot, tracks, library, camera_keys, generator))
    duration = float(shot["duration_seconds"])
    size = _first_camera_size(shot, tracks, library, camera_keys)
    camera = shot.get("camera") or {}
    moves = sorted(
        (m for m in camera.get("moves", []) if isinstance(m, dict)),
        key=lambda m: m.get("start_t", 0.0),
    )

    set_asset = library.get("sets", (shot.get("set") or {}).get("asset_id", "")) if library else {}
    set_line = (set_asset or {}).get("display_name") or (shot.get("set") or {}).get("asset_id")

    camera_bits = []
    for move in moves:
        template = _MOVE_LANGUAGE.get(move.get("type"))
        if not template:
            continue
        dolly_dir = "pushes in"
        if move.get("type") == "dolly" and "end_position" in move and "position" in move:
            start = pad3(move["position"], 1.6)
            end = pad3(move["end_position"], start[2])
            aim = pad3(move.get("target_position", [0, 0, 0]))
            dolly_dir = "pushes in" if math.dist(end, aim) < math.dist(start, aim) else "pulls back"
        camera_bits.append(template.format(size=size, dolly_dir=dolly_dir))
    camera_line = "; then ".join(camera_bits) if camera_bits else f"A {size} shot"

    action_lines = _blocking_sentences(shot, library)

    # Prefer any human-authored intent already in the shot.
    notes = shot.get("notes")

    lines = []
    lines.append(f"[REFERENCE USE]\n{profile['reference_note']}")
    lines.append("")
    lines.append("[SCENE]")
    if set_line:
        lines.append(f"Location: {set_line}.")
    lines.append(f"{camera_line}.")
    if action_lines:
        lines.append("Action: " + "; ".join(action_lines) + ".")
    lines.append("")
    lines.append("[SHOT DATA]")
    lines.append(f"Duration: {duration:g}s. Lens: {camera.get('lens_mm', 35):g}mm. "
                 f"Aspect: {profile['aspect']}.")
    cap = profile.get("max_duration_s")
    if cap and duration > cap:
        lines.append(f"NOTE: target caps at {cap:g}s; trim or retime this {duration:g}s shot.")
    if notes:
        lines.append("")
        lines.append("[DIRECTOR NOTES]")
        lines.append(notes)

    return "\n".join(lines).strip() + "\n"


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------

def _dump(path, payload):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_sidecars(out_dir, shot, tracks, camera_keys, library, fps,
                   generators=DEFAULT_GENERATORS, render_settings=None):
    """Write the host-computable bundle files (everything except the rendered
    passes and the pose capture, which need Blender).

    Returns a dict of ``{logical_name: relative_path}`` for the manifest.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}

    camera_motion = build_camera_motion(shot, camera_keys, fps)
    _dump(out_dir / "camera_motion.json", camera_motion)
    written["camera_motion"] = "camera_motion.json"

    metadata = build_metadata(shot, tracks, camera_keys, library, fps, render_settings)
    _dump(out_dir / "metadata.json", metadata)
    written["metadata"] = "metadata.json"

    for generator in generators:
        prompt = build_prompt(shot, tracks, library, camera_keys, generator)
        name = "prompt.txt" if generator == "generic" else f"prompt.{generator}.txt"
        (out_dir / name).write_text(prompt, encoding="utf-8")
        written[f"prompt_{generator}"] = name
        profile = GENERATOR_PROFILES.get(generator, {})
        if profile.get("format") == "retention":
            fragments = build_prompt_fragments(shot, tracks, library, camera_keys, generator)
            frag_name = f"prompt_fragments.{generator}.json"
            _dump(out_dir / frag_name, fragments)
            written[f"prompt_fragments_{generator}"] = frag_name
            constraints = check_target_constraints(shot, profile, render_settings)
            metadata.setdefault("target_constraints", {})[generator] = constraints
            for warning in constraints["warnings"]:
                print(f"[previs] WARNING ({generator}): {warning}")
            # re-write metadata now it carries the constraints
            _dump(out_dir / "metadata.json", metadata)

    return written, metadata


def write_manifest(out_dir, shot, entries, fps, extra=None):
    """Record what the bundle actually contains — the last thing written, so a
    consumer can trust it, and so an incomplete export is obvious."""
    out_dir = Path(out_dir)
    manifest = {
        "format": "previs.bundle_manifest",
        "version": BUNDLE_FORMAT_VERSION,
        # The orchestrator-facing contract (docs/BUNDLE_CONTRACT.md): bump on
        # any breaking change to file roles or the fields promised here.
        "contract_version": "1.0",
        "shot_id": shot.get("shot_id"),
        "fps": int(fps),
        "duration_s": float(shot["duration_seconds"]),
        "files": dict(sorted(entries.items())),
    }
    if extra:
        manifest.update(extra)
    _dump(out_dir / "bundle_manifest.json", manifest)
    return manifest


def write_readme(out_dir, shot, entries):
    """A short human-facing index of the bundle."""
    out_dir = Path(out_dir)
    lines = [
        f"Previs control-layer bundle — {shot.get('shot_id', 'shot')}",
        "=" * 48,
        "",
        "Grey-box previs exported for a downstream AI video generator.",
        "The camera track and pose landmarks are exact (authored, not solved).",
        "",
        "Contents:",
    ]
    descriptions = {
        "reference": "the grey-box control render (motion + staging reference)",
        "depth": "depth pass for depth-conditioned (ControlNet) workflows",
        "camera_motion": "per-frame camera position + pan/tilt/roll, exact",
        "pose_landmarks": "per-frame 3D and 2D joint positions, exact",
        "metadata": "machine-readable marks, lenses, timings",
        "prompt_generic": "cinematic prompt generated from the blocking",
        "blocking_diagram": "top-down staging diagram",
    }
    for name, rel in sorted(entries.items()):
        desc = descriptions.get(name, "")
        lines.append(f"  {rel:28} {desc}".rstrip())
    lines.append("")
    lines.append("Feed reference.mp4 (and depth.mp4 where supported) as the motion")
    lines.append("reference; take all appearance from the prompt text, never from")
    lines.append("the grey proxy geometry.")
    (out_dir / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return "README.txt"
