"""Shot specification schema and validator.

A shot spec is plain JSON describing *director intent* — who is where, what they
do when, and what the camera does. It never contains Blender concepts. The
compiler turns it into filmmaking-API calls; nothing here imports ``bpy``.

Stage convention (documented once, relied on everywhere):

    +X is screen-right, +Y is upstage (away from a default camera parked at -Y),
    +Z is up. Positions are metres. ``facing_deg`` is degrees counter-clockwise
    from +X, so 0 faces screen-right and 90 faces upstage/away.

Times are seconds from the start of the shot.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCHEMA_VERSION = "0.1"

STATUSES = ("needs_blocking", "blocked")

# Character actions. Each carries start_t/end_t.
ACTION_TYPES = {
    "walk_to": ("position",),
    "idle": (),
    "turn_to": (),
    "interact": (),
    "mocap_clip": ("clip_id",),
}

# Camera moves. Each carries start_t/end_t.
CAMERA_MOVE_TYPES = {
    "static": ("position",),
    "track": ("position", "target_id"),
    "dolly": ("position", "end_position"),
    "orbit": ("radius_m",),
    "pan": ("position", "start_deg", "end_deg"),
    "tilt": ("position", "start_deg", "end_deg"),
    # Named framing preset -- expanded into one of the above by
    # previs.framing.expand_presets before anything else sees it.
    "preset": ("name", "subject_id"),
}

POSES = ("stand", "crouch", "kneel", "sit", "reach")

MOCAP_ROOT_MODES = ("lock_xy", "from_clip", "blend")
MOCAP_SOURCE_UP_AXES = ("y", "z")

RENDER_ENGINES = ("WORKBENCH", "EEVEE")

# How this shot's camera relates to the shot it continues from. "match" means
# the clips are genuinely continuous and the camera must not jump; "cut" means a
# new setup, free to go anywhere. Decided from the source, not assumed.
CAMERA_CARRY_MODES = ("match", "cut")


def _is_vec(value, length_options=(2, 3)):
    return (
        isinstance(value, (list, tuple))
        and len(value) in length_options
        and all(isinstance(n, (int, float)) and not isinstance(n, bool) for n in value)
    )


def _check_span(obj, label, duration, errors):
    """Validate a start_t/end_t pair on an action or camera move."""
    start_t, end_t = obj.get("start_t"), obj.get("end_t")
    for name, value in (("start_t", start_t), ("end_t", end_t)):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            errors.append(f"{label}: {name} must be a number")
            return
    if end_t < start_t:
        errors.append(f"{label}: end_t ({end_t}) is before start_t ({start_t})")
    if start_t < -1e-6:
        errors.append(f"{label}: start_t ({start_t}) is negative")
    if duration is not None and end_t > duration + 1e-6:
        errors.append(
            f"{label}: end_t ({end_t}) runs past the shot duration ({duration}s)"
        )


def validate(shot, *, require_blocked=None):
    """Return a list of human-readable problems. Empty list means valid.

    ``require_blocked`` defaults to the shot's own ``status``: a shot marked
    "blocked" must actually have blocking, a "needs_blocking" stub need not.
    """
    errors = []
    if not isinstance(shot, dict):
        return ["shot spec must be a JSON object"]

    if shot.get("schema_version") != SCHEMA_VERSION:
        errors.append(
            f"schema_version must be {SCHEMA_VERSION!r}, got {shot.get('schema_version')!r}"
        )

    shot_id = shot.get("shot_id")
    if not isinstance(shot_id, str) or not shot_id.strip():
        errors.append("shot_id must be a non-empty string")

    status = shot.get("status")
    if status not in STATUSES:
        errors.append(f"status must be one of {STATUSES}, got {status!r}")
    if require_blocked is None:
        require_blocked = status == "blocked"

    duration = shot.get("duration_seconds")
    if not isinstance(duration, (int, float)) or isinstance(duration, bool) or duration <= 0:
        errors.append("duration_seconds must be a positive number")
        duration = None

    fps = shot.get("fps", 12)
    if not isinstance(fps, int) or isinstance(fps, bool) or fps <= 0:
        errors.append("fps must be a positive integer")

    stage = shot.get("stage") or {}
    if not isinstance(stage, dict):
        errors.append("stage must be an object")
    elif "size_m" in stage and not _is_vec(stage["size_m"], (2,)):
        errors.append("stage.size_m must be [width, depth] in metres")

    # --- characters -------------------------------------------------------
    characters = shot.get("characters", [])
    seen_ids = set()
    if not isinstance(characters, list):
        errors.append("characters must be a list")
        characters = []
    for index, char in enumerate(characters):
        label = f"characters[{index}]"
        if not isinstance(char, dict):
            errors.append(f"{label} must be an object")
            continue
        char_id = char.get("id")
        label = f"character {char_id!r}" if isinstance(char_id, str) else label
        if not isinstance(char_id, str) or not char_id.strip():
            errors.append(f"{label}: id must be a non-empty string")
        elif char_id in seen_ids:
            errors.append(f"{label}: duplicate id")
        else:
            seen_ids.add(char_id)
        if not isinstance(char.get("asset_id"), str):
            errors.append(f"{label}: asset_id must be a string")
        if "start_position" in char and not _is_vec(char["start_position"]):
            errors.append(f"{label}: start_position must be [x, y] or [x, y, z]")
        elif require_blocked and "start_position" not in char:
            errors.append(f"{label}: start_position is required once blocked")

        actions = char.get("actions", [])
        if not isinstance(actions, list):
            errors.append(f"{label}: actions must be a list")
            continue
        if require_blocked and not actions:
            errors.append(f"{label}: has no actions but the shot is marked blocked")
        for action_index, action in enumerate(actions):
            action_label = f"{label} action[{action_index}]"
            if not isinstance(action, dict):
                errors.append(f"{action_label} must be an object")
                continue
            action_type = action.get("type")
            if action_type not in ACTION_TYPES:
                errors.append(
                    f"{action_label}: type must be one of "
                    f"{tuple(ACTION_TYPES)}, got {action_type!r}"
                )
                continue
            _check_span(action, action_label, duration, errors)
            for field in ACTION_TYPES[action_type]:
                if field not in action:
                    errors.append(f"{action_label}: {action_type} requires {field!r}")
            if "position" in action and not _is_vec(action["position"]):
                errors.append(f"{action_label}: position must be [x, y] or [x, y, z]")
            if "pose" in action and action["pose"] not in POSES:
                errors.append(
                    f"{action_label}: pose must be one of {POSES}, got {action['pose']!r}"
                )
            if action_type in ("turn_to", "interact"):
                if "target_id" not in action and "facing_deg" not in action:
                    errors.append(
                        f"{action_label}: {action_type} needs target_id or facing_deg"
                    )
            if action_type == "mocap_clip":
                clip_id = action.get("clip_id")
                if not isinstance(clip_id, str) or not clip_id.strip():
                    errors.append(f"{action_label}: clip_id must be a non-empty string")

                for field in (
                    "clip_t0_s",
                    "clip_t1_s",
                    "clip_loop_from_s",
                    "clip_loop_to_s",
                    "source_fps",
                    "blend_in_s",
                    "blend_out_s",
                    "pose_weight",
                    "root_scale_m",
                ):
                    value = action.get(field)
                    if value is not None and (
                        not isinstance(value, (int, float)) or isinstance(value, bool)
                    ):
                        errors.append(f"{action_label}: {field} must be a number")

                if (
                    isinstance(action.get("source_fps"), (int, float))
                    and not isinstance(action.get("source_fps"), bool)
                    and action["source_fps"] <= 0
                ):
                    errors.append(f"{action_label}: source_fps must be > 0")

                for field in ("blend_in_s", "blend_out_s"):
                    value = action.get(field)
                    if (
                        isinstance(value, (int, float))
                        and not isinstance(value, bool)
                        and value < 0
                    ):
                        errors.append(f"{action_label}: {field} must be >= 0")

                root_scale = action.get("root_scale_m")
                if (
                    isinstance(root_scale, (int, float))
                    and not isinstance(root_scale, bool)
                    and root_scale <= 0
                ):
                    errors.append(f"{action_label}: root_scale_m must be > 0")

                pose_weight = action.get("pose_weight")
                if (
                    isinstance(pose_weight, (int, float))
                    and not isinstance(pose_weight, bool)
                    and not (0.0 <= pose_weight <= 1.0)
                ):
                    errors.append(f"{action_label}: pose_weight must be in [0, 1]")

                root_mode = action.get("root_mode")
                if root_mode is not None and root_mode not in MOCAP_ROOT_MODES:
                    errors.append(
                        f"{action_label}: root_mode must be one of {MOCAP_ROOT_MODES}, "
                        f"got {root_mode!r}"
                    )

                source_up = action.get("source_up_axis")
                if source_up is not None and source_up not in MOCAP_SOURCE_UP_AXES:
                    errors.append(
                        f"{action_label}: source_up_axis must be one of "
                        f"{MOCAP_SOURCE_UP_AXES}, got {source_up!r}"
                    )

                joint_map = action.get("joint_map")
                if joint_map is not None:
                    if not isinstance(joint_map, dict):
                        errors.append(f"{action_label}: joint_map must be an object")
                    else:
                        for source_joint, target_joint in joint_map.items():
                            if not isinstance(source_joint, str) or not source_joint.strip():
                                errors.append(
                                    f"{action_label}: joint_map keys must be non-empty strings"
                                )
                                break
                            if not isinstance(target_joint, str) or not target_joint.strip():
                                errors.append(
                                    f"{action_label}: joint_map values must be non-empty strings"
                                )
                                break

                loop_from = action.get("clip_loop_from_s")
                loop_to = action.get("clip_loop_to_s")
                if (
                    isinstance(loop_from, (int, float))
                    and not isinstance(loop_from, bool)
                    and isinstance(loop_to, (int, float))
                    and not isinstance(loop_to, bool)
                    and loop_to <= loop_from
                ):
                    errors.append(
                        f"{action_label}: clip_loop_to_s must be greater than clip_loop_from_s"
                    )

    # --- props ------------------------------------------------------------
    props = shot.get("props", [])
    if not isinstance(props, list):
        errors.append("props must be a list")
        props = []
    for index, prop in enumerate(props):
        label = f"props[{index}]"
        if not isinstance(prop, dict):
            errors.append(f"{label} must be an object")
            continue
        prop_id = prop.get("id")
        if not isinstance(prop_id, str) or not prop_id.strip():
            errors.append(f"{label}: id must be a non-empty string")
        elif prop_id in seen_ids:
            errors.append(f"{label}: id {prop_id!r} collides with another object")
        else:
            seen_ids.add(prop_id)
        if not isinstance(prop.get("asset_id"), str):
            errors.append(f"{label}: asset_id must be a string")
        if "position" in prop and not _is_vec(prop["position"]):
            errors.append(f"{label}: position must be [x, y] or [x, y, z]")

    # Fixtures declared by the set are present in every shot at that location,
    # so they are valid targets even though the shot does not restate them.
    set_asset_id = (shot.get("set") or {}).get("asset_id")
    if set_asset_id:
        try:
            from .asset_library import AssetLibrary

            for fixture in AssetLibrary().get("sets", set_asset_id).get("fixtures") or []:
                if isinstance(fixture, dict) and fixture.get("id"):
                    seen_ids.add(fixture["id"])
        except Exception:
            # Validation must still work without a readable asset library.
            pass

    # --- camera -----------------------------------------------------------
    camera = shot.get("camera")
    if camera is None:
        if require_blocked:
            errors.append("camera is required once blocked")
    elif not isinstance(camera, dict):
        errors.append("camera must be an object")
    else:
        lens = camera.get("lens_mm", 35)
        if not isinstance(lens, (int, float)) or isinstance(lens, bool) or lens <= 0:
            errors.append("camera.lens_mm must be a positive number")
        smoothing = camera.get("smoothing_s")
        if smoothing is not None:
            if not isinstance(smoothing, (int, float)) or isinstance(smoothing, bool):
                errors.append("camera.smoothing_s must be a number")
            elif smoothing < 0:
                errors.append("camera.smoothing_s must be >= 0")
        moves = camera.get("moves", [])
        if not isinstance(moves, list):
            errors.append("camera.moves must be a list")
            moves = []
        if require_blocked and not moves:
            errors.append("camera has no moves but the shot is marked blocked")
        for index, move in enumerate(moves):
            label = f"camera.moves[{index}]"
            if not isinstance(move, dict):
                errors.append(f"{label} must be an object")
                continue
            move_type = move.get("type")
            if move_type not in CAMERA_MOVE_TYPES:
                errors.append(
                    f"{label}: type must be one of "
                    f"{tuple(CAMERA_MOVE_TYPES)}, got {move_type!r}"
                )
                continue
            _check_span(move, label, duration, errors)
            for field in CAMERA_MOVE_TYPES[move_type]:
                if field not in move:
                    errors.append(f"{label}: {move_type} requires {field!r}")
            for field in ("position", "end_position", "center_position", "target_position"):
                if field in move and not _is_vec(move[field]):
                    errors.append(f"{label}: {field} must be [x, y] or [x, y, z]")
            if "look_ahead_s" in move:
                value = move["look_ahead_s"]
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    errors.append(f"{label}: look_ahead_s must be a number")
                elif value < 0:
                    errors.append(f"{label}: look_ahead_s must be >= 0")
            if move_type == "orbit" and "center_id" not in move and "center_position" not in move:
                errors.append(f"{label}: orbit needs center_id or center_position")
            if move_type == "orbit":
                for field in ("radius_m", "height_m"):
                    value = move.get(field)
                    if value is None:
                        continue
                    ok = isinstance(value, (int, float)) and not isinstance(value, bool)
                    if not ok and isinstance(value, list) and len(value) == 2:
                        ok = all(
                            isinstance(v, (int, float)) and not isinstance(v, bool)
                            for v in value
                        )
                    if not ok:
                        errors.append(
                            f"{label}: {field} must be a number, or [start, end] to "
                            "spiral/rise over the move"
                        )
            for field in ("target_id", "center_id"):
                ref = move.get(field)
                if isinstance(ref, str) and ref not in seen_ids:
                    errors.append(
                        f"{label}: {field} {ref!r} is not a character or prop in this shot"
                    )

        # Overlapping moves would fight over the camera transform.
        spans = sorted(
            (m["start_t"], m["end_t"], i)
            for i, m in enumerate(moves)
            if isinstance(m, dict)
            and isinstance(m.get("start_t"), (int, float))
            and isinstance(m.get("end_t"), (int, float))
        )
        for (_, prev_end, prev_i), (next_start, _, next_i) in zip(spans, spans[1:]):
            if next_start < prev_end - 1e-6:
                errors.append(
                    f"camera.moves[{prev_i}] and camera.moves[{next_i}] overlap in time; "
                    "camera moves must not run concurrently"
                )

    # --- continuity -------------------------------------------------------
    continuity = shot.get("continuity")
    if continuity is not None:
        if not isinstance(continuity, dict):
            errors.append("continuity must be an object")
        else:
            if "order" in continuity and not isinstance(continuity["order"], int):
                errors.append("continuity.order must be an integer")
            if "continues_from" in continuity and not isinstance(
                continuity["continues_from"], (str, type(None))
            ):
                errors.append("continuity.continues_from must be a shot_id string or null")
            if continuity.get("continues_from") == shot_id:
                errors.append("continuity.continues_from points at this shot itself")
            carry = continuity.get("carry")
            if carry is not None:
                if not isinstance(carry, dict):
                    errors.append("continuity.carry must be an object")
                else:
                    mode = carry.get("camera", "cut")
                    if mode not in CAMERA_CARRY_MODES:
                        errors.append(
                            f"continuity.carry.camera must be one of "
                            f"{CAMERA_CARRY_MODES}, got {mode!r}"
                        )
                    if "position" in carry and not isinstance(carry["position"], bool):
                        errors.append("continuity.carry.position must be true or false")

    # --- render -----------------------------------------------------------
    render = shot.get("render") or {}
    if not isinstance(render, dict):
        errors.append("render must be an object")
    else:
        engine = render.get("engine", "WORKBENCH")
        if engine not in RENDER_ENGINES:
            errors.append(f"render.engine must be one of {RENDER_ENGINES}, got {engine!r}")
        if "resolution" in render and not _is_vec(render["resolution"], (2,)):
            errors.append("render.resolution must be [width, height]")

    return errors


def load_shot(path):
    """Read and validate a shot file. Raises ValueError on invalid specs."""
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        shot = json.load(handle)
    errors = validate(shot)
    if errors:
        joined = "\n  - ".join(errors)
        raise ValueError(f"{path} is not a valid shot spec:\n  - {joined}")
    return shot


def _main(argv):
    """Validate every shot file given (or all of shots/ if none given)."""
    paths = [Path(a) for a in argv[1:]]
    if not paths:
        paths = sorted((Path(__file__).resolve().parent.parent / "shots").rglob("*.json"))
    if not paths:
        print("no shot files found")
        return 0

    failures = 0
    for path in paths:
        try:
            with path.open(encoding="utf-8") as handle:
                shot = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"FAIL {path}: {exc}")
            failures += 1
            continue
        errors = validate(shot)
        if errors:
            failures += 1
            print(f"FAIL {path}")
            for error in errors:
                print(f"     - {error}")
        else:
            print(f"ok   {path}  [{shot.get('status')}]")
    print(f"\n{len(paths) - failures}/{len(paths)} shot files valid")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
