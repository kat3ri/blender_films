"""Cross-shot continuity: carry a shot's end state into the next one.

Both source formats chain clips — Fortress marks segments ``is_chain_start:
false`` with "chains from the previous clip"; MiniMax says "continuing directly
from Job 1". A character who ends one shot at the top of the stairs must begin
the next one there, or the generated clips will not cut together.

Two kinds of carry, and they are **independent flags** — either, both or
neither:

* **spatial** — character position, facing and pose. Carried by default within a
  chain; a person does not teleport between adjacent clips. Turn it off for a
  deliberate discontinuity in the same location, such as a time skip.
* **camera** — only carried when the source says the clips are continuous. A new
  setup is a cut, and a cut is free to put the camera anywhere. The importer
  records which case it found and why; blocking can override it.

The everyday combination is ``position: true`` with ``camera: "cut"``: the scene
continues, everyone stays exactly where they were, and only the setup changes.
That is what a new angle on a continuing scene means.

Nothing here imports ``bpy``: it reuses the same trajectory maths the compiler
does, so the end state computed here is exactly the one that gets rendered.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from .asset_library import AssetLibrary, expand_fixtures
from .motion import build_camera_keys, build_tracks

# Deltas larger than these are worth telling a human about.
POSITION_TOLERANCE_M = 0.25
FACING_TOLERANCE_DEG = 20.0
CAMERA_TOLERANCE_M = 0.5

CAMERA_MODES = ("match", "cut")


def end_state(shot, library=None):
    """Where everything is when this shot finishes."""
    library = library or AssetLibrary()
    expand_fixtures(shot, library)
    duration = float(shot["duration_seconds"])
    tracks = build_tracks(shot, library)

    characters = {}
    for object_id, track in tracks.items():
        position, facing, pose = track.sample(duration)
        characters[object_id] = {
            "position": [round(v, 4) for v in position],
            "facing_deg": round(facing, 2),
            "pose": pose,
        }

    keys = build_camera_keys(shot, tracks, library)
    last = keys[-1]
    return {
        "shot_id": shot.get("shot_id"),
        "characters": characters,
        "camera": {
            "position": [round(v, 4) for v in last.position],
            "rotation_euler": [round(v, 5) for v in last.rotation_euler],
            "lens_mm": float((shot.get("camera") or {}).get("lens_mm", 35.0)),
        },
        "set_asset_id": (shot.get("set") or {}).get("asset_id"),
    }


def start_state(shot, library=None):
    """Where everything is when this shot begins."""
    library = library or AssetLibrary()
    expand_fixtures(shot, library)
    tracks = build_tracks(shot, library)

    characters = {}
    for object_id, track in tracks.items():
        position, facing, pose = track.sample(0.0)
        characters[object_id] = {
            "position": [round(v, 4) for v in position],
            "facing_deg": round(facing, 2),
            "pose": pose,
        }

    keys = build_camera_keys(shot, tracks, library)
    first = keys[0]
    return {
        "shot_id": shot.get("shot_id"),
        "characters": characters,
        "camera": {
            "position": [round(v, 4) for v in first.position],
            "rotation_euler": [round(v, 5) for v in first.rotation_euler],
            "lens_mm": float((shot.get("camera") or {}).get("lens_mm", 35.0)),
        },
        "set_asset_id": (shot.get("set") or {}).get("asset_id"),
    }


def _distance(a, b):
    return math.dist(a[:3], b[:3])


def _angle_delta(a, b):
    delta = (float(a) - float(b)) % 360.0
    return min(delta, 360.0 - delta)


def camera_mode(shot):
    """'match' or 'cut' — how this shot's camera relates to its predecessor."""
    carry = ((shot.get("continuity") or {}).get("carry") or {})
    mode = carry.get("camera", "cut")
    return mode if mode in CAMERA_MODES else "cut"


def compare(predecessor, successor, library=None):
    """Report continuity breaks between two consecutive shots.

    Returns a list of {severity, message} — 'break' for things that will read as
    a jump, 'note' for things worth knowing but probably intentional.
    """
    library = library or AssetLibrary()
    before = end_state(predecessor, library)
    after = start_state(successor, library)
    issues = []

    if before["set_asset_id"] != after["set_asset_id"]:
        issues.append(
            {
                "severity": "note",
                "message": f"set changes from {before['set_asset_id']!r} to "
                f"{after['set_asset_id']!r} — continuity carry does not apply "
                "across a location change",
            }
        )
        return issues

    # A shot that deliberately does not carry position (a time skip in the same
    # location) is allowed to move people; report it, but not as a break.
    position_severity = "break" if carries_position(successor) else "note"

    shared = set(before["characters"]) & set(after["characters"])
    for object_id in sorted(shared):
        was, now = before["characters"][object_id], after["characters"][object_id]
        gap = _distance(was["position"], now["position"])
        if gap > POSITION_TOLERANCE_M:
            issues.append(
                {
                    "severity": position_severity,
                    "message": f"{object_id} ends {predecessor['shot_id']} at "
                    f"{was['position']} but starts {successor['shot_id']} at "
                    f"{now['position']} — {gap:.2f}m apart",
                }
            )
        turn = _angle_delta(was["facing_deg"], now["facing_deg"])
        if turn > FACING_TOLERANCE_DEG:
            issues.append(
                {
                    "severity": position_severity,
                    "message": f"{object_id} faces {was['facing_deg']:.0f} deg at the end "
                    f"of {predecessor['shot_id']} but {now['facing_deg']:.0f} deg at the "
                    f"start of {successor['shot_id']} — {turn:.0f} deg snap",
                }
            )
        if was["pose"] != now["pose"]:
            issues.append(
                {
                    "severity": position_severity,
                    "message": f"{object_id} is {was['pose']!r} at the end of "
                    f"{predecessor['shot_id']} but {now['pose']!r} at the start of "
                    f"{successor['shot_id']}",
                }
            )

    for object_id in sorted(set(before["characters"]) - shared):
        issues.append(
            {
                "severity": "note",
                "message": f"{object_id} is in {predecessor['shot_id']} but not in "
                f"{successor['shot_id']}",
            }
        )

    if camera_mode(successor) == "match":
        gap = _distance(before["camera"]["position"], after["camera"]["position"])
        if gap > CAMERA_TOLERANCE_M:
            issues.append(
                {
                    "severity": "break",
                    "message": f"camera is marked 'match' but jumps {gap:.2f}m between "
                    f"shots ({before['camera']['position']} -> "
                    f"{after['camera']['position']}); the clips will not chain cleanly",
                }
            )
        if before["camera"]["lens_mm"] != after["camera"]["lens_mm"]:
            issues.append(
                {
                    "severity": "break",
                    "message": f"camera is marked 'match' but the lens changes "
                    f"{before['camera']['lens_mm']}mm -> {after['camera']['lens_mm']}mm",
                }
            )

    return issues


def carries_position(shot):
    """Whether this shot inherits where everyone was standing.

    Independent of the camera flag: the usual case for a continuing scene is
    position carried, camera cut — everyone stays exactly where they were and
    only the setup changes. Set false for a deliberate discontinuity in the same
    location, such as a time skip.
    """
    carry = ((shot.get("continuity") or {}).get("carry") or {})
    return bool(carry.get("position", True))


def apply_carry(successor, predecessor, library=None):
    """Seed ``successor``'s start state from ``predecessor``'s end state.

    Mutates and returns ``successor``. Position and camera carry independently:
    either, both or neither, according to this shot's continuity flags.
    """
    library = library or AssetLibrary()
    before = end_state(predecessor, library)
    changes = []

    if carries_position(successor):
        for character in successor.get("characters", []):
            if not isinstance(character, dict):
                continue
            was = before["characters"].get(character.get("id"))
            if not was:
                continue
            character["start_position"] = list(was["position"])
            character["start_facing_deg"] = was["facing_deg"]
            if was["pose"] != "stand":
                character["start_pose"] = was["pose"]
            changes.append(
                f"{character['id']} starts at {was['position']} facing "
                f"{was['facing_deg']:.0f} deg ({was['pose']})"
            )

    if camera_mode(successor) == "match":
        moves = (successor.get("camera") or {}).get("moves") or []
        if moves:
            moves[0]["position"] = list(before["camera"]["position"])
            successor["camera"]["lens_mm"] = before["camera"]["lens_mm"]
            changes.append(
                f"camera opens at {before['camera']['position']} on a "
                f"{before['camera']['lens_mm']}mm lens, matching the previous shot"
            )

    return successor, changes


def resolve_chain(shot_paths):
    """Order shot files into chains using their continuity metadata.

    Returns a list of chains, each a list of (path, shot) in playback order.
    Shots without continuity metadata each form their own single-shot chain.
    """
    loaded = []
    for path in shot_paths:
        with Path(path).open(encoding="utf-8") as handle:
            loaded.append((Path(path), json.load(handle)))

    by_id = {shot.get("shot_id"): (path, shot) for path, shot in loaded}
    successors = {}
    for path, shot in loaded:
        previous = (shot.get("continuity") or {}).get("continues_from")
        if previous and previous in by_id:
            successors[previous] = shot.get("shot_id")

    chained = set(successors.values())
    chains = []
    for path, shot in sorted(loaded, key=lambda item: (item[1].get("continuity") or {}).get("order", 0)):
        shot_id = shot.get("shot_id")
        if shot_id in chained:
            continue  # not a chain head; it will be reached from its predecessor
        chain, cursor = [], shot_id
        seen = set()
        while cursor in by_id and cursor not in seen:
            seen.add(cursor)
            chain.append(by_id[cursor])
            cursor = successors.get(cursor)
        chains.append(chain)
    return chains
