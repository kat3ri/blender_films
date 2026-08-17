"""Named framing presets — cinematic vocabulary over the raw move types.

The six move types in ``schema.CAMERA_MOVE_TYPES`` are the assembly language:
they say where the camera *is*, not what the shot *is*. A director says
"over-the-shoulder on Mauryl" or "low-angle two-shot", and that phrase implies
a position, a distance, a height and an aim — all derivable from where the
subjects actually stand.

A preset is therefore pure sugar. ``expand_presets`` runs before
``build_camera_keys`` and rewrites each ``{"type": "preset", ...}`` entry into
exactly the ``static``/``dolly``/``orbit`` dict a human would have authored, so
everything downstream — validation, key baking, bounds checking, the prompt's
move language — is unchanged and none of it needs to know presets exist.

Two things are kept from the preset for the prompt's benefit: ``_preset``
(the name) and ``_framing`` (human phrase), which ``bundle`` prefers over the
generic move language so the text says "an over-the-shoulder shot" rather than
"a locked-off medium shot".

Distances agree with ``bundle._SHOT_SIZE_BY_DISTANCE`` so the rendered framing
and the generated prose can never disagree about what "medium" means.
"""

from __future__ import annotations

import math

from .asset_library import aim_height
from .motion import pad3

# Distance from subject, chosen to sit mid-band of bundle's size table so
# small staging differences don't flip the reported shot size.
SHOT_DISTANCES = {
    "cu": 1.6,
    "med_close": 2.5,
    "med": 3.8,
    "wide": 6.0,
    "establishing": 9.0,
}

# Height multipliers applied to the subject's aim height.
_ANGLE_HEIGHTS = {
    "eye": 1.0,
    "low": 0.35,
    "high": 1.75,
    "top": 3.2,
}

PRESET_NAMES = (
    "single_cu", "single_med", "single_wide", "establishing",
    "ots", "two_shot",
    "low_angle", "high_angle", "top_down", "dutch",
    "push_in", "pull_back",
)

_PHRASES = {
    "single_cu": "a close-up",
    "single_med": "a medium shot",
    "single_wide": "a wide shot",
    "establishing": "an establishing wide",
    "ots": "an over-the-shoulder shot",
    "two_shot": "a two-shot",
    "low_angle": "a low-angle shot",
    "high_angle": "a high-angle shot",
    "top_down": "a top-down shot",
    "dutch": "a dutch-angle shot",
    "push_in": "a slow push-in",
    "pull_back": "a pull-back reveal",
}


def _subject_state(subject_id, tracks, t):
    """(position, facing_deg) of a subject at time ``t``."""
    track = tracks.get(subject_id)
    if track is None:
        raise ValueError(
            f"framing preset references unknown subject {subject_id!r}")
    position, facing_deg, _pose = track.sample(t)
    return list(position), float(facing_deg)


def _subject_aim(subject_id, shot, library, position):
    """Aim point at the subject's head/chest height."""
    asset = {}
    for character in shot.get("characters", []):
        if isinstance(character, dict) and character.get("id") == subject_id:
            asset = library.get("characters", character.get("asset_id", "")) if library else {}
            break
    return [position[0], position[1], position[2] + aim_height(asset or {})]


def _offset_from(position, bearing_deg, distance, height):
    """A point ``distance`` away along ``bearing_deg``, at absolute ``height``."""
    rad = math.radians(bearing_deg)
    return [
        position[0] + distance * math.cos(rad),
        position[1] + distance * math.sin(rad),
        height,
    ]


def _resolve(preset, shot, tracks, library):
    """One preset dict -> one concrete move dict."""
    name = preset.get("name")
    if name not in PRESET_NAMES:
        raise ValueError(
            f"unknown framing preset {name!r}; expected one of {PRESET_NAMES}")

    start_t = float(preset.get("start_t", 0.0))
    end_t = float(preset.get("end_t", start_t))
    subject_id = preset.get("subject_id")
    if not subject_id:
        raise ValueError(f"framing preset {name!r} needs a subject_id")

    subject_pos, subject_facing = _subject_state(subject_id, tracks, start_t)
    aim = _subject_aim(subject_id, shot, library, subject_pos)
    eye = aim[2]

    # Where the camera stands relative to the subject. Default is downstage
    # (-Y, i.e. the audience side); `bearing_deg` overrides it explicitly, and
    # `from_front` puts the camera wherever the subject is actually looking.
    if "bearing_deg" in preset:
        bearing = float(preset["bearing_deg"])
    elif preset.get("from_front"):
        bearing = subject_facing
    else:
        bearing = -90.0

    move = {
        "type": "static",
        "start_t": start_t,
        "end_t": end_t,
        "target_id": subject_id,
        "_preset": name,
        "_framing": _PHRASES[name],
    }

    if name in ("single_cu", "single_med", "single_wide", "establishing"):
        key = {"single_cu": "cu", "single_med": "med",
               "single_wide": "wide", "establishing": "establishing"}[name]
        distance = float(preset.get("distance_m", SHOT_DISTANCES[key]))
        move["position"] = _offset_from(subject_pos, bearing, distance, eye)

    elif name in ("low_angle", "high_angle", "top_down"):
        distance = float(preset.get("distance_m", SHOT_DISTANCES["med"]))
        angle = {"low_angle": "low", "high_angle": "high", "top_down": "top"}[name]
        height = aim[2] * _ANGLE_HEIGHTS[angle]
        if name == "top_down":
            # nearly overhead: stay close in plan, get the height from above
            move["position"] = _offset_from(subject_pos, bearing, distance * 0.35, height)
        else:
            move["position"] = _offset_from(subject_pos, bearing, distance, height)

    elif name == "dutch":
        distance = float(preset.get("distance_m", SHOT_DISTANCES["med"]))
        move["position"] = _offset_from(subject_pos, bearing, distance, eye)
        move["roll_deg"] = float(preset.get("roll_deg", 12.0))

    elif name == "ots":
        # Camera sits behind and just off the near subject's shoulder, aimed at
        # the far one — so the near shoulder occupies a foreground corner.
        other_id = preset.get("other_id")
        if not other_id:
            raise ValueError("framing preset 'ots' needs other_id (who we look at)")
        other_pos, _ = _subject_state(other_id, tracks, start_t)
        # bearing from the far subject, through the near one, and beyond
        dx = subject_pos[0] - other_pos[0]
        dy = subject_pos[1] - other_pos[1]
        along = math.degrees(math.atan2(dy, dx))
        shoulder = float(preset.get("shoulder_deg", 18.0))
        behind = float(preset.get("behind_m", 1.1))
        move["position"] = _offset_from(subject_pos, along + shoulder, behind, eye)
        move["target_id"] = other_id
        move["aim_offset_right_m"] = float(preset.get("aim_offset_right_m", -0.25))

    elif name == "two_shot":
        other_id = preset.get("other_id")
        if not other_id:
            raise ValueError("framing preset 'two_shot' needs other_id")
        other_pos, _ = _subject_state(other_id, tracks, start_t)
        midpoint = [(subject_pos[i] + other_pos[i]) / 2.0 for i in range(3)]
        separation = math.dist(subject_pos[:2], other_pos[:2])
        # back off far enough that both fit: the wider they stand, the further
        # the camera goes, with a floor at the medium distance.
        distance = float(preset.get("distance_m",
                                    max(SHOT_DISTANCES["med"], separation * 1.6)))
        # look perpendicular to the line between them unless told otherwise
        if "bearing_deg" not in preset:
            dx = other_pos[0] - subject_pos[0]
            dy = other_pos[1] - subject_pos[1]
            bearing = math.degrees(math.atan2(dy, dx)) - 90.0
        move["position"] = _offset_from(midpoint, bearing, distance, eye)
        move.pop("target_id")
        move["target_position"] = [midpoint[0], midpoint[1], eye]

    elif name in ("push_in", "pull_back"):
        near = float(preset.get("near_m", SHOT_DISTANCES["med_close"]))
        far = float(preset.get("far_m", SHOT_DISTANCES["wide"]))
        start_d, end_d = (far, near) if name == "push_in" else (near, far)
        move["type"] = "dolly"
        move["position"] = _offset_from(subject_pos, bearing, start_d, eye)
        move["end_position"] = _offset_from(subject_pos, bearing, end_d, eye)

    return move


def expand_presets(shot, tracks, library):
    """Rewrite framing presets in ``shot.camera.moves`` into concrete moves.

    Mutates the shot in place (it is already a working copy by this point in
    the pipeline) and returns the number of presets expanded.
    """
    camera = shot.get("camera") or {}
    moves = camera.get("moves")
    if not moves:
        return 0
    expanded = []
    count = 0
    for move in moves:
        if isinstance(move, dict) and move.get("type") == "preset":
            expanded.append(_resolve(move, shot, tracks, library))
            count += 1
        else:
            expanded.append(move)
    if count:
        camera["moves"] = expanded
    return count
