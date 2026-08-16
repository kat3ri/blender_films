"""Trajectory and camera maths — the part worth testing without Blender.

Turns the declarative shot spec into concrete animation keys:

    build_tracks(shot, library)      -> {object_id: Track}   character motion
    build_camera_keys(shot, ...)     -> [CameraKey, ...]     one per frame

Nothing here imports ``bpy``; ``blender_api`` just stamps the results onto
objects. Blender's conventions leak in at exactly one place — the look-at maths
in :func:`look_at_euler`, which encodes that a Blender camera looks down its
local -Z axis.
"""

from __future__ import annotations

import math

from .asset_library import aim_height

EPS = 1e-6

# Poses squash the proxy vertically and tip it forward. Crude on purpose: the
# downstream video model only needs to read "standing" vs "crouched".
POSE_TABLE = {
    "stand": (1.00, 0.0),
    "crouch": (0.62, 6.0),
    "kneel": (0.55, 0.0),
    "sit": (0.60, 0.0),
    "reach": (0.98, 14.0),
}

DEFAULT_FACING_DEG = -90.0  # facing downstage, toward a camera parked at -Y


def pad3(vector, default_z=0.0):
    """Accept [x, y] or [x, y, z] and always return a 3-list of floats."""
    values = list(vector)
    if len(values) == 2:
        values.append(default_z)
    return [float(values[0]), float(values[1]), float(values[2])]


def lerp(a, b, u):
    return a + (b - a) * u


def smoothstep(u):
    return u * u * (3.0 - 2.0 * u)


def _progress(move, t):
    """Normalised 0..1 position within a move, with optional easing."""
    start_t, end_t = float(move["start_t"]), float(move["end_t"])
    span = end_t - start_t
    u = 0.0 if span <= EPS else min(1.0, max(0.0, (t - start_t) / span))
    if move.get("easing", "smooth") == "smooth":
        return smoothstep(u)
    return u


def _unwrap(angle_deg, reference_deg):
    """Shift angle by whole turns so it lands within 180 degrees of reference."""
    while angle_deg - reference_deg > 180.0:
        angle_deg -= 360.0
    while angle_deg - reference_deg < -180.0:
        angle_deg += 360.0
    return angle_deg


def look_at_euler(position, target):
    """XYZ euler that points a Blender camera at ``target`` with no roll."""
    dx = target[0] - position[0]
    dy = target[1] - position[1]
    dz = target[2] - position[2]
    horizontal = math.hypot(dx, dy)
    if horizontal < EPS and abs(dz) < EPS:
        return [math.pi / 2.0, 0.0, 0.0]
    rot_x = math.atan2(horizontal, -dz)
    rot_z = math.atan2(dy, dx) - math.pi / 2.0
    return [rot_x, 0.0, rot_z]


class Track:
    """A character's motion: sparse keys, linearly interpolated."""

    def __init__(self, object_id, keys):
        self.object_id = object_id
        self.keys = keys  # each: {"t", "position", "facing_deg", "pose"}
        # Cumulative ground distance at each key. The gait cycle is a function
        # of distance rather than time, so stride always matches ground covered
        # and the feet cannot slide however fast or slow the walk is.
        self._distance = [0.0]
        for earlier, later in zip(keys, keys[1:]):
            step = math.dist(earlier["position"][:2], later["position"][:2])
            self._distance.append(self._distance[-1] + step)

    def distance_at(self, t):
        """Ground distance covered by time ``t``, ignoring vertical travel."""
        keys = self.keys
        if not keys:
            return 0.0
        if t <= keys[0]["t"]:
            return 0.0
        if t >= keys[-1]["t"]:
            return self._distance[-1]
        for index, (earlier, later) in enumerate(zip(keys, keys[1:])):
            if earlier["t"] <= t <= later["t"]:
                span = later["t"] - earlier["t"]
                u = 0.0 if span <= EPS else (t - earlier["t"]) / span
                return lerp(self._distance[index], self._distance[index + 1], u)
        return self._distance[-1]

    def pose_at(self, t, blend=0.35):
        """Return (pose, previous_pose, u) so a pose change can be eased in.

        An articulated body switching pose in a single frame reads as a glitch,
        so callers blend from the outgoing pose over ``blend`` seconds.
        """
        current, previous, changed_at = None, None, None
        for key in self.keys:
            if key["t"] > t + EPS:
                break
            if current is None:
                current = key["pose"]
            elif key["pose"] != current:
                previous, current, changed_at = current, key["pose"], key["t"]
        if current is None:
            current = self.keys[0]["pose"] if self.keys else "stand"
        if previous is None or changed_at is None or blend <= EPS:
            return current, current, 1.0
        u = min(1.0, max(0.0, (t - changed_at) / blend))
        return current, previous, u

    def speed_at(self, t, window=0.12):
        """Ground speed in m/s, sampled over a short window."""
        before = self.distance_at(max(0.0, t - window))
        after = self.distance_at(t + window)
        span = (t + window) - max(0.0, t - window)
        return (after - before) / span if span > EPS else 0.0

    def sample(self, t):
        """Return (position, facing_deg, pose_name) at time ``t``."""
        keys = self.keys
        if not keys:
            return [0.0, 0.0, 0.0], DEFAULT_FACING_DEG, "stand"
        if t <= keys[0]["t"]:
            first = keys[0]
            return list(first["position"]), first["facing_deg"], first["pose"]
        if t >= keys[-1]["t"]:
            last = keys[-1]
            return list(last["position"]), last["facing_deg"], last["pose"]
        for earlier, later in zip(keys, keys[1:]):
            if earlier["t"] <= t <= later["t"]:
                span = later["t"] - earlier["t"]
                u = 0.0 if span <= EPS else (t - earlier["t"]) / span
                position = [
                    lerp(earlier["position"][i], later["position"][i], u) for i in range(3)
                ]
                facing = lerp(earlier["facing_deg"], later["facing_deg"], u)
                # The pose in force is the one set by the most recent key at or
                # before t. Taking the *later* key's pose instead would start
                # every pose a segment early, and would lose a pose entirely
                # whenever its closing key was overwritten by the next action.
                return position, facing, earlier["pose"]
        last = keys[-1]
        return list(last["position"]), last["facing_deg"], last["pose"]


def _append_key(keys, t, position, facing_deg, pose):
    """Add a key, unwrapping facing so rotation takes the short way round."""
    if keys:
        facing_deg = _unwrap(facing_deg, keys[-1]["facing_deg"])
        if abs(t - keys[-1]["t"]) < EPS:
            # Same instant as the previous key: overwrite rather than stack.
            keys[-1] = {
                "t": t,
                "position": list(position),
                "facing_deg": facing_deg,
                "pose": pose,
            }
            return
    keys.append(
        {"t": float(t), "position": list(position), "facing_deg": facing_deg, "pose": pose}
    )


def _facing_toward(origin, target):
    dx, dy = target[0] - origin[0], target[1] - origin[1]
    if math.hypot(dx, dy) < EPS:
        return None
    return math.degrees(math.atan2(dy, dx))


def build_character_track(character, resolve_point, duration):
    """Turn one character's action list into a :class:`Track`."""
    position = pad3(character.get("start_position", [0.0, 0.0, 0.0]))
    facing = float(character.get("start_facing_deg", DEFAULT_FACING_DEG))
    pose = character.get("start_pose", "stand")

    keys = []
    _append_key(keys, 0.0, position, facing, pose)

    actions = sorted(
        (a for a in character.get("actions", []) if isinstance(a, dict)),
        key=lambda a: (a.get("start_t", 0.0), a.get("end_t", 0.0)),
    )

    for action in actions:
        start_t = float(action["start_t"])
        end_t = float(action["end_t"])
        span = max(0.0, end_t - start_t)
        action_type = action["type"]
        next_pose = action.get("pose", pose)

        # Hold the previous state up to the moment this action begins.
        if start_t > keys[-1]["t"] + EPS:
            _append_key(keys, start_t, position, facing, pose)

        if action_type == "walk_to":
            destination = pad3(action["position"], position[2])
            if "facing_deg" in action:
                # Explicit facing while moving: backing away, sidestepping, or
                # keeping eyes on something while crossing.
                travel_facing = float(action["facing_deg"])
            else:
                travel_facing = _facing_toward(position, destination)
            if travel_facing is None:
                travel_facing = facing
            # Turn into the walk over its first slice, then travel in a
            # straight line; the intermediate key sits on that same line so the
            # path is unchanged.
            turn_span = min(0.5, span * 0.25)
            if turn_span > EPS:
                u = turn_span / span if span > EPS else 1.0
                midpoint = [lerp(position[i], destination[i], u) for i in range(3)]
                _append_key(keys, start_t + turn_span, midpoint, travel_facing, next_pose)
            else:
                _append_key(keys, start_t, position, travel_facing, next_pose)
            _append_key(keys, end_t, destination, travel_facing, next_pose)
            position, facing, pose = destination, travel_facing, next_pose

        elif action_type in ("turn_to", "interact"):
            if "facing_deg" in action:
                new_facing = float(action["facing_deg"])
            else:
                target = resolve_point(action["target_id"], end_t)
                new_facing = _facing_toward(position, target)
                if new_facing is None:
                    new_facing = facing
            blend = min(0.5, span * 0.4) if span > EPS else 0.0
            if blend > EPS:
                _append_key(keys, start_t + blend, position, new_facing, next_pose)
            else:
                _append_key(keys, start_t, position, new_facing, next_pose)
            _append_key(keys, end_t, position, new_facing, next_pose)
            facing, pose = new_facing, next_pose

        else:  # idle
            _append_key(keys, start_t, position, facing, next_pose)
            _append_key(keys, end_t, position, facing, next_pose)
            pose = next_pose

    # Hold the final state to the end of the shot so nothing snaps.
    if duration > keys[-1]["t"] + EPS:
        _append_key(keys, duration, position, facing, pose)

    return Track(character["id"], keys)


def build_tracks(shot, library):
    """Build every character's track. Two passes so characters can face
    each other even when the target is itself moving."""
    duration = float(shot["duration_seconds"])
    characters = [c for c in shot.get("characters", []) if isinstance(c, dict)]
    props = {
        p["id"]: pad3(p.get("position", [0, 0, 0]))
        for p in shot.get("props", [])
        if isinstance(p, dict) and "id" in p
    }
    starts = {
        c["id"]: pad3(c.get("start_position", [0, 0, 0]))
        for c in characters
        if "id" in c
    }

    tracks = {}

    def resolve_point(object_id, t):
        if object_id in props:
            return props[object_id]
        track = tracks.get(object_id)
        if track is not None:
            return track.sample(t)[0]
        return starts.get(object_id, [0.0, 0.0, 0.0])

    for _ in range(2):
        tracks = {
            character["id"]: build_character_track(character, resolve_point, duration)
            for character in characters
            if "id" in character
        }
    return tracks


class CameraKey:
    __slots__ = ("frame", "t", "position", "rotation_euler")

    def __init__(self, frame, t, position, rotation_euler):
        self.frame = frame
        self.t = t
        self.position = position
        self.rotation_euler = rotation_euler


def _aim_point(shot, move, tracks, library, t, fallback, camera_position=None):
    """Where the camera should look during ``move`` at time ``t``.

    ``aim_offset_z`` nudges the aim up or down from the target's default head
    height — needed whenever the subject is not standing, since a crouched or
    seated figure sits well below the height the asset declares.

    ``aim_offset_right_m`` shifts the aim point sideways in the camera's own
    right vector (computed from ``camera_position`` toward the un-offset aim
    point), which pushes the subject *away* from centre in the rendered frame
    — a positive value moves the subject toward frame-left. Every move type in
    this system otherwise centres its target dead-on; this is the one knob for
    an off-centre, rule-of-thirds-style composition, which matters whenever a
    reference image places the subject somewhere other than centre.
    """
    offset_z = float(move.get("aim_offset_z", 0.0))
    offset_right = float(move.get("aim_offset_right_m", 0.0))
    if "target_position" in move:
        point = pad3(move["target_position"], 1.2)
    elif move.get("target_id"):
        point = _object_aim(shot, move["target_id"], tracks, library, t)
    else:
        point = list(fallback)
    if offset_right and camera_position is not None:
        dx = point[0] - camera_position[0]
        dy = point[1] - camera_position[1]
        horiz = math.hypot(dx, dy)
        if horiz > EPS:
            right_x, right_y = dy / horiz, -dx / horiz
            point = [point[0] + right_x * offset_right, point[1] + right_y * offset_right, point[2]]
    if offset_z:
        point = [point[0], point[1], point[2] + offset_z]
    return point


def _object_position(shot, object_id, tracks, t):
    track = tracks.get(object_id)
    if track is not None:
        return track.sample(t)[0]
    for prop in shot.get("props", []):
        if isinstance(prop, dict) and prop.get("id") == object_id:
            return pad3(prop.get("position", [0, 0, 0]))
    return [0.0, 0.0, 0.0]


def _object_aim(shot, object_id, tracks, library, t):
    """An object's position raised to a sensible height to look at."""
    position = list(_object_position(shot, object_id, tracks, t))
    kind, asset_id = "props", None
    for character in shot.get("characters", []):
        if isinstance(character, dict) and character.get("id") == object_id:
            kind, asset_id = "characters", character.get("asset_id")
            break
    else:
        for prop in shot.get("props", []):
            if isinstance(prop, dict) and prop.get("id") == object_id:
                asset_id = prop.get("asset_id")
                break
    if asset_id:
        position[2] += aim_height(library.get(kind, asset_id))
    else:
        position[2] += 1.2
    return position


def _eval_move(shot, move, tracks, library, t, stage_centre):
    """Return (camera_position, aim_point) for one move at time ``t``."""
    move_type = move["type"]
    u = _progress(move, t)

    if move_type == "static":
        position = pad3(move["position"], 1.6)
        return position, _aim_point(shot, move, tracks, library, t, stage_centre, position)

    if move_type == "dolly":
        start = pad3(move["position"], 1.6)
        end = pad3(move["end_position"], start[2])
        position = [lerp(start[i], end[i], u) for i in range(3)]
        return position, _aim_point(shot, move, tracks, library, t, stage_centre, position)

    if move_type == "track":
        start = pad3(move["position"], 1.6)
        target_id = move["target_id"]
        anchor = _object_position(shot, target_id, tracks, float(move["start_t"]))
        offset = [start[i] - anchor[i] for i in range(3)]
        current = _object_position(shot, target_id, tracks, t)
        position = [current[i] + offset[i] for i in range(3)]
        if "end_offset" in move:
            end_offset = pad3(move["end_offset"], offset[2])
            position = [
                current[i] + lerp(offset[i], end_offset[i], u) for i in range(3)
            ]
        return position, _aim_point(shot, move, tracks, library, t, stage_centre, position)

    if move_type == "orbit":
        if "center_id" in move:
            centre = _object_position(shot, move["center_id"], tracks, t)
            aim = _object_aim(shot, move["center_id"], tracks, library, t)
        else:
            centre = pad3(move["center_position"])
            aim = [centre[0], centre[1], centre[2] + 1.2]
        angle = math.radians(lerp(float(move["start_deg"]), float(move["end_deg"]), u))
        # radius_m/height_m may each be a single number (constant, as before)
        # or [start, end] to spiral in/out or rise/descend while turning --
        # "spin and pull back" instead of a flat circle, without a second move.
        radius = float(lerp(*move["radius_m"], u)) if isinstance(move["radius_m"], list) \
            else float(move["radius_m"])
        height_spec = move.get("height_m", 1.6)
        height = float(lerp(*height_spec, u)) if isinstance(height_spec, list) \
            else float(height_spec)
        position = [
            centre[0] + radius * math.cos(angle),
            centre[1] + radius * math.sin(angle),
            centre[2] + height,
        ]
        return position, _aim_point(shot, move, tracks, library, t, aim, position)

    if move_type in ("pan", "tilt"):
        position = pad3(move["position"], 1.6)
        if move_type == "pan":
            yaw = math.radians(lerp(float(move["start_deg"]), float(move["end_deg"]), u))
            pitch = math.radians(float(move.get("pitch_deg", 0.0)))
        else:
            yaw = math.radians(float(move.get("yaw_deg", 90.0)))
            pitch = math.radians(lerp(float(move["start_deg"]), float(move["end_deg"]), u))
        distance = float(move.get("aim_distance_m", 12.0))
        aim = [
            position[0] + distance * math.cos(pitch) * math.cos(yaw),
            position[1] + distance * math.cos(pitch) * math.sin(yaw),
            position[2] + distance * math.sin(pitch),
        ]
        return position, aim

    raise ValueError(f"unknown camera move type {move_type!r}")


def build_camera_keys(shot, tracks, library, fps=None):
    """One camera key per frame — cheap, and makes arcs and gaps unambiguous."""
    fps = int(fps or shot.get("fps", 12))
    duration = float(shot["duration_seconds"])
    total_frames = max(1, int(round(duration * fps)))

    stage = shot.get("stage") or {}
    size = stage.get("size_m", [10.0, 10.0])
    stage_centre = [0.0, 0.0, 1.2]

    camera = shot.get("camera") or {}
    moves = sorted(
        (m for m in camera.get("moves", []) if isinstance(m, dict)),
        key=lambda m: (m.get("start_t", 0.0), m.get("end_t", 0.0)),
    )
    if not moves:
        # No blocking yet: park a wide static camera just inside the downstage
        # edge so the stub previews without tripping the bounds check.
        moves = [
            {
                "type": "static",
                "position": [0.0, -(float(size[1]) / 2.0 - 0.5), 2.0],
                "target_position": stage_centre,
                "start_t": 0.0,
                "end_t": duration,
            }
        ]

    keys = []
    previous_rot_z = None
    for frame_index in range(total_frames):
        t = frame_index / fps

        active = None
        for move in moves:
            if float(move["start_t"]) - EPS <= t <= float(move["end_t"]) + EPS:
                active = move
                break
        if active is None:
            # In a gap or past the end: hold the nearest move's boundary state.
            earlier = [m for m in moves if float(m["end_t"]) < t]
            if earlier:
                active = earlier[-1]
                t_eval = float(active["end_t"])
            else:
                active = moves[0]
                t_eval = float(active["start_t"])
        else:
            t_eval = t

        position, aim = _eval_move(shot, active, tracks, library, t_eval, stage_centre)
        rotation = look_at_euler(position, aim)

        # Keep yaw continuous so a wrap past +/-pi doesn't spin the camera.
        if previous_rot_z is not None:
            while rotation[2] - previous_rot_z > math.pi:
                rotation[2] -= 2.0 * math.pi
            while rotation[2] - previous_rot_z < -math.pi:
                rotation[2] += 2.0 * math.pi
        previous_rot_z = rotation[2]

        keys.append(CameraKey(frame_index + 1, t, position, rotation))

    return keys


def check_camera_bounds(keys, stage, margin_m=0.15):
    """Flag frames where the camera leaves the stage.

    Sets are built at the stage edges, so a camera outside the stage is usually
    a camera behind a wall — which renders as a flat grey rectangle and wastes
    a whole review round. Cheap to detect, so detect it.
    """
    size = (stage or {}).get("size_m", [12.0, 12.0])
    half_x = float(size[0]) / 2.0 - margin_m
    half_y = float(size[1]) / 2.0 - margin_m

    outside = [
        key for key in keys
        if abs(key.position[0]) > half_x or abs(key.position[1]) > half_y
    ]
    if not outside:
        return []

    first, last = outside[0], outside[-1]
    return [
        f"camera leaves the {size[0]}x{size[1]}m stage between t={first.t:.2f}s and "
        f"t={last.t:.2f}s ({len(outside)}/{len(keys)} frames; furthest "
        f"[{max(outside, key=lambda k: abs(k.position[0])).position[0]:.2f}, "
        f"{max(outside, key=lambda k: abs(k.position[1])).position[1]:.2f}]). "
        "It is probably shooting the back of a wall."
    ]
