"""Mocap utility helpers shared by host-side and Blender-side code."""

from __future__ import annotations

import math
import os
from pathlib import Path


DEFAULT_BVH_JOINT_MAP = {
    "hips": "hips",
    "spine": "spine",
    "spine1": "chest",
    "chest": "chest",
    "neck": "neck",
    "head": "head",
    "leftshoulder": "l_shoulder",
    "leftarm": "l_shoulder",
    "leftforearm": "l_elbow",
    "lefthand": "l_hand",
    "rightshoulder": "r_shoulder",
    "rightarm": "r_shoulder",
    "rightforearm": "r_elbow",
    "righthand": "r_hand",
    "leftupleg": "l_hip",
    "leftleg": "l_knee",
    "leftfoot": "l_foot",
    "rightupleg": "r_hip",
    "rightleg": "r_knee",
    "rightfoot": "r_foot",
    # SnapMoGen skeleton (renamed_bvhs). Root is "ROOT"; joints are
    # <side>_<segment>000N_bind_JNT. ROOT itself never rotates -- the skeleton
    # forks at ROOT into a pelvis branch (legs) and a spine branch (torso) and
    # EACH branch carries the character's whole world heading as a local
    # rotation, so the retarget must go through world space (see
    # retarget_frame). The pelvis is the body reference that maps to hips.
    "c_pelvis0001_bind_jnt": "hips",
    "c_spine0001_bind_jnt": "spine",
    "c_spine0003_bind_jnt": "chest",
    "c_neck0001_bind_jnt": "neck",
    "c_head_bind_jnt": "head",
    "l_armupper0001_bind_jnt": "l_shoulder",
    "l_armlower0001_bind_jnt": "l_elbow",
    "l_hand0001_bind_jnt": "l_hand",
    "r_armupper0001_bind_jnt": "r_shoulder",
    "r_armlower0001_bind_jnt": "r_elbow",
    "r_hand0001_bind_jnt": "r_hand",
    "l_legupper0001_bind_jnt": "l_hip",
    "l_leglower0001_bind_jnt": "l_knee",
    "l_foot0001_bind_jnt": "l_foot",
    "r_legupper0001_bind_jnt": "r_hip",
    "r_leglower0001_bind_jnt": "r_knee",
    "r_foot0001_bind_jnt": "r_foot",
}


def default_mocap_cache_root():
    root = os.environ.get("PREVIS_MOCAP_CACHE")
    if root:
        return Path(root)
    return Path.home() / "previs_mocap_cache"


def resolve_clip_path(clip_id, project_root=None, cache_root=None):
    """Resolve clip_id to a .bvh file path.

    Accepted forms:
    - absolute or relative path (with or without .bvh)
    - cache id under PREVIS_MOCAP_CACHE (default ~/previs_mocap_cache)
    - project-local mocap/<clip_id>.bvh
    """
    cache_root = Path(cache_root) if cache_root else default_mocap_cache_root()
    project_root = Path(project_root) if project_root else Path(__file__).resolve().parent.parent

    raw = Path(clip_id)
    candidates = []
    if raw.is_absolute() or raw.parts:
        candidates.append(raw)
        if raw.suffix.lower() != ".bvh":
            candidates.append(raw.with_suffix(".bvh"))

    safe_rel = Path(*[part for part in clip_id.replace("\\", "/").split("/") if part])
    if safe_rel.parts:
        candidates.append(cache_root / safe_rel)
        candidates.append(cache_root / safe_rel.with_suffix(".bvh"))
        candidates.append(project_root / "mocap" / safe_rel)
        candidates.append(project_root / "mocap" / safe_rel.with_suffix(".bvh"))

    seen = set()
    for candidate in candidates:
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        f"mocap clip {clip_id!r} not found in path, cache ({cache_root}), "
        f"or project mocap directory ({project_root / 'mocap'})"
    )


def canonical_joint_map(override=None):
    mapping = {source.lower(): target for source, target in DEFAULT_BVH_JOINT_MAP.items()}
    for source, target in (override or {}).items():
        if not isinstance(source, str) or not isinstance(target, str):
            continue
        mapping[source.lower()] = target
    return mapping


# --- Coordinate-frame conversion -------------------------------------------
#
# Our proxy rig is Blender Z-up: facing +X, +Z up, +Y to the character's left,
# so a T-pose left arm rests along +Y and folds down by rotating about X.
#
# SnapMoGen BVH is Y-up: up +Y, left arm along +X, facing +Z, so the same arm
# folds down by rotating about Z. Applying those rotations to our rig unchanged
# leaves the arms splayed. The two rest poses coincide exactly under the axis
# permutation P that sends (Xsrc, Ysrc, Zsrc) -> (ourY, ourZ, ourX):
#
#     ourX <- Zsrc (forward)   ourY <- Xsrc (left)   ourZ <- Ysrc (up)
#
# Rotations cannot simply be copied joint-by-joint, because the two skeletons
# disagree about hierarchy: SnapMoGen forks at ROOT into pelvis and spine
# branches that each carry the whole-body heading, has clavicle/twist segments
# our rig lacks, and the clip's heading belongs to the authored track, not the
# clip. retarget_frame therefore transports rotations through WORLD space:
# FK the source skeleton, strip the heading yaw, conjugate into rig axes by
# R_our = P . R_src . P^T, then re-localise down the rig's own hierarchy.
# convert_euler_frame keeps the bare single-joint conjugation for
# hand-authored clips and tests.

# P maps a source-axis vector to its image in rig coordinates (columns are the
# rig-space images of source X, Y, Z).
_Y_UP_TO_Z_UP = [
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
]

SOURCE_UP_AXES = ("y", "z")


def _transpose3(m):
    return [[m[c][r] for c in range(3)] for r in range(3)]


def convert_euler_frame(xyz_deg, source_up_axis="z"):
    """Convert an XYZ-Euler rotation from a source frame into rig frame.

    ``source_up_axis="z"`` returns the angles unchanged (the rig's own
    convention, used by hand-authored clips). ``"y"`` applies the Y-up->Z-up
    similarity transform used for SnapMoGen and other Y-up BVH sources.
    """
    axis = (source_up_axis or "z").lower()
    if axis != "y":
        return list(xyz_deg)

    # Import here to keep this module import-cheap and Blender-safe.
    from . import mocap_bvh as _bvh

    # Recompose in Blender's 'XYZ' Euler convention: M = Rz . Ry . Rx, the
    # same convention _matrix_to_euler_xyz_deg decomposes with.
    matrix = _bvh._identity3()
    for label, angle in zip(("Z", "Y", "X"), reversed(xyz_deg)):
        matrix = _bvh._mul3(matrix, _bvh._axis_matrix(label, math.radians(angle)))

    p = _Y_UP_TO_Z_UP
    converted = _bvh._mul3(_bvh._mul3(p, matrix), _transpose3(p))
    return _bvh._matrix_to_euler_xyz_deg(converted)


def map_rotations(source_rotations, joint_map, source_up_axis="z"):
    mapped = {}
    for source_joint, angles in source_rotations.items():
        target = joint_map.get(source_joint.lower())
        if target:
            mapped[target] = convert_euler_frame(angles, source_up_axis)
    return mapped


def _euler_to_matrix(xyz_deg):
    """Blender 'XYZ' Euler to matrix: M = Rz . Ry . Rx."""
    from . import mocap_bvh as _bvh

    matrix = _bvh._identity3()
    for label, angle in zip(("Z", "Y", "X"), reversed(list(xyz_deg))):
        matrix = _bvh._mul3(matrix, _bvh._axis_matrix(label, math.radians(angle)))
    return matrix


def _source_world_rotations(clip, local_rotations):
    """FK the source hierarchy: joint name -> world rotation matrix."""
    from . import mocap_bvh as _bvh

    world = {}

    def resolve(name):
        cached = world.get(name)
        if cached is not None:
            return cached
        joint = clip.joints[name]
        local = _euler_to_matrix(local_rotations.get(name, (0.0, 0.0, 0.0)))
        if joint.parent is None or joint.parent not in clip.joints:
            matrix = local
        else:
            matrix = _bvh._mul3(resolve(joint.parent), local)
        world[name] = matrix
        return matrix

    for name in clip.joints:
        resolve(name)
    return world


def _heading_removal(world_matrix, source_up_axis):
    """Inverse of the body's heading yaw, as a matrix in source coordinates.

    The heading is where the body FORWARD axis points, projected on the ground
    plane: Y-up sources face +Z, the rig's own Z-up convention faces +X. The
    clip's heading belongs to the shot's authored track, so the retarget
    removes it and keeps only the pelvis's pitch/roll sway.
    """
    from . import mocap_bvh as _bvh

    if source_up_axis == "y":
        forward = [world_matrix[r][2] for r in range(3)]  # world image of +Z
        yaw = math.atan2(forward[0], forward[2])  # about +Y
        return _bvh._axis_matrix("Y", -yaw)
    forward = [world_matrix[r][0] for r in range(3)]  # world image of +X
    yaw = math.atan2(forward[1], forward[0])  # about +Z
    return _bvh._axis_matrix("Z", -yaw)


def retarget_frame(clip, local_rotations, joint_map, source_up_axis="y"):
    """Retarget one sampled frame onto the rig: rig joint -> XYZ Euler (deg).

    World-rotation transport: FK the source skeleton, strip the body heading
    (measured at the joint mapped to ``hips``), rotate into rig axes, then
    re-localise each mapped joint against its nearest mapped ancestor in the
    RIG hierarchy. Source joints with no rig counterpart (pelvis, clavicles,
    twist segments) contribute through their children's world rotations
    instead of being dropped.
    """
    from . import mocap_bvh as _bvh
    from . import rig as _rig

    axis = (source_up_axis or "z").lower()
    source_world = _source_world_rotations(clip, local_rotations)

    # source joint feeding each rig joint (first match wins, map order).
    rig_sources = {}
    lowered = {name.lower(): name for name in clip.joints}
    for source_lower, target in joint_map.items():
        if target not in rig_sources and source_lower in lowered:
            rig_sources[target] = lowered[source_lower]

    hips_source = rig_sources.get("hips")
    if hips_source is None:
        return {}
    strip = _heading_removal(source_world[hips_source], axis)

    p = _Y_UP_TO_Z_UP if axis == "y" else _bvh._identity3()
    p_t = _transpose3(p)

    rig_world = {}
    for rig_joint, source_joint in rig_sources.items():
        stripped = _bvh._mul3(strip, source_world[source_joint])
        rig_world[rig_joint] = _bvh._mul3(_bvh._mul3(p, stripped), p_t)

    locals_out = {}
    for rig_joint, world_matrix in rig_world.items():
        parent = _rig.JOINTS.get(rig_joint, {}).get("parent")
        while parent is not None and parent not in rig_world:
            parent = _rig.JOINTS.get(parent, {}).get("parent")
        if parent is None:
            local = world_matrix
        else:
            local = _bvh._mul3(_transpose3(rig_world[parent]), world_matrix)
        locals_out[rig_joint] = _bvh._matrix_to_euler_xyz_deg(local)
    return locals_out


def clip_time_for_segment(segment, t_s, clip_duration_s):
    """Map shot time to clip time with optional looping bounds."""
    start_t = float(segment["start_t"])
    end_t = float(segment["end_t"])
    clip_t0 = float(segment.get("clip_t0_s", 0.0))
    clip_t1 = segment.get("clip_t1_s")

    span = max(1e-6, end_t - start_t)
    u = min(1.0, max(0.0, (float(t_s) - start_t) / span))

    if clip_t1 is not None:
        clip_t = clip_t0 + (float(clip_t1) - clip_t0) * u
    else:
        clip_t = clip_t0 + (float(t_s) - start_t)

    loop_from = segment.get("clip_loop_from_s")
    loop_to = segment.get("clip_loop_to_s")
    if loop_from is not None and loop_to is not None:
        a = float(loop_from)
        b = float(loop_to)
        if b > a and clip_t > b:
            period = b - a
            clip_t = a + ((clip_t - a) % period)

    return max(0.0, min(float(clip_duration_s), clip_t))


def segment_blend_weight(segment, t_s):
    """Blend envelope in [0, 1], combining action envelope and pose_weight."""
    start_t = float(segment["start_t"])
    end_t = float(segment["end_t"])
    blend_in = max(0.0, float(segment.get("blend_in_s", 0.0)))
    blend_out = max(0.0, float(segment.get("blend_out_s", 0.0)))
    pose_weight = float(segment.get("pose_weight", 1.0))

    w = 1.0
    if blend_in > 1e-6 and t_s < start_t + blend_in:
        w = min(w, (t_s - start_t) / blend_in)
    if blend_out > 1e-6 and t_s > end_t - blend_out:
        w = min(w, (end_t - t_s) / blend_out)
    return max(0.0, min(1.0, w * pose_weight))