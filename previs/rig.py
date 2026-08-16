"""Articulated humanoid proxy: skeleton, poses and a procedural gait.

A capsule communicates position and timing but zero body mechanics — which is a
named authority channel for video-reference models, and the reason an "old man's
walk" could previously only be expressed through pacing. This module gives the
proxy a jointed body and drives it from data the shot already contains.

Two deliberate design choices, both aimed at what comes next:

* **Rest pose is a T-pose**, matching the SnapMoGen BVH convention, so imported
  mocap rotations will apply to these joints with minimal rest-pose correction.
* **Gait phase is driven by distance travelled, not by time.** Stride length
  therefore always agrees with ground actually covered, so the feet cannot
  slide — the hardest problem in retargeted mocap simply never arises here.

Joint names follow the common humanoid convention (hips/spine/chest/neck/head,
shoulder/elbow/hand, hip/knee/foot) so a BVH joint map is a lookup table rather
than a retargeting rig.

Stdlib only; no ``bpy``.
"""

from __future__ import annotations

import math

# Rest skeleton in T-pose, facing +X, +Z up, +Y to the character's left.
# ``offset`` is the joint's position relative to its parent, in metres, for a
# 1.77m reference figure. Everything scales linearly with the asset's height.
REFERENCE_HEIGHT_M = 1.77

JOINTS = {
    "hips":       {"parent": None,        "offset": [0.00,  0.00,  0.98]},
    "spine":      {"parent": "hips",      "offset": [0.00,  0.00,  0.17]},
    "chest":      {"parent": "spine",     "offset": [0.00,  0.00,  0.21]},
    "neck":       {"parent": "chest",     "offset": [0.00,  0.00,  0.19]},
    "head":       {"parent": "neck",      "offset": [0.00,  0.00,  0.11]},

    "l_shoulder": {"parent": "chest",     "offset": [0.00,  0.17,  0.14]},
    "l_elbow":    {"parent": "l_shoulder","offset": [0.00,  0.28,  0.00]},
    "l_hand":     {"parent": "l_elbow",   "offset": [0.00,  0.26,  0.00]},

    "r_shoulder": {"parent": "chest",     "offset": [0.00, -0.17,  0.14]},
    "r_elbow":    {"parent": "r_shoulder","offset": [0.00, -0.28,  0.00]},
    "r_hand":     {"parent": "r_elbow",   "offset": [0.00, -0.26,  0.00]},

    "l_hip":      {"parent": "hips",      "offset": [0.00,  0.10,  0.00]},
    "l_knee":     {"parent": "l_hip",     "offset": [0.00,  0.00, -0.48]},
    "l_foot":     {"parent": "l_knee",    "offset": [0.00,  0.00, -0.48]},

    "r_hip":      {"parent": "hips",      "offset": [0.00, -0.10,  0.00]},
    "r_knee":     {"parent": "r_hip",     "offset": [0.00,  0.00, -0.48]},
    "r_foot":     {"parent": "r_knee",    "offset": [0.00,  0.00, -0.48]},
}

# Limb geometry: (from_joint, to_joint, shape, thickness). The mesh hangs off
# ``from_joint`` and spans to ``to_joint``'s rest position, which need not be a
# direct child — the torso is deliberately one mass from hips to neck rather
# than a hips/spine/chest stack, because three short capsules of nearly equal
# width read as a pile of blobs instead of a body.
BONES = (
    ("hips",       "neck",     "capsule", 0.32),
    ("neck",       "head",     "capsule", 0.11),
    ("l_shoulder", "l_elbow",  "capsule", 0.105),
    ("l_elbow",    "l_hand",   "capsule", 0.088),
    ("r_shoulder", "r_elbow",  "capsule", 0.105),
    ("r_elbow",    "r_hand",   "capsule", 0.088),
    ("l_hip",      "l_knee",   "capsule", 0.145),
    ("l_knee",     "l_foot",   "capsule", 0.115),
    ("r_hip",      "r_knee",   "capsule", 0.145),
    ("r_knee",     "r_foot",   "capsule", 0.115),
)


def rest_offset(from_joint, to_joint):
    """Vector from one joint to another in ``from_joint``'s rest space."""
    chain, cursor = [], to_joint
    while cursor and cursor != from_joint:
        chain.append(cursor)
        cursor = JOINTS[cursor]["parent"]
    if cursor != from_joint:
        raise ValueError(f"{to_joint!r} is not a descendant of {from_joint!r}")
    total = [0.0, 0.0, 0.0]
    for joint in chain:
        offset = JOINTS[joint]["offset"]
        total = [total[i] + offset[i] for i in range(3)]
    return total

# Tip geometry for joints with no child bone. Each joint maps to a *list* of
# parts so the head can carry more than one piece.
TIPS = {
    "head": [
        {"shape": "uv_sphere", "size": [0.19, 0.20, 0.23], "offset": [0.01, 0.0, 0.10]},
        # A small forward-pointing nose. Without it, front and back are
        # indistinguishable on this proxy — which quietly breaks any shot whose
        # whole point is a camera reveal of the character's front (an orbit,
        # a turn-to-camera). Rest-pose front is local +X.
        {"shape": "cone", "size": [0.05, 0.05, 0.10], "offset": [0.185, 0.0, 0.07],
         "rotation_deg": [0.0, 90.0, 0.0]},
    ],
    "l_hand": [{"shape": "uv_sphere", "size": [0.09, 0.09, 0.09], "offset": [0.0, 0.05, 0.0]}],
    "r_hand": [{"shape": "uv_sphere", "size": [0.09, 0.09, 0.09], "offset": [0.0, -0.05, 0.0]}],
    "l_foot": [{"shape": "box", "size": [0.24, 0.10, 0.08], "offset": [0.06, 0.0, -0.04]}],
    "r_foot": [{"shape": "box", "size": [0.24, 0.10, 0.08], "offset": [0.06, 0.0, -0.04]}],
}

# T-pose is the rest and the mocap convention, but nobody stands in one. This
# folds the arms down to the sides; the gait and poses build on top of it.
#
# IMPORTANT: shoulder X is what swings an arm down from the T-pose, and poses
# add to it. Anything that pushes the total much past 90 degrees rotates the arm
# through vertical and out the other side of the body. So poses and the gait
# only ever adjust shoulder Y (forward/back swing) and Z, never X.
STANDING_POSE = {
    "l_shoulder": [-74.0, 0.0, 0.0],
    "l_elbow":    [0.0, 0.0, -14.0],
    "r_shoulder": [74.0, 0.0, 0.0],
    "r_elbow":    [0.0, 0.0, 14.0],
    "chest":      [0.0, 3.0, 0.0],
}

# Poses replace the old vertical squash. Angles are degrees; ``root_drop`` sinks
# the hips so the feet stay on the floor when the knees bend.
POSES = {
    "stand": {"root_drop": 0.0, "joints": {}},
    "crouch": {
        "root_drop": 0.42,
        "joints": {
            "l_hip": [0.0, -62.0, 0.0], "r_hip": [0.0, -62.0, 0.0],
            "l_knee": [0.0, 108.0, 0.0], "r_knee": [0.0, 108.0, 0.0],
            "l_foot": [0.0, -46.0, 0.0], "r_foot": [0.0, -46.0, 0.0],
            "chest": [0.0, 22.0, 0.0], "spine": [0.0, 8.0, 0.0],
            "l_shoulder": [0.0, -20.0, 0.0], "r_shoulder": [0.0, -20.0, 0.0],
            "l_elbow": [0.0, -30.0, 0.0], "r_elbow": [0.0, -30.0, 0.0],
        },
    },
    "kneel": {
        "root_drop": 0.52,
        "joints": {
            "l_hip": [0.0, -72.0, 0.0], "r_hip": [0.0, -18.0, 0.0],
            "l_knee": [0.0, 118.0, 0.0], "r_knee": [0.0, 104.0, 0.0],
            "l_foot": [0.0, -38.0, 0.0],
            "chest": [0.0, 12.0, 0.0],
            "l_shoulder": [0.0, -14.0, 0.0], "r_shoulder": [0.0, -14.0, 0.0],
        },
    },
    "sit": {
        "root_drop": 0.40,
        "joints": {
            "l_hip": [0.0, -82.0, 0.0], "r_hip": [0.0, -82.0, 0.0],
            "l_knee": [0.0, 84.0, 0.0], "r_knee": [0.0, 84.0, 0.0],
            "chest": [0.0, 6.0, 0.0],
        },
    },
    "reach": {
        "root_drop": 0.04,
        "joints": {
            "chest": [0.0, 15.0, 0.0], "spine": [0.0, 5.0, 0.0],
            "r_shoulder": [0.0, -80.0, 0.0], "r_elbow": [0.0, -20.0, 0.0],
            "l_shoulder": [0.0, 12.0, 0.0],
            "l_hip": [0.0, -8.0, 0.0], "l_knee": [0.0, 14.0, 0.0],
        },
    },
}

# A gait profile is the character's walk. Ageing a walk is mostly: shorter
# strides, less arm swing, lower knee lift, more stoop.
DEFAULT_GAIT = {
    "stride_m": 0.72,       # one step; a full cycle is two of these
    "hip_swing_deg": 24.0,
    "knee_lift_deg": 46.0,
    "arm_swing_deg": 20.0,
    "torso_bob_m": 0.028,
    "stoop_deg": 0.0,
    "lateral_sway_deg": 2.5,
}

GAIT_PRESETS = {
    "elderly": {
        "stride_m": 0.46, "hip_swing_deg": 14.0, "knee_lift_deg": 26.0,
        "arm_swing_deg": 8.0, "torso_bob_m": 0.016, "stoop_deg": 12.0,
        "lateral_sway_deg": 4.0,
    },
    "brisk": {
        "stride_m": 0.82, "hip_swing_deg": 29.0, "knee_lift_deg": 54.0,
        "arm_swing_deg": 27.0, "torso_bob_m": 0.034, "stoop_deg": 0.0,
        "lateral_sway_deg": 2.0,
    },
    "child": {
        "stride_m": 0.40, "hip_swing_deg": 26.0, "knee_lift_deg": 52.0,
        "arm_swing_deg": 24.0, "torso_bob_m": 0.030, "stoop_deg": 0.0,
        "lateral_sway_deg": 3.5,
    },
}


def resolve_gait(asset):
    """Gait profile for a character asset: preset name or inline overrides."""
    gait = dict(DEFAULT_GAIT)
    spec = asset.get("gait")
    if isinstance(spec, str):
        gait.update(GAIT_PRESETS.get(spec, {}))
    elif isinstance(spec, dict):
        gait.update(GAIT_PRESETS.get(spec.get("preset", ""), {}))
        gait.update({k: v for k, v in spec.items() if k != "preset"})
    return gait


def scale_for(asset):
    """Uniform scale mapping the reference skeleton to this character's height."""
    height = asset.get("height_m")
    if not isinstance(height, (int, float)) or height <= 0:
        return 1.0
    return float(height) / REFERENCE_HEIGHT_M


def is_rigged(asset):
    return asset.get("rig") == "humanoid"


def _add(into, joint, angles):
    current = into.setdefault(joint, [0.0, 0.0, 0.0])
    for i in range(3):
        current[i] += angles[i]


def evaluate(distance_m, speed_mps, pose_name, gait, previous_pose=None, pose_u=1.0):
    """Joint angles (degrees) and root drop for one instant.

    ``distance_m`` is ground covered so far — the gait cycle is a function of
    it, which is what keeps footfall honest at any speed. ``speed_mps`` only
    scales how emphatic the cycle is, and fades it out when standing still.

    ``previous_pose``/``pose_u`` ease one pose into the next, so a change reads
    as a movement rather than a single-frame glitch.
    """
    joints = {name: list(angles) for name, angles in STANDING_POSE.items()}

    pose = POSES.get(pose_name, POSES["stand"])
    root_drop = pose["root_drop"]
    target = pose["joints"]

    if previous_pose and previous_pose != pose_name and pose_u < 1.0:
        outgoing = POSES.get(previous_pose, POSES["stand"])
        u = max(0.0, min(1.0, pose_u))
        u = u * u * (3.0 - 2.0 * u)  # ease both ends of the transition
        root_drop = outgoing["root_drop"] + (root_drop - outgoing["root_drop"]) * u
        blended = {}
        for joint in set(target) | set(outgoing["joints"]):
            a = outgoing["joints"].get(joint, [0.0, 0.0, 0.0])
            b = target.get(joint, [0.0, 0.0, 0.0])
            blended[joint] = [a[i] + (b[i] - a[i]) * u for i in range(3)]
        target = blended

    for joint, angles in target.items():
        _add(joints, joint, angles)

    # Below a slow shuffle there is no walk cycle to speak of.
    walking = max(0.0, min(1.0, (speed_mps - 0.12) / 0.5))
    if walking <= 1e-4:
        return joints, root_drop

    stride = max(0.15, float(gait["stride_m"]))
    phase = math.pi * distance_m / stride  # pi per step, 2*pi per full cycle

    swing = math.sin(phase)
    hip = gait["hip_swing_deg"] * swing * walking
    arm = gait["arm_swing_deg"] * swing * walking

    # Knees flex on the backswing, the leg straightening as it reaches forward.
    l_flex = gait["knee_lift_deg"] * max(0.0, -math.sin(phase - 0.7)) * walking
    r_flex = gait["knee_lift_deg"] * max(0.0, -math.sin(phase + math.pi - 0.7)) * walking

    _add(joints, "l_hip", [0.0, -hip, 0.0])
    _add(joints, "r_hip", [0.0, hip, 0.0])
    _add(joints, "l_knee", [0.0, l_flex, 0.0])
    _add(joints, "r_knee", [0.0, r_flex, 0.0])
    _add(joints, "l_foot", [0.0, -l_flex * 0.35, 0.0])
    _add(joints, "r_foot", [0.0, -r_flex * 0.35, 0.0])

    # Arms swing opposite the leg on the same side.
    _add(joints, "l_shoulder", [0.0, arm, 0.0])
    _add(joints, "r_shoulder", [0.0, -arm, 0.0])

    _add(joints, "chest", [0.0, gait["stoop_deg"] * walking, 0.0])
    _add(joints, "hips", [gait["lateral_sway_deg"] * walking * math.sin(phase), 0.0, 0.0])

    # Two bobs per cycle — one per footfall.
    bob = gait["torso_bob_m"] * walking * (0.5 - 0.5 * math.cos(2.0 * phase))
    return joints, root_drop + bob
