"""Import an ea_worlds room export as a previs set asset.

`ea_worlds/out/<world>/export/room.json` is the richest scene source available:
every object carries a label, a MEASURED metric position and scale, and a full
`matrix_world`. Its own note is worth repeating, because it says exactly how far
to trust each part::

    position and scale are MEASURED from the point map;
    only the rotation comes from the SAM 3D pose

So placement is solid and rotation is the soft part — a sofa may sit in exactly
the right spot facing the wrong way. For a grey blockout that costs silhouette
accuracy, not spatial truth, which is the right trade.

Two conventions matter and both are handled here:

* **Canonical meshes.** `objects/<inst>/v0_uv40k.glb` is origin-centred and
  unit-ish (bounds about +/-0.5); `matrix_world` is what places it. We decompose
  that matrix into previs's `position` / `rotation_deg` / `scale`.
* **Y-up vs Z-up.** The export declares `up: gltf` (Y-up), and `matrix_world`'s
  rotation block already folds in `placements.json`'s
  `gltf_to_gaussian = [[1,0,0],[0,0,-1],[0,1,0]]`, i.e. (x,y,z) -> (x,-z,y).
  The matrix therefore lands the mesh in a Z-up world already, which is previs's
  frame, so it composes with no extra conversion. Verified against obj_013,
  whose matrix is exactly `scale * gltf_to_gaussian`.

Objects flagged `reliable: false` (low coverage or a poor fit) fall back to a
box at the measured centre and extent — better an honest proxy than a mesh
placed on bad evidence. The room shell is a separate entry and can be omitted
entirely, which is the cheapest way to see whether object placement alone holds
up.
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

# Mesh LOD to prefer. v0_uv40k is ~1 MB against v0's ~17 MB and v0_uv's ~22 MB;
# a Workbench blockout only needs the silhouette, and Blender import time is the
# real cost when a room has twenty of them.
MESH_CANDIDATES = ("v0_uv40k.glb", "v0_uv.glb", "v0.glb")


def _decompose(matrix):
    """4x4 (row-major, translation in the last column) -> position, XYZ euler deg, scale."""
    position = [float(matrix[i][3]) for i in range(3)]
    cols = [[float(matrix[r][c]) for r in range(3)] for c in range(3)]
    scale = [math.sqrt(sum(v * v for v in col)) for col in cols]
    rot = [[cols[c][r] / scale[c] if scale[c] > 1e-9 else (1.0 if r == c else 0.0)
            for c in range(3)] for r in range(3)]

    # Blender 'XYZ' euler from a rotation matrix (R = Rz @ Ry @ Rx).
    sy = -rot[2][0]
    if abs(sy) < 0.999999:
        rx = math.atan2(rot[2][1], rot[2][2])
        ry = math.asin(max(-1.0, min(1.0, sy)))
        rz = math.atan2(rot[1][0], rot[0][0])
    else:  # gimbal lock: fold the spin into rz
        rx = math.atan2(-rot[1][2], rot[1][1])
        ry = math.asin(max(-1.0, min(1.0, sy)))
        rz = 0.0
    return position, [math.degrees(a) for a in (rx, ry, rz)], scale


def _glb_position_bounds(path):
    """(min, max) of every float VEC3 POSITION accessor in a .glb, or None.

    Enough of the container to read the JSON chunk -- no glTF library, since
    Blender's bundled Python has no pip and previs stays stdlib-plus-numpy.
    """
    try:
        with open(path, "rb") as handle:
            magic, _version, _length = struct.unpack("<III", handle.read(12))
            if magic != 0x46546C67:  # 'glTF'
                return None
            chunk_len, _chunk_type = struct.unpack("<II", handle.read(8))
            gltf = json.loads(handle.read(chunk_len))
    except (OSError, ValueError, struct.error):
        return None

    lows, highs = [], []
    for accessor in gltf.get("accessors", []):
        if (accessor.get("type") == "VEC3" and accessor.get("componentType") == 5126
                and "min" in accessor and "max" in accessor):
            lows.append(accessor["min"])
            highs.append(accessor["max"])
    if not lows:
        return None
    return ([min(v[i] for v in lows) for i in range(3)],
            [max(v[i] for v in highs) for i in range(3)])


def _extent(obj, mesh_path=None):
    """Full extent in metres for a box proxy.

    Preferred source is the canonical mesh's own bounds times the MEASURED
    scale: the meshes are origin-centred and unit-ish, so this recovers real
    dimensions. Falling back to a uniform cube of `scale` makes a carpet a
    3.6 m box, which is worse than useless in a blockout -- it occludes.
    """
    if mesh_path is not None:
        bounds = _glb_position_bounds(mesh_path)
        if bounds:
            lo, hi = bounds
            scale = float(obj.get("scale", 1.0))
            # local Y-up -> world Z-up, same swap the matrices encode
            span = [abs(hi[i] - lo[i]) * scale for i in range(3)]
            return [span[0], span[2], span[1]]
    scale = float(obj.get("scale", 1.0))
    return [scale, scale, scale]


def _part_height(part, obj):
    """World-Z extent of a built part, for floor detection."""
    if part["shape"] == "box":
        return float(part["size"][2])
    bounds = _glb_position_bounds(part["file"])
    if not bounds:
        return 0.0
    lo, hi = bounds
    # canonical meshes are glTF Y-up; local Y becomes world Z
    return abs(hi[1] - lo[1]) * float(obj.get("scale", 1.0))


def build_set_asset(world_dir, set_id, include_shell=True, display_name=None,
                    pano_yaw_offset_deg=0.0, mesh_lod=None, notes=None,
                    floor_z=None):
    """room.json -> a previs set asset dict. Pure data; writes nothing."""
    world_dir = Path(world_dir)
    export = world_dir / "export"
    room_path = export / "room.json"
    if not room_path.is_file():
        raise FileNotFoundError(f"{room_path} not found -- is this an ea_worlds world dir?")
    with room_path.open(encoding="utf-8") as handle:
        room = json.load(handle)

    candidates = (mesh_lod,) + MESH_CANDIDATES if mesh_lod else MESH_CANDIDATES
    parts, placed, boxed, missing = [], [], [], []

    for obj in room.get("objects", []):
        inst = obj.get("inst")
        label = obj.get("label") or inst
        position, rotation_deg, scale = _decompose(obj["matrix_world"])

        available = None
        for name in candidates:
            path = world_dir / "objects" / inst / name
            if path.is_file():
                available = path
                break
        # An unreliable fit still has a usable mesh for MEASURING the proxy --
        # we distrust the placement, not the geometry.
        mesh_file = available if obj.get("reliable", True) else None

        if mesh_file is not None:
            parts.append({
                "shape": "mesh",
                "file": str(mesh_file),
                "position": [round(v, 4) for v in position],
                "rotation_deg": [round(v, 3) for v in rotation_deg],
                "scale": [round(v, 5) for v in scale],
                "_inst": inst, "_label": label,
            })
            placed.append(f"{inst}/{label}")
        else:
            # Unreliable fit, or no mesh on disk: an honest box at the measured
            # centre and extent rather than a mesh placed on bad evidence.
            parts.append({
                "shape": "box",
                "position": [round(v, 4) for v in obj.get("centre_m", position)],
                "size": [round(v, 4) for v in _extent(obj, available)],
                "rotation_deg": [round(v, 3) for v in rotation_deg],
                "_inst": inst, "_label": label,
            })
            (boxed if obj.get("reliable", True) is False else missing).append(
                f"{inst}/{label}")

    # The ea_worlds origin is the PANO CAMERA, roughly 1.1 m above the floor,
    # so every object sits at negative Z. previs puts its ground plane at z=0
    # and that plane then hides the entire room -- the first render of this set
    # showed nothing but the floor grid. Shift the room so its lowest object
    # rests on z=0. `floor_z` overrides the detection when one object is a
    # badly-fit outlier dangling below the real floor.
    if parts:
        detected = min(
            part["position"][2] - _part_height(part, obj) / 2.0
            for part, obj in zip(parts, room.get("objects", []))
        )
        lift = -(float(floor_z) if floor_z is not None else detected)
        for part in parts:
            part["position"] = [round(part["position"][0], 4),
                                round(part["position"][1], 4),
                                round(part["position"][2] + lift, 4)]
    else:
        detected, lift = 0.0, 0.0

    shell_used = None
    if include_shell:
        shell = room.get("shell")
        if shell:
            shell_path = Path(shell)
            if not shell_path.is_absolute():
                shell_path = export / shell
            if shell_path.is_file():
                parts.insert(0, {
                    "shape": "mesh", "file": str(shell_path),
                    "position": [0.0, 0.0, round(lift, 4)],
                    # The shell mesh is exported in the same glTF Y-up frame the
                    # object matrices convert from, so it needs that conversion
                    # applied on its own: (x, y, z) -> (x, -z, y) is +90 about X.
                    "rotation_deg": [90.0, 0.0, 0.0],
                    "scale": [1.0, 1.0, 1.0],
                    "_inst": "shell", "_label": "room shell",
                })
                shell_used = str(shell_path)

    asset = {
        "asset_id": set_id,
        "kind": "set",
        "display_name": display_name or set_id.replace("_", " "),
        "color": [0.55, 0.5, 0.45],
        "parts": parts,
        # Consumed by previs.pano to cut background plates for whatever the
        # camera looks at; the offset is calibrated with `previs pano-check`.
        "pano": str(world_dir / "pano" / "base.png"),
        "pano_yaw_offset_deg": float(pano_yaw_offset_deg),
        "_source": {
            "world": str(world_dir),
            "metric_scale": room.get("metric_scale"),
            "up": room.get("up"),
            "note": room.get("note"),
            "shell": shell_used,
            "floor_z_detected": round(detected, 4),
            "floor_lift_applied": round(lift, 4),
        },
    }
    if notes:
        asset["notes"] = notes

    return asset, {"placed": placed, "boxed_unreliable": boxed,
                   "boxed_no_mesh": missing, "shell": shell_used,
                   "floor_z_detected": detected, "floor_lift_applied": lift}
