"""Reusable proxy-geometry asset definitions.

An asset is a JSON file under ``assets/<kind>/<asset_id>.json`` describing crude
stand-in geometry — a character is a couple of cylinders, a set is a floor and
some walls. Sets in particular are authored *once* per location and reused by
every shot there; nothing re-derives them per shot.

Asset shape::

    {
      "asset_id": "generic_human",
      "kind": "character",
      "display_name": "Generic Human",
      "color": [0.85, 0.35, 0.25],
      "aim_height_m": 1.4,          # optional; where a camera should look
      "parts": [
        {"shape": "cylinder", "position": [0, 0, 0.45], "size": [0.4, 0.4, 0.9]},
        {"shape": "uv_sphere", "position": [0, 0, 1.6], "size": [0.24, 0.24, 0.28],
         "color": [0.9, 0.7, 0.5]}
      ]
    }

``position`` is the part centre in asset-local metres, ``size`` its full extent
on each axis. Character and prop assets are built with their origin at floor
level (z=0) so poses can squash them vertically and they stay on the ground.

``shape`` is an open enum — ``box``, ``cylinder``, ``uv_sphere``, ``cone``,
``capsule``, ``plane``. It is deliberately extensible: a future asset-sourcing
step that downloads real geometry only has to add ``{"shape": "mesh", "file":
"..."}`` here, with no change to the shot schema, compiler or filmmaking API.

Any part (or set fixture) may carry a ``repeat`` instead of being hand-copied::

    {"shape": "box", "position": [0, 0, 1.06], "size": [0.40, 0.40, 0.12],
     "repeat": {"axis": "x", "count": 6, "spacing": 0.40}}

This expands to ``count`` copies centred on ``position``, spaced along the
given axis. It exists because hand-placing every repeated element makes
authors quietly under-count real-world density (a stone wall's coping course,
bottles on a shelf, planks, rivets) — and a video model conditioned on a
control clip tends to take a too-small count *literally*, then hallucinate or
warp geometry trying to reconcile it with what the scene "should" have.
Reaching for a plausible count via ``repeat`` costs one line; hand-placing it
costs one line per copy, which is exactly the gap that produces this bug.
"""

from __future__ import annotations

import json
from pathlib import Path

ASSET_KINDS = ("characters", "sets", "props")
PART_SHAPES = ("box", "cylinder", "uv_sphere", "cone", "capsule", "plane")

DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "assets"

# Used when a shot references an asset that has not been authored yet, so a
# freshly imported stub still renders instead of blowing up.
_FALLBACKS = {
    "characters": {
        "display_name": "Unknown character (placeholder)",
        "color": [0.85, 0.35, 0.25],
        "parts": [
            {"shape": "capsule", "position": [0, 0, 0.85], "size": [0.45, 0.45, 1.7]},
        ],
    },
    "props": {
        "display_name": "Unknown prop (placeholder)",
        "color": [0.55, 0.55, 0.6],
        "parts": [{"shape": "box", "position": [0, 0, 0.25], "size": [0.5, 0.5, 0.5]}],
    },
    "sets": {
        "display_name": "Empty stage (placeholder)",
        "color": [0.35, 0.35, 0.38],
        "parts": [],
    },
}


class AssetLibrary:
    """Loads and caches asset definitions from disk."""

    def __init__(self, root=None):
        self.root = Path(root) if root else DEFAULT_ROOT
        self._cache = {}
        self.missing = []  # (kind, asset_id) pairs that fell back to placeholders

    def path_for(self, kind, asset_id):
        return self.root / kind / f"{asset_id}.json"

    def exists(self, kind, asset_id):
        return self.path_for(kind, asset_id).is_file()

    def get(self, kind, asset_id):
        """Return an asset definition, falling back to a placeholder if absent."""
        if kind not in ASSET_KINDS:
            raise ValueError(f"unknown asset kind {kind!r}, expected one of {ASSET_KINDS}")
        key = (kind, asset_id)
        if key in self._cache:
            return self._cache[key]

        path = self.path_for(kind, asset_id)
        if path.is_file():
            with path.open(encoding="utf-8") as handle:
                asset = json.load(handle)
            asset.setdefault("asset_id", asset_id)
            asset.setdefault("kind", kind.rstrip("s"))
        else:
            asset = dict(_FALLBACKS[kind])
            asset["asset_id"] = asset_id
            asset["kind"] = kind.rstrip("s")
            asset["parts"] = [dict(p) for p in asset["parts"]]
            asset["placeholder"] = True
            if key not in self.missing:
                self.missing.append(key)

        asset.setdefault("color", [0.6, 0.6, 0.6])
        asset.setdefault("parts", [])
        asset["parts"] = _expand_part_repeats(asset["parts"])
        self._cache[key] = asset
        return asset

    def list_assets(self, kind):
        directory = self.root / kind
        if not directory.is_dir():
            return []
        return sorted(p.stem for p in directory.glob("*.json"))


_REPEAT_AXES = {"x": 0, "y": 1, "z": 2}


def _repeat_positions(position, repeat):
    """Expand one position into ``count`` positions along one axis.

    The row is centred on ``position``, so the original coordinate stays the
    row's midpoint — an author picks where the *row* goes, not where copy zero
    starts.
    """
    axis = _REPEAT_AXES[repeat.get("axis", "x")]
    count = max(1, int(repeat.get("count", 1)))
    spacing = float(repeat.get("spacing", 0.3))
    base = list(position)
    if len(base) == 2:
        base.append(0.0)
    start = base[axis] - spacing * (count - 1) / 2.0
    positions = []
    for i in range(count):
        copy = list(base)
        copy[axis] = start + i * spacing
        positions.append(copy)
    return positions


def _expand_part_repeats(parts):
    expanded = []
    for part in parts:
        repeat = part.get("repeat")
        if not repeat:
            expanded.append(part)
            continue
        for position in _repeat_positions(part.get("position", [0, 0, 0]), repeat):
            copy = {k: v for k, v in part.items() if k != "repeat"}
            copy["position"] = position
            expanded.append(copy)
    return expanded


def expand_fixtures(shot, library):
    """Merge a set's fixtures into the shot's prop list.

    A set may declare ``fixtures`` — named, interactable objects that are part
    of the location itself (a door, a hearth, a wall panel). They appear in
    every shot at that location automatically, which is both less tedious than
    restating them per shot and the only way to stop one shot in a chain from
    silently missing a piece of its own set.

    A shot can still override a fixture by declaring a prop with the same id.
    Idempotent, so it is safe to call more than once on the same shot.
    """
    set_asset_id = (shot.get("set") or {}).get("asset_id")
    if not set_asset_id:
        return shot

    fixtures = library.get("sets", set_asset_id).get("fixtures") or []
    if not fixtures:
        return shot

    props = shot.setdefault("props", [])
    existing = {p.get("id") for p in props if isinstance(p, dict)}
    for fixture in fixtures:
        if not isinstance(fixture, dict) or not fixture.get("id"):
            continue
        repeat = fixture.get("repeat")
        if repeat:
            # Idempotency check uses the first derived id, since the
            # un-suffixed base id is never itself added to `existing` below.
            if f"{fixture['id']}_0" in existing:
                continue
            for index, position in enumerate(
                _repeat_positions(fixture.get("position", [0, 0, 0]), repeat)
            ):
                copy = {k: v for k, v in fixture.items() if k != "repeat"}
                copy["id"] = f"{fixture['id']}_{index}"
                copy["position"] = position
                copy["_from_set"] = set_asset_id
                props.append(copy)
                existing.add(copy["id"])
            continue
        if fixture["id"] in existing:
            continue
        merged = dict(fixture)
        merged["_from_set"] = set_asset_id
        props.append(merged)
        existing.add(fixture["id"])
    return shot


def asset_height(asset):
    """Top of the asset's proxy geometry, in metres."""
    top = 0.0
    for part in asset.get("parts", []):
        position = list(part.get("position", [0, 0, 0]))
        size = list(part.get("size", [1, 1, 1]))
        centre_z = position[2] if len(position) > 2 else 0.0
        height = size[2] if len(size) > 2 else 0.0
        top = max(top, centre_z + height / 2.0)
    return top


def aim_height(asset):
    """Height a camera should look at for this asset — roughly head/chest level."""
    if isinstance(asset.get("aim_height_m"), (int, float)):
        return float(asset["aim_height_m"])
    height = asset_height(asset)
    return height * 0.8 if height else 1.0
