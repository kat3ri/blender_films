"""Shared helpers for turning story-pipeline formats into shot stubs.

Importers do the *mechanical* half of the job only: durations, who is present,
which location, and the prose for each beat carried through verbatim as
``_source_text``. They never invent positions — neither source format contains
spatial or camera data, so blocking is a judgement call made afterwards (see
the ``previs-blocking`` skill).

Stubs come out as ``status: "needs_blocking"`` with empty ``actions``/``moves``.
That is the handoff point.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from .. import SCHEMA_VERSION

# Reused so imported characters get proxies with the same build as the seed
# asset, just recoloured so several people on stage stay tellable apart.
_CHARACTER_PARTS = [
    {"shape": "capsule", "position": [0.0, 0.0, 0.48], "size": [0.34, 0.30, 0.96]},
    {"shape": "capsule", "position": [0.0, 0.0, 1.20], "size": [0.44, 0.34, 0.62]},
    {"shape": "uv_sphere", "position": [0.0, 0.0, 1.62], "size": [0.22, 0.22, 0.26]},
    {
        "shape": "cone",
        "position": [0.10, 0.0, 1.42],
        "size": [0.16, 0.16, 0.16],
        "rotation_deg": [0.0, 90.0, 0.0],
    },
]

_PALETTE = [
    [0.85, 0.38, 0.28],
    [0.35, 0.60, 0.85],
    [0.55, 0.78, 0.42],
    [0.86, 0.72, 0.30],
    [0.72, 0.45, 0.82],
    [0.38, 0.78, 0.74],
    [0.88, 0.55, 0.68],
    [0.62, 0.62, 0.68],
]


def slugify(text):
    """'Mauryl Gestaurien' -> 'mauryl_gestaurien'."""
    text = unicodedata.normalize("NFKD", str(text))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def colour_for(asset_id):
    """Deterministic per-character colour so proxies stay distinguishable."""
    return _PALETTE[sum(ord(c) for c in asset_id) % len(_PALETTE)]


def make_stub(
    shot_id,
    source_format,
    source_ref,
    duration_seconds,
    *,
    set_asset_id=None,
    characters=(),
    camera_source_text=(),
    fps=12,
    stage_size=(12.0, 12.0),
    notes="",
    continuity=None,
):
    """Build a needs_blocking shot spec."""
    return {
        "schema_version": SCHEMA_VERSION,
        "shot_id": shot_id,
        "status": "needs_blocking",
        "source": {"format": source_format, "ref": source_ref},
        "continuity": continuity or {"order": 0, "continues_from": None,
                                     "carry": {"position": True, "camera": "cut"}},
        "duration_seconds": round(float(duration_seconds), 2),
        "fps": fps,
        "set": {"asset_id": set_asset_id} if set_asset_id else {},
        "stage": {"size_m": list(stage_size), "ground_grid": True},
        "characters": list(characters),
        "props": [],
        "camera": {"lens_mm": 35, "moves": [], "_source_text": list(camera_source_text)},
        "render": {"engine": "WORKBENCH", "resolution": [960, 544], "fps": fps},
        "notes": notes,
    }


def make_character(object_id, asset_id, source_text=()):
    return {
        "id": object_id,
        "asset_id": asset_id,
        "actions": [],
        "_source_text": list(source_text),
    }


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return path


def write_stub(path, stub):
    """Write a stub, but never destroy hand-authored blocking.

    Re-importing a scene is routine — the source changes, or a new segment is
    added. Silently overwriting a blocked shot would throw away the one part of
    this pipeline a human actually made.
    """
    path = Path(path)
    if path.is_file():
        try:
            with path.open(encoding="utf-8") as handle:
                existing = json.load(handle)
        except (OSError, json.JSONDecodeError):
            existing = {}
        if existing.get("status") == "blocked":
            # Refresh the source-derived fields, keep everything authored.
            for key in ("continuity", "source", "duration_seconds",
                        "_beats", "_summary", "_scene", "_identity_locks",
                        "_acting", "_negatives"):
                if key in stub:
                    existing[key] = stub[key]
            write_json(path, existing)
            return path, "refreshed"
    write_json(path, stub)
    return path, "written"


def ensure_character_asset(assets_root, asset_id, display_name, notes="", source_ref=""):
    """Create a proxy stub for a character, but never clobber an authored one."""
    path = Path(assets_root) / "characters" / f"{asset_id}.json"
    if path.is_file():
        return path, False
    write_json(
        path,
        {
            "asset_id": asset_id,
            "kind": "character",
            "display_name": display_name,
            "notes": notes,
            "source_ref": source_ref,
            "color": colour_for(asset_id),
            "aim_height_m": 1.35,
            "parts": [dict(part) for part in _CHARACTER_PARTS],
        },
    )
    return path, True


def ensure_set_asset(
    assets_root, asset_id, display_name, notes="", source_ref="", reference_image=""
):
    """Create a set stub. Authored sets are reused untouched — a location is
    blocked out once and every later shot there picks up the same geometry."""
    path = Path(assets_root) / "sets" / f"{asset_id}.json"
    if path.is_file():
        return path, False
    write_json(
        path,
        {
            "asset_id": asset_id,
            "kind": "set",
            "display_name": display_name,
            "notes": notes
            or "Placeholder box room. Replace with a real blockout of this "
            "location before rendering anything you intend to use.",
            "source_ref": source_ref,
            "reference_image": reference_image,
            "needs_blockout": True,
            "color": [0.42, 0.44, 0.48],
            "parts": [
                {"shape": "box", "position": [0.0, 6.0, 1.6], "size": [12.0, 0.2, 3.2]},
                {"shape": "box", "position": [-6.0, 0.0, 1.6], "size": [0.2, 12.0, 3.2]},
                {"shape": "box", "position": [6.0, 0.0, 1.6], "size": [0.2, 12.0, 3.2]},
            ],
        },
    )
    return path, True
