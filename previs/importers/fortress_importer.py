"""Import the Fortress worldbuild reference format.

Source: ``.../fortress_in_the_eye_of_time/worldbuild/references/<scene>.json``
— a scene of chained segments, each with an explicit ``duration_seconds``,
``subjects``, ``location``, and a ``detailed_description`` broken into
``[Shot N]`` beats with ``At MM:SS.mmm`` offsets.

The beat structure and durations are exact, so they import cleanly. Positions
and camera moves do not exist in this format and are left for the blocking pass.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from . import (
    ensure_character_asset,
    ensure_set_asset,
    make_character,
    make_stub,
    slugify,
    write_stub,
)

# "[Shot 2] At 00:04.000, the shot cuts to ..."
_SHOT_MARKER = re.compile(r"\[Shot\s+(\d+)\]\s*", re.IGNORECASE)
_TIMECODE = re.compile(r"^At\s+(\d+):(\d+(?:\.\d+)?)\s*,?\s*", re.IGNORECASE)

# Camera language worth surfacing to whoever blocks the shot.
_CAMERA_HINT = re.compile(
    r"(camera|shot|close-?up|wide|medium|eye-?level|pans?|tilts?|tracks?|"
    r"dollys?|orbits?|static|angle|perspective|frame)",
    re.IGNORECASE,
)


def split_beats(description, duration):
    """Split a detailed_description into timed [Shot N] beats."""
    if not description:
        return []

    pieces = _SHOT_MARKER.split(description)
    # split() yields [preamble, num, text, num, text, ...]
    beats = []
    for index in range(1, len(pieces) - 1, 2):
        text = pieces[index + 1].strip()
        start_t = None
        match = _TIMECODE.match(text)
        if match:
            start_t = int(match.group(1)) * 60 + float(match.group(2))
            text = text[match.end():].strip()
        beats.append({"shot": int(pieces[index]), "start_t": start_t, "text": text})

    if not beats:
        return [{"start_t": 0.0, "end_t": float(duration), "text": description.strip()}]

    # The first beat is implicitly at 0; fill any other gaps by even division.
    if beats[0]["start_t"] is None:
        beats[0]["start_t"] = 0.0
    for index, beat in enumerate(beats):
        if beat["start_t"] is None:
            beat["start_t"] = duration * index / len(beats)

    for index, beat in enumerate(beats):
        beat["end_t"] = (
            beats[index + 1]["start_t"] if index + 1 < len(beats) else float(duration)
        )
        beat.pop("shot", None)
    return beats


def _world_bible_root(scene_path):
    """Locate the sibling world_bible_generated tree, if this scene has one."""
    for parent in Path(scene_path).resolve().parents:
        candidate = parent / "world_bible_generated"
        if candidate.is_dir():
            return candidate
    return None


def _find_bible_entry(bible_root, kind, name):
    """Match 'Mauryl' to mauryl_gestaurien.md — exact slug first, then prefix."""
    if not bible_root:
        return None
    directory = bible_root / kind
    if not directory.is_dir():
        return None
    slug = slugify(name)
    exact = directory / f"{slug}.md"
    if exact.is_file():
        return exact
    matches = sorted(p for p in directory.glob(f"{slug}*.md"))
    return matches[0] if matches else None


def _visual_signature(path):
    """Pull the 'Visual signature' section out of a world-bible entry."""
    if not path or not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8")
    match = re.search(
        r"##\s*Visual signature\s*\n+(.+?)(?=\n##\s|\Z)", text, re.DOTALL | re.IGNORECASE
    )
    return " ".join(match.group(1).split()) if match else ""


def import_scene(scene_path, out_dir, assets_root, stage_size=(12.0, 12.0)):
    """Import every segment of a Fortress scene. Returns written shot paths."""
    scene_path = Path(scene_path)
    with scene_path.open(encoding="utf-8") as handle:
        scene = json.load(handle)

    bible_root = _world_bible_root(scene_path)
    scene_id = scene.get("scene_id", scene_path.stem)
    written, created_assets = [], []

    segments = sorted(scene.get("segments", []), key=lambda s: s.get("order", 0))
    previous_id = None

    for segment in segments:
        segment_id = segment.get("segment_id", f"{scene_id}_seg")
        duration = float(segment.get("duration_seconds", 10.0))
        beats = split_beats(segment.get("detailed_description", ""), duration)

        # Set / location
        location = segment.get("location", "")
        set_asset_id = None
        if location:
            set_asset_id = slugify(location.split(",")[0])
            entry = _find_bible_entry(bible_root, "locations", location.split(",")[0])
            _, created = ensure_set_asset(
                assets_root,
                set_asset_id,
                location,
                notes=_visual_signature(entry),
                source_ref=str(entry) if entry else "",
            )
            if created:
                created_assets.append(f"sets/{set_asset_id}")

        # Characters
        characters = []
        for subject in segment.get("subjects", []):
            object_id = slugify(subject)
            entry = _find_bible_entry(bible_root, "characters", subject)
            asset_id = entry.stem if entry else object_id
            _, created = ensure_character_asset(
                assets_root,
                asset_id,
                subject,
                notes=_visual_signature(entry),
                source_ref=str(entry) if entry else "",
            )
            if created:
                created_assets.append(f"characters/{asset_id}")
            mentions = [
                {"start_t": b["start_t"], "end_t": b["end_t"], "text": b["text"]}
                for b in beats
                if subject.lower() in b["text"].lower()
            ]
            characters.append(make_character(object_id, asset_id, mentions or beats))

        camera_text = [
            {"start_t": b["start_t"], "end_t": b["end_t"], "text": b["text"]}
            for b in beats
            if _CAMERA_HINT.search(b["text"])
        ]

        # The format states chaining outright: is_chain_start plus a 'reason'
        # naming continuation. A chained segment is a genuine video continuation,
        # so its camera must not jump; a chain start is a fresh setup.
        chain_start = bool(segment.get("is_chain_start", False))
        continues_from = None if chain_start else previous_id
        stub = make_stub(
            segment_id.upper(),
            "fortress",
            f"{scene_path.name}#{segment_id}",
            duration,
            set_asset_id=set_asset_id,
            characters=characters,
            camera_source_text=camera_text or [
                {"start_t": 0.0, "end_t": duration, "text": segment.get("summary", "")}
            ],
            stage_size=stage_size,
            notes=f"Imported from {scene_path.name} segment {segment_id}. "
            "Blocking and camera moves still to be authored.",
            continuity={
                "order": int(segment.get("order", 0)),
                "chain_start": chain_start,
                "continues_from": continues_from,
                "carry": {
                    "position": continues_from is not None,
                    "camera": "match" if continues_from else "cut",
                },
                "_reason": segment.get("reason", ""),
            },
        )
        stub["_beats"] = beats
        stub["_summary"] = segment.get("summary", "")

        path, _ = write_stub(Path(out_dir) / f"{segment_id}.json", stub)
        written.append(path)
        previous_id = segment_id.upper()

    return written, created_assets
