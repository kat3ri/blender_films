"""Import the MiniMax H3 prompt-package format.

Source: ``D:/story_framework/06_prompts/minimax/<EPISODE>.md`` — one ``## Job N``
per clip, each with a fenced block containing ``[SHOT LIST]`` ranges
("0-4s - cold open wide on..."), plus ``[CAMERA]``, ``[SCENE]`` and
``[IDENTITY / CONTINUITY LOCKS]`` sections, and a Reference Map table naming the
location plate.

Durations, beat ranges and the location plate import exactly. Character
identities are *guessed* from the continuity-locks prose and flagged for review
— that block is written for a human, not a parser.
"""

from __future__ import annotations

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

_JOB_HEADING = re.compile(r"^##\s+Job\s+(\d+)\s*[—–-]\s*(.+?)\s*$", re.MULTILINE)
_FENCE = re.compile(r"```(?:text)?\s*\n(.*?)```", re.DOTALL)
_SECTION = re.compile(r"^\[([A-Z][A-Z /]+)\]\s*$", re.MULTILINE)
# "0–4s — cold open wide on..." (en dash, em dash or hyphen, either position)
_RANGE = re.compile(r"^\s*(\d+)\s*[–—-]\s*(\d+)\s*s\s*[–—-]\s*(.*)$")
_LOCATION_PLATE = re.compile(r"`(07_visual_refs/locations/([A-Za-z0-9_\-]+)\.png)`")

# Deliberately loose: the locks block is prose. Anything these miss gets caught
# during blocking, where the full text is available verbatim.
_IDENTITY_PATTERNS = (
    re.compile(r"\ban?\s+unnamed\s+([a-z][a-z \-]{2,28}?)(?=[,;.]|\s+—|\s+shown|\s+wrapped)"),
    re.compile(r"\bSame\s+([a-z][a-z \-]{2,28}?)\s+as\s+Job\b"),
    re.compile(r"\bexactly\s+one\s+(?:visible\s+|named\s+)?([a-z][a-z \-]{2,28}?)(?=[,;.]|\s+—|\s+and\b)"),
    re.compile(r"\bOne\s+([a-z][a-z \-]{2,24}?)(?=[,;.])"),
)
# "continuing directly from Job 1" / "Same as Job 3 —" / "after Job 6" /
# "continuing from Job 3's knock". The phrasing varies per episode, so keep the
# alternatives broad; a false positive only mis-sets a carry flag that blocking
# can override, while a miss silently loses continuity.
_CONTINUES_FROM = re.compile(
    r"continu\w*\s+(?:directly\s+)?from\s+Job\s+(\d+)"
    r"|^Same\s+as\s+Job\s+(\d+)"
    r"|\b(?:after|following)\s+Job\s+(\d+)",
    re.IGNORECASE | re.MULTILINE,
)
# A job only inherits the camera if it says the setup itself carries over.
# MiniMax jobs are separate generations, so the default is a cut.
_CAMERA_HOLDS = re.compile(
    r"(same\s+(?:camera\s+)?setup|camera\s+(?:remains|holds|continues)\b.*\bfrom\s+Job|"
    r"unbroken\s+(?:held\s+)?composition\s+continuing)",
    re.IGNORECASE,
)

_IDENTITY_STOPWORDS = {
    "person",
    "figure",
    "human figure",
    "other figures",
    "no other figures",
    "subject",
    "named subject",
    "visible subject",
}


def _split_sections(block):
    """Turn a fenced prompt block into {SECTION NAME: text}."""
    parts = _SECTION.split(block)
    sections = {}
    for index in range(1, len(parts) - 1, 2):
        sections[parts[index].strip()] = parts[index + 1].strip()
    return sections


def parse_shot_list(text):
    """Parse '0-4s - ...' ranges, rejoining wrapped continuation lines."""
    beats = []
    for line in text.splitlines():
        match = _RANGE.match(line)
        if match:
            beats.append(
                {
                    "start_t": float(match.group(1)),
                    "end_t": float(match.group(2)),
                    "text": match.group(3).strip(),
                }
            )
        elif beats and line.strip():
            beats[-1]["text"] += " " + line.strip()
    for beat in beats:
        beat["text"] = " ".join(beat["text"].split())
    return beats


def guess_identities(locks_text):
    """Best-effort character names from the continuity-locks prose."""
    found = []
    for pattern in _IDENTITY_PATTERNS:
        for match in pattern.finditer(locks_text):
            name = " ".join(match.group(1).split()).strip(" -")
            # "bartender and a few background patrons" is two roles in one
            # phrase; keep the first, which is always the named one.
            name = re.split(r"\band\b", name)[0].strip()
            name = re.sub(r"^(the|a|an)\s+", "", name)
            if not name or name in _IDENTITY_STOPWORDS or len(name.split()) > 3:
                continue
            if name not in found:
                found.append(name)
    return found[:4]


def canonicalise_identities(all_names):
    """Collapse the same person named differently across jobs.

    Job 1 says "an unnamed night cleaner", Job 2 says "Same cleaner as Job 1".
    Both mean one character, so the shorter name folds into the longer one.
    """
    unique = sorted(set(all_names), key=lambda n: (-len(n.split()), n))
    mapping = {}
    for name in unique:
        canonical = name
        for longer in unique:
            if longer == name:
                continue
            if longer.endswith(" " + name) or longer == name:
                canonical = longer
                break
        mapping[name] = canonical
    return mapping


def _repo_root(path):
    """Walk up to the story_framework root that holds 07_visual_refs."""
    for parent in Path(path).resolve().parents:
        if (parent / "07_visual_refs").is_dir():
            return parent
    return None


def import_episode(episode_path, out_dir, assets_root, stage_size=(12.0, 12.0)):
    """Import every job in a MiniMax episode. Returns written shot paths."""
    episode_path = Path(episode_path)
    text = episode_path.read_text(encoding="utf-8")
    root = _repo_root(episode_path)
    episode_id = episode_path.stem.split("_")[0].upper()

    headings = list(_JOB_HEADING.finditer(text))
    written, created_assets, review_notes = [], [], []

    # --- pass 1: parse every job -----------------------------------------
    jobs, previous_set = [], None
    for index, heading in enumerate(headings):
        start = heading.end()
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        body = text[start:end]
        job_number, job_title = heading.group(1), heading.group(2)

        fence = _FENCE.search(body)
        if not fence:
            continue
        sections = _split_sections(fence.group(1))
        beats = parse_shot_list(sections.get("SHOT LIST", ""))
        if not beats:
            continue

        # Location plate — jobs that reuse the previous plate say so in prose.
        plate = _LOCATION_PLATE.search(body)
        reference_image = ""
        if plate:
            set_asset_id = plate.group(2)
            reference_image = str(root / plate.group(1)) if root else plate.group(1)
            if root and not (root / plate.group(1)).is_file():
                reference_image = ""
                review_notes.append(
                    f"Job {job_number}: reference plate {plate.group(1)} is missing on disk"
                )
            previous_set = set_asset_id
        else:
            set_asset_id = previous_set

        # Continuation is stated in prose, not metadata, so read it from the
        # sections that talk about the previous job.
        continuation_text = " ".join(
            [body[: fence.start()]]
            + [sections.get(key, "") for key in
               ("REFERENCE USE", "IDENTITY / CONTINUITY LOCKS", "SCENE")]
        )
        continues_match = _CONTINUES_FROM.search(continuation_text)
        continues_job = None
        if continues_match:
            continues_job = next(g for g in continues_match.groups() if g)

        jobs.append(
            {
                "number": job_number,
                "title": job_title,
                "sections": sections,
                "beats": beats,
                "duration": max(beat["end_t"] for beat in beats),
                "set_asset_id": set_asset_id,
                "reference_image": reference_image,
                "names": guess_identities(sections.get("IDENTITY / CONTINUITY LOCKS", "")),
                "continues_job": continues_job,
                "camera_holds": bool(_CAMERA_HOLDS.search(sections.get("CAMERA", ""))),
                "continuation_note": continues_match.group(0).strip() if continues_match else "",
            }
        )

    # --- pass 2: one identity per person across the whole episode ---------
    canonical = canonicalise_identities([n for job in jobs for n in job["names"]])

    # --- pass 3: write assets and stubs ----------------------------------
    for job in jobs:
        job_number = job["number"]
        sections, beats = job["sections"], job["beats"]
        duration, set_asset_id = job["duration"], job["set_asset_id"]

        if set_asset_id:
            _, created = ensure_set_asset(
                assets_root,
                set_asset_id,
                set_asset_id.replace("_", " ").title(),
                source_ref=f"{episode_path.name}#job{job_number}",
                reference_image=job["reference_image"],
            )
            if created:
                created_assets.append(f"sets/{set_asset_id}")

        locks = sections.get("IDENTITY / CONTINUITY LOCKS", "")
        characters, seen = [], set()
        for raw_name in job["names"]:
            name = canonical.get(raw_name, raw_name)
            object_id = slugify(name)
            if object_id in seen:
                continue
            seen.add(object_id)
            _, created = ensure_character_asset(
                assets_root,
                object_id,
                name.title(),
                notes=f"Identity guessed from continuity locks: {locks}",
                source_ref=f"{episode_path.name}#job{job_number}",
            )
            if created:
                created_assets.append(f"characters/{object_id}")
            mentions = [
                {"start_t": b["start_t"], "end_t": b["end_t"], "text": b["text"]}
                for b in beats
                if name.split()[-1].lower() in b["text"].lower()
            ]
            characters.append(make_character(object_id, object_id, mentions or beats))

        job_title = job["title"]
        shot_id = f"{episode_id}_JOB{int(job_number):02d}"
        camera_text = [{"start_t": 0.0, "end_t": duration, "text": sections["CAMERA"]}] \
            if sections.get("CAMERA") else []
        camera_text += [
            {"start_t": b["start_t"], "end_t": b["end_t"], "text": b["text"]} for b in beats
        ]

        continues_from = (
            f"{episode_id}_JOB{int(job['continues_job']):02d}"
            if job["continues_job"]
            else None
        )
        stub = make_stub(
            shot_id,
            "minimax",
            f"{episode_path.name}#job{job_number}",
            duration,
            set_asset_id=set_asset_id,
            characters=characters,
            camera_source_text=camera_text,
            stage_size=stage_size,
            notes=f"Imported from {episode_path.name} Job {job_number} ({job_title}). "
            "Character identities are guessed from the continuity-locks prose — "
            "verify them during blocking.",
            continuity={
                "order": int(job_number) - 1,
                "chain_start": continues_from is None,
                "continues_from": continues_from,
                "carry": {
                    "position": continues_from is not None,
                    # Each job is a separate generation with its own setup, so a
                    # continuing scene still cuts unless the prose says the
                    # camera itself carries over.
                    "camera": "match" if (continues_from and job["camera_holds"]) else "cut",
                },
                "_reason": job["continuation_note"],
            },
        )
        stub["_beats"] = beats
        stub["_scene"] = sections.get("SCENE", "")
        stub["_identity_locks"] = locks
        stub["_acting"] = sections.get("ACTING", "")
        stub["_negatives"] = sections.get("NEGATIVES", "")

        path, _ = write_stub(Path(out_dir) / f"{shot_id}.json", stub)
        written.append(path)

    return written, created_assets, review_notes
