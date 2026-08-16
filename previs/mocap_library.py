"""SnapMoGen caption-library search, host-side and Blender-safe (no bpy).

The SnapMoGen dataset ships two pieces:

- ``renamed_bvhs/*.bvh`` -- the motion clips themselves.
- ``all_caption_clean.json`` -- natural-language captions keyed by a clip and a
  frame range, e.g. ``"ep1_00000#0#281"`` describes frames 0..281 of clip
  ``ep1_00000``.

This module turns that JSON into a searchable index so a director can ask for
"picks something up off the floor" instead of memorising clip ids, and maps a
chosen caption entry into the ``clip_id`` + ``clip_t0_s``/``clip_t1_s`` fields a
``mocap_clip`` action wants.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from . import mocap

# SnapMoGen BVH is captured/retargeted at 30 fps. Individual clips carry their
# own Frame Time; this is only the fallback used to turn a caption's frame range
# into seconds when the BVH itself has not been consulted.
DEFAULT_SNAPMOGEN_FPS = 30.0

# Where the caption JSON and BVH folder live inside the cache root.
DATASET_SUBDIR = "SnapMoGen"
CAPTION_FILENAME = "all_caption_clean.json"
BVH_SUBDIR = "renamed_bvhs"

_KEY_RE = re.compile(r"^(?P<clip>.+)#(?P<start>\d+)#(?P<end>\d+)$")
_WORD_RE = re.compile(r"[a-z0-9']+")

# Words too common to be worth scoring; keeps "the person walks" from matching
# every clip via "the"/"person".
_STOPWORDS = frozenset(
    """a an and the of to in on with her his their they them she he it its at as
    is are was were be been being for from into onto over under by then this that
    person individual man woman figure someone body slowly quickly while then""".split()
)


@dataclass
class LibraryEntry:
    """One caption-annotated frame range of one clip."""

    key: str
    clip_name: str
    start_frame: int
    end_frame: int
    captions: tuple

    @property
    def frame_span(self):
        return max(0, self.end_frame - self.start_frame)

    def duration_s(self, fps=DEFAULT_SNAPMOGEN_FPS):
        return self.frame_span / float(fps or DEFAULT_SNAPMOGEN_FPS)

    def clip_id(self, cache_relative=True):
        """The clip id a shot's ``mocap_clip`` action should reference.

        Returns a cache-relative id like ``SnapMoGen/renamed_bvhs/ep1_00000`` so
        ``mocap.resolve_clip_path`` finds it under the cache root with no config.
        """
        if cache_relative:
            return f"{DATASET_SUBDIR}/{BVH_SUBDIR}/{self.clip_name}"
        return self.clip_name

    def best_caption(self):
        return self.captions[0] if self.captions else ""


def default_caption_path(cache_root=None):
    cache_root = Path(cache_root) if cache_root else mocap.default_mocap_cache_root()
    return cache_root / DATASET_SUBDIR / CAPTION_FILENAME


def default_bvh_root(cache_root=None):
    cache_root = Path(cache_root) if cache_root else mocap.default_mocap_cache_root()
    return cache_root / DATASET_SUBDIR / BVH_SUBDIR


def parse_key(key):
    """Split ``"ep1_00000#0#281"`` into (clip_name, start_frame, end_frame).

    Also accepts a key whose clip portion carries the cache-relative prefix
    (``"SnapMoGen/renamed_bvhs/ep1_00000#0#281"``, as printed by `mocap-search`)
    and strips it back to the bare clip name the caption JSON is keyed by.
    """
    match = _KEY_RE.match(key)
    if not match:
        return None
    clip = match.group("clip").replace("\\", "/")
    prefix = f"{DATASET_SUBDIR}/{BVH_SUBDIR}/"
    if clip.startswith(prefix):
        clip = clip[len(prefix):]
    clip = clip.rsplit("/", 1)[-1]
    return (
        clip,
        int(match.group("start")),
        int(match.group("end")),
    )


def _entry_captions(value):
    """Flatten the gpt/manual caption lists of one JSON value into a tuple."""
    if isinstance(value, str):
        return (value,)
    captions = []
    if isinstance(value, dict):
        for group in ("manual", "gpt"):
            for item in value.get(group, []) or []:
                if isinstance(item, str) and item.strip():
                    captions.append(item.strip())
        # Any other list-valued fields (defensive; schema may evolve).
        for field, items in value.items():
            if field in ("manual", "gpt"):
                continue
            if isinstance(items, list):
                captions.extend(s.strip() for s in items if isinstance(s, str) and s.strip())
    elif isinstance(value, list):
        captions.extend(s.strip() for s in value if isinstance(s, str) and s.strip())
    return tuple(captions)


def load_library(caption_path=None, cache_root=None):
    """Load the caption JSON into a list of :class:`LibraryEntry`."""
    path = Path(caption_path) if caption_path else default_caption_path(cache_root)
    if not path.is_file():
        raise FileNotFoundError(
            f"SnapMoGen caption file not found: {path}\n"
            "Download all_caption_clean.json from "
            "https://huggingface.co/datasets/Ericguo5513/SnapMoGen into the cache."
        )
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)

    entries = []
    for key, value in raw.items():
        parsed = parse_key(key)
        if not parsed:
            continue
        clip_name, start_frame, end_frame = parsed
        entries.append(
            LibraryEntry(
                key=key,
                clip_name=clip_name,
                start_frame=start_frame,
                end_frame=end_frame,
                captions=_entry_captions(value),
            )
        )
    return entries


def _tokens(text):
    return [w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS]


def score_entry(entry, query_tokens):
    """Rank one entry against query tokens.

    Coverage (how many distinct query words appear at all) dominates, so a clip
    that mentions every requested word beats one that mentions a single word many
    times; frequency and phrase adjacency break ties.
    """
    if not query_tokens:
        return 0.0
    haystacks = [c.lower() for c in entry.captions]
    if not haystacks:
        return 0.0

    covered = 0
    frequency = 0
    for token in query_tokens:
        hits = sum(h.count(token) for h in haystacks)
        if hits:
            covered += 1
            frequency += hits

    if not covered:
        return 0.0

    coverage_ratio = covered / len(query_tokens)
    phrase = " ".join(query_tokens)
    phrase_bonus = 3.0 if any(phrase in h for h in haystacks) else 0.0
    # Normalise frequency so a very long caption set does not dominate.
    freq_component = min(2.0, frequency / (len(haystacks) + 1.0))
    return coverage_ratio * 10.0 + phrase_bonus + freq_component


def search(entries, query, limit=15, min_frames=1):
    """Return up to ``limit`` (score, entry) pairs best matching ``query``."""
    query_tokens = _tokens(query)
    scored = []
    for entry in entries:
        if entry.frame_span < min_frames:
            continue
        score = score_entry(entry, query_tokens)
        if score > 0.0:
            scored.append((score, entry))
    # Prefer higher score, then longer (more usable) ranges for equal scores.
    scored.sort(key=lambda pair: (pair[0], pair[1].frame_span), reverse=True)
    return scored[:limit]


def entry_to_mocap_action(entry, actor_start_t=0.0, actor_end_t=None,
                          fps=DEFAULT_SNAPMOGEN_FPS, root_mode="lock_xy",
                          loop=False):
    """Build a ``mocap_clip`` action dict for a chosen library entry.

    The caption's frame range becomes ``clip_t0_s``/``clip_t1_s`` so only the
    described slice of the source BVH plays. ``actor_end_t`` defaults to the
    clip's own duration so the motion plays at natural speed.
    """
    fps = float(fps or DEFAULT_SNAPMOGEN_FPS)
    clip_t0 = entry.start_frame / fps
    clip_t1 = entry.end_frame / fps
    duration = clip_t1 - clip_t0
    if actor_end_t is None:
        actor_end_t = actor_start_t + max(0.1, duration)

    action = {
        "type": "mocap_clip",
        "clip_id": entry.clip_id(),
        "start_t": round(float(actor_start_t), 3),
        "end_t": round(float(actor_end_t), 3),
        "clip_t0_s": round(clip_t0, 3),
        "clip_t1_s": round(clip_t1, 3),
        "source_fps": fps,
        "source_up_axis": "y",
        "root_mode": root_mode,
        "blend_in_s": 0.15,
        "blend_out_s": 0.15,
        "pose_weight": 1.0,
    }
    if loop:
        action["clip_loop_from_s"] = round(clip_t0, 3)
        action["clip_loop_to_s"] = round(clip_t1, 3)
    return action
