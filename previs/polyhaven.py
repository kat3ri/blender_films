"""Poly Haven asset sourcing — host-side, no bpy, no third-party deps.

Poly Haven's API (https://github.com/Poly-Haven/Public-API) is fully open —
no auth, no API key, just a `User-Agent` header — and every asset is CC0:
public domain, free for any use including commercial, no attribution
required. Confirmed live against the real API before this module was
written, not assumed from memory.

This is the search/download half. The Blender-side half that imports a
downloaded file and flattens its materials to a flat colour lives in
`blender_api.py` (`_import_mesh_part`) — it needs `bpy`, so it can't live
here. Everything in this module runs on plain Python, host-side or inside
Blender equally, since it's just `urllib` + `json`.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

API_ROOT = "https://api.polyhaven.com"
DOWNLOAD_ROOT = "https://dl.polyhaven.org"
USER_AGENT = "previs-blender-films (github.com/kat3ri/blender_films)"
DEFAULT_RESOLUTION = "1k"  # plenty for a flat-shaded control-video proxy


def default_cache_root():
    """Where fetched assets are cached — outside the repo by default, same
    convention as the mocap cache (`PREVIS_MOCAP_CACHE`): third-party binary
    data doesn't belong in a git-tracked project folder."""
    root = os.environ.get("PREVIS_ASSET_CACHE")
    if root:
        return Path(root) / "polyhaven"
    return Path.home() / "previs_asset_cache" / "polyhaven"


def _get_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Poly Haven API request failed ({exc.code}): {url}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"could not reach Poly Haven API: {url} ({exc.reason})") from exc


def _download(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        dest.write_bytes(response.read())


def search(query, kind="models", limit=15):
    """Rank Poly Haven assets by how many query words their id/name/tags
    contain. No embeddings, no ML — plain substring scoring, matching the
    "pure retrieval, no model" spirit this whole system favours. Returns a
    list of dicts for a human (or me) to review and pick from — this never
    auto-selects a "best" match, the same way `previs mocap search` doesn't.
    """
    assets = _get_json(f"{API_ROOT}/assets?t={kind}")
    terms = [t.lower() for t in query.split() if t.strip()]
    if not terms:
        return []
    scored = []
    for asset_id, info in assets.items():
        # Exact word matches only. A substring-count fallback was tried first
        # and made things worse, not better: "bar" as a raw substring matches
        # "barrel" too, and since id+name+tags get concatenated (so "barrel"
        # legitimately repeats across near-duplicate tag entries), an asset
        # with *zero* real "bar" tags out-scored an actual bar_chair_round_01
        # match on the very first real query this ran against. Tags are
        # discrete words, so match them as words.
        words = re.findall(
            r"[a-z0-9]+",
            " ".join(
                [asset_id.lower(), str(info.get("name", "")).lower()]
                + [str(t).lower() for t in info.get("tags", [])]
            ),
        )
        score = sum(words.count(term) for term in terms)
        if score:
            scored.append((score, asset_id, info))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [
        {
            "id": asset_id,
            "name": info.get("name"),
            "tags": info.get("tags", []),
            "polycount": info.get("polycount"),
            "categories": info.get("categories", []),
        }
        for _, asset_id, info in scored[:limit]
    ]


def info(asset_id):
    """Full metadata for one asset, including licence/author (always CC0 for
    Poly Haven, but fetched live rather than hardcoded in case that ever
    changes for a specific asset)."""
    return _get_json(f"{API_ROOT}/info/{asset_id}")


def available_resolutions(asset_id):
    files = _get_json(f"{API_ROOT}/files/{asset_id}")
    return sorted(files.get("gltf", {}))


def fetch(asset_id, resolution=DEFAULT_RESOLUTION, cache_root=None):
    """Download one model's glTF plus every file it depends on (.bin,
    textures). Idempotent — files already on disk are not re-downloaded, so
    this is safe to call every time an asset is used, not just once.

    Returns the local path to the main .gltf file.
    """
    cache_root = Path(cache_root) if cache_root else default_cache_root()
    asset_dir = cache_root / asset_id

    files = _get_json(f"{API_ROOT}/files/{asset_id}")
    if "gltf" not in files:
        raise ValueError(f"{asset_id!r} has no gltf export available")
    if resolution not in files["gltf"]:
        available = sorted(files["gltf"])
        raise ValueError(
            f"{asset_id!r} has no {resolution!r} gltf; available: {available}"
        )

    manifest = files["gltf"][resolution]["gltf"]
    main_path = asset_dir / Path(manifest["url"]).name
    if not main_path.is_file():
        _download(manifest["url"], main_path)
    for rel_path, dep in manifest.get("include", {}).items():
        dest = asset_dir / rel_path
        if not dest.is_file():
            _download(dep["url"], dest)
    return main_path
