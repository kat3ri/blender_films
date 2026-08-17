"""Host-side launcher. Finds Blender, runs the driver, reports the result.

    python -m previs.cli render shots/examples/synth_test_shot.json
    python -m previs.cli validate
    python -m previs.cli info

This is the only piece the director agent needs to invoke; it never imports
``bpy`` and runs on the system Python.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DRIVER = PROJECT_ROOT / "previs" / "driver.py"
DEFAULT_RENDER_DIR = PROJECT_ROOT / "renders"
REGISTRY = PROJECT_ROOT / "projects.json"

# Searched newest-first when Blender is not on PATH and PREVIS_BLENDER is unset.
_WINDOWS_GLOBS = (
    "C:/Program Files/Blender Foundation/Blender */blender.exe",
    "C:/Program Files (x86)/Blender Foundation/Blender */blender.exe",
)


def mocap_library_default_fps():
    """SnapMoGen's default capture rate, resolved lazily so the module import
    stays cheap and reads no files unless a mocap command runs."""
    from previs import mocap_library

    return mocap_library.DEFAULT_SNAPMOGEN_FPS


def find_blender(explicit=None):
    """Locate a Blender executable, or raise with a useful message."""
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"--blender path does not exist: {path}")
        return path

    from_env = os.environ.get("PREVIS_BLENDER")
    if from_env and Path(from_env).is_file():
        return Path(from_env)

    on_path = shutil.which("blender")
    if on_path:
        return Path(on_path)

    candidates = []
    for pattern in _WINDOWS_GLOBS:
        root = Path(pattern).anchor
        relative = Path(pattern).relative_to(root)
        candidates.extend(Path(root).glob(str(relative)))
    if candidates:
        # "Blender 4.5" sorts after "Blender 4.4"; good enough for point releases.
        return sorted(candidates, key=lambda p: p.parent.name)[-1]

    raise FileNotFoundError(
        "Could not find Blender. Install it (winget install "
        "BlenderFoundation.Blender.LTS.4.5), put it on PATH, set PREVIS_BLENDER, "
        "or pass --blender."
    )


def camera_path(args):
    """Render the shot's actual computed camera trajectory as a visible
    trail, viewed from a static camera placed outside the shot -- the only
    way to see a camera path rather than infer it from what that camera
    itself renders."""
    blender = find_blender(args.blender)
    shot_path = Path(args.shot).resolve()
    if not shot_path.is_file():
        print(f"error: no such shot file: {shot_path}", file=sys.stderr)
        return 2

    default_name = f"{shot_path.stem}_campath_{args.mode}.png"
    output = (Path(args.out) if args.out else DEFAULT_RENDER_DIR / default_name).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    command = [
        str(blender), "--background", "--factory-startup", "--python", str(DRIVER),
        "--", str(shot_path), "--out", str(output), "--camera-path", args.mode,
    ]
    if args.assets:
        command += ["--assets", str(Path(args.assets).resolve())]

    print(f"[previs] blender     {blender}")
    result = subprocess.run(command)
    if result.returncode != 0:
        print(f"error: Blender exited with code {result.returncode}", file=sys.stderr)
        return result.returncode
    if not output.is_file():
        print(f"error: expected output not written: {output}", file=sys.stderr)
        return 1
    print(f"[previs] camera path view: {output}  ({output.stat().st_size / 1024:.0f} KB)")
    return 0


def bundle(args):
    """Export a full control-layer bundle for a downstream AI video generator:
    the reference render, a depth pass, exact camera_motion + pose landmarks,
    machine-readable metadata, and per-generator prompts — the producer side
    of a Blockout-style shot package, with camera and pose as ground truth
    rather than solved from footage."""
    blender = find_blender(args.blender)
    shot_path = Path(args.shot).resolve()
    if not shot_path.is_file():
        print(f"error: no such shot file: {shot_path}", file=sys.stderr)
        return 2

    with shot_path.open(encoding="utf-8") as handle:
        shot = json.load(handle)
    if shot.get("status") == "needs_blocking" and not args.allow_unblocked:
        print(
            f"error: {shot_path.name} is still status 'needs_blocking'.\n"
            "       Author its blocking first, or pass --allow-unblocked.",
            file=sys.stderr,
        )
        return 2

    shot_id = shot.get("shot_id", shot_path.stem)
    out_dir = (Path(args.out) if args.out else DEFAULT_RENDER_DIR / "bundles" / shot_id).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    command = [
        str(blender), "--background", "--factory-startup", "--python", str(DRIVER),
        "--", str(shot_path), "--out", str(out_dir), "--bundle",
    ]
    if args.assets:
        command += ["--assets", str(Path(args.assets).resolve())]
    if args.generators:
        command += ["--generators", args.generators]
    if getattr(args, "pano", None):
        command += ["--pano", args.pano]
    if args.no_depth:
        command.append("--no-depth")
    if args.no_pose:
        command.append("--no-pose")
    if args.no_stills:
        command.append("--no-stills")

    print(f"[previs] blender     {blender}")
    result = subprocess.run(command)
    if result.returncode != 0:
        print(f"error: Blender exited with code {result.returncode}", file=sys.stderr)
        return result.returncode

    manifest_path = out_dir / "bundle_manifest.json"
    if not manifest_path.is_file():
        print(f"error: bundle manifest not written: {manifest_path}", file=sys.stderr)
        return 1
    print(f"[previs] bundle: {out_dir}")
    return 0



def render(args):
    blender = find_blender(args.blender)
    shot_path = Path(args.shot).resolve()
    if not shot_path.is_file():
        print(f"error: no such shot file: {shot_path}", file=sys.stderr)
        return 2

    with shot_path.open(encoding="utf-8") as handle:
        shot = json.load(handle)
    if shot.get("status") == "needs_blocking" and not args.allow_unblocked:
        print(
            f"error: {shot_path.name} is still status 'needs_blocking'.\n"
            "       Author its blocking first (see the previs-blocking skill), "
            "or pass --allow-unblocked\n"
            "       to render the stub with a default wide static camera.",
            file=sys.stderr,
        )
        return 2

    default_name = f"{shot.get('shot_id', shot_path.stem)}"
    default_name += "_preview.png" if args.preview else ".mp4"
    output = Path(args.out) if args.out else DEFAULT_RENDER_DIR / default_name
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    command = [
        str(blender),
        "--background",
        "--factory-startup",
        "--python",
        str(DRIVER),
        "--",
        str(shot_path),
        "--out",
        str(output),
    ]
    if args.assets:
        command += ["--assets", str(Path(args.assets).resolve())]
    if args.preview is not None:
        command += ["--preview-frame", str(args.preview)]

    print(f"[previs] blender     {blender}")
    result = subprocess.run(command)
    if result.returncode != 0:
        print(f"error: Blender exited with code {result.returncode}", file=sys.stderr)
        return result.returncode
    if not output.is_file():
        print(f"error: expected output not written: {output}", file=sys.stderr)
        return 1

    size_kb = output.stat().st_size / 1024.0
    label = "preview frame" if args.preview is not None else "control video"
    print(f"[previs] {label}: {output}  ({size_kb:.0f} KB)")
    return 0


def load_registry():
    if not REGISTRY.is_file():
        return {}
    with REGISTRY.open(encoding="utf-8") as handle:
        return json.load(handle).get("projects", {})


def resolve_source(target, name):
    """Resolve (project-or-format, name-or-path) into (format, path, shots_dir).

    Accepts either a registry project name plus a short source name, or a raw
    format plus a full path, so the long-path form keeps working.
    """
    registry = load_registry()

    if target in registry:
        entry = registry[target]
        root = Path(entry["source_root"])
        if not root.is_dir():
            raise FileNotFoundError(
                f"project {target!r} points at a source_root that does not exist: {root}\n"
                f"       fix it in {REGISTRY}"
            )
        direct = Path(name)
        if direct.is_file():
            matches = [direct]
        else:
            matches = sorted(root.glob(entry.get("pattern", "{name}*").format(name=name)))
        if not matches:
            available = sorted(p.name for p in root.iterdir() if p.is_file())[:12]
            raise FileNotFoundError(
                f"no source matching {name!r} in {root}\n"
                f"       available: {', '.join(available)}"
            )
        shots_dir = PROJECT_ROOT / entry.get("shots_dir", f"shots/{target}")
        return entry["format"], matches[0].resolve(), shots_dir

    # Raw form: `import fortress <path>`
    path = Path(name)
    if not path.is_file():
        known = ", ".join(sorted(registry)) or "none configured"
        raise FileNotFoundError(
            f"no such source file: {path}\n"
            f"       (and {target!r} is not a known project; known projects: {known})"
        )
    return target, path.resolve(), PROJECT_ROOT / "shots" / target


def import_source(args):
    """Turn a story-pipeline file into needs_blocking shot stubs."""
    try:
        source_format, source, default_out = resolve_source(args.target, args.name)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    assets_root = Path(args.assets).resolve() if args.assets else PROJECT_ROOT / "assets"
    out_dir = Path(args.out).resolve() if args.out else default_out

    review_notes = []
    if source_format == "fortress":
        from previs.importers.fortress_importer import import_scene

        written, created = import_scene(source, out_dir, assets_root)
    else:
        from previs.importers.minimax_importer import import_episode

        written, created, review_notes = import_episode(source, out_dir, assets_root)

    if not written:
        print(f"error: no shots could be parsed out of {source}", file=sys.stderr)
        return 1

    print(f"[previs] imported {len(written)} shot stub(s) from {source.name} -> {out_dir}")
    for path in written:
        print(f"           {path.name}")
    if created:
        print(f"[previs] created {len(created)} placeholder asset(s):")
        for asset in created:
            print(f"           {asset}")
    for note in review_notes:
        print(f"[previs] NOTE     {note}")
    print(
        "\n[previs] These stubs have no blocking yet. Author positions, actions and\n"
        "         camera moves (see .claude/skills/previs-blocking), flip status to\n"
        "         'blocked', then render."
    )
    return 0


def _survey_move(args, span_x, span_y, radius, stand):
    """Pick the survey camera move.

    Default is a push from the open -Y end down the length of the space: it
    matches the viewpoint reference plates are usually shot from, so the survey
    is directly comparable to the image the set was built from, and it travels
    along the open axis so it cannot drive through furniture.
    """
    duration = float(args.duration)
    if args.mode == "orbit":
        return {
            "type": "orbit", "center_position": [0.0, 0.0, 0.0], "radius_m": radius,
            "start_deg": -90, "end_deg": 270, "height_m": args.height,
            "start_t": 0.0, "end_t": duration, "easing": "linear",
        }
    if args.mode == "pan":
        return {
            "type": "pan", "position": [stand[0], stand[1], args.height],
            "start_deg": -90, "end_deg": 270, "pitch_deg": -6.0,
            "start_t": 0.0, "end_t": duration, "easing": "linear",
        }
    return {
        "type": "dolly",
        "position": [stand[0], -(span_y - 0.4), args.height],
        "end_position": [stand[0], -(span_y - 3.0), args.height - 0.05],
        "target_position": [stand[0], span_y * 0.25, 1.25],
        "start_t": 0.0, "end_t": duration, "easing": "smooth",
    }


def survey(args):
    """Render a turntable of a set so you can see the space you just built.

    Sets are authored once and reused by every shot at that location, so it is
    worth looking at one on its own — with a human-scale figure standing in it —
    before blocking anything inside it.
    """
    import tempfile

    from previs.asset_library import AssetLibrary

    library = AssetLibrary(args.assets)
    if not library.exists("sets", args.set_id):
        available = ", ".join(library.list_assets("sets")) or "none"
        print(f"error: no set asset {args.set_id!r}\n       available: {available}",
              file=sys.stderr)
        return 2
    asset = library.get("sets", args.set_id)

    # Size the turntable from the set's own footprint.
    span_x = span_y = 6.0
    for part in asset.get("parts", []):
        position, size = part.get("position", [0, 0, 0]), part.get("size", [1, 1, 1])
        span_x = max(span_x, abs(position[0]) + size[0] / 2.0)
        span_y = max(span_y, abs(position[1]) + size[1] / 2.0)
    radius = args.radius or max(2.5, min(span_x, span_y) * 0.62)
    stand = list(args.camera_at) if args.camera_at else [0.0, 0.0]

    shot = {
        "schema_version": "0.1",
        "shot_id": f"SURVEY_{args.set_id.upper()}",
        "status": "blocked",
        "source": {"format": "manual", "ref": f"survey of set {args.set_id}"},
        "duration_seconds": float(args.duration),
        "fps": 12,
        "set": {"asset_id": args.set_id},
        "stage": {"size_m": [span_x * 2, span_y * 2], "ground_grid": True},
        "characters": [
            {
                "id": "scale_figure",
                "asset_id": args.figure,
                "start_position": list(args.figure_at) if args.figure_at else [0.0, 0.0, 0.0],
                "start_facing_deg": 90,
                "actions": [{"type": "idle", "start_t": 0.0, "end_t": float(args.duration)}],
            }
        ],
        "props": [],
        "camera": {"lens_mm": args.lens, "moves": [_survey_move(args, span_x, span_y, radius, stand)]},
        "render": {"engine": "WORKBENCH", "resolution": [960, 544], "fps": 12},
        "notes": "Auto-generated set survey.",
    }

    output = Path(args.out) if args.out else DEFAULT_RENDER_DIR / f"survey_{args.set_id}.mp4"
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as handle:
        json.dump(shot, handle, indent=2)
        temp_path = handle.name
    try:
        render_args = argparse.Namespace(
            shot=temp_path, out=str(output), assets=args.assets,
            blender=args.blender, allow_unblocked=True, preview=None,
        )
        how = {"orbit": f"orbit r={radius:.1f}m", "pan": f"pan from {stand}"}.get(
            args.mode, f"push down the room from y={-(span_y - 0.4):.1f}")
        print(f"[previs] survey      {args.set_id}: {how} at h={args.height}m, "
              f"scale figure at {shot['characters'][0]['start_position']}")
        return render(render_args)
    finally:
        Path(temp_path).unlink(missing_ok=True)


def asset_search(args):
    """List ranked Poly Haven candidates for a query. Never auto-picks one --
    same principle as `mocap search`: asset choice is a judgment call."""
    from previs import polyhaven

    try:
        results = polyhaven.search(args.query, kind=args.kind, limit=args.limit)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not results:
        print(f"no {args.kind} matched {args.query!r}")
        return 0
    for item in results:
        print(f"{item['id']:<28} poly={item['polycount']:<8} tags={item['tags'][:6]}")
    print(f"\n{len(results)} result(s). CC0 -- free for any use, no attribution required.")
    print(f"Fetch one with: previs asset-fetch-polyhaven <id> --kind <props|sets|characters> --as <name>")
    return 0


def asset_fetch(args):
    """Download a Poly Haven asset and write a ready-to-use asset JSON that
    references it -- the mesh-part equivalent of hand-writing a primitive
    asset def, just with the file already fetched and positioned at the
    origin, ready for position/rotation/scale tuning."""
    from previs import polyhaven

    try:
        gltf_path = polyhaven.fetch(args.asset_id, resolution=args.resolution)
        meta = polyhaven.info(args.asset_id)
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    name = args.as_name or args.asset_id.lower()
    asset = {
        "asset_id": name,
        "kind": args.kind.rstrip("s"),
        "display_name": meta.get("name", args.asset_id),
        "notes": f"Fetched from Poly Haven ({args.asset_id}), CC0. Real imported "
        "geometry -- position/rotation_deg/scale below are a starting guess, "
        "check with `previs render --preview` and adjust to fit the scene.",
        "source_ref": f"https://polyhaven.com/a/{args.asset_id}",
        "color": [0.6, 0.6, 0.6],
        "parts": [
            {
                "shape": "mesh",
                "file": str(gltf_path),
                "position": [0.0, 0.0, 0.0],
                "rotation_deg": [0.0, 0.0, 0.0],
                "scale": 1.0,
            }
        ],
    }
    out = Path(args.out) if args.out else PROJECT_ROOT / "assets" / args.kind / f"{name}.json"
    if out.is_file() and not args.force:
        print(f"error: {out} already exists (pass --force to overwrite)", file=sys.stderr)
        return 2
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(asset, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"[previs] fetched     {args.asset_id} -> {gltf_path}")
    print(f"[previs] wrote       {out}")
    print("[previs] scale/position are a starting guess -- preview and adjust before using in a shot.")
    return 0


def _collect_shot_paths(paths):
    if not paths:
        return sorted((PROJECT_ROOT / "shots").rglob("*.json"))
    collected = []
    for entry in paths:
        path = Path(entry)
        collected.extend(sorted(path.rglob("*.json")) if path.is_dir() else [path])
    return collected


def continuity_cmd(args):
    """Report or repair continuity breaks between chained shots."""
    from previs.asset_library import AssetLibrary
    from previs.continuity import apply_carry, camera_mode, compare, resolve_chain

    library = AssetLibrary(args.assets)
    chains = [c for c in resolve_chain(_collect_shot_paths(args.paths)) if len(c) > 1]
    if not chains:
        print("no multi-shot chains found (nothing has continuity.continues_from set)")
        return 0

    breaks = 0
    for chain in chains:
        names = " -> ".join(shot.get("shot_id", path.stem) for path, shot in chain)
        print(f"\nchain: {names}")

        for (prev_path, previous), (next_path, successor) in zip(chain, chain[1:]):
            label = f"  {previous.get('shot_id')} -> {successor.get('shot_id')}"
            if successor.get("status") != "blocked" or previous.get("status") != "blocked":
                print(f"{label}: skipped (both shots must be blocked)")
                continue

            if args.action == "apply":
                _, changes = apply_carry(successor, previous, library)
                if changes:
                    with Path(next_path).open("w", encoding="utf-8") as handle:
                        json.dump(successor, handle, indent=2, ensure_ascii=False)
                        handle.write("\n")
                    print(f"{label}: carried [camera={camera_mode(successor)}]")
                    for change in changes:
                        print(f"      {change}")
                else:
                    print(f"{label}: nothing to carry")
                continue

            issues = compare(previous, successor, library)
            hard = [i for i in issues if i["severity"] == "break"]
            breaks += len(hard)
            if not issues:
                print(f"{label}: ok [camera={camera_mode(successor)}]")
            else:
                print(f"{label}: [camera={camera_mode(successor)}]")
                for issue in issues:
                    marker = "BREAK" if issue["severity"] == "break" else "note "
                    print(f"      {marker} {issue['message']}")

    if args.action == "check":
        print(f"\n{breaks} continuity break(s)")
        if breaks:
            print("run 'previs continuity apply' to seed start states from the previous shot")
    return 1 if (args.action == "check" and breaks) else 0


def validate(args):
    from previs.schema import _main as validate_main

    # Accept directories as well as files, matching the other subcommands.
    return validate_main(["validate"] + [str(p) for p in _collect_shot_paths(args.paths)])


def info(args):
    try:
        blender = find_blender(args.blender)
    except FileNotFoundError as exc:
        print(f"blender: NOT FOUND ({exc})")
        return 1
    print(f"blender:      {blender}")
    print(f"project root: {PROJECT_ROOT}")
    print(f"renders:      {DEFAULT_RENDER_DIR}")
    result = subprocess.run(
        [str(blender), "--background", "--factory-startup", "--python-expr",
         "import bpy; print('version:      ' + bpy.app.version_string)"],
        capture_output=True,
        text=True,
    )
    for line in result.stdout.splitlines():
        if line.startswith("version:"):
            print(line)
    return 0


def mocap_inspect(args):
    """Inspect a BVH mocap clip and report mapping coverage for this rig."""
    from previs import mocap
    from previs import mocap_bvh

    cache_root = Path(args.cache).resolve() if args.cache else None
    try:
        clip_path = mocap.resolve_clip_path(
            args.clip_id,
            project_root=PROJECT_ROOT,
            cache_root=cache_root,
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        clip = mocap_bvh.load_bvh(clip_path)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"[previs] clip path    {clip_path}")
    print(f"[previs] root joint   {clip.root_joint}")
    print(f"[previs] joints       {len(clip.joints)}")
    print(f"[previs] frames       {clip.frame_count}")
    print(f"[previs] frame time   {clip.frame_time_s:.6f}s")
    print(f"[previs] duration     {clip.duration_seconds:.3f}s")

    mapping = mocap.canonical_joint_map()
    mapped = {}
    for source_name in clip.joints:
        target = mapping.get(source_name.lower())
        if target:
            mapped[source_name] = target
    unique_targets = sorted(set(mapped.values()))
    print(f"[previs] mapped       {len(mapped)} source joints -> "
          f"{len(unique_targets)} rig joints")
    if unique_targets:
        print(f"[previs] rig targets  {', '.join(unique_targets)}")

    if args.sample is not None:
        t = max(0.0, float(args.sample))
        sample = clip.sample_joint_rotations(min(t, clip.duration_seconds))
        mapped_sample = mocap.map_rotations(sample, mapping, source_up_axis=args.source_up)
        print(f"[previs] sample t={min(t, clip.duration_seconds):.3f}s "
              f"(source_up={args.source_up})")
        for joint_name in sorted(mapped_sample)[:20]:
            x, y, z = mapped_sample[joint_name]
            print(f"           {joint_name:>10}: [{x:7.2f}, {y:7.2f}, {z:7.2f}]")

    return 0


def mocap_search(args):
    """Search the SnapMoGen caption library in plain language.

    Prints ranked clip/frame-range candidates with their captions. Never
    auto-picks one -- like `asset-search-polyhaven`, choosing a performance is a
    judgment call. Copy a clip id into a shot yourself, or use `mocap-fetch` to
    print a ready-to-paste action for one result.
    """
    from previs import mocap_library

    cache_root = Path(args.cache).resolve() if args.cache else None
    try:
        entries = mocap_library.load_library(
            caption_path=args.captions, cache_root=cache_root
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    results = mocap_library.search(
        entries, args.query, limit=args.limit, min_frames=args.min_frames
    )
    if not results:
        print(f"no motion matched {args.query!r} in {len(entries)} caption entries")
        return 0

    for score, entry in results:
        secs = entry.duration_s(fps=args.fps)
        caption = entry.best_caption()
        if len(caption) > 140:
            caption = caption[:137] + "..."
        print(
            f"{entry.clip_id():<34} "
            f"f{entry.start_frame}-{entry.end_frame} ({secs:4.1f}s) "
            f"score={score:5.1f}"
        )
        print(f"    {caption}")
    print(
        f"\n{len(results)} result(s) of {len(entries)} entries. "
        f"Wire one into a shot with:\n"
        f"  previs mocap-fetch \"<clip_id>#<start>#<end>\"   "
        f"(prints a ready mocap_clip action)"
    )
    return 0


def mocap_fetch(args):
    """Print a ready-to-paste ``mocap_clip`` action for one caption entry.

    Takes a full caption key (``clip#start#end``, as shown by `mocap-search`) and
    resolves it into the clip_id + clip_t0_s/clip_t1_s a shot's actor action
    wants, at the clip's own frame rate when the BVH is present.
    """
    from previs import mocap
    from previs import mocap_bvh
    from previs import mocap_library

    cache_root = Path(args.cache).resolve() if args.cache else None
    parsed = mocap_library.parse_key(args.key)
    if not parsed:
        print(
            f"error: {args.key!r} is not a caption key of the form clip#start#end\n"
            "       (run 'previs mocap-search <query>' to find one)",
            file=sys.stderr,
        )
        return 2
    clip_name, start_frame, end_frame = parsed

    # Look up the entry so the printed action carries the caption as a comment.
    caption = ""
    try:
        entries = mocap_library.load_library(
            caption_path=args.captions, cache_root=cache_root
        )
        match = next(
            (
                e for e in entries
                if e.clip_name == clip_name
                and e.start_frame == start_frame
                and e.end_frame == end_frame
            ),
            None,
        )
        if match is None:
            match = next((e for e in entries if e.clip_name == clip_name), None)
        entry = match or mocap_library.LibraryEntry(
            key=args.key, clip_name=clip_name,
            start_frame=start_frame, end_frame=end_frame, captions=(),
        )
        caption = entry.best_caption()
    except FileNotFoundError:
        entry = mocap_library.LibraryEntry(
            key=args.key, clip_name=clip_name,
            start_frame=start_frame, end_frame=end_frame, captions=(),
        )

    # Prefer the clip's own frame rate if we can read the BVH.
    fps = args.fps
    try:
        clip_path = mocap.resolve_clip_path(
            entry.clip_id(), project_root=PROJECT_ROOT, cache_root=cache_root
        )
        clip = mocap_bvh.load_bvh(clip_path)
        if clip.frame_time_s > 0:
            fps = 1.0 / clip.frame_time_s
        print(f"[previs] clip file    {clip_path}")
        print(f"[previs] clip fps     {fps:.3f}  ({clip.frame_count} frames)")
    except (FileNotFoundError, ValueError) as exc:
        print(f"[previs] note         BVH not read ({exc}); using fps={fps}")

    action = mocap_library.entry_to_mocap_action(
        entry, fps=fps, root_mode=args.root_mode, loop=args.loop
    )
    if caption:
        print(f"[previs] caption      {caption}")
    print("\n// paste into a shot character's \"actions\": [ ... ]\n")
    print(json.dumps(action, indent=2))
    return 0



def pose_render(args):
    """Draw a bundle's exact pose data as an OpenPose-style skeleton video."""
    from .pose_render import render_from_bundle

    out = render_from_bundle(
        args.bundle, args.out,
        fps=args.fps, limb_width=args.limb_width, joint_radius=args.joint_radius,
    )
    print(f"[previs] wrote pose video -> {out}")
    return 0



def import_world(args):
    """ea_worlds room export -> a previs set asset."""
    import json as _json
    from .importers.ea_world import build_set_asset

    asset, report = build_set_asset(
        args.world, args.set_id,
        include_shell=not args.no_shell,
        display_name=args.display_name,
        pano_yaw_offset_deg=args.pano_yaw,
        mesh_lod=args.mesh,
        notes=args.notes,
    )
    root = Path(args.assets) if args.assets else (PROJECT_ROOT / "assets")
    out = root / "sets" / f"{args.set_id}.json"
    if out.exists() and not args.force:
        print(f"[previs] {out} exists; pass --force to overwrite")
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_json.dumps(asset, indent=2) + "\n", encoding="utf-8")

    print(f"[previs] wrote set  {out}")
    print(f"[previs] meshes     {len(report['placed'])}")
    if report["boxed_unreliable"]:
        print(f"[previs] boxed      {len(report['boxed_unreliable'])} unreliable fit(s): "
              + ", ".join(report["boxed_unreliable"]))
    if report["boxed_no_mesh"]:
        print(f"[previs] boxed      {len(report['boxed_no_mesh'])} without a mesh: "
              + ", ".join(report["boxed_no_mesh"]))
    print(f"[previs] shell      {report['shell'] or 'omitted'}")
    print(f"[previs] pano       {asset['pano']}  (yaw offset {asset['pano_yaw_offset_deg']}deg)")
    print("[previs] next       previs pano-check %s --yaw <deg> to calibrate the offset"
          % args.set_id)
    return 0


def pano_check(args):
    """Cut a pano plate at one yaw -- the calibration gate for pano_yaw_offset_deg.

    Compare against a blockout frame shot from the same yaw: if the plate and
    the render show the same corner of the room, the offset is right. Getting
    this wrong is the failure that is *confidently* wrong rather than obviously
    broken, so it is worth doing before anything downstream.
    """
    import json as _json
    import numpy as np
    from PIL import Image
    from .pano import fov_from_lens, render_plate

    root = Path(args.assets) if args.assets else (PROJECT_ROOT / "assets")
    set_path = root / "sets" / f"{args.set_id}.json"
    if not set_path.is_file():
        print(f"[previs] no set asset at {set_path}")
        return 1
    asset = _json.loads(set_path.read_text(encoding="utf-8"))
    pano_path = args.pano or asset.get("pano")
    if not pano_path:
        print(f"[previs] set {args.set_id!r} has no `pano` -- pass --pano")
        return 1

    offset = args.offset if args.offset is not None else asset.get("pano_yaw_offset_deg", 0.0)
    fov_x, _ = fov_from_lens(args.lens, args.width / float(args.height))
    pano = np.asarray(Image.open(pano_path).convert("RGB"))
    image = render_plate(pano, args.yaw + offset, args.pitch, fov_x, args.width, args.height)

    out = Path(args.out) if args.out else (PROJECT_ROOT / "renders"
                                           / f"panocheck_{args.set_id}_yaw{int(args.yaw)}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(out)
    print(f"[previs] pano       {pano_path}")
    print(f"[previs] yaw        {args.yaw}deg + offset {offset}deg = {args.yaw + offset}deg")
    print(f"[previs] wrote      {out}")
    print("[previs] compare    previs render <shot> --preview 1  with a camera at the same yaw")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="previs", description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    render_parser = subparsers.add_parser("render", help="render a shot's control video")
    render_parser.add_argument("shot")
    render_parser.add_argument("--out", default=None)
    render_parser.add_argument("--assets", default=None)
    render_parser.add_argument("--blender", default=None)
    render_parser.add_argument(
        "--allow-unblocked",
        action="store_true",
        help="render a needs_blocking stub with placeholder camera/blocking",
    )
    render_parser.add_argument(
        "--preview", type=int, nargs="?", const=1, default=None, metavar="FRAME",
        help="render one still frame (default: frame 1) plus a scene manifest, "
        "instead of the full animated video -- a fast sanity check before "
        "paying for the whole render",
    )
    render_parser.set_defaults(func=render)

    camera_path_parser = subparsers.add_parser(
        "camera-path", help="visualize a shot's camera trajectory from outside the shot"
    )
    camera_path_parser.add_argument("shot")
    camera_path_parser.add_argument("--mode", choices=("top", "angle"), default="angle")
    camera_path_parser.add_argument("--out", default=None)
    camera_path_parser.add_argument("--assets", default=None)
    camera_path_parser.add_argument("--blender", default=None)
    camera_path_parser.set_defaults(func=camera_path)

    bundle_parser = subparsers.add_parser(
        "bundle", help="export a full control-layer bundle for an AI video generator"
    )
    bundle_parser.add_argument("shot")
    bundle_parser.add_argument("--out", default=None, help="output bundle directory")
    bundle_parser.add_argument("--assets", default=None)
    bundle_parser.add_argument("--blender", default=None)
    bundle_parser.add_argument(
        "--pano", default=None,
        help="override the set's panorama for background plates")
    bundle_parser.add_argument(
        "--generators", default=None,
        help="comma-separated target generators (default: generic,seedance,minimax)",
    )
    bundle_parser.add_argument("--no-depth", action="store_true", help="skip the depth pass")
    bundle_parser.add_argument("--no-pose", action="store_true", help="skip pose landmark capture")
    bundle_parser.add_argument("--no-stills", action="store_true", help="skip mark stills")
    bundle_parser.add_argument(
        "--allow-unblocked", action="store_true",
        help="bundle a needs_blocking stub with placeholder camera/blocking",
    )
    bundle_parser.set_defaults(func=bundle)


    import_parser = subparsers.add_parser(
        "import", help="import shot stubs from a story-pipeline file"
    )
    import_parser.add_argument(
        "target", help="a project name from projects.json, or a raw format (fortress/minimax)"
    )
    import_parser.add_argument(
        "name", help="short source name (e.g. a1s1, EP01) or a full path"
    )
    import_parser.add_argument("--out", default=None, help="output directory for stubs")
    import_parser.add_argument("--assets", default=None)
    import_parser.set_defaults(func=import_source)

    asset_search_parser = subparsers.add_parser(
        "asset-search-polyhaven", help="search Poly Haven's CC0 asset catalogue"
    )
    asset_search_parser.add_argument("query")
    asset_search_parser.add_argument("--kind", default="models", choices=("models", "hdris", "textures"))
    asset_search_parser.add_argument("--limit", type=int, default=15)
    asset_search_parser.set_defaults(func=asset_search)

    asset_fetch_parser = subparsers.add_parser(
        "asset-fetch-polyhaven", help="download a Poly Haven asset and write its asset JSON"
    )
    asset_fetch_parser.add_argument("asset_id")
    asset_fetch_parser.add_argument("--kind", default="props", choices=("props", "sets", "characters"))
    asset_fetch_parser.add_argument("--as", dest="as_name", default=None, metavar="NAME")
    asset_fetch_parser.add_argument("--resolution", default="1k", choices=("1k", "2k", "4k"))
    asset_fetch_parser.add_argument("--out", default=None)
    asset_fetch_parser.add_argument("--force", action="store_true")
    asset_fetch_parser.set_defaults(func=asset_fetch)

    survey_parser = subparsers.add_parser(
        "survey", help="render a turntable of a set, with a figure in it for scale"
    )
    survey_parser.add_argument("set_id")
    survey_parser.add_argument("--out", default=None)
    survey_parser.add_argument("--assets", default=None)
    survey_parser.add_argument("--blender", default=None)
    survey_parser.add_argument("--duration", type=float, default=10.0)
    survey_parser.add_argument("--radius", type=float, default=None)
    survey_parser.add_argument("--height", type=float, default=1.7)
    survey_parser.add_argument("--lens", type=float, default=28.0)
    survey_parser.add_argument("--figure", default="generic_human")
    survey_parser.add_argument(
        "--mode", choices=("push", "pan", "orbit"), default="push",
        help="push down the space (default), pan in place, or orbit (clips indoors)",
    )
    survey_parser.add_argument(
        "--camera-at", nargs=2, type=float, default=None, metavar=("X", "Y"),
        help="where to stand the panning camera (default 0 0)",
    )
    survey_parser.add_argument(
        "--figure-at", nargs=2, type=float, default=None, metavar=("X", "Y"),
        help="where to stand the scale figure (default 0 0)",
    )
    survey_parser.set_defaults(func=survey)

    continuity_parser = subparsers.add_parser(
        "continuity", help="check or apply cross-shot continuity within a chain"
    )
    continuity_parser.add_argument(
        "action", choices=("check", "apply"), help="report breaks, or seed start states"
    )
    continuity_parser.add_argument(
        "paths", nargs="*", help="shot files or a directory (default: every shot in shots/)"
    )
    continuity_parser.add_argument("--assets", default=None)
    continuity_parser.set_defaults(func=continuity_cmd)

    validate_parser = subparsers.add_parser("validate", help="validate shot JSON files")
    validate_parser.add_argument("paths", nargs="*")
    validate_parser.set_defaults(func=validate)

    info_parser = subparsers.add_parser("info", help="show the resolved Blender install")
    info_parser.add_argument("--blender", default=None)
    info_parser.set_defaults(func=info)

    mocap_inspect_parser = subparsers.add_parser(
        "mocap-inspect", help="inspect a BVH clip and report joint-map coverage"
    )
    mocap_inspect_parser.add_argument("clip_id")
    mocap_inspect_parser.add_argument(
        "--sample", type=float, default=None,
        help="print mapped joint angles at this clip time (seconds)",
    )
    mocap_inspect_parser.add_argument(
        "--source-up", dest="source_up", default="y", choices=("y", "z"),
        help="source up-axis for frame conversion (y=SnapMoGen, z=rig-native)",
    )
    mocap_inspect_parser.add_argument(
        "--cache", default=None,
        help="override PREVIS_MOCAP_CACHE for clip resolution",
    )
    mocap_inspect_parser.set_defaults(func=mocap_inspect)

    mocap_search_parser = subparsers.add_parser(
        "mocap-search", help="search the SnapMoGen caption library in plain language"
    )
    mocap_search_parser.add_argument("query")
    mocap_search_parser.add_argument("--limit", type=int, default=15)
    mocap_search_parser.add_argument(
        "--min-frames", type=int, default=20,
        help="ignore caption ranges shorter than this many frames",
    )
    mocap_search_parser.add_argument(
        "--fps", type=float, default=mocap_library_default_fps(),
        help="frame rate used to report clip durations",
    )
    mocap_search_parser.add_argument(
        "--captions", default=None,
        help="path to all_caption_clean.json (default: <cache>/SnapMoGen/...)",
    )
    mocap_search_parser.add_argument(
        "--cache", default=None, help="override PREVIS_MOCAP_CACHE"
    )
    mocap_search_parser.set_defaults(func=mocap_search)

    mocap_fetch_parser = subparsers.add_parser(
        "mocap-fetch", help="print a ready mocap_clip action for a caption entry"
    )
    mocap_fetch_parser.add_argument(
        "key", help="caption key clip#start#end (from mocap-search)"
    )
    mocap_fetch_parser.add_argument(
        "--root-mode", default="lock_xy", choices=("lock_xy", "from_clip", "blend"),
        help="how the clip's root translation drives the actor",
    )
    mocap_fetch_parser.add_argument(
        "--loop", action="store_true", help="loop the caption range to fill the action"
    )
    mocap_fetch_parser.add_argument(
        "--fps", type=float, default=mocap_library_default_fps(),
        help="fallback frame rate if the BVH cannot be read",
    )
    mocap_fetch_parser.add_argument(
        "--captions", default=None, help="path to all_caption_clean.json"
    )
    mocap_fetch_parser.add_argument(
        "--cache", default=None, help="override PREVIS_MOCAP_CACHE"
    )
    mocap_fetch_parser.set_defaults(func=mocap_fetch)

    pose_parser = subparsers.add_parser(
        "pose-render",
        help="draw a bundle's pose_landmarks.json as an OpenPose-style video",
    )
    pose_parser.add_argument("bundle", help="a bundle directory (or any dir with pose_landmarks.json)")
    pose_parser.add_argument("--out", default=None, help="default: <bundle>/openpose_pose.mp4")
    pose_parser.add_argument("--fps", type=int, default=None, help="default: the pose file's fps")
    pose_parser.add_argument("--limb-width", type=int, default=4)
    pose_parser.add_argument("--joint-radius", type=int, default=3)
    pose_parser.set_defaults(func=pose_render)

    world_parser = subparsers.add_parser(
        "import-world", help="import an ea_worlds room export as a previs set")
    world_parser.add_argument("world", help="ea_worlds world dir (contains export/room.json)")
    world_parser.add_argument("set_id", help="asset id to write, e.g. wooden_lounge")
    world_parser.add_argument("--no-shell", action="store_true",
                              help="omit the room shell -- objects only")
    world_parser.add_argument("--display-name", default=None)
    world_parser.add_argument("--pano-yaw", type=float, default=0.0,
                              help="initial pano_yaw_offset_deg (calibrate with pano-check)")
    world_parser.add_argument("--mesh", default=None,
                              help="preferred mesh filename (default: v0_uv40k.glb)")
    world_parser.add_argument("--notes", default=None)
    world_parser.add_argument("--assets", default=None)
    world_parser.add_argument("--force", action="store_true")
    world_parser.set_defaults(func=import_world)

    panocheck_parser = subparsers.add_parser(
        "pano-check", help="cut one pano plate at a yaw, to calibrate pano_yaw_offset_deg")
    panocheck_parser.add_argument("set_id")
    panocheck_parser.add_argument("--yaw", type=float, required=True)
    panocheck_parser.add_argument("--pitch", type=float, default=0.0)
    panocheck_parser.add_argument("--offset", type=float, default=None,
                                  help="override the set's pano_yaw_offset_deg")
    panocheck_parser.add_argument("--lens", type=float, default=35.0)
    panocheck_parser.add_argument("--width", type=int, default=1024)
    panocheck_parser.add_argument("--height", type=int, default=576)
    panocheck_parser.add_argument("--pano", default=None)
    panocheck_parser.add_argument("--assets", default=None)
    panocheck_parser.add_argument("--out", default=None)
    panocheck_parser.set_defaults(func=pano_check)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
